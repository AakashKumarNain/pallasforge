"""Implements matmul for bf16 matrices using Pallas on Hopper devices.
Showcases how to profile and optimize the same. For auto-tuning we use `tune-jax`
package.
"""

import jax
import argparse
import jax.numpy as jnp
from functools import partial
from jax.extend import backend
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from pallasforge.common import get_max_smem_bytes


def matmul(
    lhs,
    rhs,
    tile_m,
    tile_n,
    tile_k,
    num_pipeline_stages,
    panel_width,
    is_persistent,
):
    """Kernel for matrix multiplication of two matrices with bfloat16 dtype.

    Args:
        lhs: Left-hand operand of shape (M, K) with bf16 dtype.
        rhs: Right-hand operand of shape (K, N) with bf16 dtype.
        tile_m: Number of rows in the output tile computed by the kernel (M dimension).
        tile_n: Number of columns in the output tile computed by the kernel (N dimension).
        tile_k: Size of the reduction step (K dimension).
        num_pipeline_stages: Number of reduction stages staged concurrently in SMEM.
        panel_width: Width (in tiles) of a vertical panel for swizzled L2 cache reuse.
        is_persistent: If enabled, persistent GPU programs iterate over multiple output tiles.

    Returns:
        output of lhs @ rhs of bfloat16 dtype with shape (M, N).
    """

    m, k = lhs.shape
    k_rhs, n = rhs.shape

    # Some sanity checks: Input shapes and dtype validation
    if lhs.dtype != jnp.bfloat16 or rhs.dtype != jnp.bfloat16:
        raise ValueError("Inputs must be of dtype `jnp.bfloat16`")
    if k != k_rhs:
        raise ValueError(
            f"Reduction dimension must match. Got {k} for LHS and {k_rhs} for RHS"
        )

    if (
        tile_m <= 0
        or tile_n <= 0
        or tile_k <= 0
        or num_pipeline_stages <= 0
        or panel_width <= 0
    ):
        raise ValueError(
            "Tile dimensions, pipeline stages, and panel width must be positive!"
        )

    if m % tile_m or n % tile_n or k % tile_k:
        raise ValueError(
            "Tile sizes must evenly divide the corresponding input matrix dimensions (M, N, K)"
        )

    # Adhere to Hopper WGMMA tile alignment constraints
    if tile_m % 64 or tile_n % 8:
        raise ValueError(
            "Output tile sizes (tile_m, tile_n) must be multiples of 64 and 8 respectively!"
        )

    bytes_per_elem = 2  # bf16 consumes 2 bytes
    num_warpgroup_threads = 128

    # 2D Grid extents in units of tiles
    num_tiles_m = m // tile_m
    num_tiles_n = n // tile_n
    num_tiles_k = k // tile_k
    total_tiles_mn = num_tiles_m * num_tiles_n

    # Input and output memory layouts (WGMMA Hopper swizzles)
    input_transforms = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))
    output_transforms = (plgpu.TilingTransform((1, 64)), plgpu.SwizzleTransform(128))

    # Shared Memory (SMEM) allocation sizing
    input_smem_bytes = num_pipeline_stages * tile_k * (tile_m + tile_n) * bytes_per_elem
    out_smem_bytes = tile_m * tile_n * bytes_per_elem
    max_smem_bytes = get_max_smem_bytes()

    # Register allocation sizing per thread
    acc_reg_per_thread = tile_m * tile_n // num_warpgroup_threads

    if max_smem_bytes is None:
        raise ValueError("Unable to find out max shared memory size for this GPU!")
    if input_smem_bytes + out_smem_bytes > max_smem_bytes:
        raise ValueError(
            "The current configuration exceeds the total available shared memory!"
        )
    if acc_reg_per_thread > 192:  # Hardware ceiling for Hopper H100/H200
        raise ValueError("The accumulator leaves too few registers!")

    def kernel(lhs_gmem, rhs_gmem, out_gmem, out_smem):
        def compute_one_output_tile(tile_idx):
            # -------------------------------------------------------------------
            # Swizzled 1D -> 2D Panel Rasterization:
            # Traverses the matrix in vertical "Panels" using a snake-like path
            # to maximize L2 cache hit rates on rhs columns.
            # -------------------------------------------------------------------
            tiles_per_panel = num_tiles_m * panel_width

            # Identify which vertical panel we belong to and our local index within it
            panel_idx = tile_idx // tiles_per_panel
            tile_in_panel_idx = tile_idx % tiles_per_panel

            # Identify panel boundary and clamped width (handles edge tiles)
            panel_start_col = panel_idx * panel_width
            effective_panel_width = jnp.minimum(
                num_tiles_n - panel_start_col, panel_width
            )

            # Determine local 2D coordinate inside the active panel
            row_idx = tile_in_panel_idx // effective_panel_width
            col_offset = tile_in_panel_idx % effective_panel_width

            # Snake pattern: even rows sweep left-to-right; odd rows sweep right-to-left
            col_offset = jnp.where(
                row_idx % 2 == 0,
                col_offset,
                effective_panel_width - col_offset - 1,
            )
            col_idx = panel_start_col + col_offset

            def accumulate_over_reduction_dim(acc):
                def pipeline_step(_, lhs_smem, rhs_smem):
                    # Tensor Core MMA: acc += lhs_tile @ rhs_tile
                    plgpu.wgmma(acc, lhs_smem, rhs_smem)
                    plgpu.wgmma_wait(1)

                tile_spec = partial(
                    plgpu.BlockSpec, transforms=input_transforms, delay_release=1
                )
                lhs_tile_spec = tile_spec((tile_m, tile_k), lambda k_idx: (row_idx, k_idx))  # fmt: skip
                rhs_tile_spec = tile_spec((tile_k, tile_n), lambda k_idx: (k_idx, col_idx))  # fmt: skip

                # Asynchronous SMEM staged pipeline
                plgpu.emit_pipeline(
                    pipeline_step,
                    grid=(num_tiles_k,),
                    in_specs=(lhs_tile_spec, rhs_tile_spec),
                    max_concurrent_steps=num_pipeline_stages,
                )(lhs_gmem, rhs_gmem)

                return acc[...]

            # Run reduction in accumulator registers (float32 precision)
            acc = pl.run_scoped(
                accumulate_over_reduction_dim, plgpu.ACC((tile_m, tile_n), jnp.float32)
            )

            # Cast accumulator to bf16 in SMEM and commit
            out_smem[...] = acc[...].astype(jnp.bfloat16)
            plgpu.commit_smem()

            # Write result from SMEM to GMEM slice
            m_slice = pl.ds(row_idx * tile_m, tile_m)
            n_slice = pl.ds(col_idx * tile_n, tile_n)
            plgpu.copy_smem_to_gmem(out_smem, out_gmem.at[m_slice, n_slice])
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)

        # Grid Dispatch: Persistent Worker Loop vs. 1-to-1 Threadblock Launch
        if is_persistent:

            def persistent_loop_body(loop_info):
                (tile_idx,) = loop_info.index
                compute_one_output_tile(tile_idx)

            plgpu.nd_loop((total_tiles_mn,), collective_axes="sm")(persistent_loop_body)
        else:
            tile_idx = jax.lax.axis_index("out_tile")
            compute_one_output_tile(tile_idx)

    # Launch Configuration & Compilation
    if is_persistent:
        launch_grid = (backend.get_default_device().core_count,)
        grid_names = ("sm",)
    else:
        launch_grid = (total_tiles_mn,)
        grid_names = ("out_tile",)

    return plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((m, n), dtype=jnp.bfloat16),
        scratch_types={
            "out_smem": plgpu.SMEM(
                (tile_m, tile_n),
                jnp.bfloat16,
                transforms=output_transforms,
            ),
        },
        grid=launch_grid,
        grid_names=grid_names,
        kernel_name="hopper_bf16_matmul",
        compiler_params=plgpu.CompilerParams(
            approx_math=True, unsafe_no_auto_barriers=True
        ),
    )(lhs, rhs)


def main(args):
    key = jax.random.PRNGKey(0)
    key, lhs_key, rhs_key = jax.random.split(key, 3)

    lhs = jax.random.normal(shape=(args.m, args.k), key=lhs_key, dtype=jnp.bfloat16)
    rhs = jax.random.normal(shape=(args.k, args.n), key=rhs_key, dtype=jnp.bfloat16)

    out1 = matmul(
        lhs,
        rhs,
        tile_m=args.tile_m,
        tile_k=args.tile_k,
        tile_n=args.tile_n,
        num_pipeline_stages=args.num_pipeline_stages,
        panel_width=args.panel_width,
        is_persistent=args.is_persistent,
    )

    out2 = jnp.matmul(lhs, rhs)
    print("Checking correctness ... ", end=" ")
    print(jnp.allclose(out1, out2, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for bf16 matmul kernel")
    parser.add_argument("-m", default=64, type=int)
    parser.add_argument("-k", default=4096, type=int)
    parser.add_argument("-n", default=4096, type=int)
    parser.add_argument("--tile_m", default=64, type=int)
    parser.add_argument("--tile_n", default=128, type=int)
    parser.add_argument("--tile_k", default=128, type=int)
    parser.add_argument("--num_pipeline_stages", default=4, type=int)
    parser.add_argument("--panel_width", default=4, type=int)
    parser.add_argument("--is_persistent", default=True, type=bool)

    arguments = parser.parse_args()
    main(arguments)
