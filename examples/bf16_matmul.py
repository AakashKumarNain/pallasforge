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


def matmul(x, y, out_row_tile_size, out_col_tile_size, reduction_tile_size, num_pipeline_stages, col_group_size, is_persistent):
    """Kernel for matrix multiplication for two matrices with bfloat16 dtype.
    
    Args:
        x, y: Input matrices each with bf16 dtype.
        out_row_tile_size: Number of rows in the output tile computed by the kernel
        out_col_tile_size: Number of columns in the output tile computed by the kernel
        reduction_tile_size: Size of the reduction tile
        num_pipeline_stages: Number of reduction tiles to be staged concurrently in the
            shared memory to overlap compute and loading stages
        is_persistent: If enabled, one GPU program computes many output tiles as opposed
            to launching one program per output tile
    Returns:
        output of x @ y of bfloat16 dtype
    """

    m, k = x.shape
    y_k, n = y.shape

    # Ensure dtypes are bfloat16 and reduction dimension match
    if x.dtype != jnp.bfloat16 or y.dtype !=jnp.bfloat16:
        raise ValueError("Inputs must be of dtype `jnp.bfloat16`")
    if k != y_k:
        raise ValueError(f"Reduction dimension must be same. Got {k} for left operand and {y_k} for right operand")

    # Ensure tile sizes, pipeline stages, and column group sizes are all positive
    if (
        out_row_tile_size <= 0 or
        out_col_tile_size <= 0 or
        reduction_tile_size <= 0 or
        num_pipeline_stages <=0 or 
        col_group_size <= 0
    ):
        raise ValueError("Output tile sizes, number of pipeline stages, and column group sizes should be positive!")

    # Ensure tile sizes evenly divide the dimensions of the input matrices
    if m % out_row_tile_size or n % out_col_tile_size or k % reduction_tile_size:
        raise ValueError("Tile sizes must evenly divide the corresponding input dimensions of the matrices")

    # Ensure we are adhering to Hopper tile constraints
    if out_row_tile_size % 64 or out_col_tile_size % 8:
        raise("Output tile sizes should be evenly divisible by 64 and 8 respectively!")

    bytes_per_elem = 2  # bf16 consumes 2 bytes
    num_warpgroup_threads = 128

    num_out_row_tiles = m // out_row_tile_size
    num_out_col_tiles = n // out_col_tile_size
    num_reduction_tiles = k // reduction_tile_size
    total_output_tiles = num_out_row_tiles * num_out_col_tiles

    # Input and output transforms
    input_transforms = (
        plgpu.TilingTransform((8, 64)),
        plgpu.SwizzleTransform(128)
    )
    output_transforms = (
        plgpu.TilingTransform((1, 64)),
        plgpu.SwizzleTransform(128)
    )

    # Bytes consumed by one pipeline:(m * k + n*k) * 2
    input_smem_bytes = num_pipeline_stages * reduction_tile_size * (out_row_tile_size * out_col_tile_size) * bytes_per_elem
    out_smem_bytes = out_row_tile_size * out_col_tile_size * bytes_per_elem

    # Total registers per thread
    acc_reg_per_thread = out_row_tile_size * out_col_tile_size // num_warpgroup_threads

    if input_smem_bytes + out_smem_bytes > get_max_smem_bytes():
        raise ValueError("The current configuration exceeds the total available shared memory!")
    if acc_reg_per_thread > 192: # For H200
        raise ValueError("The accumulator leaves too few registers!")


    def kernel(x_gmem, y_gmem, out_gmem, out_smem):
        def compute_one_output_tile(tile_idx):
            # We need to map the linear tile number to the actual 2D grid

            # Number of output tiles covered by one column across every row
            col_group_span = num_out_row_tiles * col_group_size

            # Find out the column group of the current tile, and its index in the group
            col_group_idx = tile_idx // col_group_span
            local_tile_idx = tile_idx % col_group_span

            # Find the first column (the start) and width of the active column group
            active_col_group_start = col_group_idx % col_group_size
            active_col_group_stride = jnp.minimum(num_out_col_tiles - active_col_group_start, col_group_size)

            # Convert linear tile index to 2D (row, column) format
            row_idx = local_tile_idx // active_col_group_stride
            col_offset = local_tile_idx % active_col_group_stride
            # Because traversal within one column group is done in snake fashion,
            # we need to check if we are at even tile (left -> right) or odd tile (right -> left)
            col_offset = jnp.where(row_idx % 2 == 0, col_offset, active_col_group_stride - col_offset - 1)
            col_idx = active_col_group_start + col_offset

            def accumulate_over_reduction_dim(acc):
                def pipeline_step(_, x_smem, y_smem):
                    # acc += x[row_idx, k_tile] @ y [k_tile, col_idx]
                    plgpu.wgmma(acc, x_smem, y_smem)
                    plgpu.wgmma_wait(1)

                tile_spec = partial(plgpu.BlockSpec, transforms=input_transforms, delay_release=1)
                x_tile_spec = tile_spec((out_row_tile_size, reduction_tile_size), lambda k_tile: (row_idx, k_tile))
                y_tile_spec = tile_spec((reduction_tile_size, out_col_tile_size), lambda k_tile: (k_tile, col_idx))

                # Overlap memory load and computations
                plgpu.emit_pipeline(
                    pipeline_step,
                    grid=(num_reduction_tiles,),
                    in_specs=(x_tile_spec, y_tile_spec),
                    max_concurrent_steps=num_pipeline_stages,
                )(x_gmem, y_gmem)

                # Return accumulated values
                return acc[...]


            # Zero initialize the accumulator for the output tile
            acc = pl.run_scoped(
                accumulate_over_reduction_dim,
                plgpu.ACC((out_row_tile_size, out_col_tile_size)), jnp.float32
            )

            # Once the reduction is complete, we need to write the results to the output shared memory
            out_smem[...] = acc[...].astype(jnp.bfloat16)
            plgpu.commit_smem()

            # Copy the values from shared memory to the current slices in the global memory
            out_row_slice = pl.ds(row_idx * out_row_tile_size, out_row_tile_size)
            out_col_slice = pl.ds(col_idx * out_col_tile_size, out_col_tile_size)
            plgpu.copy_smem_to_gmem(out_smem, out_gmem.at[out_row_slice, out_col_slice])
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)


        if is_persistent:
            # Each persistent program compute multiple output tiles
            def persistent_loop_body(loop_info):
                (tile_idx,) = loop_info.index
                compute_one_output_tile(tile_idx)

            plgpu.nd_loop((total_output_tiles,), collective_axes="sm")(persistent_loop_body)
        else:
            # Once gpu program computes one output tile
            tile_idx = jax.lax.axis_index("out_tile")
            compute_one_output_tile(tile_idx)

    if is_persistent:
        launch_grid = (backend.get_default_device().core_count,)
        grid_names = ("sm",)
    else:
        launch_grid = (total_output_tiles,)
        grid_names = ("out_tile",)

    return plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((m, n), dtype=jnp.bfloat16),
        scratch_types={
            "out_smem": plgpu.SMEM(
                (out_row_tile_size, out_col_tile_size),
                jnp.bfloat16,
                transforms=output_transforms,
            ),
        },
        grid=launch_grid,
        grid_names=grid_names,
        kernel_name="hopper_bf16_matmul",
        compiler_params=plgpu.CompilerParams(
            approx_math=True,
            unsafe_no_auto_barriers=True
        ),
    )(x, y)

