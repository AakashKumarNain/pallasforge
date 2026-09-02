from functools import partial

import jax
import jax.numpy as jnp
from jax.extend import backend
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from pallasforge.common import get_max_smem_bytes
from pallasforge.common import benchmark


def quantize_weight_per_output_channel(weight):
    """Quantizes weights symmetrically with one BF16 scale per output channel."""

    weight = weight.astype(jnp.float32)
    maxval = jnp.max(jnp.abs(weight), axis=1)
    scale = jnp.where(maxval == 0.0, 1.0, maxval / 127.0).astype(jnp.bfloat16)
    quantized = jnp.round(weight / scale[:, None]).astype(jnp.float32)
    quantized = jnp.clip(quantized, -127, 127).astype(jnp.int8)
    return quantized, scale


def simple_w8a16_matmul(quantized_weight, weight_scale, activations):
    """Performs W8A16 op with INT8 -> BF16 weight dequantization."""
    dequantized = quantized_weight.astype(jnp.bfloat16) * weight_scale[:, None]
    return jnp.matmul(activations, dequantized.T)


def matmul(
    weights,
    scale,
    activations,
    tile_m=64,
    tile_n=64,
    tile_k=128,
    num_pipeline_stages=5,
    panel_width=4,
    is_persistent=False,
):
    m, k = activations.shape
    n, k_weight = weights.shape

    activations_bytes_per_elem = 2
    weights_bytes_per_elem = 1
    out_bytes_per_elem = 2
    num_warpgroup_threads = 128

    # M can be smaller then WGMMA tile during decoding (e.g. bsz=1). Pad only activation rows
    padded_m = ((m + tile_m - 1) // tile_m) * tile_m
    m_padding = padded_m - m

    if m_padding:
        padded_activations = jnp.pad(activations, ((0, m_padding), (0, 0)))
    else:
        padded_activations = activations

    # 2D Grid extents in units of tiles
    num_tiles_m = padded_m // tile_m
    num_tiles_n = n // tile_n
    num_tiles_k = k // tile_k
    total_tiles_mn = num_tiles_m * num_tiles_n

    # Some validations
    if activations.dtype != jnp.bfloat16:
        raise ValueError("Activations must be of dtype `jnp.bfloat16`")
    if weights.dtype != jnp.int8:
        raise ValueError("Quantized weights must be of dtype `jnp.int8`")
    if scale.dtype != jnp.bfloat16:
        raise ValueError("Weight scales must be of dtype `jnp.bfloat16`")

    if k != k_weight:
        raise ValueError(
            f"Reduction dimension must match. Got {k} for activations and {k_weight} for weights"
        )

    if scale.shape != (n,):
        raise ValueError(f"Weight scales must have shape ({n},). Got {scale.shape}")

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

    if n % tile_n or k % tile_k:
        raise ValueError("tile_n and tile_k must evenly divide the N and K dimensions")

    # weight @ activations.T produces [tile_n, tile_m], so tile_n is WGMMA's M dimension.
    if tile_n % 64 or tile_m % 8:
        raise ValueError(
            "tile_n and tile_m must be multiples of 64 and 8 respectively for Hopper WGMMA"
        )

    if tile_k % 128:
        raise ValueError("tile_k must be a multiple of 128 for the INT8 weight layout")

    # GPU Transforms or the layouts (WGMMA Hopper swizzles)
    activation_transforms = (
        plgpu.TilingTransform((8, 64)),
        plgpu.SwizzleTransform(128),
    )
    weight_transforms = (plgpu.TilingTransform((8, 128)), plgpu.SwizzleTransform(128))
    output_transforms = (
        plgpu.TilingTransform((1, 64)),
        plgpu.SwizzleTransform(128),
    )

    # Shared Memory (SMEM) allocation sizing
    activation_stage_bytes = tile_m * tile_k * activations_bytes_per_elem
    weight_stage_bytes = tile_n * tile_k * weights_bytes_per_elem

    input_smem_bytes = num_pipeline_stages * (
        activation_stage_bytes + weight_stage_bytes
    )
    out_smem_bytes = tile_m * tile_n * out_bytes_per_elem
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

    def kernel(activations_gmem, weight_gmem, scale_gmem, out_gmem, out_smem):
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

            # one scale is shared by every K elements belonging to the same tile
            scale_slice = pl.ds(col_idx * tile_n, tile_n)
            weight_scale = plgpu.load(
                scale_gmem.at[scale_slice],
                layout=plgpu.Layout.WGMMA.reduce(1),
                optimized=False,
            )

            def accumulate_over_reduction_dim(acc):
                def pipeline_step(_, activation_smem, weight_smem):
                    # Load int8 weight tile into registers using the packed WGMMA layout
                    weight_fragment = plgpu.load(
                        weight_smem,
                        layout=plgpu.Layout.WGMMA_UPCAST_2X,
                    )

                    # Int8 -> Bf16 dequantization happens only inside the kernel
                    weight_fragment = plgpu.layout_cast(
                        weight_fragment, plgpu.Layout.WGMMA
                    ).astype(jnp.bfloat16)
                    # Apply per channel scale to these weights now
                    weight_fragment *= jax.lax.broadcast_in_dim(
                        weight_scale, weight_fragment.shape, (0,)
                    )

                    # Compute [tile_n, tile_k] @ [tile_k, tile_m]
                    #               weight          activation
                    plgpu.wgmma(acc, weight_fragment, activation_smem.T)
                    plgpu.wgmma_wait(1)

                tile_spec = partial(plgpu.BlockSpec, delay_release=1)
                activation_tile_spec = tile_spec(
                    (tile_m, tile_k),
                    lambda k_idx: (row_idx, k_idx),
                    transforms=activation_transforms,
                )
                weight_tile_spec = tile_spec(
                    (tile_n, tile_k),
                    lambda k_idx: (col_idx, k_idx),
                    transforms=weight_transforms,
                )

                # Async GMEM -> SMEM staged pipeline over the K dimension
                plgpu.emit_pipeline(
                    pipeline_step,
                    grid=(num_tiles_k,),
                    in_specs=(activation_tile_spec, weight_tile_spec),
                    max_concurrent_steps=num_pipeline_stages,
                )(activations_gmem, weight_gmem)

                return acc[...]

            acc = pl.run_scoped(
                accumulate_over_reduction_dim,
                plgpu.ACC((tile_n, tile_m), jnp.float32),
            )

            # Convert [tile_n, tile_m] -> [tile_m, tile_n]
            acc = acc.astype(jnp.bfloat16)
            out_smem.T[...] = plgpu.layout_cast(acc, plgpu.Layout.WGMMA_TRANSPOSED)
            plgpu.commit_smem()

            # Copy from SMEM to GMEM
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

    if is_persistent:
        launch_grid = (backend.get_default_device().core_count,)
        grid_names = ("sm",)
    else:
        launch_grid = (total_tiles_mn,)
        grid_names = ("out_tile",)

    output = plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((padded_m, n), dtype=jnp.bfloat16),
        scratch_types={
            "out_smem": plgpu.SMEM(
                (tile_m, tile_n),
                jnp.bfloat16,
                transforms=output_transforms,
            ),
        },
        grid=launch_grid,
        grid_names=grid_names,
        kernel_name="hopper_w8a16_matmul",
        compiler_params=plgpu.CompilerParams(
            approx_math=True, unsafe_no_auto_barriers=True
        ),
    )(padded_activations, weights, scale)
    return output[:m, :]


def compute_memory_bandwidth_gbps(m, k, n, time_ms):
    """Calculates effective memory bandwidth (GB/s) for W8A16 GEMM."""
    # Bytes: Activations (BF16: 2B) + Weights (INT8: 1B) + Scales (BF16: 2B) + Output (BF16: 2B)
    total_bytes = (m * k * 2) + (n * k * 1) + (n * 2) + (m * n * 2)
    time_sec = time_ms / 1000.0
    return (total_bytes / 1e9) / time_sec


def format_relative_perf(t_kernel: float, t_ref: float) -> str:
    """Formats relative performance as Nx faster or Nx slower."""
    if t_kernel <= 0 or t_ref <= 0:
        return "N/A"

    if t_kernel <= t_ref:
        factor = t_ref / t_kernel
        return f"{factor:.2f}x faster"
    else:
        factor = t_kernel / t_ref
        return f"{factor:.2f}x slower"


def run_benchmark_and_profiling():
    jax.config.update("jax_default_matmul_precision", "highest")
    key = jax.random.PRNGKey(0)

    scenarios = [
        # --- LLaMA-3 8B (Hidden: 4096, Intermediate: 14336) ---
        {"desc": "Llama-8B Gate/Up (M=1)", "m": 1, "k": 4096, "n": 14336},
        {"desc": "Llama-8B Down (M=1)", "m": 1, "k": 14336, "n": 4096},
        {"desc": "Llama-8B Gate/Up (M=4)", "m": 4, "k": 4096, "n": 14336},
        {"desc": "Llama-8B Down (M=8)", "m": 8, "k": 14336, "n": 4096},
        {"desc": "Llama-8B Gate/Up (M=16)", "m": 16, "k": 4096, "n": 14336},
        # --- LLaMA-3 70B (Hidden: 8192, Intermediate: 28672) ---
        {"desc": "Llama-70B QKV (M=1)", "m": 1, "k": 8192, "n": 8192},
        {"desc": "Llama-70B Gate/Up (M=1)", "m": 1, "k": 8192, "n": 28672},
        {"desc": "Llama-70B Down (M=1)", "m": 1, "k": 28672, "n": 8192},
        {"desc": "Llama-70B Down (M=8)", "m": 8, "k": 28672, "n": 8192},
        {"desc": "Llama-70B Gate/Up (M=16)", "m": 16, "k": 8192, "n": 28672},
    ]

    print("=" * 120)
    print(
        f"{'Workload':<24} | {'Shape (M, K, N)':<18} | {'Correct':<8} | "
        f"{'Pallas (ms)':<12} | {'Ref (ms)':<10} | {'Relative Perf':<15} | {'Bandwidth'}"
    )
    print("=" * 120)

    for sc in scenarios:
        m, k, n = sc["m"], sc["k"], sc["n"]
        k_act, k_w, key = jax.random.split(key, 3)

        # 1. Inputs generation
        act = (jax.random.normal(k_act, (m, k)) * 0.1).astype(jnp.bfloat16)
        raw_w = jax.random.normal(k_w, (n, k), dtype=jnp.float32) * 0.05
        weights, scale = quantize_weight_per_output_channel(raw_w)

        # 2. Correctness validation
        ref_out = simple_w8a16_matmul(weights, scale, act)
        kernel_out = matmul(act, weights, scale)
        is_correct = jnp.allclose(ref_out, kernel_out, rtol=1e-2, atol=5e-2)

        # 3. CUPTI Measurements
        pallas_report = benchmark(
            matmul,
            args=(act, weights, scale),
            warmup=5,
            iterations=15,
        )

        ref_report = benchmark(
            simple_w8a16_matmul,
            args=(weights, scale, act),
            warmup=5,
            iterations=15,
        )

        t_pallas = pallas_report.median_kernel_time_ms
        t_ref = ref_report.median_kernel_time_ms

        # Explicit fast vs slow calculation
        perf_str = format_relative_perf(t_pallas, t_ref)
        bw_gbps = compute_memory_bandwidth_gbps(m, k, n, t_pallas)

        shape_str = f"({m}, {k}, {n})"
        status_str = "Pass" if is_correct else "Fail"
        bw_str = (
            f"{bw_gbps / 1000.0:.2f} TB/s" if bw_gbps >= 1000 else f"{bw_gbps:.1f} GB/s"
        )

        print(
            f"{sc['desc']:<24} | {shape_str:<18} | {status_str:<8} | "
            f"{t_pallas:<12.4f} | {t_ref:<10.4f} | {perf_str:<15} | {bw_str}"
        )

    print("=" * 120)


def main():
    key = jax.random.PRNGKey(0)

    # LLM decode shapes (M, K, N) across QKV, MLP-Gate/Up, and MLP-Down
    # M represents decode batch size (1 for single-stream, 4-16 for batched decode)
    shapes = [
        (1, 4096, 4096),  # LLaMA-8B Single-token Attention
        (1, 4096, 14336),  # LLaMA-8B Single-token MLP Gate/Up
        (1, 14336, 4096),  # LLaMA-8B Single-token MLP Down
        (4, 4096, 4096),  # LLaMA-8B Small Batched Decode
        (16, 4096, 14336),  # LLaMA-8B Batched MLP
        (1, 8192, 8192),  # LLaMA-70B Single-token Attention
        (1, 8192, 28672),  # LLaMA-70B Single-token MLP Gate/Up
    ]

    for m, k, n in shapes:
        k_act, k_w, key = jax.random.split(key, 3)

        # Activations: (M, K) in BF16
        activations = (jax.random.normal(k_act, (m, k)) * 0.1).astype(jnp.bfloat16)

        # Weights: (N, K) quantized symmetrically to INT8 with BF16 scales
        raw_weight = jax.random.normal(k_w, (n, k), dtype=jnp.float32) * 0.05
        weights, scale = quantize_weight_per_output_channel(raw_weight)

        # Reference vs Kernel
        ref = simple_w8a16_matmul(weights, scale, activations)
        out = matmul(weights, scale, activations)

        # BF16 tolerance: rtol=1e-2 matches 7-bit mantissa precision; atol=5e-2 covers zero-floor
        passed = jnp.allclose(ref, out, rtol=1e-2, atol=5e-2)
        print(f"Shape (M={m:<2}, K={k:<5}, N={n:<5}) : {'Pass' if passed else 'Fail'}")

        assert passed, f"Kernel mismatch on shape (M={m}, K={k}, N={n})"


if __name__ == "__main__":
    # main()
    run_benchmark_and_profiling()
