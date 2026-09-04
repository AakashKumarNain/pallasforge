import time
import argparse
from pathlib import Path
from functools import partial
from itertools import product

import tune_jax
from tune_jax import tune_logger

import jax
import jax.numpy as jnp
from jax.extend import backend
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from pallasforge.common import get_max_smem_bytes
from pallasforge.common import benchmark
from pallasforge.common import format_relative_perf


# A deliberately bounded starter search space. Expand it after the first pass if
# the winner lands on one of the boundaries.
DEFAULT_TUNE_SPACE = {
    "tile_m": (8, 16, 32, 64),
    "tile_n": (64, 128, 256),
    "tile_k": (128, 256),
    "num_pipeline_stages": (2, 4, 6),
    "panel_width": (1, 2, 4, 8),
    "persistent": (False, True),
}


# Shared workload list so tune/benchmark/profile exercise the same shapes.
SCENARIOS = (
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
)



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
    persistent=False,
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

                    # TODO: Explain why tha value of zero instead of anything else
                    plgpu.wgmma_wait(0)

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
        if persistent:

            def persistent_loop_body(loop_info):
                (tile_idx,) = loop_info.index
                compute_one_output_tile(tile_idx)

            plgpu.nd_loop((total_tiles_mn,), collective_axes="sm")(persistent_loop_body)
        else:
            tile_idx = jax.lax.axis_index("out_tile")
            compute_one_output_tile(tile_idx)

    if persistent:
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


def make_inputs(key, m, k, n):
    """Create one deterministic W8A16 input set and return the updated PRNG key."""
    key, act_key, weight_key = jax.random.split(key, 3)
    activations = (jax.random.normal(act_key, (m, k)) * 0.1).astype(jnp.bfloat16)
    raw_weight = jax.random.normal(weight_key, (n, k), dtype=jnp.float32) * 0.05
    weights, scale = quantize_weight_per_output_channel(raw_weight)
    return key, weights, scale, activations


def check_correctness(weights, scale, activations, kernel_fn=matmul):
    """Compare a kernel invocation against the simple BF16 reference."""
    ref_out = simple_w8a16_matmul(weights, scale, activations)
    kernel_out = kernel_fn(weights, scale, activations)
    ref_out.block_until_ready()
    kernel_out.block_until_ready()
    return bool(jnp.allclose(ref_out, kernel_out, rtol=1e-2, atol=5e-2))


def enumerate_valid_tuning_configs(m, k, n, search_space=DEFAULT_TUNE_SPACE):
    """Build correlated, shape-valid configs before handing them to tune-jax.

    tune-jax normally evaluates a Cartesian product. This kernel has correlated
    constraints (divisibility, WGMMA layout, accumulator registers, and SMEM),
    so we pre-filter configs and tune a single ``config_id`` instead.
    """
    max_smem_bytes = get_max_smem_bytes()
    if max_smem_bytes is None:
        raise ValueError("Unable to find out max shared memory size for this GPU!")

    configs = []
    for (
        tile_m,
        tile_n,
        tile_k,
        num_pipeline_stages,
        panel_width,
        persistent,
    ) in product(
        search_space["tile_m"],
        search_space["tile_n"],
        search_space["tile_k"],
        search_space["num_pipeline_stages"],
        search_space["panel_width"],
        search_space["persistent"],
    ):
        # Kernel/WGMMA validity constraints.
        if tile_m <= 0 or tile_n <= 0 or tile_k <= 0:
            continue
        if tile_m % 8 or tile_n % 64 or tile_k % 128:
            continue
        if n % tile_n or k % tile_k:
            continue

        # The kernel uses BlockSpec(delay_release=1), therefore the pipeline
        # must have at least two concurrent stages.
        if num_pipeline_stages <= 1:
            continue

        # Widths larger than the number of N tiles are equivalent/redundant.
        num_tiles_n = n // tile_n
        if panel_width <= 0 or panel_width > num_tiles_n:
            continue

        # Same register and SMEM constraints enforced by matmul().
        acc_reg_per_thread = tile_m * tile_n // 128
        if acc_reg_per_thread > 192:
            continue

        activation_stage_bytes = tile_m * tile_k * 2  # BF16 activations
        weight_stage_bytes = tile_n * tile_k  # INT8 weights
        input_smem_bytes = num_pipeline_stages * (
            activation_stage_bytes + weight_stage_bytes
        )
        out_smem_bytes = tile_m * tile_n * 2  # BF16 output staging
        if input_smem_bytes + out_smem_bytes > max_smem_bytes:
            continue

        configs.append(
            {
                "tile_m": tile_m,
                "tile_n": tile_n,
                "tile_k": tile_k,
                "num_pipeline_stages": num_pipeline_stages,
                "panel_width": panel_width,
                "persistent": persistent,
            }
        )

    if not configs:
        raise ValueError(f"No valid tuning configs for shape (M={m}, K={k}, N={n})")
    return configs


def tune_matmul_for_shape(
    weights,
    scale,
    activations,
    *,
    search_space=DEFAULT_TUNE_SPACE,
    max_workers=16,
):
    """Tune matmul for one concrete input shape.

    Returns:
      tuned_fn: tune-jax wrapped function containing timing results/cache.
      best_config: concrete kernel parameters for the winning config.
      output: output produced by the winning config.
    """

    m, k = activations.shape
    n, k_weight = weights.shape
    if k != k_weight:
        raise ValueError(f"K mismatch: activations has {k}, weights has {k_weight}")

    configs = enumerate_valid_tuning_configs(m, k, n, search_space)

    def candidate(weights, scale, activations, *, config_id):
        return matmul(weights, scale, activations, **configs[config_id])

    tuned_fn = tune_jax.tune(
        candidate,
        hyperparams={"config_id": tuple(range(len(configs)))},
        max_workers=max_workers,
        example_args=(weights, scale, activations),
    )
    tuned_fn_jit = jax.jit(tuned_fn)
    output = tuned_fn_jit(weights, scale, activations)
    output.block_until_ready()

    hyperparams = tuned_fn_jit.optimal_hyperparams
    best_config_id = int(hyperparams["config_id"])
    return tuned_fn_jit, configs[best_config_id], output


def run_tuning(
    scenarios=SCENARIOS,
    *,
    search_space=DEFAULT_TUNE_SPACE,
    max_workers=16,
):
    """Tune all requested workloads, verify winners, and return shape->config."""
    tune_logger.setLevel("INFO")
    key = jax.random.PRNGKey(0)
    winners = {}

    print(
        f"Tuning {len(scenarios)} workload(s) with up to {max_workers} compile workers."
    )

    for index, scenario in enumerate(scenarios, start=1):
        m, k, n = scenario["m"], scenario["k"], scenario["n"]
        key, weights, scale, activations = make_inputs(key, m, k, n)

        configs = enumerate_valid_tuning_configs(m, k, n, search_space)
        print(
            f"\n[{index}/{len(scenarios)}] {scenario['desc']} "
            f"shape=({m}, {k}, {n}) candidates={len(configs)}"
        )

        tuned_fn, best_config, out = tune_matmul_for_shape(
            weights,
            scale,
            activations,
            search_space=search_space,
            max_workers=max_workers,
        )

        ref = simple_w8a16_matmul(weights, scale, activations)
        ref.block_until_ready()
        passed = bool(jnp.allclose(ref, out, rtol=1e-2, atol=5e-2))
        if not passed:
            raise AssertionError(
                f"Winning tune-jax config failed correctness for shape "
                f"{(m, k, n)}: {best_config}"
            )

        winners[(m, k, n)] = best_config
        print(f"\nBest config             : {best_config}")
        print(f"Kernel correctness passed : {passed}")
        # print("tune-jax results:")
        # print(tune_jax.tabulate(tuned_fn.timing_results))

    print("\nReusable winners:")
    for shape, config in winners.items():
        print(f"    {shape}: {config},")

    return winners


def run_benchmark(
    scenarios=SCENARIOS,
    *,
    configs_by_shape=None,
    warmup=5,
    iterations=15,
):
    """Benchmark the Pallas kernel against the simple JAX reference."""
    configs_by_shape = configs_by_shape or {}
    key = jax.random.PRNGKey(0)

    print("=" * 132)
    print(
        f"{'Workload':<28} | {'Shape (M, K, N)':<20} | {'Config':<8} | {'Correct':<8} | "
        f"{'Pallas (ms)':<12} | {'Ref (ms)':<10} | {'Relative Perf':<15} | {'Bandwidth'}"
    )
    print("=" * 132)

    for scenario in scenarios:
        m, k, n = scenario["m"], scenario["k"], scenario["n"]
        shape = (m, k, n)
        key, weights, scale, activations = make_inputs(key, m, k, n)

        config = configs_by_shape.get(shape)
        kernel_fn = partial(matmul, **config) if config is not None else matmul
        config_label = "tuned" if config is not None else "default"

        is_correct = check_correctness(weights, scale, activations, kernel_fn)

        pallas_report = benchmark(
            kernel_fn,
            args=(weights, scale, activations),
            warmup=warmup,
            iterations=iterations,
        )
        ref_report = benchmark(
            simple_w8a16_matmul,
            args=(weights, scale, activations),
            warmup=warmup,
            iterations=iterations,
        )

        t_pallas = pallas_report.median_kernel_time_ms
        t_ref = ref_report.median_kernel_time_ms
        perf_str = format_relative_perf(t_pallas, t_ref)
        bw_gbps = compute_memory_bandwidth_gbps(m, k, n, t_pallas)

        shape_str = f"({m}, {k}, {n})"
        status_str = "Pass" if is_correct else "Fail"
        bw_str = (
            f"{bw_gbps / 1000.0:.2f} TB/s" if bw_gbps >= 1000 else f"{bw_gbps:.1f} GB/s"
        )

        print(
            f"{scenario['desc']:<28} | {shape_str:<20} | {config_label:<8} | "
            f"{status_str:<8} | {t_pallas:<12.4f} | {t_ref:<10.4f} | "
            f"{perf_str:<15} | {bw_str}"
        )

    print("=" * 132)


def run_profile(
    scenarios=SCENARIOS,
    configs_by_shape=None,
    static_argnames=("tile_m", "tile_n", "tile_k", "num_pipeline_stages", "panel_width","persistent"),
    profile_dir="/tmp/w8a16_matmul_profile",
    warmup=5,
    repetitions=10,
):
    """Capture one JAX profiler trace per workload.

    Compilation and input creation happen before each trace. The trace itself
    contains only repeated kernel executions, with explicit trace annotations.
    """
    configs_by_shape = configs_by_shape or {}
    root = Path(profile_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(0)


    print(f"Writing JAX profiler traces under: {root}")

    for index, scenario in enumerate(scenarios, start=1):
        m, k, n = scenario["m"], scenario["k"], scenario["n"]
        shape = (m, k, n)
        key, weights, scale, activations = make_inputs(key, m, k, n)

        config = configs_by_shape.get(shape)
        kernel_fn = partial(matmul, **config) if config is not None else matmul

        print("\nLowering and compiling kernel function...")
        start = time.perf_counter()
        jitted_fn = jitted_fn = jax.jit(kernel_fn, static_argnames=static_argnames)
        lowered = jitted_fn.lower(weights, scale, activations)
        compiled = lowered.compile()
        end = time.perf_counter()
        print(f"Kernel function compiled. Time taken:  {(end - start)*1000:.3f} ms")
        
        config_label = "tuned" if config is not None else "default"

        # Compile/warm up before starting the trace so compilation and random
        # input generation do not dominate the captured computation profile.
        for _ in range(warmup):
            compiled(weights, scale, activations).block_until_ready()

        workload_dir = root / f"m{m}_k{k}_n{n}"
        workload_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[{index}/{len(scenarios)}] Profiling {scenario['desc']} "
            f"shape={shape} config={config_label} -> {workload_dir}"
        )

        with jax.profiler.trace(str(workload_dir)):
            for step in range(repetitions):
                with jax.profiler.StepTraceAnnotation("w8a16_matmul", step_num=step):
                    compiled(weights, scale, activations).block_until_ready()

    print(f"Profile capture complete: {root}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Tune, benchmark, and profile the Hopper Fused W8A16 Pallas kernel."
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Tune kernel hyperparameters with tune-jax.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark the Pallas kernel against the JAX reference.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture JAX profiler traces for the kernel workloads.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Maximum tune-jax parallel compilation workers (default: 16).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations used by benchmark/profile (default: 5).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=15,
        help="Measured benchmark iterations (default: 15).",
    )
    parser.add_argument(
        "--profile-repetitions",
        type=int,
        default=10,
        help="Kernel executions recorded in each profiler trace (default: 10).",
    )
    parser.add_argument(
        "--profile-dir",
        default="/tmp/w8a16_matmul_profile",
        help="Root directory for JAX profiler traces.",
    )
    args = parser.parse_args(argv)

    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.profile_repetitions <= 0:
        parser.error("--profile-repetitions must be positive")

    return parser, args


def main(argv=None):
    parser, args = parse_args(argv)

    if not (args.tune or args.benchmark or args.profile):
        raise ValueError("No action selected for kernel!")

    winners = None
    if args.tune:
        winners = run_tuning(max_workers=args.max_workers)

    # When actions are combined, reuse winners from this process. This makes
    # `--tune --benchmark` and `--tune --profile` exercise the tuned configs.
    if args.benchmark:
        run_benchmark(
            configs_by_shape=winners,
            warmup=args.warmup,
            iterations=args.iterations,
        )

    if args.profile:
        run_profile(
            configs_by_shape=winners,
            profile_dir=args.profile_dir,
            warmup=args.warmup,
            repetitions=args.profile_repetitions,
        )


if __name__ == "__main__":
    main()
