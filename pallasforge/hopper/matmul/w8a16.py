"""Fused W8A16 decoding kernels for one Hopper GPU.

Inputs: INT8 weights [N, K], BF16 scales [N], BF16 activations [M, K].
Output: BF16 [M, N]. Inputs must be finite; N and K must fit the chosen tiles.

Numerical contract: accumulate A @ Q.T in FP32, multiply by each output-channel
scale in FP32, then round to BF16. This differs from rounding Q * scale to BF16
before the matrix multiply, as the original implementation did.

All custom kernels use the Mosaic GPU API, including GEMV and split-K reduction.
The WGMMA path retains wait(0) because delay release when set to 1 is resulting
in large numerical mismatch for some reason. Split-K stores unscaled FP32 partials
without transposing the accumulator layout; a second Mosaic kernel reduces and scales.
Tuning validates each candidate before timing it; a GPU execution error aborts.

Examples:
    python w8a16_decode.py --check --scenario 0
    python w8a16_decode.py --tune --benchmark --configs w8a16_configs.json
    python w8a16_decode.py --profile --configs w8a16_configs.json
    MOSAIC_GPU_DUMP_PTXAS=1 python w8a16_decode.py --tune --scenario 0

Reported latency includes host dispatch, synchronization, and every GPU kernel
in the compiled call. It is not a device-only kernel time or an HBM measurement.
Use --profile to inspect device execution, padding, reduction, and slicing.
"""

import argparse
import json
import time
from functools import partial
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

import tune_jax
from tune_jax import tune, tune_logger

from pallasforge.common import benchmark
from pallasforge.common import format_relative_perf

tune_logger.setLevel("INFO")


# The Q example is a single projection; gate/up shapes each describe one projection.
SCENARIOS = (
    {"desc": "8B gate or up, M=1", "m": 1, "k": 4096, "n": 14336},
    {"desc": "8B down, M=1", "m": 1, "k": 14336, "n": 4096},
    {"desc": "8B gate or up, M=4", "m": 4, "k": 4096, "n": 14336},
    {"desc": "8B down, M=8", "m": 8, "k": 14336, "n": 4096},
    {"desc": "8B gate or up, M=16", "m": 16, "k": 4096, "n": 14336},
    {"desc": "70B Q, M=1", "m": 1, "k": 8192, "n": 8192},
    {"desc": "70B gate or up, M=1", "m": 1, "k": 8192, "n": 28672},
    {"desc": "70B down, M=1", "m": 1, "k": 28672, "n": 8192},
    {"desc": "70B down, M=8", "m": 8, "k": 28672, "n": 8192},
    {"desc": "70B gate or up, M=16", "m": 16, "k": 8192, "n": 28672},
)

CONTRACT_NAME = "int8_bf16_fp32_accumulate_then_scale_v1"
HOPPER_BLOCK_SMEM_BYTES = 227 * 1024  # Tested on H200


def quantize_weight_per_output_channel(weight):
    """Quantize offline using the same BF16 scales that inference will receive."""
    weight = weight.astype(jnp.float32)
    maxval = jnp.max(jnp.abs(weight), axis=1)
    scale = jnp.maximum(maxval / 127.0, jnp.finfo(jnp.bfloat16).tiny)
    scale = jnp.where(maxval == 0, 1.0, scale).astype(jnp.bfloat16)
    quantized = jnp.round(weight / scale.astype(jnp.float32)[:, None])
    return jnp.clip(quantized, -127, 127).astype(jnp.int8), scale


@jax.jit
def reference_fp32(weights, scale, activations):
    """Correctness reference"""
    result = jnp.matmul(
        activations.astype(jnp.float32),
        weights.astype(jnp.float32).T,
        precision=jax.lax.Precision.HIGHEST,
    )
    return result * scale.astype(jnp.float32)[None, :]


def simple_w8a16_matmul(weights, scale, activations):
    """Unfused JAX implementation of the new numerical contract."""
    result = jnp.matmul(
        activations, weights.astype(jnp.bfloat16).T, preferred_element_type=jnp.float32
    )
    return (result * scale.astype(jnp.float32)[None, :]).astype(jnp.bfloat16)


def bf16_matmul(prepared_weight, activations):
    """BF16 inference baseline: weight preparation happens outside timing."""
    return jnp.matmul(
        activations, prepared_weight.T, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)


def default_tile_m(m):
    return next((size for size in (8, 16, 32) if m <= size), 64)


def check_shapes(weights, scale, activations):
    m, k = activations.shape
    n, weight_k = weights.shape
    if min(m, n, k) <= 0 or weight_k != k or scale.shape != (n,):
        raise ValueError("Expected positive shapes A[M,K], Q[N,K], and scales[N].")
    return m, k, n


def output_tile_coordinates(tile_idx, num_tiles_m, num_tiles_n, panel_width):
    if num_tiles_m == 1:
        return 0, tile_idx
    panel_size = num_tiles_m * panel_width
    panel_start = (tile_idx // panel_size) * panel_width
    width = jnp.minimum(panel_width, num_tiles_n - panel_start)
    local_idx = tile_idx % panel_size
    row = local_idx // width
    col = local_idx % width
    col = jnp.where(row % 2 == 0, col, width - col - 1)
    return row, panel_start + col


def estimated_smem_bytes(tile_m, tile_n, tile_k, stages, split_k):
    # A prefilter only: reserve space for barriers/alignment and trust compilation.
    # Registers, spills, and the actual allocation appear in the PTXAS report.
    # FP32 split-K partials use ordinary global stores and need no output SMEM.
    output_bytes = 2 if split_k == 1 else 0
    return (
        stages * (2 * tile_m * tile_k + tile_n * tile_k)
        + output_bytes * tile_m * tile_n
        + 2048
    )


def reduce_split_k(partials, scale, m, split_k):
    """Reduce FP32 [split_k * N, padded_M] partials to BF16 [M, N] in Mosaic."""
    n = scale.shape[0]
    tile_n = 64
    if n % tile_n or partials.shape[0] != split_k * n or not 0 < m <= partials.shape[1]:
        raise ValueError(
            "Split-K reduction requires [split_k * N, padded_M] partials and N divisible by 64."
        )
    row_layout = plgpu.Layout.WGMMA.reduce(1)

    def kernel(partials_gmem, scale_gmem, out_gmem):
        row = jax.lax.axis_index("row")
        col_start = jax.lax.axis_index("column_tile") * tile_n
        initial = plgpu.layout_cast(jnp.zeros((tile_n,), jnp.float32), row_layout)

        def add_split(split_idx, acc):
            channels = pl.ds(split_idx * n + col_start, tile_n)
            values = plgpu.load(
                partials_gmem.at[channels, row], layout=row_layout, optimized=False
            )
            return acc + values

        result = jax.lax.fori_loop(0, split_k, add_split, initial)
        cols = pl.ds(col_start, tile_n)
        channel_scale = plgpu.load(
            scale_gmem.at[cols], layout=row_layout, optimized=False
        ).astype(jnp.float32)
        result = (result * channel_scale).astype(jnp.bfloat16)
        # Mosaic stores use Ref assignment; there is no plgpu.store function.
        out_gmem[row, cols] = result

    return plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
        grid=(m, n // tile_n),
        grid_names=("row", "column_tile"),
        kernel_name="hopper_w8a16_reduce_split_k",
        compiler_params=plgpu.CompilerParams(approx_math=False),
    )(partials, scale)


def matmul(
    weights,
    scale,
    activations,
    tile_m=None,
    tile_n=64,
    tile_k=128,
    num_pipeline_stages=3,
    panel_width=1,
    persistent=False,
    split_k=1,
    num_sms=None,
):
    """Mosaic WGMMA path; pass tuning parameters as static arguments to JIT."""
    m, k, n = check_shapes(weights, scale, activations)
    tile_m = default_tile_m(m) if tile_m is None else tile_m

    if min(tile_m, tile_n, tile_k, num_pipeline_stages, panel_width, split_k) <= 0:
        raise ValueError(
            "Tile sizes, stages, panel width, and split_k must be positive."
        )
    if tile_m % 8 or tile_m > 256 or tile_n % 64 or tile_k % 128:
        raise ValueError(
            "Require tile_m in 8..256 by 8, tile_n divisible by 64, and tile_k divisible by 128."
        )
    if n % tile_n or k % (tile_k * split_k):
        raise ValueError(
            "N must divide into tile_n; K must divide into tile_k * split_k."
        )

    padded_m = ((m + tile_m - 1) // tile_m) * tile_m
    num_tiles_m, num_tiles_n = padded_m // tile_m, n // tile_n
    output_tiles = num_tiles_m * num_tiles_n
    k_steps = k // (tile_k * split_k)
    stages = min(num_pipeline_stages, k_steps)
    if (
        estimated_smem_bytes(tile_m, tile_n, tile_k, stages, split_k)
        > HOPPER_BLOCK_SMEM_BYTES
    ):
        raise ValueError("Estimated SMEM exceeds the Hopper per-block budget.")

    # These operations belong to the outer JIT and are included in whole-call timing.
    padded_activations = (
        jnp.pad(activations, ((0, padded_m - m), (0, 0)))
        if padded_m != m
        else activations
    )
    output_dtype = jnp.bfloat16 if split_k == 1 else jnp.float32
    activation_transforms = (
        plgpu.TilingTransform((8, 64)),
        plgpu.SwizzleTransform(128),
    )
    weight_transforms = (plgpu.TilingTransform((8, 128)), plgpu.SwizzleTransform(128))
    output_transforms = (plgpu.TilingTransform((1, 64)), plgpu.SwizzleTransform(128))
    if split_k == 1:
        output_shape = (padded_m, n)
        scratch = {
            "out_smem": plgpu.SMEM(
                (tile_m, tile_n), jnp.bfloat16, transforms=output_transforms
            )
        }
    else:
        # Keep [N, M] accumulator orientation in the workspace, with splits stacked along N.
        output_shape = (split_k * n, padded_m)
        scratch = {}

    def kernel(a_gmem, q_gmem, scale_gmem, out_gmem, out_smem=None):
        def compute_tile(task_idx):
            split_idx = task_idx // output_tiles
            tile_idx = task_idx % output_tiles
            row_idx, col_idx = output_tile_coordinates(
                tile_idx, num_tiles_m, num_tiles_n, panel_width
            )

            def accumulate(acc):
                def pipeline_step(_, a_smem, q_smem):
                    q = plgpu.load(q_smem, layout=plgpu.Layout.WGMMA_UPCAST_2X)
                    q = plgpu.layout_cast(q, plgpu.Layout.WGMMA).astype(jnp.bfloat16)
                    plgpu.wgmma(acc, q, a_smem.T)
                    # Wait before registers or activation SMEM can be reused.
                    # Changing this to 1 requires a different, verified operand schedule.
                    plgpu.wgmma_wait(0)

                specs = (
                    plgpu.BlockSpec(
                        (tile_m, tile_k),
                        lambda ki: (row_idx, split_idx * k_steps + ki),
                        transforms=activation_transforms,
                        delay_release=0,
                    ),
                    plgpu.BlockSpec(
                        (tile_n, tile_k),
                        lambda ki: (col_idx, split_idx * k_steps + ki),
                        transforms=weight_transforms,
                        delay_release=0,
                    ),
                )
                plgpu.emit_pipeline(
                    pipeline_step,
                    grid=(k_steps,),
                    in_specs=specs,
                    max_concurrent_steps=stages,
                )(a_gmem, q_gmem)
                return acc[...]

            result = pl.run_scoped(accumulate, plgpu.ACC((tile_n, tile_m), jnp.float32))
            if split_k == 1:
                channel_scale = plgpu.load(
                    scale_gmem.at[pl.ds(col_idx * tile_n, tile_n)],
                    layout=plgpu.Layout.WGMMA.reduce(1),
                    optimized=False,
                ).astype(jnp.float32)
                result *= jax.lax.broadcast_in_dim(channel_scale, result.shape, (0,))
                # Retain the working BF16 transpose/store path only for the final output.
                result = result.astype(jnp.bfloat16)
                out_smem.T[...] = plgpu.layout_cast(
                    result, plgpu.Layout.WGMMA_TRANSPOSED
                )
                plgpu.commit_smem()
                rows = pl.ds(row_idx * tile_m, tile_m)
                cols = pl.ds(col_idx * tile_n, tile_n)
                plgpu.copy_smem_to_gmem(out_smem, out_gmem.at[rows, cols])
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)
            else:
                # Ordinary FP32 stores preserve the WGMMA layout; no stmatrix or SMEM transpose.
                channels = pl.ds(split_idx * n + col_idx * tile_n, tile_n)
                rows = pl.ds(row_idx * tile_m, tile_m)
                out_gmem[channels, rows] = result

        if persistent:

            def worker(loop_info):
                compute_tile(loop_info.index[0])

            plgpu.nd_loop((split_k * output_tiles,), collective_axes="worker")(worker)
        else:
            compute_tile(jax.lax.axis_index("output_tile"))
        # Drain all stores before leaving the kernel, after allowing SMEM reuse above.
        if split_k == 1:
            plgpu.wait_smem_to_gmem(0)

    task_count = split_k * output_tiles
    if persistent:
        num_sms = (
            jax.local_devices(backend="gpu")[0].core_count
            if num_sms is None
            else num_sms
        )
        grid, grid_names = (min(num_sms, task_count),), ("worker",)
    else:
        grid, grid_names = (task_count,), ("output_tile",)

    result = plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct(output_shape, output_dtype),
        scratch_types=scratch,
        grid=grid,
        grid_names=grid_names,
        kernel_name="hopper_w8a16_split_k",
        compiler_params=plgpu.CompilerParams(approx_math=False),
    )(padded_activations, weights, scale)

    if split_k == 1:
        return result[:m, :]
    return reduce_split_k(result, scale, m, split_k)


def gemv(weights, scale, activations, tile_n=64, tile_k=256, split_k=1):
    """Batch-1 Mosaic kernel using FP32 vector arithmetic, without WGMMA operations."""

    m, k, n = check_shapes(weights, scale, activations)

    if m != 1 or min(tile_n, tile_k, split_k) <= 0:
        raise ValueError("GEMV requires M=1 and positive tile sizes and split_k.")
    if tile_n % 64 or tile_k % 128:
        raise ValueError(
            "GEMV uses tiles with N divisible by 64 and K divisible by 128."
        )
    if n % tile_n or k % (tile_k * split_k):
        raise ValueError("GEMV requires exact N and K tile coverage.")

    steps = k // (tile_k * split_k)
    output_dtype = jnp.bfloat16 if split_k == 1 else jnp.float32
    output_shape = (1, n) if split_k == 1 else (split_k * n, 1)
    row_layout = plgpu.Layout.WGMMA.reduce(1)
    column_layout = plgpu.Layout.WGMMA.reduce(0)

    def kernel(q_gmem, scale_gmem, a_gmem, out_gmem):
        channel_start = jax.lax.axis_index("channel_tile") * tile_n
        split_idx = jax.lax.axis_index("split")
        channels = pl.ds(channel_start, tile_n)
        initial = plgpu.layout_cast(jnp.zeros((tile_n,), jnp.float32), row_layout)

        def step(ki, acc):
            cols = pl.ds((split_idx * steps + ki) * tile_k, tile_k)
            # Use one common register layout for multiplication and row reduction.
            # The layout name does not issue a WGMMA instruction or pad activation rows.
            q = plgpu.load(
                q_gmem.at[channels, cols], layout=plgpu.Layout.WGMMA, optimized=False
            )
            q = q.astype(jnp.float32)
            a = plgpu.load(
                a_gmem.at[0, cols], layout=column_layout, optimized=False
            ).astype(jnp.float32)
            a = jax.lax.broadcast_in_dim(a, (tile_n, tile_k), (1,))
            return acc + jnp.sum(q * a, axis=1, dtype=jnp.float32)

        result = jax.lax.fori_loop(0, steps, step, initial)
        if split_k == 1:
            channel_scale = plgpu.load(
                scale_gmem.at[channels], layout=row_layout, optimized=False
            )
            result = (result * channel_scale.astype(jnp.float32)).astype(jnp.bfloat16)
            out_gmem[0, channels] = result
        else:
            partial_channels = pl.ds(split_idx * n + channel_start, tile_n)
            out_gmem[partial_channels, 0] = result

    result = plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct(output_shape, output_dtype),
        grid=(n // tile_n, split_k),
        grid_names=("channel_tile", "split"),
        compiler_params=plgpu.CompilerParams(approx_math=False),
        kernel_name="hopper_w8a16_gemv",
    )(weights, scale, activations)
    if split_k == 1:
        return result
    return reduce_split_k(result, scale, 1, split_k)


def candidate_function(config, num_sms):
    params = dict(config)
    method = params.pop("method")
    if method == "gemv":
        return partial(gemv, **params)
    if method == "wgmma":
        return partial(matmul, num_sms=num_sms, **params)
    raise ValueError(f"Unknown method: {method}")


def default_config(m):
    # This is a safe starting point, not a measured winner.
    return dict(
        method="wgmma",
        tile_m=default_tile_m(m),
        tile_n=64,
        tile_k=128,
        num_pipeline_stages=3,
        panel_width=1,
        persistent=False,
        split_k=1,
    )


def enumerate_configs(m, k, n, num_sms, include_gemv=True):
    base_m = default_tile_m(m)
    m_tiles = (base_m, min(64, base_m * 2))
    configs, seen = [], set()
    for tm, tn, tk, stages, split, persistent in product(
        sorted(set(m_tiles)),
        (64, 128),
        (128, 256),
        (2, 3, 4),
        (1, 2, 4, 8),
        (False, True),
    ):
        if n % tn or k % (tk * split):
            continue
        actual_stages = min(stages, k // (tk * split))
        tiles_m, tiles_n = (m + tm - 1) // tm, n // tn
        if persistent and split * tiles_m * tiles_n <= num_sms:
            continue
        if (
            estimated_smem_bytes(tm, tn, tk, actual_stages, split)
            > HOPPER_BLOCK_SMEM_BYTES
        ):
            continue
        panels = (
            (1,)
            if tiles_m == 1
            else tuple(width for width in (1, 4) if width <= tiles_n)
        )
        for panel in panels:
            config = dict(
                method="wgmma",
                tile_m=tm,
                tile_n=tn,
                tile_k=tk,
                num_pipeline_stages=actual_stages,
                panel_width=panel,
                persistent=persistent,
                split_k=split,
            )
            identity = tuple(config.items())
            if identity not in seen:
                seen.add(identity)
                configs.append(config)
    if m == 1 and include_gemv:
        for tn, tk, split in product((64, 128), (128, 256, 512), (1, 2, 4, 8)):
            if n % tn == 0 and k % (tk * split) == 0:
                configs.append(dict(method="gemv", tile_n=tn, tile_k=tk, split_k=split))
    if not configs:
        raise ValueError(f"No configurations cover {(m, k, n)}.")
    return configs


def make_inputs(key, m, k, n):
    key, act_key, weight_key = jax.random.split(key, 3)
    activations = (jax.random.normal(act_key, (m, k)) * 0.1).astype(jnp.bfloat16)
    raw_weight = jax.random.normal(weight_key, (n, k)) * 0.05
    weights, scale = quantize_weight_per_output_channel(raw_weight)
    # A true BF16 inference baseline prepared once, never inside a timed function.
    prepared_weight = raw_weight.astype(jnp.bfloat16)
    jax.block_until_ready((weights, scale, activations, prepared_weight))
    return key, weights, scale, activations, prepared_weight


def validation_cases(weights, scale, activations):
    m, k = activations.shape
    n = weights.shape[0]
    rows, cols = jnp.arange(n)[:, None], jnp.arange(k)[None, :]
    zero_channels = weights.at[::17, :].set(0)
    # Adjacent columns cancel exactly except for a small, representable residual.
    extremes = jnp.where((rows + cols // 2) % 2 == 0, 127, -128).astype(jnp.int8)
    alternating = jnp.where(jnp.arange(k) % 2 == 0, 0.125, -0.125)
    cancellation = (
        jnp.broadcast_to(alternating, (m, k))
        .at[:, 0]
        .add(1.0 / 1024)
        .astype(jnp.bfloat16)
    )
    varied_scales = jnp.exp2((jnp.arange(n) % 9 - 8).astype(jnp.float32)).astype(
        jnp.bfloat16
    )
    inputs = (
        ("random", weights, scale, activations),
        ("zero_channels", zero_channels, scale, activations),
        ("zero_activations", weights, scale, jnp.zeros_like(activations)),
        ("extremes_and_cancellation", extremes, varied_scales, cancellation),
    )
    cases = []
    for name, q, s, a in inputs:
        expected = np.asarray(jax.device_get(reference_fp32(q, s, a)), dtype=np.float32)
        cases.append((name, (q, s, a), expected))
    return cases


def error_metrics(actual, expected, rtol=5e-3, atol=1e-4):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        return dict(
            passed=False,
            max_abs=float("inf"),
            nrmse=float("inf"),
            worst_ratio=float("inf"),
        )
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    rms = np.sqrt(np.mean(expected.astype(np.float64) ** 2))
    nrmse = float(np.sqrt(np.mean(error**2)) / max(rms, 1e-12))
    ratios = error / (atol + rtol * np.abs(expected))
    passed = bool(np.all(np.isfinite(actual)) and np.all(ratios <= 1) and nrmse <= 3e-3)
    return dict(
        passed=passed,
        max_abs=float(error.max()),
        nrmse=nrmse,
        worst_ratio=float(ratios.max()),
    )


def validate(compiled, cases, repetitions=2):
    reports = []
    for name, args, expected in cases:
        for _ in range(repetitions):
            output = compiled(*args)
            metrics = error_metrics(jax.device_get(output), expected)
            reports.append((name, metrics))
            if not metrics["passed"]:
                return False, reports
    return True, reports


def benchmark_calls(functions, warmup=5, iterations=31):
    """Interleave complete calls; return host-observed synchronized latency in us."""
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations must be positive.")
    names = list(functions)
    samples = {name: [] for name in names}
    for fn, args in functions.values():
        jax.block_until_ready(args)
        for _ in range(warmup):
            jax.block_until_ready(fn(*args))
    rng = np.random.default_rng(0)
    for _ in range(iterations):
        for idx in rng.permutation(len(names)):
            name = names[idx]
            fn, args = functions[name]
            start = time.perf_counter_ns()
            output = fn(*args)
            jax.block_until_ready(output)
            samples[name].append((time.perf_counter_ns() - start) / 1000)
    return {
        name: dict(
            median_us=float(np.median(values)),
            p10_us=float(np.percentile(values, 10)),
            p90_us=float(np.percentile(values, 90)),
        )
        for name, values in samples.items()
    }


def compile_candidate(config, num_sms, args):
    # Do not close over input tensors: weights remain runtime arguments.
    return jax.jit(candidate_function(config, num_sms)).lower(*args).compile()


def check_kernel_paths(args, cases, num_sms, include_gemv=True):
    """Check representative direct and split-K paths without hiding compilation failures."""
    _, _, activations = args
    m, k = activations.shape
    configurations = []
    for split_k in (1, 2):
        if k % (128 * split_k):
            continue
        config = default_config(m)
        config.update(tile_k=128, split_k=split_k, num_pipeline_stages=2)
        configurations.append(config)
        if m == 1 and include_gemv:
            configurations.append(
                dict(method="gemv", tile_n=64, tile_k=128, split_k=split_k)
            )
    for config in configurations:
        print(f"Checking kernel path: {config}", flush=True)
        compiled = compile_candidate(config, num_sms, args)
        passed, reports = validate(compiled, cases)
        if not passed:
            raise AssertionError(f"Kernel path failed: {config}: {reports[-1]}")
    print(f"All {len(configurations)} representative kernel paths passed.")


def benchmark_gpu(function, inputs, warmup=5, iterations=31):
    report = benchmark(
        function,
        args=inputs,
        warmup=max(1, warmup),
        iterations=iterations,
    )
    return {"median_us": report.median_kernel_time_ms * 1000.0}


def tunable_w8a16(
    weights,
    scale,
    activations,
    method,
    tile_m,
    tile_n,
    tile_k,
    num_pipeline_stages,
    panel_width,
    persistent,
    split_k,
    num_sms,
):
    # tune-jax tries every combination of the grid. Raising here marks a combination as failed, so tune skips it.
    # These checks replace the pruning that enumerate_configs used to do.
    m, k = activations.shape
    n = weights.shape[0]
    if n % tile_n or k % (tile_k * split_k):
        raise ValueError("Tiles do not cover N and K exactly.")

    if method == "gemv":
        # GEMV ignores these four; allow only the first grid value of each so one kernel is not timed many times
        if (tile_m, num_pipeline_stages, panel_width, persistent) != (
            default_tile_m(m),
            2,
            1,
            False,
        ):
            raise ValueError("Duplicate GEMV candidate.")
        return gemv(
            weights, scale, activations, tile_n=tile_n, tile_k=tile_k, split_k=split_k
        )

    tiles_m, tiles_n = pl.cdiv(m, tile_m), n // tile_n
    k_steps = k // (tile_k * split_k)
    if tile_k > 256:
        raise ValueError("tile_k=512 is GEMV-only.")
    if num_pipeline_stages > max(k_steps, 2):
        raise ValueError(
            "matmul clamps stages to k_steps; this would duplicate a smaller stage count."
        )
    if persistent and split_k * tiles_m * tiles_n <= num_sms:
        raise ValueError("Fewer tasks than SMs; persistent adds nothing.")
    if panel_width > 1 and (tiles_m == 1 or panel_width > tiles_n):
        raise ValueError("Panel swizzle has no effect for this tile grid.")
    return matmul(
        weights,
        scale,
        activations,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        num_pipeline_stages=num_pipeline_stages,
        panel_width=panel_width,
        persistent=persistent,
        split_k=split_k,
        num_sms=num_sms,
    )


def tune_for_shape(args, num_sms, include_gemv=True, warmup=3, iterations=15):
    _, _, activations = args
    m = activations.shape[0]
    base_m = default_tile_m(m)

    hyperparams = dict(
        method=["wgmma", "gemv"] if m == 1 and include_gemv else ["wgmma"],
        tile_m=sorted({base_m, min(64, 2 * base_m)}),
        tile_n=[64, 128],
        tile_k=[128, 256, 512],
        num_pipeline_stages=[2, 3, 4],
        panel_width=[1, 4],
        persistent=[False, True],
        split_k=[1, 2, 4, 8],
    )

    tune_logger.setLevel("INFO")
    # example_args: tune on the real INT8 weights and scales rather than on generated inputs
    tuned = jax.jit(
        tune(
            partial(tunable_w8a16, num_sms=num_sms),
            hyperparams=hyperparams,
            example_args=args,
        )
    )
    jax.block_until_ready(tuned(*args))  # first call runs the tuning
    print(tune_jax.tabulate(tuned))

    best = dict(tuned.optimal_hyperparams)
    if (
        best["method"] == "gemv"
    ):  # keep the saved config in the format candidate_function expects
        config = dict(
            method="gemv",
            tile_n=best["tile_n"],
            tile_k=best["tile_k"],
            split_k=best["split_k"],
        )
    else:
        config = best

    compiled = compile_candidate(config, num_sms, args)
    # Re-time the winner with CUPTI so the reported number matches metadata["timing"]
    timing = benchmark_gpu(
        candidate_function(config, num_sms), args, warmup, max(31, iterations)
    )
    return config, compiled, timing


def environment_metadata(device):
    return dict(
        contract=CONTRACT_NAME,
        device=device.device_kind,
        sms=device.core_count,
        jax=jax.__version__,
        jaxlib=jaxlib.__version__,
        timing="cupti_gpu_us",
    )


def read_configs(path, metadata, ignore_stale=False):
    if not path.exists():
        return {}
    stored = json.loads(path.read_text())
    if stored["environment"] != metadata:
        if ignore_stale:
            print(
                "Ignoring configurations from an older implementation or environment."
            )
            return {}
        raise ValueError(
            "Saved configurations are from a different implementation or environment. Run with --tune."
        )
    return stored["configs"]


def save_configs(path, metadata, configs):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(environment=metadata, configs=configs), indent=2) + "\n"
    )
    temporary.replace(path)


def profile(compiled, args, directory, warmup, repetitions):
    directory.mkdir(parents=True, exist_ok=True)
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    with jax.profiler.trace(str(directory)):
        for step in range(repetitions):
            with jax.profiler.StepTraceAnnotation("w8a16_decode", step_num=step):
                jax.block_until_ready(compiled(*args))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check direct and split-K WGMMA/GEMV kernel paths.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Validate and tune WGMMA and batch-1 GEMV candidates.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Compare full-call latency with prepared BF16.",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Capture complete operator traces."
    )
    parser.add_argument(
        "--scenario",
        nargs="+",
        type=int,
        help="Zero-based scenario indices; default: all.",
    )
    parser.add_argument(
        "--configs",
        default="w8a16_configs.json",
        help="Read/save tuned configurations.",
    )
    parser.add_argument(
        "--no-gemv", action="store_true", help="Restrict tuning to Mosaic WGMMA."
    )
    parser.add_argument("--device", type=int, default=0, help="Local GPU index.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--tune-iterations", type=int, default=10)
    parser.add_argument("--profile-repetitions", type=int, default=10)
    parser.add_argument("--profile-dir", default="w8a16_profiles")
    args = parser.parse_args(argv)

    if not any((args.check, args.tune, args.benchmark, args.profile)):
        parser.error("Select --check, --tune, --benchmark, or --profile.")
    if (
        args.warmup < 0
        or min(args.iterations, args.tune_iterations, args.profile_repetitions) <= 0
    ):
        parser.error(
            "Warmup must be nonnegative and repetition counts must be positive."
        )
    if args.scenario is not None and any(
        idx < 0 or idx >= len(SCENARIOS) for idx in args.scenario
    ):
        parser.error("Scenario index is outside the workload list.")
    if args.profile and (args.tune or args.benchmark):
        parser.error("Run --profile separately from --tune and --benchmark.")
    return args


def main(argv=None):
    args = parse_args(argv)
    devices = jax.local_devices(backend="gpu")
    if args.device < 0 or args.device >= len(devices):
        raise ValueError("Invalid local GPU index.")
    device = devices[args.device]
    metadata = environment_metadata(device)
    config_path = Path(args.configs).expanduser()
    configs = read_configs(config_path, metadata, ignore_stale=args.tune or args.check)
    indices = range(len(SCENARIOS)) if args.scenario is None else args.scenario
    key = jax.random.key(0)
    print(json.dumps(metadata, indent=2))
    print("Timing metric: CUPTI GPU execution time per complete call, in microseconds.")

    with jax.default_device(device):
        for idx in indices:
            scenario = SCENARIOS[idx]
            m, k, n = scenario["m"], scenario["k"], scenario["n"]
            shape_key = f"{m},{k},{n}"
            print(f"\n[{idx}] {scenario['desc']}: M={m}, K={k}, N={n}")
            key, weights, scale, activations, prepared_weight = make_inputs(
                key, m, k, n
            )
            inputs = (weights, scale, activations)
            cases = validation_cases(*inputs)
            if args.check:
                check_kernel_paths(
                    inputs, cases, device.core_count, include_gemv=not args.no_gemv
                )
            if args.tune:
                config, compiled, _ = tune_for_shape(
                    inputs,
                    device.core_count,
                    not args.no_gemv,
                    args.warmup,
                    args.tune_iterations,
                )
            else:
                config = configs.get(shape_key, default_config(m))
                compiled = compile_candidate(config, device.core_count, inputs)
                print(f"Config: {config}")

            passed, reports = validate(compiled, cases)
            if not passed:
                raise AssertionError(f"Numerical check failed: {reports[-1]}")
            print(
                f"Checks passed; max absolute error={max(report['max_abs'] for _, report in reports):.6g}"
            )
            print(f"Max NRMSE={max(report['nrmse'] for _, report in reports):.6g}")
            del cases

            if args.benchmark:
                # Pass plain functions; benchmark() handles JIT, compilation, and warmup.
                # Bind only configuration values; keep weights and activations as runtime inputs.
                bf16_args = (prepared_weight, activations)
                functions = {
                    "fused_w8a16": (
                        candidate_function(config, device.core_count),
                        inputs,
                    ),
                    "prepared_bf16": (bf16_matmul, bf16_args),
                    "jax_w8a16": (simple_w8a16_matmul, inputs),
                }

                # Measure GPU work, including split-K reduction and other kernels in each call.
                # The existing tuner still selects configurations using synchronized host timing.
                reports = {}
                for name, (function, call_args) in functions.items():
                    report = benchmark(
                        function,
                        args=call_args,
                        warmup=max(1, args.warmup),
                        iterations=args.iterations,
                    )
                    reports[name] = report
                    print(
                        f"{name:18} CUPTI median GPU time: {report.median_kernel_time_ms * 1000:.2f} us"
                    )

                fused_ms = reports["fused_w8a16"].median_kernel_time_ms
                bf16_ms = reports["prepared_bf16"].median_kernel_time_ms
                print(
                    f"Fused versus prepared BF16: {format_relative_perf(fused_ms, bf16_ms)}"
                )

                # Preserve the quantization-quality comparison outside timing.
                original = np.asarray(
                    jax.device_get(jax.jit(bf16_matmul)(*bf16_args)), dtype=np.float32
                )
                quantized = np.asarray(
                    jax.device_get(compiled(*inputs)), dtype=np.float32
                )
                relative_l2 = np.linalg.norm(quantized - original) / max(
                    np.linalg.norm(original), 1e-12
                )
                print(
                    f"Output relative L2 error versus original BF16 weights: {relative_l2:.6g}"
                )
            if args.profile:
                directory = Path(args.profile_dir).expanduser() / f"m{m}_k{k}_n{n}"
                profile(
                    compiled, inputs, directory, args.warmup, args.profile_repetitions
                )
                print(f"Trace: {directory}")


if __name__ == "__main__":
    main()
