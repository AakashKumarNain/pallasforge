import inspect
import math
import operator
import pathlib
import shutil
import tempfile
import time
from dataclasses import dataclass
from contextlib import contextmanager

import jax
import numpy as np
from jax.extend import backend
from jax.experimental.mosaic.gpu import profiler


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Compilation costs, optional compiler memory estimate, and GPU timings."""

    lower_time_ms: float
    compile_time_ms: float
    peak_memory_mb: float | None
    cupti_times_ms: tuple[float, ...]

    @property
    def median_kernel_time_ms(self):
        return float(np.median(self.cupti_times_ms))

    @property
    def max_kernel_time_ms(self):
        return float(np.max(self.cupti_times_ms))

    @property
    def min_kernel_time_ms(self):
        return float(np.min(self.cupti_times_ms))

    def print_summary(self):
        memory = (
            "unavailable"
            if self.peak_memory_mb is None
            else f"{self.peak_memory_mb:.2f} MB"
        )
        print(f"Static Peak Memory  : {memory}\n")
        print(f"Lowering Time       : {self.lower_time_ms:.3f} ms")
        print(f"Compilation Time    : {self.compile_time_ms:.3f} ms\n")
        print(f"Min GPU Time        : {self.min_kernel_time_ms:.4f} ms")
        print(f"Max GPU Time        : {self.max_kernel_time_ms:.4f} ms")
        print(f"Median GPU Time     : {self.median_kernel_time_ms:.4f} ms")


def get_max_smem_bytes(device=None):
    """Return the device's exposed per-block opt-in limit, or None if unavailable."""
    device = backend.get_default_device() if device is None else device
    if device.platform != "gpu":
        raise ValueError("Shared-memory limits require a GPU device.")
    limit = getattr(device, "shared_memory_per_block_optin", None)
    return int(limit) if limit is not None and limit > 0 else None


def format_relative_perf(kernel_ms, reference_ms):
    if not all(
        math.isfinite(duration) and duration > 0
        for duration in (kernel_ms, reference_ms)
    ):
        return "N/A"
    if kernel_ms == reference_ms:
        return "same speed"
    if kernel_ms < reference_ms:
        return f"{reference_ms / kernel_ms:.2f}x faster"
    return f"{kernel_ms / reference_ms:.2f}x slower"


def benchmark(
    function,
    *,
    args=(),
    kwargs=None,
    static_argnames=None,
    static_argnums=None,
    warmup=5,
    iterations=10,
    **jit_kwargs,
):
    """Benchmark a plain, pure function on one local NVIDIA GPU.

    Static argument options follow jax.jit exactly. At least one warmup is required.
    Donation is unsupported because every execution reuses the same inputs.
    Compilation times include any effects of JAX's compilation caches.
    CUPTI uses finalize=False. Run XProf in a separate process.
    """

    warmup = operator.index(warmup)
    iterations = operator.index(iterations)
    if warmup < 1 or iterations < 1:
        raise ValueError("warmup and iterations must both be at least 1.")

    for option in ("donate_argnums", "donate_argnames"):
        donation = jit_kwargs.pop(option, None)
        if donation is not None and np.size(donation):
            raise ValueError(
                "Donation is unsupported because benchmark inputs are reused."
            )

    args = tuple(args)
    kwargs = {} if kwargs is None else dict(kwargs)

    if static_argnums is not None:
        static_argnums = tuple(
            map(operator.index, np.atleast_1d(static_argnums).tolist())
        )
    if static_argnames is not None:
        static_argnames = tuple(np.atleast_1d(static_argnames).tolist())

    jitted = jax.jit(
        function,
        static_argnames=static_argnames,
        static_argnums=static_argnums,
        **jit_kwargs,
    )

    start = time.perf_counter()
    lowered = jitted.lower(*args, **kwargs)
    lower_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    compiled = lowered.compile()
    compile_ms = (time.perf_counter() - start) * 1000.0

    # Compiled calls omit static arguments. JAX infers positions from
    # names only when argnums is None.
    static_positions = set(static_argnums or ())
    if static_argnums is None and static_argnames:
        try:
            parameters = inspect.signature(function).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        static_positions = {
            index
            for index, parameter in enumerate(parameters)
            if parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            and parameter.name in static_argnames
        }
    static_positions = {
        index if index >= 0 else len(args) + index for index in static_positions
    }
    call_args = tuple(
        value for index, value in enumerate(args) if index not in static_positions
    )
    # The compiled metadata already reflects JAX's static-name inference and retained keyword arguments.
    call_kwargs = {name: kwargs[name] for name in compiled.args_info[1]}

    # Inspect the actual executable target, which can differ from the default backend.
    executable = compiled.runtime_executable()

    if executable is None:
        raise RuntimeError("The compiled executable is unavailable.")

    devices = executable.local_devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("This benchmark requires exactly one local NVIDIA GPU.")
    if "cuda" not in devices[0].client.platform_version.lower():
        raise ValueError("CUPTI requires an NVIDIA CUDA backend.")

    # Memory analysis is a compiler estimate, not a live GPU memory measurement.
    try:
        memory = compiled.memory_analysis()
    except NotImplementedError:
        memory = None
    peak_bytes = getattr(memory, "peak_memory_in_bytes", None)
    peak_mb = peak_bytes / 1e6 if peak_bytes is not None and peak_bytes >= 0 else None

    for iteration in range(warmup):
        jax.block_until_ready(compiled(*call_args, **call_kwargs))

    # Construct one timer and call it separately for each sample.
    # Keep finalize=False; do not pass iterations to the CUPTI helper.
    measure = profiler.Cupti(finalize=False).measure(compiled)
    timings = []
    for iteration in range(iterations):
        duration = measure(*call_args, **call_kwargs)[1]
        if duration is None:
            raise RuntimeError("CUPTI recorded no kernel launches.")
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise RuntimeError("CUPTI returned no positive, finite GPU execution time.")
        timings.append(duration)

    return BenchmarkReport(lower_ms, compile_ms, peak_mb, tuple(timings))


@contextmanager
def profile_xprof(profile_dir=None, event_filter_regex=None, retain_trace=False):
    """Collect device timing and optionally retain the trace.

    Compile and warm up before entering. Inside the context, call
    jax.block_until_ready(result) for every output tree whose work must be captured.
    Read the yielded dictionary only after leaving the context.
    The trace is deleted unless retain_trace is True; then stats includes trace_dir.
    Run this in a separate process from CUPTI benchmarks to isolate profiler state.
    Do not nest this context with another profiling session.
    """

    if jax.default_backend() not in ("gpu", "tpu"):
        raise ValueError("XProf profiling requires a GPU or TPU backend.")

    from xprof.cli.tools import get_kernel_stats_tool

    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 0
    options.enable_hlo_proto = False
    if jax.default_backend() == "tpu":
        options.advanced_configuration = {
            "tpu_trace_mode": "TRACE_ONLY_XLA",
            "tpu_perf_counters": True,
        }

    parent = (
        None
        if profile_dir is None
        else pathlib.Path(profile_dir).expanduser().resolve()
    )
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)

    run_dir = pathlib.Path(tempfile.mkdtemp(prefix="xprof_profile_", dir=parent))
    stats = {}

    try:
        with jax.profiler.trace(str(run_dir), profiler_options=options):
            yield stats

        trace_files = list(run_dir.glob("**/*.xplane.pb"))
        if len(trace_files) != 1:
            raise RuntimeError(
                f"Expected one XPlane trace, found {len(trace_files)} in {run_dir}."
            )

        profile = jax.profiler.ProfileData.from_serialized_xspace(
            trace_files[0].read_bytes()
        )
        matchers = None if event_filter_regex is None else (event_filter_regex,)
        summary = get_kernel_stats_tool.compute_kernel_stats(
            profile,
            output_format="dict",
            include_summary=True,
            trace_matchers=matchers,
        )

        if "total_device_duration_us" not in summary:
            raise RuntimeError(
                "XProf did not return total_device_duration_us; check your XProf version."
            )

        duration_us = float(summary["total_device_duration_us"])

        if not math.isfinite(duration_us) or duration_us <= 0:
            raise RuntimeError(
                "No positive device time was captured. Check blocking and the event filter."
            )

        # The metric merges overlapping device intervals and excludes gaps.
        stats.update(total_device_time_ms=duration_us / 1000.0, summary=summary)
        if retain_trace:
            stats["trace_dir"] = run_dir
        else:
            shutil.rmtree(run_dir, ignore_errors=True)
    except BaseException:
        # Cleanup must not hide the original tracing or user-code exception.
        shutil.rmtree(run_dir, ignore_errors=True)
