import time
import pathlib
import shutil
import tempfile
import dataclasses
from contextlib import contextmanager
from typing import Any, Callable, Sequence, Iterator

import jax
import jax.numpy as jnp
from jax.experimental.mosaic.gpu import profiler
from xprof.cli.tools import get_kernel_stats_tool


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Container for compilation metrics and GPU hardware execution timings.

    Attributes:
        lower_time_ms: Wall-clock time taken to trace and lower the function into
            HLO/StableHLO (in milliseconds).
        compile_time_ms: Wall-clock time taken by XLA backend compilation into
            machine code (in milliseconds).
        peak_memory_mb: Static peak memory requirement of the compiled executable
            in megabytes.
        cupti_times_ms: Tuple of individual GPU device execution durations measured
            via CUPTI (in milliseconds).
    """

    lower_time_ms: float
    compile_time_ms: float
    peak_memory_mb: float
    cupti_times_ms: tuple[float, ...]

    @property
    def median_kernel_time_ms(self):
        """Computes the median GPU execution time across measured iterations."""
        return float(jnp.median(jnp.array(self.cupti_times_ms)))

    def print_summary(self):
        """Prints a formatted summary of the benchmark metrics."""
        print(f"Lowering Time       :  {self.lower_time_ms:.3f} ms")
        print(f"Compilation Time    :  {self.compile_time_ms:.3f} ms")
        print(f"Peak Device Memory  :  {self.peak_memory_mb:.2f} MB")
        print(f"Median Kernel Time  :  {self.median_kernel_time_ms:.4f} ms")


def benchmark(
    fn: Callable[..., Any],
    *,
    args: Sequence[Any] = (),
    kwargs: dict[str, Any] | None = None,
    static_argnames: str | Sequence[str] | None = None,
    static_argnums: int | Sequence[int] | None = None,
    warmup: int = 5,
    iterations: int = 10,
    **jit_kwargs: Any,
) -> BenchmarkReport:
    """JIT-compiles, lowers, and benchmarks a plain Python function using Cupti.

    This utility wraps a standard Python function with `jax.jit`, lowers it for
    the provided inputs, compiles the executable, does static memory analysis,
    and measures on-device kernel runtimes via NVIDIA CUPTI.

    Args:
        fn: Plain Python function to JIT, lower, compile, and benchmark.
        args: Positional arguments to forward to `fn` (both dynamic arrays and
            positional static values).
        kwargs: Keyword arguments to forward to `fn` (both dynamic arrays and
            keyword static configurations).
        static_argnames: Names of keyword arguments that should be treated as
            compile-time static constants.
        static_argnums: Indices of positional arguments that should be treated as
            compile-time static constants.
        warmup: Number of warmup executions.
        iterations: Number of timed executions to record using the CUPTI profiler.
        **jit_kwargs: Additional options to pass to `jax.jit` (e.g., `donate_argnums`,
            `inline=True`).

    Returns:
        A `BenchmarkReport` containing lowering duration, XLA compile time, static
        peak device memory usage, and recorded CUPTI kernel durations.

    Raises:
        ValueError: If called on a non-GPU platform.
        TypeError: If arguments provided in `args`/`kwargs` do not match `fn`'s signature.

    Example:
        >>> def tiled_reduction(x: jax.Array, tile_size: int, scale: float = 1.0):
        ...   return jnp.sum(x.reshape(-1, tile_size), axis=0) * scale
        ...
        >>> data = jnp.ones((2048, 2048), dtype=jnp.float32)
        >>> report = benchmark(
        ...     tiled_reduction,
        ...     args=(data,),
        ...     kwargs={"tile_size": 64, "scale": 2.0},
        ...     static_argnames=("tile_size",),
        ...     warmup=3,
        ...     iterations=15,
        ... )
        >>> report.print_summary()
    """

    kwargs = kwargs or {}

    # 1. JIT compile the plain function with specified static args and options
    jitted_fn = jax.jit(
        fn,
        static_argnames=static_argnames,
        static_argnums=static_argnums,
        **jit_kwargs,
    )

    # 2. Lowering stage (traces into HLO / StableHLO IR)
    t0 = time.perf_counter()
    lowered = jitted_fn.lower(*args, **kwargs)
    lower_time_ms = (time.perf_counter() - t0) * 1000.0

    # 3. XLA compilation stage (compiles HLO into GPU machine code)
    t0 = time.perf_counter()
    compiled = lowered.compile()
    compile_time_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Static peak device memory analysis
    mem_info = compiled.memory_analysis()
    peak_mem_mb = mem_info.peak_memory_in_bytes / 1e6 if mem_info is not None else 0.0

    # 5. Warmup executions (lowered handles dynamic/static dispatch internally)
    for _ in range(warmup):
        compiled(*args).block_until_ready()

    # 6. Device kernel runtime measurement with CUPTI
    runner = profiler.Cupti(finalize=False).measure(compiled)

    timings = []
    for _ in range(iterations):
        res, duration_ms = runner(*args)
        res.block_until_ready()
        timings.append(duration_ms)

    return BenchmarkReport(
        lower_time_ms=lower_time_ms,
        compile_time_ms=compile_time_ms,
        peak_memory_mb=peak_mem_mb,
        cupti_times_ms=tuple(timings),
    )


@contextmanager
def profile_xprof(profile_dir=None, event_filter_regex=None):
    """Profiles XLA device operations and collects XProf kernel statistics.

    Creates a dedicated profiling run directory and records a JAX device trace
    while the context manager body executes. After tracing completes, the
    resulting XPlane trace is processed using XProf's kernel statistics tool.

    Profiling artifacts are retained only after a successful run. If tracing or
    post-processing fails, the run directory is removed so incomplete profiling
    artifacts are not left behind.

    The yielded dictionary is populated after the context manager body completes
    successfully.

    Args:
        profile_dir: Parent directory in which to create the profiling run
            directory. If not provided, a directory is created in the system
            temporary directory. Defaults to `None`
        event_filter_regex: Optional regular expression used to restrict which trace
            events are included when computing kernel statistics.

    Yields:
        A mutable dictionary populated with profiling results after successful
        completion. It contains:

        - "total_device_time_ms": Total matched device execution time in milliseconds.
        - "summary": Full dictionary returned by XProf kernel statistics.
        - "trace_dir": Path to the retained profiling run directory.

    Raises:
        ValueError: If the active JAX backend is CPU.
        RuntimeError: If profiling completes without producing an XPlane trace file.
        Exception: Propagates exceptions raised by the profiled code or XProf
            post-processing after cleaning up the incomplete run directory.

    Example:
        >>> with profile_xprof() as stats:
        ...    result = jax.jit(fn)(inputs)
        ...    jax.block_until_ready(result)
        ...
        >>> print(stats["total_device_time_ms"])
        >>> print(stats["trace_dir"])
    """

    if jax.default_backend() == "cpu":
        raise ValueError("XProf profiling requires GPU or TPU backend.")

    if profile_dir is not None:
        parent_path = pathlib.Path(profile_dir)
        parent_path.mkdir(parents=True, exist_ok=True)
        run_dir = pathlib.Path(tempfile.mkdtemp(prefix="run_", dir=parent_path))
    else:
        run_dir = pathlib.Path(tempfile.mkdtemp(prefix="xprof_profile_"))

    stats = {}
    completed = False

    try:
        with jax.profiler.trace(str(run_dir)):
            yield stats

        trace_files = list(run_dir.glob("**/*.xplane.pb"))
        if not trace_files:
            raise RuntimeError(
                f"No profile trace file found in {run_dir}. Ensure device operations "
                "were executed and blocked using `jax.block_until_ready()`."
            )

        profile_data = jax.profiler.ProfileData.from_serialized_xspace(
            trace_files[0].read_bytes()
        )

        matchers = (event_filter_regex,) if event_filter_regex else None
        summary = get_kernel_stats_tool.compute_kernel_stats(
            profile_data,
            output_format="dict",
            include_summary=True,
            trace_matchers=matchers,
        )

        device_time_us = summary.get("total_device_duration_us", 0.0)

        stats["total_device_time_ms"] = device_time_us / 1000.0
        stats["summary"] = summary
        stats["trace_dir"] = run_dir

        completed = True

    finally:
        if not completed and run_dir.exists():
            shutil.rmtree(run_dir)
