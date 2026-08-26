import time
import dataclasses
import jax.numpy as jnp
from jax.extend import backend
from jax.experimental.mosaic.gpu import profiler


@dataclasses.dataclass(frozen=True)
class BenchmarkReport:
    lower_time_ms: float
    compile_time_ms: float
    peak_memory_mb: float
    cupti_times_ms: list[float]

    @property
    def median_kernel_time_ms(self):
        return float(jnp.median(jnp.array(self.cupti_times_ms)))

    @property
    def print_benchmarks(self):
        print(f"lowering time : {self.lower_time_ms:.3f} ms")
        print(f"compile time  : {self.compile_time_ms:.3f} ms")
        print(f"kernel time   : {self.median_kernel_time_ms():.3f} ms")
        print(f"memory usage  : {self.peak_memory_mb:.2f} MB")


def get_max_smem_bytes():
    """Get the maximum available shared memory on a single device!"""
    gpu = backend.get_default_device()
    smem_bytes = getattr(gpu, "shared_memory_per_block_optin", None)
    return int(smem_bytes) if smem_bytes else None


def get_compile_time(lowered_fn):
    """Measures the compilation time (in ms) of a lowered function.

    Args:
        lowered_fn: Function that has already been jitted and lowered
    Returns:
        compiled_fn, and compilation time in milliseconds
    """

    start = time.perf_counter()
    compiled = lowered_fn.compile()
    end = time.perf_counter()
    duration_ms = (end - start) * 1000
    return compiled, duration_ms


def get_lowering_time(jitted_fn, inputs):
    """Measures the lowering time (in ms) of a jitted function.

    Args:
        jitted_fn: Function that has already been jitted
        inputs: Inputs to be passed to the jitted function
    Returns:
        lowered_fn, and lowering time in milliseconds
    """

    start = time.perf_counter()
    lowered = jitted_fn.lower(*inputs)
    end = time.perf_counter()
    duration_ms = (end - start) * 1000
    return lowered, duration_ms


def benchmark_cupti_runtime(jitted_fn, inputs, warmup_iteratons=5, iterations=10):
    for _ in range(warmup_iteratons):
        jitted_fn(inputs).block_until_ready()

    timings = []
    for _ in range(iterations):
        res, duration_ms = profiler.Cupti(finalize=False)(jitted_fn)(*inputs)
        res.block_until_ready()
        timings.append(duration_ms)
    return timings


def get_memory_usage(compiled_fn):
    """Return the peak memory usage by the compiled function"""
    memory_analysis = compiled_fn.memory_analysis()
    peak_mem_mb = memory_analysis.peak_memory_in_bytes / 1e6 if memory_analysis else 0.0
    return peak_mem_mb


def benchmark(jitted_fn, inputs, warmup_iterations=5, iterations=100):
    """Create a benchmark report for a jitted function.
    Args:
        jitted_fn: Function that has already been jitted
        inputs: Inputs to be passed to the jitted function
        warmup_iterations: Num of times to run the function to warm up
        iterations: Number of times to measure the actual performance

    Returns:
        A benchmark report for the jitted function
    """

    lowered_fn, lowered_time = get_lowering_time(jitted_fn, inputs)
    compiled_fn, compiled_time = get_compile_time(lowered_fn)
    memory_usage = get_memory_usage(compiled_fn)
    kernel_time = benchmark_cupti_runtime(
        jitted_fn, inputs, warmup_iteratons=warmup_iterations, iterations=iterations
    )
    return BenchmarkReport(
        compile_time_ms=compiled_time,
        lower_time_ms=lowered_time,
        peak_memory_mb=memory_usage,
        cupti_times_ms=kernel_time,
    )
