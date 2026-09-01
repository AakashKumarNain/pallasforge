# Contributor Guidelines

Thanks for contributing! We want to keep this codebase clean, readable, and consistent across all GPU kernels and benchmarking utilities. Please follow these guidelines before opening a pull request.

---

## 1. Kernel Variable Naming (CUDA / Generic Convention)

When writing GPU kernels, stick to standard CUDA and public GPU naming conventions. This helps everyone read and optimize memory access patterns and thread hierarchies without confusion. This is not a hard constraint, but it helps the end user to get a grasp of kernels better when moving from one example to another. You can look at the examples provided for reference.

---

## 2. Metric & Timing Variable Conventions

Always include units as suffixes for time and memory variables across all benchmarking and profiling code:

* **Time measurements:** Always append the unit: `lower_time_ms`, `compile_time_ms`, `cupti_times_ms`, `duration_us`.
* **Memory measurements:** Always append the unit: `peak_memory_mb`, `smem_bytes`, `max_smem_bytes`.
* **Profiling targets:** Stick to NVIDIA CUPTI naming for hardware metrics (e.g., `cupti_times_ms`, `cupti_runner`).

---

## 3. Linting and Formatting with Ruff

We use **[Ruff](https://docs.astral.sh/ruff/)** to keep formatting and code style consistent.

Before submitting a pull request, run the following commands:

```bash
ruff check
ruff format
```