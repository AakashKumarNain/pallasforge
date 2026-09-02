# PallasForge

Pallas is an extension to JAX for writing custom kernels for GPUs and TPUs. Though many features are still `experimental`, it has matured a lot over the past few years. Though Pallas became better in terms of functionality and documentation, there are still many blockers for people trying to use Pallas, especially on GPUs. For example, the [JAX documentation](https://docs.jax.dev/en/latest/pallas/gpu/index.html) despite being extremely detailed, only contains a few specific generic examples.

There are three major gaps that we want to address:

1. **Examples**: We want to target real-world examples here for different families of GPUs. Instead of showcasing how to write a generic matmul kernel using pallas for Hopper or Blackwell, we want to showcase things like fused dequantization schemes, flash-attention variants, fused-norm, etc.
2. **GPU performance benchmarks**: Writing kernel alone is not enough. People can look up the API, and use a coding agent like codex or claude to come up with a kernel for a specific use case, but it does not provide a full picture on performance. Unless you know the baseline, it is hard to know whether your kernel is efficient or not. For example, even for a simple matmul kernel, you need to know whether you are making full use of the hardware. This exercise has to be repeated for different GPUs (e.g. Hopper, Blackwell, etc.)
3. **Profiling**: Hardly any tutorial out there that teaches you the right way to profile your kernels written in Pallas on GPUs. We want this repo to become the goto example for profiling section. Profiling should be intuitive for the end user, and the current resources hardly cut it. 

## Project Structure

```
pallasforge/
├── docs
│   ├── GUIDELINES.md                   # Contributor guide / style guide
│   └── PROPOSALS.md                    # Future kernel RFCs / roadmap
├── examples
├── LICENSE
├── pallasforge                         # Main package containing all tutorials & shared tooling
│   ├── __init__.py
│   ├── common                          # Common utilities related to pallas, device, and profiling
│   │   ├── __init__.py
│   │   └── utils.py
│   └── hopper                          # Architecture track (Hopper: H100/H200)
│       ├── __init__.py
│       ├── matmul
│       └── media
├── pyproject.toml

```

## Tutorials

Tutorials are provided as a reference for starters along with educational material to help understand the concepts.

1. **Hopper (H100, H200)**
    - Matrix Multiplication
        - [Overview and Fundamentals](./pallasforge/hopper/matmul/hopper-wgmma-pipeline.md)
        - [Basic Matrix Multiplication (BF16 @ BF16)](./pallasforge/hopper/matmul/bf16_matmul.py)
        - [Quantized Matrix Multiplication (W8 @ A16 with Fused Dequantization)](./pallasforge/hopper/matmul/fused_w8a16.py)

2. **Blackwell (B100, B200)**

<br><br>

---

<br>

We are taking a slightly different approach to open source with this repository. Instead of building the project behind closed doors and asking for contributions later, we want to develop it *with the community* from day one. This is an open invitation to anyone who loves writing GPU kernels; whether you are a solo developer learning the ropes, or part of the core JAX-Pallas and NVIDIA JAX teams.

If you want to get involved, you can find a list of planned kernels and ideas listed in the [PROPOSALS](./docs/PROPOSALS.md) document. Please take a look at the [code guidelines](./docs/GUIDELINES.md) before opening a PR. 
