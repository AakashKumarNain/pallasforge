# Pallas Forge

Pallas is an extension to JAX for writing custom kernels for GPUs and TPUs. Though many features are still `experimental`, it has matured a lot over the past few years. Though Pallas became better in terms of functionality and documentation, there are still many blockers for people trying to use Pallas, especially on GPUs. For example, the [JAX documentation](https://docs.jax.dev/en/latest/pallas/gpu/index.html) despite being extremely detailed, only contains a few specific generic examples.

There are three major gaps that we want to address:

1. **Examples**: We want to target real-world examples here for different families of GPUs. Instead of showcasing how to write a generic matmul kernel using pallas for Hopper or Blackwell, we want to showcase things like fused dequantization schemes, flash-attention variants, fused-norm, etc.
2. **GPU performance benchmarks**: Writing kernel alone is not enough. People can look up the API, and use a coding agent like codex or claude to come up with a kernel for a specific use case, but it does not provide a full picture on performance. Unless you know the baseline, it is hard to know whether your kernel is efficient or not. For example, even for a simple matmul kernel, you need to know whether you are making full use of the hardware. This exercise has to be repeated for different GPUs (e.g. Hopper, Blackwell, etc.)
3. **Profiling**: Hardly any tutorial out there that teaches you the right way to profile your kernels written in Pallas on GPUs. We want this repo to become the goto example for profiling section. Profiling should be intuitive for the end user, and the current resources hardly cut it. 


## Community-driven project

Given how people have changed the meaning of OSS, we want to take a different approach here. Instead of starting a project and then calling for community contributions, we want to develop it *with the community* from day one. Community here does not mean only individuals, but it also includes big teams who love writing kernels, and want to contribute to OSS, JAX/Pallas, and GPUs. We are eager and very much looking forward to contributions from the JAX-Pallas team, and Nvidia's JAX teams.

Please refer to the ideas listed [here](./PROPOSALS.md) for contributions.
