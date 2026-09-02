---
title: "How a Hopper GPU Keeps Matrix Multiplication Moving"
description: "A visual guide to matrix tiling, cache-friendly traversal, pipelined data movement, and writeback in a Pallas-JAX WGMMA kernel."
---

Fast matrix multiplication is not only about doing more math. It is also about keeping data close to the hardware that needs it and moving the next block before the current one is finished.

This visualization follows one output tile through a Hopper WGMMA kernel written in Pallas. It shows the full path:
- split the matrices into tiles
- visit those tiles in a cache-friendly order
- overlap data loading with Tensor Core work, and
- write the result back to global memory.

[![Hopper WGMMA pipeline](../visualizations/simple_matmul-preview.gif)](
  https://aakashkumarnain.github.io/pallasforge/visualizations/simple_matmul.html
)

## 1. Break the problem into tiles

The kernel multiplies a left matrix, `LHS (M × K)`, by a right matrix, `RHS (K × N)`, to produce `Output (M × N)`.

These matrices are too large to process at once, so the kernel divides them into smaller tiles. Each worker handles one output tile. It loads a small block from each input, multiplies them, and adds the partial result to a running sum. It repeats this along K until the tile is complete. Tiling gives the hardware a manageable unit of work and lets the kernel keep frequently used data in shared memory and registers. 

In this example, each output tile has 64 rows and 128 columns. One reduction step combines a `64 × 128` LHS tile with a `128 × 128` RHS tile. The 128-wide K dimension is the shared dimension being summed. These sizes also meet the alignment rules used by the Hopper WGMMA path. Here, `tile_m` is a multiple of 64 for Hopper WGMMA, while `tile_n` is a multiple of 8 for Tensor Core alignment. If K is larger than 128, the kernel repeats this work across `K / tile_k` steps and adds every partial result to the same output tile.

## 2. Visit tiles in a cache-friendly order

After choosing the tile size, we still need to decide which output tile to compute next. That order affects how much input data can be reused from the GPU's L2 cache. A simple row-by-row scan moves left to right, then jumps back to the first column at the start of every new row. That jump can leave behind the RHS data that was just loaded. 

A snake path avoids the jump. It moves left to right across one row, then right to left across the next. A persistent worker stays alive on the same SM, and the snake order lets it begin the next row where the last one ended. The RHS columns it just used therefore have a better chance of still being warm in L2.

The work is also split into narrow panels. The example uses a panel width of four output columns. This limits the number of RHS columns in play at once, keeps the working set smaller, and improves the odds of cache reuse. Four is not a universal rule; the best width depends on the matrix shape and the hardware.

## 3. Overlap loading with Tensor Core work

Global memory, also called HBM, is large but far from the compute units. If the kernel waits for every load before doing more math, the Tensor Cores sit idle. TMA can move tiles from HBM into shared memory without tying up the main compute path. WGMMA can then run a warp-group matrix multiply-accumulate on data that is already in shared memory.

The kernel avoids that wait with a four-stage circular pipeline in shared memory, or SMEM. TMA, Hopper's Tensor Memory Accelerator, moves LHS and RHS tiles from HBM to SMEM without making the main compute path wait for each copy. At the same time, WGMMA runs warp-group matrix multiply-accumulate work on a tile that is already ready.

To overlap those jobs, the kernel uses a circular pipeline. The example has four shared-memory slots. At any moment, one slot can feed WGMMA, some can hold tiles that are ready, and another can receive a future `K` tile. Once a slot has been consumed, the kernel wraps around and fills it again. Four slots do not promise that every memory delay will disappear, but they can hide much of it when loading and compute are well balanced.

Each WGMMA step updates the same (`64 × 128` in this example) accumulator:

`acc += lhs_smem @ rhs_smem`

The accumulator stays in registers and uses FP32. That avoids sending partial sums back to memory, and FP32 gives the running total more precision than bfloat16.


## 4. Finish, cast and write the tile back

After all K steps are complete, the kernel enters its final stage. It converts the FP32 accumulator to bfloat16, places the result in `out_smem`, and commits that shared-memory data so it is ready to copy.

That buffer uses `SwizzleTransform(128)`, which lays out the values to reduce shared-memory bank conflicts. The kernel commits the shared-memory writes, starts an asynchronous copy to the correct output slice in HBM, and waits for that copy to finish before it reuses the buffer.

That is the full pattern: tile the matrices, visit the tiles in an order that encourages reuse, overlap memory traffic with Tensor Core work, keep the running sum in registers, and write the finished tile back only once. The arithmetic matters, but on a GPU, moving data well is what keeps the arithmetic moving.
