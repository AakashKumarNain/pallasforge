---
title: "How a Hopper GPU Keeps Matrix Multiplication Moving"
description: "A visual guide to matrix tiling, cache-friendly traversal, pipelined data movement, and writeback in a Pallas-JAX WGMMA kernel."
---

Whne you do a matrix multiplication in JAX as $C = AB$, the compiler chooses how to run it on the acceleration (a GPU in our case). You can think of Pallas kernel for the same operation as a function that specifies how the GPU computes the result. You choose how much of each matrix to load at a time, where to store it, and when to load the next part. These choices matter because the GPU can spend time waiting for data even when it has plenty of multiplication work left to do.

This visualization follows one small part of the output through a Pallas kernel for NVIDIA Hopper GPUs. That part is called a tile. The kernel uses WGMMA, a Hopper operation that multiplies two input tiles and adds their product to a running sum. We will follow four steps:
- split the matrices into tiles
- visit those tiles in a cache-friendly order
- overlap data loading with Tensor Core work, and
- write the result back to global memory.

https://github.com/user-attachments/assets/1054424d-a09f-4e0d-b60b-4fbecd2527e3


To view the interactive video, open [this](../media/bf16_matmul.html) in your browser.

*Note: This visualization was generated using Gemini and may contain minor inaccuracies. Use it to understand the basic ideas; it may not match every detail of the implementation.*

---

## 1. Break the problem into tiles

The left matrix, LHS, has shape $M \times K$. The right matrix, RHS, has shape $K \times N$. Their product has shape $M \times N$. One output value is calculated as $C_{m,n} = \sum_{k=0}^{K-1} A_{m,k} B_{k,n}$, where $A$ is LHS and $B$ is RHS.

The kernel computes one output tile at a time. It starts with a tile of zeros, loads a block from each input, and adds their product to it. The kernel then moves along the columns of LHS and the matching rows of RHS to get the next pair of blocks. Each pair contributes to the same output tile. The tile is complete only after the kernel has covered all $K$ entries. Computing several output values together also lets the kernel reuse each loaded input value across multiple products. 

Here, the output tile has shape ($64 \times 128$). Each step multiplies a ($64 \times 128$) LHS tile by a ($128 \times 128$) RHS tile, producing a ($64 \times 128$) partial result. In the code, `tile_m=64` and `tile_n=128` set the output tile size, while `tile_k=128` sets how far each step moves along $K$. If $K=256$, the kernel takes two steps and adds their results. More generally, it takes ($K / \mathrm{tile\_k}$) steps when $K$ is a multiple of `tile_k`. The tile sizes also have to fit the multiplication operation: the Hopper WGMMA path used here requires `tile_m` to be a multiple of 64 and `tile_n` to be a multiple of 8.

## 2. Visit tiles in a cache-friendly order

Two output tiles in the same tile column use the same columns of RHS, even though they use different rows of LHS. Computing them close together in time can avoid another read from the other memory: the RHS data may still be in L2, a cache shared across the GPU. A row-by-row scan works against this at the row boundary. After computing the rightmost tile, it jumps back to the leftmost tile of the next row, which needs a different set of RHS columns. 

The snake order goes left to right across one row of output tiles, then right to left across the next. At the turn, the next output tile needs the same RHS columns as the previous one. This makes it more likely that those values are still in L2. The example also uses persistent workers: each worker computes several output tiles before it finishes, rather than stopping after one. A worker runs on an SM, or streaming multiprocessor, one of the GPU units that executes the kernel. Persistence lets a worker continue taking tiles from the chosen order, but it does not guarantee that the data will stay in cache.

The kernel applies this order within narrow panels of output tiles. Each panel in the example is four tile columns wide. Since one output tile is 128 columns wide, a panel covers 512 output columns. Finishing work within that panel keeps the kernel returning to the same group of RHS columns before moving to the next panel. This reduces the amount of RHS data competing for cache space. A width of four is a choice for this example; a different matrix or GPU may work better with another width.

## 3. Overlap loading with Tensor Core work

The full input matrices live in global memory, also called HBM. Before multiplying a pair of tiles, this kernel copies them into shared memory, or SMEM, a smaller and faster memory available to the threads in a block. The Tensor Cores, which perform the matrix multiplication, read the tiles from there. If every step waits to load its inputs until the previous multiplication finishes, loading and multiplication take turns. The kernel tries to do both at once.

TMA, Hopper's Tensor Memory Accelerator, handles the copies into shared memory. A copy is asynchronous: the kernel can start it and do other work while it runs. WGMMA, short for warp-group matrix multiply-accumulate, performs the multiplication and adds the result to the running sum. A warp-group is a group of 128 GPU threads that issue this operation together. Here, both input tiles come from shared memory. Once one pair of tiles is ready, WGMMA can use it while TMA loads a later pair into a different buffer.

The four-stage pipeline provides four slots, each with space for one LHS tile and one RHS tile. For example, WGMMA can read the first slot while TMA fills later slots. After the fourth slot, the pipeline returns to the first and reuses it for a later step along $K$. Reuse must wait until WGMMA has finished reading the old tiles, and a multiplication must wait until its input copies are complete. Those waits prevent the two operations from overwriting or reading unfinished data. The extra slots give loading a head start, but the Tensor Cores will still have to wait if the copies fall behind.

The running sum is called the accumulator. It starts at zero and has the same shape as the output tile, $64 \times 128$ in this example. Each WGMMA step performs this update, where $A_{\mathrm{tile}}$ and $B_{\mathrm{tile}}$ are the current input tiles:

$\mathrm{acc} \leftarrow \mathrm{acc} + A_{\mathrm{tile}} B_{\mathrm{tile}}$

The accumulator stays in registers, the storage used directly by the executing threads. The kernel keeps updating it there instead of writing each partial result back to global memory. It uses FP32, a 32-bit floating-point format, for the sum. FP32 retains more precision than bfloat16, which matters when many partial results are added together.


## 4. Finish, cast and write the tile back

Once all steps along $K$ have finished, the accumulator contains the complete output tile. The kernel converts it from FP32 to bfloat16, the 16-bit format used for the output, and places it in the shared-memory buffer `out_smem`. This buffer is the source for the copy back to global memory. Before starting that copy, the kernel commits the shared-memory writes so the copy engine can see the values just written.

The buffer uses `SwizzleTransform(128)` to control how its values are arranged in shared memory. Shared memory is divided into banks that can serve accesses in parallel. Some access patterns send multiple requests to the same bank, forcing them to wait. This is a bank conflict. The swizzle rearranges the storage locations to reduce those conflicts without changing the matrix values or their logical positions. The kernel then starts an asynchronous copy from `out_smem` to the tile's position in the output matrix. It waits for the copy to finish before reusing the buffer, so a later tile cannot overwrite values still being copied.

---

<br>

The full process is:

    - tile the matrices
    - visit the tiles in an order that encourages reuse
    - overlap memory traffic with Tensor Core work
    - keep the running sum in registers, and
    - write the finished tile back only once.
