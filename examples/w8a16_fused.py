from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from utils import get_max_smem_bytes



def quantize_weight_per_output_channel(weight):
    """Quantizes weights symmetrically with one BF16 scale per output channel."""

    weight = weight.asype(jnp.float32)
    maxval = jnp.max(jnp.abs(weight), axis=1)
    scale = jnp.where(weight == 0.0, 1.0, maxval / 127.0).astype(jnp.bfloat16)
    quantized = jnp.round(weight / scale[:, None]).astype(jnp.float32)
    quantized = jnp.clip(quantized, -127, 127).astype(jnp.int8)
    return quantized, scale


def simple_w816_matmul(quantized_weights, weight_scale, activations):
    """Performs W8A16 op with INT8 -> BF16 weight dequantization."""
    dequnatized = quantized_weights.astype(jnp.bfloat16) * weight_scale[:, None]
    return jnp.matmul(dequnatized, activations)


def matmul(activations, weights, scale, tile_m=64, tile_n=64, tile_k=128, num_pipeline_stages=5):
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
        padded_activations = jnp.pad(activations, (0, m_padding), (0, 0))
    else:
        padded_activations = activations

    # 2D Grid extents in units of tiles
    num_tiles_m = padded_m // tile_m
    num_tiles_n = n // tile_n
    num_tiles_k = k // tile_k

    # GPU Transforms or the layouts (WGMMA Hopper swizzles)
    activation_transforms = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))
    weight_transforms     = (plgpu.TilingTransform((8, 128)), plgpu.SwizzleTransform(128))
    output_transforms     = (plgpu.SwizzleTransform(128),)

    # Shared Memory (SMEM) allocation sizing
    activation_stage_bytes = tile_m * tile_k * activations_bytes_per_elem
    weight_stage_bytes     = tile_n * tile_k * weights_bytes_per_elem

    input_smem_bytes = num_pipeline_stages * (activation_stage_bytes + weight_stage_bytes)
    out_smem_bytes   = tile_m * tile_n * out_bytes_per_elem
    max_smem_bytes   = get_max_smem_bytes()

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


