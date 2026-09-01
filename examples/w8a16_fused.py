from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu


DEFAULT_TILE_M = 64
DEFAULT_TILE_N = 64
DEFAULT_TILE_K = 128
DEFAULT_NUM_PIPELINE_STAGES = 5


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


