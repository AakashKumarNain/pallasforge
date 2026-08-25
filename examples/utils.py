from jax.extend import backend


def get_max_smem_bytes():
    gpu = backend.get_default_device()
    smem_bytes = getattr(gpu, "shared_memory_per_block_optin", None)
    return int(smem_bytes) if smem_bytes else None
