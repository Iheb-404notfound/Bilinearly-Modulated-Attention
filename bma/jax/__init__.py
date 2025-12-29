# bma/jax/__init__.py
"""JAX/Flax implementation of BMA."""

from .attention import (
    BilinearlyModulatedAttention,
    MultiHeadAttention,
    GatedAttention,
    init_attention
)

__all__ = [
    'BilinearlyModulatedAttention',
    'MultiHeadAttention',
    'GatedAttention',
    'init_attention',
]
