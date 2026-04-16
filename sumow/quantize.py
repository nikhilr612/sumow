"""Quantization routines in pure JAX.

Implements asymmetric round-to-nearest (RTN) quantization with support for:
- Per-tensor and blockwise quantization
- Outlier clipping (z-score, percentage-based)
- Super-weight-aware quantization (clip → quantize → restore SW)
- Super-activation-aware quantization (replace SA → quantize → restore SA)

All functions are pure and JIT-compatible where noted.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


# ---------------------------------------------------------------------------
# Core RTN quantization
# ---------------------------------------------------------------------------


def quantize_rtn(
    x: Float[Array, "*dims"],
    nbits: int = 4,
) -> tuple[Int[Array, "*dims"], Float[Array, ""], Float[Array, ""]]:
    """Asymmetric round-to-nearest quantization.

    Maps x into [0, 2^nbits - 1] integers using:
        q = Round((x - min) / Δ)
    where Δ = (max - min) / (2^nbits - 1).

    Returns (quantized_ints, scale_delta, zero_point_min).
    """
    x_min = jnp.min(x)
    x_max = jnp.max(x)
    qmax = (1 << nbits) - 1
    delta = (x_max - x_min) / qmax
    # Avoid division by zero for constant tensors
    delta = jnp.where(delta == 0, jnp.ones_like(delta), delta)
    q = jnp.round((x - x_min) / delta).astype(jnp.int32)
    q = jnp.clip(q, 0, qmax)
    return q, delta, x_min


def dequantize_rtn(
    q: Int[Array, "*dims"],
    delta: Float[Array, ""],
    x_min: Float[Array, ""],
) -> Float[Array, "*dims"]:
    """Inverse of quantize_rtn: Δ * q + min."""
    return delta * q.astype(jnp.float32) + x_min


def quantize_dequantize(
    x: Float[Array, "*dims"],
    nbits: int = 4,
) -> Float[Array, "*dims"]:
    """Quantize and immediately dequantize (simulated quantization)."""
    q, delta, x_min = quantize_rtn(x, nbits)
    return dequantize_rtn(q, delta, x_min)


# ---------------------------------------------------------------------------
# Blockwise quantization
# ---------------------------------------------------------------------------


def quantize_dequantize_blockwise(
    weight: Float[Array, "rows cols"],
    nbits: int = 4,
    blocksize: int = 128,
) -> Float[Array, "rows cols"]:
    """Blockwise simulated quantization.

    Reshapes the weight into blocks of `blocksize` elements, quantizes each
    block independently, then reshapes back. If the total number of elements
    is not divisible by blocksize, falls back to per-tensor quantization.
    """
    shape = weight.shape
    numel = weight.size

    if blocksize <= 0 or numel % blocksize != 0:
        return quantize_dequantize(weight, nbits).reshape(shape)

    flat = weight.reshape(-1, blocksize)
    # Per-block min/max
    block_min = jnp.min(flat, axis=1, keepdims=True)
    block_max = jnp.max(flat, axis=1, keepdims=True)
    qmax = (1 << nbits) - 1
    delta = (block_max - block_min) / qmax
    delta = jnp.where(delta == 0, jnp.ones_like(delta), delta)

    q = jnp.round((flat - block_min) / delta).astype(jnp.int32)
    q = jnp.clip(q, 0, qmax)
    deq = delta * q.astype(jnp.float32) + block_min
    return deq.reshape(shape)


# ---------------------------------------------------------------------------
# Outlier clipping
# ---------------------------------------------------------------------------


def clip_zscore(
    weight: Float[Array, "*dims"],
    z_threshold: float = 9.0,
) -> Float[Array, "*dims"]:
    """Clip values beyond `z_threshold` standard deviations from the mean.

    The paper uses z-score clipping to remove outliers before quantization,
    then restores the super weight afterwards.
    """
    abs_weight = jnp.abs(weight)
    mean = jnp.mean(abs_weight)
    std = jnp.std(abs_weight)
    threshold = mean + z_threshold * std
    return jnp.clip(weight, -threshold, threshold)


def clip_percentage(
    weight: Float[Array, "*dims"],
    percentage: float = 1e-6,
) -> Float[Array, "*dims"]:
    """Clip the top `percentage` of values by magnitude (per-tensor).

    Values strictly above the threshold (the k-th largest |value|) are clamped.
    """
    abs_weight = jnp.abs(weight)
    k = max(int(weight.size * percentage), 1)
    # Threshold = the (k+1)-th largest abs value, so the top-k are clipped
    sorted_abs = jnp.sort(abs_weight.reshape(-1))
    threshold = sorted_abs[-(k + 1)] if k < weight.size else sorted_abs[0]
    return jnp.clip(weight, -threshold, threshold)


# ---------------------------------------------------------------------------
# Super-weight-aware quantization (Section 4.2 of the paper)
# ---------------------------------------------------------------------------


def quantize_weight_sw_aware(
    weight: Float[Array, "rows cols"],
    sw_coords: list[tuple[int, int]],
    nbits: int = 4,
    blocksize: int = 128,
    z_threshold: float = 9.0,
) -> Float[Array, "rows cols"]:
    """Super-weight-aware weight quantization.

    From the paper (Equation 2):
        Ŵ = Restore(Q⁻¹(Q(Clip_z(W))))

    Steps:
    1. Save super weight values at sw_coords
    2. Clip outliers using z-score
    3. Quantize and dequantize (blockwise RTN)
    4. Restore super weights in original precision

    Args:
        weight: A single down_proj weight matrix.
        sw_coords: List of (row, col) coordinates of super weights in this matrix.
        nbits: Quantization bit-width.
        blocksize: Block size for blockwise quantization.
        z_threshold: Z-score threshold for outlier clipping.

    Returns:
        Quantized weight with super weights restored.
    """
    # 1. Save super weight values
    sw_values = [(r, c, float(weight[r, c])) for r, c in sw_coords]

    # 2. Clip outliers
    clipped = clip_zscore(weight, z_threshold)

    # 3. Quantize-dequantize
    if blocksize > 0 and weight.size % blocksize == 0:
        quantized = quantize_dequantize_blockwise(clipped, nbits, blocksize)
    else:
        quantized = quantize_dequantize(clipped, nbits)

    # 4. Restore super weights in original precision
    for r, c, val in sw_values:
        quantized = quantized.at[r, c].set(val)

    return quantized


# ---------------------------------------------------------------------------
# Super-activation-aware quantization (Section 4.1 of the paper)
# ---------------------------------------------------------------------------


def quantize_activation_sa_aware(
    activation: Float[Array, "batch seq hidden"],
    sa_positions: list[tuple[int, int]],
    nbits: int = 8,
) -> Float[Array, "batch seq hidden"]:
    """Super-activation-aware activation quantization.

    From the paper (Equation 1):
        Â = Restore(Q⁻¹(Q(Replace(A))))

    Steps:
    1. Save super activation values at sa_positions (seq_idx, hidden_idx)
    2. Replace super activations with median value
    3. Quantize and dequantize
    4. Restore super activations in original precision

    Args:
        activation: Activation tensor [batch, seq_len, hidden_dim].
        sa_positions: List of (seq_idx, hidden_idx) for super activations.
        nbits: Quantization bit-width.

    Returns:
        Quantized activations with super activations restored.
    """
    # 1. Save super activation values (per batch element)
    sa_values = []
    for seq_idx, hid_idx in sa_positions:
        sa_values.append((seq_idx, hid_idx, activation[:, seq_idx, hid_idx]))

    # 2. Replace super activations with median
    median_val = jnp.median(activation)
    replaced = activation
    for seq_idx, hid_idx, _ in sa_values:
        replaced = replaced.at[:, seq_idx, hid_idx].set(median_val)

    # 3. Quantize-dequantize (per-tensor for activations)
    quantized = quantize_dequantize(replaced, nbits)

    # 4. Restore super activations
    for seq_idx, hid_idx, original_val in sa_values:
        quantized = quantized.at[:, seq_idx, hid_idx].set(original_val)

    return quantized
