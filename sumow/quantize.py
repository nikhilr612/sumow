"""Quantization routines in pure JAX.

Implements asymmetric round-to-nearest (RTN) quantization with support for:
- Per-tensor and blockwise quantization
- Normal float (NF4/NF3) quantization
- Outlier clipping (z-score, percentage, block-percentage, IQR)
- Scale-shift rounding variant
- Super-weight-aware quantization (clip → quantize → restore SW)
- Super-activation-aware quantization (replace SA → quantize → restore SA)
- 4-bit packing utilities for storage efficiency
- Super weight scaling/amplification

All functions are pure and JIT-compatible where noted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


# ---------------------------------------------------------------------------
# NF4 / NF3 quantization levels (from QLoRA / bitsandbytes)
# ---------------------------------------------------------------------------

NF4_LEVELS: list[float] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]

NF3_LEVELS: list[float] = [
    -1.0,
    -0.5350227355957031,
    -0.2469314038753510,
    0.0,
    0.1833375245332718,
    0.3819939494132996,
    0.6229856610298157,
    1.0,
]


@dataclass
class QuantizeResult:
    """Result of quantization with metadata."""

    weight: Float[Array, "*dims"]
    num_outliers: int = 0


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
    clip_method: str = "none",
    clip_threshold: float = 9.0,
    use_normal_float: bool = False,
    scale_shift: bool = False,
) -> QuantizeResult:
    """Blockwise simulated quantization.

    Reshapes the weight into blocks of `blocksize` elements, optionally clips
    outliers, then quantizes each block independently.

    Args:
        weight: Weight matrix to quantize.
        nbits: Bit width (3 or 4 for NF, 2-8 for INT).
        blocksize: Elements per block (0 or indivisible falls back to per-tensor).
        clip_method: One of "none", "zscore", "tensor_percentage",
                     "block_percentage", "iqr".
        clip_threshold: Threshold parameter for the clip method.
        use_normal_float: Use NF4/NF3 quantization instead of INT.
        scale_shift: Use scale-shift rounding variant (INT only).

    Returns:
        QuantizeResult with quantized weight and outlier count.
    """
    shape = weight.shape
    numel = weight.size
    num_outliers = 0

    # Reshape into blocks
    if blocksize > 0 and numel % blocksize == 0:
        flat = weight.reshape(-1, blocksize)
    else:
        flat = weight.reshape(1, -1)

    # Clipping
    if clip_method != "none":
        if clip_method == "block_percentage":
            flat, num_outliers = clip_block_percentage(flat, clip_threshold)
        elif clip_method == "tensor_percentage":
            clipped = clip_percentage(flat.reshape(shape), clip_threshold)
            num_outliers = int(jnp.sum(jnp.abs(flat) > jnp.abs(clipped.reshape(flat.shape))))
            flat = clipped.reshape(flat.shape)
        elif clip_method == "zscore":
            abs_flat = jnp.abs(flat)
            means = jnp.mean(abs_flat, axis=1, keepdims=True)
            stds = jnp.std(abs_flat, axis=1, keepdims=True)
            threshold = means + clip_threshold * stds
            num_outliers = int(jnp.sum(abs_flat > threshold))
            flat = jnp.clip(flat, -threshold, threshold)
        elif clip_method == "iqr":
            clipped, num_outliers = clip_iqr(flat, factor=clip_threshold, per_block=True)
            flat = clipped
        else:
            raise ValueError(f"Unknown clip method: {clip_method}")

    # Quantize
    if use_normal_float:
        # NF4/NF3 path
        result = quantize_dequantize_nf(flat.reshape(shape), nbits, blocksize)
    elif scale_shift:
        result = quantize_dequantize_scale_shift(flat.reshape(shape), nbits, blocksize)
    else:
        # Standard INT RTN path
        block_min = jnp.min(flat, axis=1, keepdims=True)
        block_max = jnp.max(flat, axis=1, keepdims=True)
        qmax = (1 << nbits) - 1
        delta = (block_max - block_min) / qmax
        delta = jnp.where(delta == 0, jnp.ones_like(delta), delta)

        q = jnp.round((flat - block_min) / delta).astype(jnp.int32)
        q = jnp.clip(q, 0, qmax)
        result = (delta * q.astype(jnp.float32) + block_min).reshape(shape)

    return QuantizeResult(weight=result, num_outliers=num_outliers)


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


def clip_block_percentage(
    weight: Float[Array, "blocks blocksize"],
    percentage: float = 1e-6,
) -> tuple[Float[Array, "blocks blocksize"], int]:
    """Clip the top `percentage` of values by magnitude, independently per block.

    Each block (row) gets its own threshold from its top-k absolute values.
    Returns (clipped_weight, num_outliers).
    """
    abs_weight = jnp.abs(weight)
    k = max(int(weight.shape[1] * percentage + 1), 1)

    # Per-block: sort each row descending, take the k-th value as threshold
    sorted_abs = jnp.sort(abs_weight, axis=1)  # ascending
    threshold = sorted_abs[:, -k : -(k - 1) if k > 1 else None][:, 0:1]

    num_outliers = int(jnp.sum(abs_weight > threshold))
    clipped = jnp.clip(weight, -threshold, threshold)
    return clipped, num_outliers


def clip_iqr(
    weight: Float[Array, "*dims"],
    factor: float = 1.5,
    per_block: bool = False,
) -> tuple[Float[Array, "*dims"], int]:
    """Clip outliers using the interquartile range (IQR) method.

    Threshold = Q3 + factor * (Q3 - Q1), applied to absolute values.
    Returns (clipped_weight, num_outliers).
    """
    abs_weight = jnp.abs(weight).astype(jnp.float32)

    if per_block and weight.ndim == 2:
        q1 = jnp.quantile(abs_weight, 0.25, axis=1, keepdims=True)
        q3 = jnp.quantile(abs_weight, 0.75, axis=1, keepdims=True)
    else:
        q1 = jnp.quantile(abs_weight.reshape(-1), 0.25)
        q3 = jnp.quantile(abs_weight.reshape(-1), 0.75)

    iqr = q3 - q1
    threshold = q3 + factor * iqr

    num_outliers = int(jnp.sum(abs_weight > threshold))
    clipped = jnp.clip(weight, -threshold, threshold)
    return clipped, num_outliers


# ---------------------------------------------------------------------------
# Normal Float (NF4/NF3) quantization
# ---------------------------------------------------------------------------


def round_to_nearest_pole(
    x: Float[Array, "*dims"],
    poles: Float[Array, "num_poles"],
) -> Float[Array, "*dims"]:
    """Round each element of x to the nearest value in `poles`."""
    # Broadcast: x[..., None] vs poles[None, ...]
    shape = x.shape
    flat = x.reshape(-1)
    diffs = jnp.abs(flat[:, None] - poles[None, :])
    nearest_idx = jnp.argmin(diffs, axis=1)
    return poles[nearest_idx].reshape(shape)


def quantize_dequantize_nf(
    weight: Float[Array, "rows cols"],
    nbits: int = 4,
    blocksize: int = 128,
) -> Float[Array, "rows cols"]:
    """Normal float quantization (NF4 or NF3).

    Maps weights to [-1, 1] per block, rounds to nearest quantization pole,
    then maps back. Uses the quantization levels from QLoRA/bitsandbytes.
    """
    if nbits == 4:
        levels = NF4_LEVELS
    elif nbits == 3:
        levels = NF3_LEVELS
    else:
        raise ValueError(f"Normal float quantization only supports 3 and 4 bits, got {nbits}")

    poles = jnp.array(levels, dtype=jnp.float32)
    shape = weight.shape
    numel = weight.size

    if blocksize > 0 and numel % blocksize == 0:
        flat = weight.reshape(-1, blocksize)
    else:
        flat = weight.reshape(1, -1)

    block_min = jnp.min(flat, axis=1, keepdims=True)
    block_max = jnp.max(flat, axis=1, keepdims=True)

    # Scale to [0, 2], then shift to [-1, 1]
    scale = 2.0 / (block_max - block_min)
    scale = jnp.where(scale == jnp.inf, jnp.ones_like(scale), scale)

    normalized = (flat - block_min) * scale - 1.0
    quantized = round_to_nearest_pole(normalized, poles)

    # Map back: (q + 1) / scale + min
    dequantized = (quantized + 1.0) / scale + block_min
    return dequantized.reshape(shape)


# ---------------------------------------------------------------------------
# Scale-shift rounding variant
# ---------------------------------------------------------------------------


def quantize_dequantize_scale_shift(
    weight: Float[Array, "rows cols"],
    nbits: int = 4,
    blocksize: int = 128,
) -> Float[Array, "rows cols"]:
    """Blockwise quantization with scale-shift rounding.

    Maps to [-0.4999, 2^nbits - 0.51] then rounds, instead of standard [0, 2^nbits-1].
    """
    shape = weight.shape
    numel = weight.size

    if blocksize > 0 and numel % blocksize == 0:
        flat = weight.reshape(-1, blocksize)
    else:
        flat = weight.reshape(1, -1)

    block_min = jnp.min(flat, axis=1, keepdims=True)
    block_max = jnp.max(flat, axis=1, keepdims=True)

    scale = ((1 << nbits) - 0.01) / (block_max - block_min)
    scale = jnp.where(scale == jnp.inf, jnp.ones_like(scale), scale)

    q = (flat - block_min) * scale - 0.49
    q = jnp.round(q)
    dequantized = (q + 0.49) / scale + block_min
    return dequantized.reshape(shape)


# ---------------------------------------------------------------------------
# Per-channel quantization
# ---------------------------------------------------------------------------


def quantize_dequantize_per_channel(
    weight: Float[Array, "rows cols"],
    nbits: int = 4,
) -> Float[Array, "rows cols"]:
    """Per-channel (per-row) quantization.

    Each output channel (row) is quantized independently with its own scale/zero.
    Equivalent to blockwise with blocksize = number of columns.
    """
    return quantize_dequantize_blockwise(weight, nbits, blocksize=weight.shape[1])


# ---------------------------------------------------------------------------
# 4-bit packing utilities
# ---------------------------------------------------------------------------


def pack_4bit_to_int8(
    quantized: Int[Array, "rows cols"],
) -> Int[Array, "rows packed_cols"]:
    """Pack pairs of 4-bit values into single int8 values.

    If cols is odd, pads with a zero column before packing.
    """
    rows, cols = quantized.shape
    q = quantized.astype(jnp.uint8)

    if cols % 2 != 0:
        q = jnp.concatenate([q, jnp.zeros((rows, 1), dtype=jnp.uint8)], axis=1)

    high = q[:, ::2]
    low = q[:, 1::2]
    packed = (high << 4) | (low & 0xF)
    return packed.astype(jnp.int8)


def unpack_int8_to_4bit(
    packed: Int[Array, "rows packed_cols"],
) -> Int[Array, "rows unpacked_cols"]:
    """Unpack int8 values into pairs of 4-bit values."""
    p = packed.astype(jnp.uint8)
    high = (p >> 4) & 0xF
    low = p & 0xF
    # Interleave: [h0, l0, h1, l1, ...]
    interleaved = jnp.stack([high, low], axis=2).reshape(packed.shape[0], -1)
    return interleaved.astype(jnp.int32)


# ---------------------------------------------------------------------------
# Super weight scaling / amplification
# ---------------------------------------------------------------------------


def scale_super_weights(
    weight: Float[Array, "rows cols"],
    sw_coords: list[tuple[int, int]],
    scaling_factor: float = 1.0,
) -> Float[Array, "rows cols"]:
    """Scale (amplify) super weights by a given factor.

    From Tables 5-6 of the paper: multiplying super weights by factors
    0.0-3.0 can slightly improve or destroy model quality.
    """
    result = weight
    for r, c in sw_coords:
        result = result.at[r, c].set(weight[r, c] * scaling_factor)
    return result


# ---------------------------------------------------------------------------
# Super-weight-aware quantization (Section 4.2 of the paper)
# ---------------------------------------------------------------------------


def quantize_weight_sw_aware(
    weight: Float[Array, "rows cols"],
    sw_coords: list[tuple[int, int]],
    nbits: int = 4,
    blocksize: int = 128,
    clip_method: str = "zscore",
    clip_threshold: float = 9.0,
    use_normal_float: bool = False,
    scale_shift: bool = False,
) -> QuantizeResult:
    """Super-weight-aware weight quantization.

    From the paper (Equation 2):
        Ŵ = Restore(Q⁻¹(Q(Clip_z(W))))

    Steps:
    1. Save super weight values at sw_coords
    2. Clip outliers using specified method
    3. Quantize and dequantize (blockwise)
    4. Restore super weights in original precision

    Args:
        weight: A single down_proj weight matrix.
        sw_coords: List of (row, col) coordinates of super weights in this matrix.
        nbits: Quantization bit-width.
        blocksize: Block size for blockwise quantization.
        clip_method: Clipping method ("zscore", "tensor_percentage",
                     "block_percentage", "iqr", "none").
        clip_threshold: Threshold parameter for the clip method.
        use_normal_float: Use NF4/NF3 instead of INT.
        scale_shift: Use scale-shift rounding (INT only).

    Returns:
        QuantizeResult with quantized weight (super weights restored) and outlier count.
    """
    # 1. Save super weight values
    sw_values = [(r, c, float(weight[r, c])) for r, c in sw_coords]

    # 2+3. Clip and quantize-dequantize
    result = quantize_dequantize_blockwise(
        weight,
        nbits=nbits,
        blocksize=blocksize,
        clip_method=clip_method,
        clip_threshold=clip_threshold,
        use_normal_float=use_normal_float,
        scale_shift=scale_shift,
    )

    # 4. Restore super weights in original precision
    quantized = result.weight
    for r, c, val in sw_values:
        quantized = quantized.at[r, c].set(val)

    return QuantizeResult(weight=quantized, num_outliers=result.num_outliers)


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
