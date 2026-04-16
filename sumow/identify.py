"""Super weight identification via activation spike detection.

Implements the paper's data-free identification method (Section 3.1):
  1. Run a single forward pass with any prompt
  2. Record max-magnitude activations at each layer's down_proj input and output
  3. Detect spikes: layers where max activation is far above the cross-layer median
  4. Input spike channel → super weight column (k)
  5. Output spike channel → super weight row (j)
  6. Repeat after removing found SW until no spikes remain

All detection logic operates on pre-computed activation statistics (pure JAX).
The forward-pass hook infrastructure is intentionally separated so it can work
with any model framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float


@dataclass(frozen=True)
class SuperWeight:
    """A single identified super weight."""

    layer: int
    row: int  # output dimension index of down_proj
    col: int  # input dimension index of down_proj
    input_magnitude: float  # magnitude of the input activation spike
    output_magnitude: float  # magnitude of the output activation spike


@dataclass(frozen=True)
class LayerActivationStats:
    """Max-magnitude activation statistics for one layer's down_proj."""

    layer: int
    input_max_magnitude: float
    input_max_channel: int  # channel (hidden dim index) of max input activation
    output_max_magnitude: float
    output_max_channel: int  # channel (feature dim index) of max output activation


def detect_spikes(
    stats: list[LayerActivationStats],
    spike_threshold: float = 100.0,
    spike_ratio: float = 6.0,
) -> list[LayerActivationStats]:
    """Find layers with activation spikes indicating super weights.

    A layer is flagged if:
      - Its max input OR output magnitude exceeds `spike_threshold`, AND
      - Its magnitude exceeds `spike_ratio` × the median across all layers.

    Args:
        stats: Per-layer activation statistics from a forward pass.
        spike_threshold: Minimum absolute magnitude to consider.
        spike_ratio: Minimum ratio above the cross-layer median.

    Returns:
        List of LayerActivationStats that are identified as spikes, sorted by
        input magnitude descending.
    """
    if not stats:
        return []

    input_mags = jnp.array([s.input_max_magnitude for s in stats])
    output_mags = jnp.array([s.output_max_magnitude for s in stats])

    input_median = float(jnp.median(input_mags))
    output_median = float(jnp.median(output_mags))

    spikes = []
    for s in stats:
        input_is_spike = (
            s.input_max_magnitude > spike_threshold
            and s.input_max_magnitude > spike_ratio * max(input_median, 1e-8)
        )
        output_is_spike = (
            s.output_max_magnitude > spike_threshold
            and s.output_max_magnitude > spike_ratio * max(output_median, 1e-8)
        )
        if input_is_spike or output_is_spike:
            spikes.append(s)

    return sorted(spikes, key=lambda s: s.input_max_magnitude, reverse=True)


def identify_super_weights(
    stats: list[LayerActivationStats],
    spike_threshold: float = 100.0,
    spike_ratio: float = 6.0,
) -> list[SuperWeight]:
    """Identify super weights from activation statistics.

    Uses the paper's method: input spike at channel k gives the SW column,
    output spike at channel j gives the SW row. The layer is the spike layer.

    Args:
        stats: Per-layer activation statistics from a single forward pass.
        spike_threshold: Minimum magnitude for spike detection.
        spike_ratio: Minimum ratio vs. cross-layer median.

    Returns:
        List of identified SuperWeight instances.
    """
    spikes = detect_spikes(stats, spike_threshold, spike_ratio)

    super_weights = []
    for spike in spikes:
        sw = SuperWeight(
            layer=spike.layer,
            row=spike.output_max_channel,
            col=spike.input_max_channel,
            input_magnitude=spike.input_max_magnitude,
            output_magnitude=spike.output_max_magnitude,
        )
        super_weights.append(sw)

    return super_weights


def compute_layer_stats(
    input_act: Float[Array, "seq hidden"],
    output_act: Float[Array, "seq features"],
    layer: int,
) -> LayerActivationStats:
    """Compute activation statistics for a single layer's down_proj.

    Args:
        input_act: Input activations to down_proj [seq_len, hidden_dim].
        output_act: Output activations from down_proj [seq_len, feature_dim].
        layer: Layer index.

    Returns:
        LayerActivationStats with max magnitudes and their channel indices.
    """
    # Max magnitude across all positions for input
    input_abs = jnp.abs(input_act)
    input_max_per_channel = jnp.max(input_abs, axis=0)
    input_max_channel = int(jnp.argmax(input_max_per_channel))
    input_max_magnitude = float(input_max_per_channel[input_max_channel])

    # Max magnitude across all positions for output
    output_abs = jnp.abs(output_act)
    output_max_per_channel = jnp.max(output_abs, axis=0)
    output_max_channel = int(jnp.argmax(output_max_per_channel))
    output_max_magnitude = float(output_max_per_channel[output_max_channel])

    return LayerActivationStats(
        layer=layer,
        input_max_magnitude=input_max_magnitude,
        input_max_channel=input_max_channel,
        output_max_magnitude=output_max_magnitude,
        output_max_channel=output_max_channel,
    )
