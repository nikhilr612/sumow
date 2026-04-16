"""Tests for super weight identification."""

import jax.numpy as jnp

from sumow.identify import (
    LayerActivationStats,
    SuperWeight,
    compute_layer_stats,
    detect_spikes,
    identify_super_weights,
)


def _make_stats(
    n_layers: int = 32,
    spike_layers: dict[int, tuple[float, int, float, int]] | None = None,
) -> list[LayerActivationStats]:
    """Helper to create synthetic activation stats.

    Args:
        n_layers: Number of layers.
        spike_layers: Dict mapping layer_idx -> (input_mag, input_ch, output_mag, output_ch).
                      Non-spike layers get small random magnitudes.
    """
    if spike_layers is None:
        spike_layers = {}

    stats = []
    for i in range(n_layers):
        if i in spike_layers:
            in_mag, in_ch, out_mag, out_ch = spike_layers[i]
        else:
            in_mag = 2.0 + i * 0.1
            in_ch = i % 8
            out_mag = 1.5 + i * 0.1
            out_ch = (i + 1) % 8
        stats.append(
            LayerActivationStats(
                layer=i,
                input_max_magnitude=in_mag,
                input_max_channel=in_ch,
                output_max_magnitude=out_mag,
                output_max_channel=out_ch,
            )
        )
    return stats


class TestDetectSpikes:
    def test_single_spike(self):
        stats = _make_stats(32, spike_layers={2: (500.0, 7003, 400.0, 3968)})
        spikes = detect_spikes(stats, spike_threshold=100.0, spike_ratio=6.0)
        assert len(spikes) == 1
        assert spikes[0].layer == 2

    def test_no_spikes(self):
        stats = _make_stats(32)  # all values small
        spikes = detect_spikes(stats, spike_threshold=100.0, spike_ratio=6.0)
        assert len(spikes) == 0

    def test_multiple_spikes(self):
        stats = _make_stats(
            32,
            spike_layers={
                1: (600.0, 269, 500.0, 7467),
                2: (550.0, 269, 450.0, 8275),
                7: (300.0, 269, 250.0, 453),
            },
        )
        spikes = detect_spikes(stats, spike_threshold=100.0, spike_ratio=6.0)
        assert len(spikes) == 3
        # Should be sorted by input magnitude descending
        assert spikes[0].layer == 1
        assert spikes[1].layer == 2

    def test_empty_stats(self):
        assert detect_spikes([], spike_threshold=100.0) == []


class TestIdentifySuperWeights:
    def test_single_super_weight(self):
        """Simulates Llama-7B: single spike at layer 2."""
        stats = _make_stats(32, spike_layers={2: (500.0, 7003, 400.0, 3968)})
        sws = identify_super_weights(stats, spike_threshold=100.0, spike_ratio=6.0)
        assert len(sws) == 1
        sw = sws[0]
        assert sw.layer == 2
        assert sw.col == 7003  # from input spike channel
        assert sw.row == 3968  # from output spike channel

    def test_multiple_super_weights(self):
        """Simulates OLMo-7B: multiple spikes."""
        stats = _make_stats(
            32,
            spike_layers={
                1: (600.0, 7467, 500.0, 269),
                2: (550.0, 8275, 450.0, 269),
            },
        )
        sws = identify_super_weights(stats)
        assert len(sws) == 2
        assert all(isinstance(sw, SuperWeight) for sw in sws)

    def test_no_super_weights(self):
        stats = _make_stats(32)
        sws = identify_super_weights(stats)
        assert sws == []


class TestComputeLayerStats:
    def test_basic(self):
        input_act = jnp.zeros((10, 16))
        input_act = input_act.at[3, 7].set(999.0)

        output_act = jnp.zeros((10, 8))
        output_act = output_act.at[5, 2].set(-777.0)

        stats = compute_layer_stats(input_act, output_act, layer=2)
        assert stats.layer == 2
        assert stats.input_max_channel == 7
        assert stats.input_max_magnitude == 999.0
        assert stats.output_max_channel == 2
        assert stats.output_max_magnitude == 777.0

    def test_all_zeros(self):
        input_act = jnp.zeros((4, 8))
        output_act = jnp.zeros((4, 8))
        stats = compute_layer_stats(input_act, output_act, layer=0)
        assert stats.input_max_magnitude == 0.0
        assert stats.output_max_magnitude == 0.0

    def test_uniform_values(self):
        input_act = jnp.ones((4, 8)) * 5.0
        output_act = jnp.ones((4, 8)) * 3.0
        stats = compute_layer_stats(input_act, output_act, layer=1)
        assert stats.input_max_magnitude == 5.0
        assert stats.output_max_magnitude == 3.0
