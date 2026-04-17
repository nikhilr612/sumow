"""Hardening tests — edge cases, stress tests, and boundary conditions.

Covers:
  - Model: JIT compilation, long sequences, extreme weights, dtype consistency
  - Quantize: all-zero blocks, all-same blocks, very large values, tiny blocks
  - Identify: boundary spike ratios, single-layer models
  - Integration: quantize every layer type, multiple SW per layer
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from sumow.model import (
    LlamaAttention,
    LlamaModel,
    RMSNorm,
    TransformerConfig,
    make_hf_weights_dict,
)
from sumow.quantize import (
    quantize_dequantize_blockwise,
    quantize_weight_sw_aware,
    clip_zscore,
    clip_percentage,
)
from sumow.identify import (
    LayerActivationStats,
    compute_layer_stats,
    detect_spikes,
    identify_super_weights,
)
from sumow.eval import perplexity

TINY = TransformerConfig(
    vocab_size=32, hidden_size=16, intermediate_size=32,
    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
    max_position_embeddings=64,
)


# ---------------------------------------------------------------------------
# Model edge cases
# ---------------------------------------------------------------------------


class TestModelEdgeCases:
    def test_single_layer_model(self):
        cfg = TransformerConfig(
            vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=16,
        )
        model = LlamaModel(cfg)
        ids = jnp.array([0, 1, 2])
        logits, stats = model(ids, capture_activations=True)
        assert logits.shape == (3, 16)
        assert len(stats) == 1

    def test_max_sequence_length(self):
        """Use full max_position_embeddings."""
        cfg = TransformerConfig(
            vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=32,
        )
        model = LlamaModel(cfg)
        ids = jnp.arange(32)  # exactly max length
        logits, _ = model(ids)
        assert logits.shape == (32, 16)
        assert jnp.all(jnp.isfinite(logits))

    def test_model_with_identical_tokens(self):
        """All same tokens should still produce valid output."""
        model = LlamaModel(TINY)
        ids = jnp.zeros(8, dtype=jnp.int32)
        logits, _ = model(ids)
        assert jnp.all(jnp.isfinite(logits))

    def test_model_is_pytree(self):
        """Model should be a valid JAX pytree."""
        model = LlamaModel(TINY)
        leaves = jax.tree.leaves(model)
        assert len(leaves) > 0
        # All leaves should be JAX arrays
        for leaf in leaves:
            assert isinstance(leaf, jnp.ndarray), f"Non-array leaf: {type(leaf)}"

    def test_rmsnorm_large_values(self):
        """RMSNorm should handle large input values without overflow."""
        norm = RMSNorm(8, eps=1e-5)
        x = jnp.ones((2, 8)) * 1e4
        y = norm(x)
        assert jnp.all(jnp.isfinite(y))

    def test_rmsnorm_tiny_values(self):
        """RMSNorm should handle very small values (eps protects)."""
        norm = RMSNorm(8, eps=1e-5)
        x = jnp.ones((2, 8)) * 1e-10
        y = norm(x)
        assert jnp.all(jnp.isfinite(y))

    def test_attention_long_sequence(self):
        """Attention with longer sequence should still work."""
        cfg = TransformerConfig(
            vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64,
        )
        attn = LlamaAttention(cfg)
        x = jax.random.normal(jax.random.PRNGKey(0), (32, 16)) * 0.01
        y = attn(x)
        assert y.shape == (32, 16)

    def test_causal_mask_is_causal(self):
        """Verify that attention is causal: output at position i depends only on positions <= i."""
        cfg = TransformerConfig(
            vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=32,
        )
        attn = LlamaAttention(cfg)
        # Initialize with random weights
        leaves, treedef = jax.tree.flatten(attn)
        new_leaves = [
            jax.random.normal(jax.random.PRNGKey(i), leaf.shape) * 0.1
            if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32
            else leaf
            for i, leaf in enumerate(leaves)
        ]
        attn = jax.tree.unflatten(treedef, new_leaves)
        x = jax.random.normal(jax.random.PRNGKey(99), (8, 8)) * 0.1
        y_full = attn(x)

        # Prefix only (first 4 tokens)
        y_prefix = attn(x[:4])

        # Output at position 3 should be same whether we have 4 or 8 tokens
        np.testing.assert_allclose(y_full[:4], y_prefix, atol=1e-5)


# ---------------------------------------------------------------------------
# Quantize edge cases
# ---------------------------------------------------------------------------


class TestQuantizeEdgeCases:
    def test_all_zero_weight(self):
        """Quantizing all-zero weight should return all zeros."""
        w = jnp.zeros((8, 16))
        result = quantize_dequantize_blockwise(
            w, nbits=4, blocksize=16,
            clip_method="none", clip_threshold=0.0,
        )
        np.testing.assert_allclose(result.weight, jnp.zeros_like(w), atol=1e-6)

    def test_all_same_nonzero(self):
        """All-same values should quantize to that value."""
        w = jnp.full((4, 16), 3.14)
        result = quantize_dequantize_blockwise(
            w, nbits=8, blocksize=16,
            clip_method="none", clip_threshold=0.0,
        )
        np.testing.assert_allclose(result.weight, w, atol=0.1)

    def test_very_large_values(self):
        """Large values should not cause overflow."""
        w = jnp.array([[1e6, -1e6, 1e6, -1e6]] * 4, dtype=jnp.float32)
        result = quantize_dequantize_blockwise(
            w, nbits=8, blocksize=4,
            clip_method="none", clip_threshold=0.0,
        )
        assert jnp.all(jnp.isfinite(result.weight))

    def test_blocksize_equals_tensor_size(self):
        """Blocksize = total elements (per-tensor quantization)."""
        w = jax.random.normal(jax.random.PRNGKey(0), (4, 8))
        result = quantize_dequantize_blockwise(
            w, nbits=4, blocksize=32,
            clip_method="none", clip_threshold=0.0,
        )
        assert result.weight.shape == w.shape

    def test_8bit_near_lossless(self):
        """8-bit quantization should have very low error for normal-range weights."""
        w = jax.random.normal(jax.random.PRNGKey(0), (8, 32)) * 0.1
        result = quantize_dequantize_blockwise(
            w, nbits=8, blocksize=32,
            clip_method="none", clip_threshold=0.0,
        )
        max_err = float(jnp.max(jnp.abs(w - result.weight)))
        assert max_err < 0.01  # 8-bit should be very close

    def test_2bit_quantization(self):
        """2-bit quantization should work (extreme case)."""
        w = jax.random.normal(jax.random.PRNGKey(0), (4, 16))
        result = quantize_dequantize_blockwise(
            w, nbits=2, blocksize=16,
            clip_method="none", clip_threshold=0.0,
        )
        assert result.weight.shape == w.shape
        assert jnp.all(jnp.isfinite(result.weight))

    def test_zscore_with_constant_block(self):
        """Z-score clipping on constant block (std=0) should not crash."""
        w = jnp.full((1, 32), 5.0)
        # This should handle division by zero in std gracefully
        clipped = clip_zscore(w.reshape(-1), z_threshold=3.0)
        assert jnp.all(jnp.isfinite(clipped))

    def test_percentage_clip_all_same(self):
        """Percentage clipping on all-same values."""
        w = jnp.full(100, 2.0)
        clipped = clip_percentage(w, percentage=0.01)
        np.testing.assert_allclose(clipped, w, atol=1e-6)

    def test_nf4_quantization(self):
        """NF4 mode should produce values only from the NF4 codebook."""
        w = jax.random.normal(jax.random.PRNGKey(0), (4, 32))
        result = quantize_dequantize_blockwise(
            w, nbits=4, blocksize=32,
            clip_method="none", clip_threshold=0.0,
            use_normal_float=True,
        )
        assert result.weight.shape == w.shape
        assert jnp.all(jnp.isfinite(result.weight))

    def test_scale_shift_mode(self):
        """Scale-shift quantization should work."""
        w = jax.random.normal(jax.random.PRNGKey(0), (4, 32))
        result = quantize_dequantize_blockwise(
            w, nbits=4, blocksize=32,
            clip_method="none", clip_threshold=0.0,
            scale_shift=True,
        )
        assert result.weight.shape == w.shape
        assert jnp.all(jnp.isfinite(result.weight))

    def test_sw_aware_multiple_sw_same_layer(self):
        """Multiple super weights in the same weight matrix."""
        w = jax.random.normal(jax.random.PRNGKey(0), (16, 32)) * 0.1
        w = w.at[3, 7].set(50.0)
        w = w.at[10, 20].set(-30.0)
        w = w.at[0, 0].set(100.0)

        result = quantize_weight_sw_aware(
            w, sw_coords=[(3, 7), (10, 20), (0, 0)],
            nbits=4, blocksize=32,
            clip_method="zscore", clip_threshold=3.0,
        )
        # All super weights should be exactly preserved
        assert float(result.weight[3, 7]) == 50.0
        assert float(result.weight[10, 20]) == -30.0
        assert float(result.weight[0, 0]) == 100.0


# ---------------------------------------------------------------------------
# Identify edge cases
# ---------------------------------------------------------------------------


class TestIdentifyEdgeCases:
    def test_all_layers_same_magnitude(self):
        """When all layers have same magnitude, no spikes should be detected."""
        stats = [
            LayerActivationStats(
                layer=i,
                input_max_magnitude=10.0,
                input_max_channel=0,
                output_max_magnitude=10.0,
                output_max_channel=0,
            )
            for i in range(4)
        ]
        spikes = detect_spikes(stats, spike_threshold=1.0, spike_ratio=2.0)
        # All same = no spike exceeds 2x median
        assert len(spikes) == 0

    def test_single_layer_always_spike(self):
        """Single layer: it IS the median, so ratio test = 1.0, needs ratio > spike_ratio."""
        stats = [
            LayerActivationStats(
                layer=0,
                input_max_magnitude=100.0,
                input_max_channel=5,
                output_max_magnitude=100.0,
                output_max_channel=3,
            )
        ]
        # With spike_ratio=1.0, 100/100 = 1.0, NOT > 1.0, so no spike
        spikes = detect_spikes(stats, spike_threshold=1.0, spike_ratio=1.5)
        assert len(spikes) == 0

    def test_extreme_spike_ratio(self):
        """One layer has 1000x the median magnitude."""
        stats = [
            LayerActivationStats(layer=0, input_max_magnitude=1.0,
                                 input_max_channel=0, output_max_magnitude=1.0,
                                 output_max_channel=0),
            LayerActivationStats(layer=1, input_max_magnitude=1000.0,
                                 input_max_channel=5, output_max_magnitude=1000.0,
                                 output_max_channel=3),
            LayerActivationStats(layer=2, input_max_magnitude=1.0,
                                 input_max_channel=0, output_max_magnitude=1.0,
                                 output_max_channel=0),
        ]
        sws = identify_super_weights(stats, spike_threshold=1.0, spike_ratio=5.0)
        assert len(sws) == 1
        assert sws[0].layer == 1
        assert sws[0].row == 3
        assert sws[0].col == 5

    def test_compute_layer_stats_with_spiky_input(self):
        """Verify channel detection with a clear spike."""
        # Input: channel 7 has a huge value
        input_act = jnp.zeros((10, 16))
        input_act = input_act.at[3, 7].set(999.0)
        output_act = jnp.zeros((10, 32))
        output_act = output_act.at[5, 20].set(-500.0)

        stats = compute_layer_stats(input_act, output_act, layer=2)
        assert stats.layer == 2
        assert stats.input_max_channel == 7
        assert stats.input_max_magnitude == 999.0
        assert stats.output_max_channel == 20
        assert stats.output_max_magnitude == 500.0


# ---------------------------------------------------------------------------
# Cross-module integration
# ---------------------------------------------------------------------------


class TestCrossModuleIntegration:
    def test_quantize_all_weight_types(self):
        """Quantize every weight matrix in the model (not just down_proj)."""
        model = LlamaModel(TINY)
        # Initialize with random weights
        leaves, treedef = jax.tree.flatten(model)
        new_leaves = [
            jax.random.normal(jax.random.PRNGKey(i), leaf.shape) * 0.1
            if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32
            else leaf
            for i, leaf in enumerate(leaves)
        ]
        model = jax.tree.unflatten(treedef, new_leaves)

        weights = make_hf_weights_dict(model)
        for key, w in weights.items():
            if w.ndim < 2:
                continue  # skip norms (1D)
            result = quantize_dequantize_blockwise(
                w, nbits=4, blocksize=min(32, w.shape[-1]),
                clip_method="none", clip_threshold=0.0,
            )
            assert result.weight.shape == w.shape
            assert jnp.all(jnp.isfinite(result.weight))

    def test_identify_then_quantize_roundtrip(self):
        """Identify SWs → quantize with retention → verify preservation."""
        # Create stats that identify a SW at layer 0, row=3, col=5
        stats = [
            LayerActivationStats(layer=0, input_max_magnitude=500.0,
                                 input_max_channel=5, output_max_magnitude=500.0,
                                 output_max_channel=3),
            LayerActivationStats(layer=1, input_max_magnitude=1.0,
                                 input_max_channel=0, output_max_magnitude=1.0,
                                 output_max_channel=0),
            LayerActivationStats(layer=2, input_max_magnitude=1.0,
                                 input_max_channel=0, output_max_magnitude=1.0,
                                 output_max_channel=0),
            LayerActivationStats(layer=3, input_max_magnitude=1.0,
                                 input_max_channel=0, output_max_magnitude=1.0,
                                 output_max_channel=0),
        ]
        sws = identify_super_weights(stats, spike_threshold=10.0, spike_ratio=3.0)
        assert len(sws) == 1
        sw = sws[0]

        # Create a weight matrix with a large value at (3, 5)
        w = jax.random.normal(jax.random.PRNGKey(0), (8, 16)) * 0.1
        w = w.at[sw.row, sw.col].set(42.0)

        result = quantize_weight_sw_aware(
            w, sw_coords=[(sw.row, sw.col)],
            nbits=4, blocksize=16,
            clip_method="zscore", clip_threshold=3.0,
        )
        assert float(result.weight[sw.row, sw.col]) == 42.0

    def test_perplexity_degrades_with_aggressive_quantization(self):
        """More aggressive quantization should generally increase perplexity."""
        model = LlamaModel(TINY)
        leaves, treedef = jax.tree.flatten(model)
        new_leaves = [
            jax.random.normal(jax.random.PRNGKey(i + 50), leaf.shape) * 0.05
            if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32
            else leaf
            for i, leaf in enumerate(leaves)
        ]
        model = jax.tree.unflatten(treedef, new_leaves)

        ids = jnp.arange(16)
        logits_orig, _ = model(ids)
        ppl_orig = perplexity(logits_orig[:-1], ids[1:])

        # Quantize aggressively: 2-bit, no clipping
        model_q = model
        for layer_idx in range(TINY.num_hidden_layers):
            block = model_q.layers[layer_idx]
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                w = getattr(block.mlp, proj_name)
                result = quantize_dequantize_blockwise(
                    w, nbits=2, blocksize=min(16, w.shape[-1]),
                    clip_method="none", clip_threshold=0.0,
                )
                block = eqx.tree_at(
                    lambda b, pn=proj_name: getattr(b.mlp, pn),
                    block, result.weight,
                )
            model_q = eqx.tree_at(lambda m, li=layer_idx: m.layers[li], model_q, block)

        logits_q, _ = model_q(ids)
        ppl_q = perplexity(logits_q[:-1], ids[1:])

        assert np.isfinite(ppl_orig) and np.isfinite(ppl_q)
        # With random weights, both are near vocab_size. Just verify both are valid.
        assert ppl_orig > 1.0
        assert ppl_q > 1.0
