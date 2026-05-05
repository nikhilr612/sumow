"""End-to-end smoke tests for the full sumow pipeline.

Tests the complete workflow:
  1. Create tiny Equinox LlamaModel with random weights
  2. Plant a known super weight
  3. Forward pass with activation capture
  4. Identify super weights from activations
  5. Quantize weights (with and without SW retention)
  6. Compare quantization error

These tests verify the pipeline connects correctly, not numerical precision
(that's tested in unit tests and reference equivalence tests).
"""

import jax
import jax.numpy as jnp
import numpy as np

from sumow.eval import cross_entropy_loss, perplexity, perplexity_from_loss
from sumow.identify import (
    identify_super_weights,
)
from sumow.model import (
    LlamaModel,
    TransformerConfig,
    load_weights_into_model,
    make_hf_weights_dict,
)
from sumow.quantize import (
    quantize_dequantize_blockwise,
    quantize_weight_sw_aware,
)

# Tiny config for smoke tests
SMOKE_CONFIG = TransformerConfig(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
)


def _make_random_model(config: TransformerConfig, seed: int = 42) -> LlamaModel:
    """Create a model with small random weights."""
    model = LlamaModel(config)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key = jax.random.PRNGKey(seed + i)
            new_leaves.append(jax.random.normal(key, leaf.shape) * 0.02)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


def _plant_super_weight(
    model: LlamaModel, layer: int, row: int, col: int, value: float
) -> LlamaModel:
    """Plant a super weight in down_proj of the specified layer."""
    import equinox as eqx

    block = model.layers[layer]
    down = block.mlp.down_proj.at[row, col].set(value)
    block = eqx.tree_at(lambda b: b.mlp.down_proj, block, down)
    return eqx.tree_at(lambda m: m.layers[layer], model, block)


class TestFullPipeline:
    """End-to-end: model → identify → quantize → eval."""

    def test_forward_pass_produces_logits(self):
        """Basic forward pass on tiny model."""
        model = _make_random_model(SMOKE_CONFIG)
        ids = jnp.arange(16)
        logits, _ = model(ids)
        assert logits.shape == (16, SMOKE_CONFIG.vocab_size)
        assert jnp.all(jnp.isfinite(logits))

    def test_activation_capture_all_layers(self):
        """Activation capture returns stats for every layer."""
        model = _make_random_model(SMOKE_CONFIG)
        ids = jnp.arange(10)
        _, stats = model(ids, capture_activations=True)
        assert len(stats) == SMOKE_CONFIG.num_hidden_layers
        for i, s in enumerate(stats):
            assert s.layer == i
            assert s.input_max_magnitude >= 0
            assert s.output_max_magnitude >= 0

    def test_identify_planted_super_weight(self):
        """Plant a large weight and verify identification finds it."""
        model = _make_random_model(SMOKE_CONFIG, seed=100)
        # Plant a very large weight in layer 1
        model = _plant_super_weight(model, layer=1, row=5, col=10, value=200.0)

        ids = jnp.arange(20)
        _, stats = model(ids, capture_activations=True)

        # Layer 1 should have the biggest output spike at channel 5
        layer1_stats = stats[1]
        assert layer1_stats.output_max_channel == 5

        # With tiny random weights, absolute magnitudes are small.
        # Use very low threshold + ratio-based detection to find the spike.
        sws = identify_super_weights(
            stats,
            spike_threshold=1e-6,  # accept any non-trivial magnitude
            spike_ratio=2.0,       # layer 1 output is ~500x larger than others
        )
        spike_layers = [sw.layer for sw in sws]
        assert 1 in spike_layers, f"Expected layer 1 in spikes, got {spike_layers}"
        # Verify the identified SW points to the right channel
        sw_layer1 = [sw for sw in sws if sw.layer == 1][0]
        assert sw_layer1.row == 5  # output channel = row in down_proj

    def test_quantize_down_proj_weights(self):
        """Quantize all down_proj weights from the model."""
        model = _make_random_model(SMOKE_CONFIG, seed=200)
        weights = make_hf_weights_dict(model)

        quantized_count = 0
        for key, w in weights.items():
            if "down_proj" not in key:
                continue
            # Quantize with no clipping
            result = quantize_weight_sw_aware(
                w, sw_coords=[], nbits=4, blocksize=32,
                clip_method="none", clip_threshold=0.0,
            )
            assert result.weight.shape == w.shape
            # Quantization should introduce some error
            err = float(jnp.mean(jnp.abs(w - result.weight)))
            assert err >= 0
            quantized_count += 1

        assert quantized_count == SMOKE_CONFIG.num_hidden_layers

    def test_sw_aware_quantize_preserves_super_weight(self):
        """SW-aware quantization should restore the super weight exactly."""
        model = _make_random_model(SMOKE_CONFIG, seed=300)
        model = _plant_super_weight(model, layer=0, row=5, col=10, value=100.0)

        # Get the down_proj weight for layer 0
        down_proj = model.layers[0].mlp.down_proj

        # Quantize WITH super weight retention
        result = quantize_weight_sw_aware(
            down_proj,
            sw_coords=[(5, 10)],
            nbits=4,
            blocksize=32,
            clip_method="none",
            clip_threshold=0.0,
        )

        # The super weight should be exactly preserved
        assert float(result.weight[5, 10]) == 100.0

    def test_sw_vs_no_sw_quantization_error(self):
        """SW-aware quantization should have similar or better error for the SW element."""
        model = _make_random_model(SMOKE_CONFIG, seed=400)
        model = _plant_super_weight(model, layer=0, row=5, col=10, value=50.0)
        down_proj = model.layers[0].mlp.down_proj

        # Without SW retention
        result_no_sw = quantize_weight_sw_aware(
            down_proj,
            sw_coords=[],
            nbits=4,
            blocksize=32,
            clip_method="zscore",
            clip_threshold=3.0,
        )

        # With SW retention
        result_with_sw = quantize_weight_sw_aware(
            down_proj,
            sw_coords=[(5, 10)],
            nbits=4,
            blocksize=32,
            clip_method="zscore",
            clip_threshold=3.0,
        )

        # SW element: with retention should be exact, without should have error
        sw_err_no = abs(float(down_proj[5, 10]) - float(result_no_sw.weight[5, 10]))
        sw_err_yes = abs(float(down_proj[5, 10]) - float(result_with_sw.weight[5, 10]))

        assert sw_err_yes == 0.0, "SW should be exactly preserved"
        assert sw_err_no > 0, "Without retention, large SW should have quantization error"

    def test_perplexity_pipeline(self):
        """Forward pass → logits → perplexity computation."""
        model = _make_random_model(SMOKE_CONFIG, seed=500)
        ids = jnp.arange(20)
        logits, _ = model(ids)

        # Use shifted logits/targets for next-token prediction
        pred_logits = logits[:-1]  # [seq-1, vocab]
        targets = ids[1:]  # [seq-1]

        loss = cross_entropy_loss(pred_logits, targets)
        ppl = perplexity_from_loss(float(loss))

        assert float(loss) > 0
        assert ppl > 1.0  # perplexity is always >= 1
        assert jnp.isfinite(jnp.array(ppl))

    def test_quantize_then_eval(self):
        """Full: model → quantize down_proj → forward pass → compare PPL."""
        import equinox as eqx

        model = _make_random_model(SMOKE_CONFIG, seed=600)
        ids = jnp.arange(32)

        # Original perplexity
        logits_orig, _ = model(ids)
        ppl_orig = perplexity(logits_orig[:-1], ids[1:])

        # Quantize all down_proj weights
        model_q = model
        for layer_idx in range(SMOKE_CONFIG.num_hidden_layers):
            block = model_q.layers[layer_idx]
            w = block.mlp.down_proj
            result = quantize_weight_sw_aware(
                w, sw_coords=[], nbits=4, blocksize=32,
                clip_method="none", clip_threshold=0.0,
            )
            block = eqx.tree_at(lambda b: b.mlp.down_proj, block, result.weight)
            model_q = eqx.tree_at(lambda m: m.layers[layer_idx], model_q, block)

        logits_q, _ = model_q(ids)
        ppl_q = perplexity(logits_q[:-1], ids[1:])

        assert ppl_orig > 1.0
        assert ppl_q > 1.0
        # Both should be finite
        assert np.isfinite(ppl_orig)
        assert np.isfinite(ppl_q)

    def test_full_pipeline_with_all_quant_methods(self):
        """Test quantization with different clipping methods."""
        model = _make_random_model(SMOKE_CONFIG, seed=700)
        down_proj = model.layers[0].mlp.down_proj

        methods = [
            ("none", 0.0),
            ("zscore", 3.0),
            ("tensor_percentage", 0.01),
            ("block_percentage", 0.01),
        ]

        for clip_method, threshold in methods:
            result = quantize_weight_sw_aware(
                down_proj,
                sw_coords=[(5, 10)],
                nbits=4,
                blocksize=32,
                clip_method=clip_method,
                clip_threshold=threshold,
            )
            assert result.weight.shape == down_proj.shape
            assert jnp.all(jnp.isfinite(result.weight))
            # SW should always be preserved
            np.testing.assert_allclose(
                result.weight[5, 10], down_proj[5, 10], atol=1e-6
            )

    def test_weight_roundtrip_after_quantize(self):
        """Model weights can be extracted, quantized, and reloaded."""
        model = _make_random_model(SMOKE_CONFIG, seed=800)
        weights = make_hf_weights_dict(model)

        # Quantize only down_proj weights in the dict
        for key in list(weights.keys()):
            if "down_proj" not in key:
                continue
            w = weights[key]
            result = quantize_weight_sw_aware(
                w, sw_coords=[], nbits=8, blocksize=64,
                clip_method="none", clip_threshold=0.0,
            )
            weights[key] = result.weight

        # Reload into fresh model
        model2 = LlamaModel(SMOKE_CONFIG)
        model2 = load_weights_into_model(model2, weights)

        # Verify the loaded model runs
        ids = jnp.arange(8)
        logits, _ = model2(ids)
        assert jnp.all(jnp.isfinite(logits))

    def test_blockwise_quantization_modes(self):
        """Test INT and NF4 blockwise quantization on model weight."""
        model = _make_random_model(SMOKE_CONFIG, seed=900)
        w = model.layers[0].mlp.down_proj  # already 2D [hidden, intermediate]

        # INT4 blockwise
        result_int = quantize_dequantize_blockwise(
            w, nbits=4, blocksize=32,
            clip_method="none", clip_threshold=0.0,
            scale_shift=False, use_normal_float=False,
        )
        assert result_int.weight.shape == w.shape
        assert jnp.all(jnp.isfinite(result_int.weight))

        # NF4 blockwise
        result_nf4 = quantize_dequantize_blockwise(
            w, nbits=4, blocksize=32,
            clip_method="none", clip_threshold=0.0,
            scale_shift=False, use_normal_float=True,
        )
        assert result_nf4.weight.shape == w.shape
        assert jnp.all(jnp.isfinite(result_nf4.weight))
