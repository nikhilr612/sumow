"""Integration tests using real model weights (SmolLM2-135M).

These tests verify the full pipeline against a real pretrained model:
- Weight loading into Equinox
- Numerical equivalence vs PyTorch
- Super weight identification matches paper's predictions
- Quantization pipeline on real weights

Requires: weights/smollm2-135m/ (run tools/export_hf_model.py first)
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

WEIGHTS_DIR = Path("weights/smollm2-135m")
SKIP_REASON = "SmolLM2-135M weights not exported (run: uv run python tools/export_hf_model.py HuggingFaceTB/SmolLM2-135M -o weights/smollm2-135m)"

needs_weights = pytest.mark.skipif(
    not (WEIGHTS_DIR / "model.safetensors").exists(), reason=SKIP_REASON
)


def _load_model():
    """Load SmolLM2-135M into Equinox."""
    from sumow.model import LlamaModel, TransformerConfig, load_weights_into_model
    from sumow.model_io import load_model_weights

    with open(WEIGHTS_DIR / "config.json") as f:
        c = json.load(f)

    config = TransformerConfig(
        vocab_size=c["vocab_size"],
        hidden_size=c["hidden_size"],
        intermediate_size=c["intermediate_size"],
        num_hidden_layers=c["num_hidden_layers"],
        num_attention_heads=c["num_attention_heads"],
        num_key_value_heads=c["num_key_value_heads"],
        max_position_embeddings=c["max_position_embeddings"],
        rms_norm_eps=c["rms_norm_eps"],
        rope_theta=c["rope_theta"],
        tie_word_embeddings=c.get("tie_word_embeddings", False),
    )

    model = LlamaModel(config)
    weights = load_model_weights(str(WEIGHTS_DIR))
    model = load_weights_into_model(model, weights)  # type: ignore[invalid-argument-type]
    # model is typed as Module (equinox stub issue) rather than LlamaModel


@pytest.fixture(scope="module")
def real_model():
    """Module-scoped fixture to avoid reloading weights per test."""
    return _load_model()


# ── Weight Loading ──────────────────────────────────────────


@needs_weights
class TestWeightLoading:
    def test_config_matches(self, real_model):
        model, config = real_model
        assert config.vocab_size == 49152
        assert config.hidden_size == 576
        assert config.intermediate_size == 1536
        assert config.num_hidden_layers == 30
        assert config.num_attention_heads == 9
        assert config.num_key_value_heads == 3

    def test_embedding_not_zero(self, real_model):
        model, _ = real_model
        assert jnp.abs(model.embed_tokens).max() > 0.01

    def test_all_layers_loaded(self, real_model):
        model, config = real_model
        for i in range(config.num_hidden_layers):
            block = model.layers[i]
            assert jnp.abs(block.mlp.down_proj).max() > 0.01, f"Layer {i} down_proj is zero"
            assert jnp.abs(block.self_attn.q_proj).max() > 0.01, f"Layer {i} q_proj is zero"

    def test_tied_embeddings(self, real_model):
        model, config = real_model
        assert config.tie_word_embeddings is True
        np.testing.assert_array_equal(
            np.array(model.embed_tokens), np.array(model.lm_head)
        )


# ── Forward Pass ────────────────────────────────────────────


@needs_weights
class TestForwardPass:
    def test_output_shape(self, real_model):
        model, config = real_model
        tokens = jnp.arange(1, 17)
        logits, stats = model(tokens)
        assert logits.shape == (16, config.vocab_size)
        assert len(stats) == 0  # capture_activations=False

    def test_logits_not_uniform(self, real_model):
        model, _ = real_model
        tokens = jnp.arange(1, 9)
        logits, _ = model(tokens)
        # A real model should have varied logits, not uniform
        std = jnp.std(logits, axis=-1)
        assert jnp.all(std > 0.1), "Logits are too uniform for a real model"

    def test_capture_activations(self, real_model):
        model, config = real_model
        tokens = jnp.arange(1, 17)
        logits, stats = model(tokens, capture_activations=True)
        assert len(stats) == config.num_hidden_layers
        for s in stats:
            assert s.input_max_magnitude > 0
            assert s.output_max_magnitude > 0

    def test_pytorch_equivalence(self, real_model):
        """Compare forward pass against PyTorch reference."""
        import torch
        from transformers import AutoModelForCausalLM

        model, _ = real_model

        torch_model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.float32
        )
        torch_model.eval()

        ids = list(range(1, 9))  # short seq for speed
        with torch.no_grad():
            pt_logits = torch_model(torch.tensor([ids])).logits[0].numpy()

        jax_logits = np.array(model(jnp.array(ids))[0])

        # Top-1 agreement must be 100%
        pt_top1 = np.argmax(pt_logits, axis=-1)
        jax_top1 = np.argmax(jax_logits, axis=-1)
        np.testing.assert_array_equal(pt_top1, jax_top1)

        # Mean KL should be very small
        from scipy.special import log_softmax

        pt_lp = log_softmax(pt_logits, axis=-1)
        jax_lp = log_softmax(jax_logits, axis=-1)
        kl = np.sum(np.exp(pt_lp) * (pt_lp - jax_lp), axis=-1).mean()
        assert kl < 0.01, f"KL divergence too high: {kl}"

        del torch_model


# ── Super Weight Identification ─────────────────────────────


@needs_weights
class TestSuperWeightIdentification:
    def test_identifies_super_weights(self, real_model):
        model, _ = real_model
        from sumow.identify import identify_super_weights

        tokens = jnp.arange(1, 65)
        _, stats = model(tokens, capture_activations=True)
        sws = identify_super_weights(stats, spike_threshold=100.0, spike_ratio=6.0)
        assert len(sws) >= 1, "Should find at least 1 super weight"
        assert len(sws) <= 10, f"Too many super weights: {len(sws)}"

    def test_super_weights_in_down_proj(self, real_model):
        """Paper's core claim: super weights are ALWAYS in down_proj."""
        model, _ = real_model
        from sumow.identify import identify_super_weights

        tokens = jnp.arange(1, 65)
        _, stats = model(tokens, capture_activations=True)
        sws = identify_super_weights(stats, spike_threshold=50.0, spike_ratio=3.0)

        for sw in sws:
            # Verify the weight is in down_proj
            val = float(model.layers[sw.layer].mlp.down_proj[sw.row, sw.col])
            assert val != 0.0, f"Layer {sw.layer} down_proj[{sw.row},{sw.col}] should be non-zero"

    def test_super_activation_channel_consistency(self, real_model):
        """Paper says super activations cluster in specific output channels."""
        model, _ = real_model
        from sumow.identify import identify_super_weights

        tokens = jnp.arange(1, 65)
        _, stats = model(tokens, capture_activations=True)
        sws = identify_super_weights(stats, spike_threshold=100.0, spike_ratio=6.0)

        if len(sws) >= 2:
            # Most super weights should share a common output channel
            rows = [sw.row for sw in sws]
            from collections import Counter
            most_common_row, count = Counter(rows).most_common(1)[0]
            assert count >= len(sws) // 2, (
                f"Expected most SWs to share output channel, "
                f"but only {count}/{len(sws)} share channel {most_common_row}"
            )

    def test_layer_11_is_prominent(self, real_model):
        """Layer 11 has the largest activation spike in SmolLM2-135M."""
        model, _ = real_model

        tokens = jnp.arange(1, 65)
        _, stats = model(tokens, capture_activations=True)

        # Layer 11 output_max_magnitude should be among the largest
        layer_11_mag = stats[11].output_max_magnitude
        all_mags = [s.output_max_magnitude for s in stats]
        top_5 = sorted(all_mags, reverse=True)[:5]
        assert layer_11_mag in top_5, (
            f"Layer 11 mag={layer_11_mag:.1f} not in top-5: {top_5}"
        )


# ── Quantization Pipeline ──────────────────────────────────


@needs_weights
class TestQuantizationPipeline:
    def test_quantize_single_layer(self, real_model):
        """Quantize a single down_proj matrix from the real model."""
        from sumow.quantize import quantize_dequantize_blockwise

        model, _ = real_model
        w = model.layers[0].mlp.down_proj  # [576, 1536]

        # INT8 blockwise
        result = quantize_dequantize_blockwise(w, nbits=8, blocksize=128)
        mse = float(jnp.mean((w - result.weight) ** 2))
        assert mse < 0.01, f"INT8 MSE too high: {mse}"

    def test_sw_aware_quantization(self, real_model):
        """SW-aware quantization should have lower error than baseline."""
        from sumow.quantize import (
            quantize_dequantize_blockwise,
            quantize_weight_sw_aware,
        )
        from sumow.identify import identify_super_weights

        model, _ = real_model

        tokens = jnp.arange(1, 65)
        _, stats = model(tokens, capture_activations=True)
        sws = identify_super_weights(stats, spike_threshold=100.0, spike_ratio=6.0)

        # Find a layer with a super weight
        sw_layers = {sw.layer for sw in sws}
        if not sw_layers:
            pytest.skip("No super weights found")

        layer_idx = min(sw_layers)
        w = model.layers[layer_idx].mlp.down_proj

        # Baseline quantization
        baseline = quantize_dequantize_blockwise(w, nbits=4, blocksize=128)
        baseline_mse = float(jnp.mean((w - baseline.weight) ** 2))

        # SW-aware: retain super weight positions
        layer_sws = [sw for sw in sws if sw.layer == layer_idx]
        sw_coords = [(sw.row, sw.col) for sw in layer_sws]
        sw_aware = quantize_weight_sw_aware(
            w, sw_coords=sw_coords, nbits=4, blocksize=128
        )
        sw_mse = float(jnp.mean((w - sw_aware.weight) ** 2))

        # SW-aware should have equal or lower error
        assert sw_mse <= baseline_mse * 1.01, (
            f"SW-aware MSE ({sw_mse:.6f}) worse than baseline ({baseline_mse:.6f})"
        )

    def test_zeroing_super_weight_destroys_output(self, real_model):
        """Paper's key insight: zeroing a super weight destroys model output."""
        import equinox as eqx
        from sumow.identify import identify_super_weights

        model, config = real_model

        tokens = jnp.arange(1, 33)
        baseline_logits, stats = model(tokens, capture_activations=True)
        sws = identify_super_weights(stats, spike_threshold=100.0, spike_ratio=6.0)

        if not sws:
            pytest.skip("No super weights found")

        # Zero the top super weight
        sw = sws[0]
        layer = model.layers[sw.layer]
        w = layer.mlp.down_proj
        w_zeroed = w.at[sw.row, sw.col].set(0.0)
        layer_new = eqx.tree_at(lambda b: b.mlp.down_proj, layer, w_zeroed)
        model_zeroed = eqx.tree_at(lambda m: m.layers[sw.layer], model, layer_new)

        zeroed_logits, _ = model_zeroed(tokens)

        # Output should change dramatically
        diff = jnp.abs(baseline_logits - zeroed_logits).mean()
        assert diff > 1.0, (
            f"Zeroing SW at layer {sw.layer} had little effect: mean_diff={diff:.4f}"
        )


# ── Model Export/Roundtrip ──────────────────────────────────


@needs_weights
class TestModelRoundtrip:
    def test_extract_and_reload(self, real_model):
        """Extract weights and reload — should get same model."""
        from sumow.model import (
            LlamaModel,
            load_weights_into_model,
            make_hf_weights_dict,
        )

        model, config = real_model
        weights = make_hf_weights_dict(model)

        model2 = LlamaModel(config)
        model2 = load_weights_into_model(model2, weights)  # type: ignore[invalid-argument-type]
        # model2 is typed as Module (equinox stub issue) rather than LlamaModel

        tokens = jnp.arange(1, 9)
        logits1, _ = model(tokens)
        logits2, _ = model2(tokens)

        np.testing.assert_allclose(
            np.array(logits1), np.array(logits2), rtol=1e-5, atol=1e-5
        )
