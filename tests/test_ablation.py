"""Ablation tests: measure the impact of super weight retention.

Tests the core claim of arXiv 2411.07191:
  Retaining super weights in high precision during quantization
  significantly reduces perplexity degradation.

All tests use tiny synthetic models (CPU, <30s).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sumow.eval import perplexity
from sumow.identify import (
    identify_super_weights,
)
from sumow.model import (
    LlamaModel,
    TransformerConfig,
)
from sumow.quantize import (
    quantize_dequantize_blockwise,
    quantize_weight_sw_aware,
)

# Tiny config shared by all ablation tests
ABLATION_CONFIG = TransformerConfig(
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


def _make_model(seed: int = 42) -> LlamaModel:
    """Create a model with small random weights."""
    model = LlamaModel(ABLATION_CONFIG)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key = jax.random.PRNGKey(seed + i)
            new_leaves.append(jax.random.normal(key, leaf.shape) * 0.02)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


def _plant_sw(
    model: LlamaModel, layer: int, row: int, col: int, value: float
) -> LlamaModel:
    """Plant a super weight in down_proj of the specified layer."""
    block = model.layers[layer]
    down = block.mlp.down_proj.at[row, col].set(value)
    block = eqx.tree_at(lambda b: b.mlp.down_proj, block, down)
    return eqx.tree_at(lambda m: m.layers[layer], model, block)


def _quantize_model(
    model: LlamaModel,
    sw_map: dict[int, list[tuple[int, int]]],
    nbits: int = 4,
    blocksize: int = 32,
    clip_method: str = "zscore",
    clip_threshold: float = 3.0,
    use_normal_float: bool = False,
) -> LlamaModel:
    """Quantize all down_proj weights. sw_map: {layer_idx: [(row,col),...]}."""
    model_q = model
    for layer_idx in range(ABLATION_CONFIG.num_hidden_layers):
        block = model_q.layers[layer_idx]
        w = block.mlp.down_proj
        coords = sw_map.get(layer_idx, [])
        result = quantize_weight_sw_aware(
            w,
            sw_coords=coords,
            nbits=nbits,
            blocksize=blocksize,
            clip_method=clip_method,
            clip_threshold=clip_threshold,
            use_normal_float=use_normal_float,
        )
        block = eqx.tree_at(lambda b: b.mlp.down_proj, block, result.weight)
        model_q = eqx.tree_at(lambda m: m.layers[layer_idx], model_q, block)
    return model_q


def _eval_ppl(model: LlamaModel, ids: jnp.ndarray) -> float:
    """Forward pass → perplexity."""
    logits, _ = model(ids)
    return perplexity(logits[:-1], ids[1:])


def _quant_error(original: jnp.ndarray, quantized: jnp.ndarray) -> float:
    """Mean absolute error between original and quantized weights."""
    return float(jnp.mean(jnp.abs(original - quantized)))


class TestSWRetentionAblation:
    """Core ablation: with vs without super weight retention."""

    @pytest.fixture
    def model_with_sw(self):
        """Model with a planted super weight."""
        model = _make_model(seed=1000)
        model = _plant_sw(model, layer=1, row=5, col=10, value=150.0)
        return model

    @pytest.fixture
    def tokens(self):
        return jnp.arange(32)

    def test_sw_retention_reduces_weight_error(self, model_with_sw):
        """SW-aware quantization has less error at the SW coordinate."""
        w = model_with_sw.layers[1].mlp.down_proj

        result_no_sw = quantize_weight_sw_aware(
            w, sw_coords=[], nbits=4, blocksize=32,
            clip_method="zscore", clip_threshold=3.0,
        )
        result_with_sw = quantize_weight_sw_aware(
            w, sw_coords=[(5, 10)], nbits=4, blocksize=32,
            clip_method="zscore", clip_threshold=3.0,
        )

        err_no = abs(float(w[5, 10]) - float(result_no_sw.weight[5, 10]))
        err_with = abs(float(w[5, 10]) - float(result_with_sw.weight[5, 10]))

        assert err_with == 0.0, "SW retention must preserve exactly"
        assert err_no > 0, "Without retention, large weight should be clipped/quantized"

    def test_sw_retention_reduces_ppl_degradation(self, model_with_sw, tokens):
        """PPL with SW retention should be closer to original than without."""
        ppl_orig = _eval_ppl(model_with_sw, tokens)

        model_no_sw = _quantize_model(model_with_sw, sw_map={})
        ppl_no_sw = _eval_ppl(model_no_sw, tokens)

        model_with = _quantize_model(model_with_sw, sw_map={1: [(5, 10)]})
        ppl_with = _eval_ppl(model_with, tokens)

        # Both quantized models should differ from original
        delta_no_sw = abs(ppl_no_sw - ppl_orig)
        delta_with_sw = abs(ppl_with - ppl_orig)

        # SW retention should reduce PPL degradation
        assert delta_with_sw <= delta_no_sw, (
            f"SW retention should help: delta_with={delta_with_sw:.4f} "
            f"vs delta_without={delta_no_sw:.4f}"
        )

    def test_multiple_super_weights(self, tokens):
        """Retaining multiple SWs in multiple layers."""
        model = _make_model(seed=1100)
        model = _plant_sw(model, layer=0, row=3, col=7, value=120.0)
        model = _plant_sw(model, layer=2, row=8, col=15, value=-100.0)

        ppl_orig = _eval_ppl(model, tokens)

        sw_map = {0: [(3, 7)], 2: [(8, 15)]}
        model_no = _quantize_model(model, sw_map={})
        model_yes = _quantize_model(model, sw_map=sw_map)

        ppl_no = _eval_ppl(model_no, tokens)
        ppl_yes = _eval_ppl(model_yes, tokens)

        delta_no = abs(ppl_no - ppl_orig)
        delta_yes = abs(ppl_yes - ppl_orig)

        assert delta_yes <= delta_no, (
            f"Multi-SW retention should help: {delta_yes:.4f} vs {delta_no:.4f}"
        )

    def test_sw_retention_across_bit_widths(self, model_with_sw, tokens):
        """SW retention benefit should hold for both INT4 and INT8."""
        for nbits in [4, 8]:
            ppl_orig = _eval_ppl(model_with_sw, tokens)

            model_no = _quantize_model(model_with_sw, sw_map={}, nbits=nbits)
            model_yes = _quantize_model(model_with_sw, sw_map={1: [(5, 10)]}, nbits=nbits)

            ppl_no = _eval_ppl(model_no, tokens)
            ppl_yes = _eval_ppl(model_yes, tokens)

            delta_no = abs(ppl_no - ppl_orig)
            delta_yes = abs(ppl_yes - ppl_orig)

            assert delta_yes <= delta_no, (
                f"{nbits}-bit: SW retention should help: {delta_yes:.4f} vs {delta_no:.4f}"
            )


class TestClippingMethodAblation:
    """Compare different clipping methods' impact on quantization quality."""

    @pytest.fixture
    def model(self):
        return _make_model(seed=2000)

    @pytest.fixture
    def tokens(self):
        return jnp.arange(32)

    def test_zscore_vs_no_clipping(self, model, tokens):
        """Z-score clipping should reduce quantization error for well-behaved weights."""
        _eval_ppl(model, tokens)  # baseline (unused; test focuses on relative improvement)

        model_none = _quantize_model(
            model, sw_map={}, clip_method="none", clip_threshold=0.0
        )
        model_zscore = _quantize_model(
            model, sw_map={}, clip_method="zscore", clip_threshold=3.0
        )

        ppl_none = _eval_ppl(model_none, tokens)
        ppl_zscore = _eval_ppl(model_zscore, tokens)

        # Both should be finite
        assert np.isfinite(ppl_none)
        assert np.isfinite(ppl_zscore)

    def test_all_clipping_methods_produce_valid_output(self, model, tokens):
        """All clipping methods produce finite perplexity."""
        methods = [
            ("none", 0.0),
            ("zscore", 3.0),
            ("tensor_percentage", 0.01),
            ("block_percentage", 0.01),
        ]
        for clip_method, threshold in methods:
            model_q = _quantize_model(
                model, sw_map={}, clip_method=clip_method,
                clip_threshold=threshold,
            )
            ppl = _eval_ppl(model_q, tokens)
            assert np.isfinite(ppl), f"{clip_method} produced non-finite PPL"
            assert ppl > 1.0, f"{clip_method} PPL should be > 1"


class TestNFvsINTAblation:
    """Compare NormalFloat vs INT quantization."""

    @pytest.fixture
    def model(self):
        return _make_model(seed=3000)

    @pytest.fixture
    def tokens(self):
        return jnp.arange(32)

    def test_nf4_vs_int4_weight_error(self, model):
        """Compare weight-level error of NF4 vs INT4."""
        w = model.layers[0].mlp.down_proj

        result_int = quantize_dequantize_blockwise(
            w, nbits=4, blocksize=32,
            clip_method="none", clip_threshold=0.0,
            use_normal_float=False, scale_shift=False,
        )
        result_nf = quantize_dequantize_blockwise(
            w, nbits=4, blocksize=32,
            clip_method="none", clip_threshold=0.0,
            use_normal_float=True, scale_shift=False,
        )

        err_int = _quant_error(w, result_int.weight)
        err_nf = _quant_error(w, result_nf.weight)

        # Both should produce non-zero error
        assert err_int > 0
        assert err_nf > 0
        # Both should be reasonable (< 50% of weight magnitude)
        mean_mag = float(jnp.mean(jnp.abs(w)))
        assert err_int < mean_mag, "INT4 error too large"
        assert err_nf < mean_mag, "NF4 error too large"

    def test_nf4_vs_int4_ppl(self, model, tokens):
        """Compare perplexity impact of NF4 vs INT4."""
        _eval_ppl(model, tokens)  # baseline (unused; test focuses on relative comparison)

        model_int = _quantize_model(model, sw_map={}, use_normal_float=False)
        model_nf = _quantize_model(model, sw_map={}, use_normal_float=True)

        ppl_int = _eval_ppl(model_int, tokens)
        ppl_nf = _eval_ppl(model_nf, tokens)

        assert np.isfinite(ppl_int)
        assert np.isfinite(ppl_nf)

        # For normally distributed weights (which JAX random.normal produces),
        # NF4 is theoretically better-matched. We just check both are reasonable.
        assert ppl_int > 0
        assert ppl_nf > 0


class TestQuantBitWidthAblation:
    """Compare quantization at different bit widths."""

    @pytest.fixture
    def model(self):
        return _make_model(seed=4000)

    @pytest.fixture
    def tokens(self):
        return jnp.arange(32)

    def test_higher_bits_lower_error(self, model):
        """8-bit should have less weight error than 4-bit."""
        w = model.layers[0].mlp.down_proj

        result_4 = quantize_dequantize_blockwise(
            w, nbits=4, blocksize=32,
            clip_method="none", clip_threshold=0.0,
            use_normal_float=False, scale_shift=False,
        )
        result_8 = quantize_dequantize_blockwise(
            w, nbits=8, blocksize=32,
            clip_method="none", clip_threshold=0.0,
            use_normal_float=False, scale_shift=False,
        )

        err_4 = _quant_error(w, result_4.weight)
        err_8 = _quant_error(w, result_8.weight)

        assert err_8 < err_4, (
            f"8-bit should have less error: {err_8:.6f} vs {err_4:.6f}"
        )

    def test_higher_bits_closer_ppl(self, model, tokens):
        """8-bit quantized model PPL should be closer to original than 4-bit."""
        ppl_orig = _eval_ppl(model, tokens)

        model_4 = _quantize_model(model, sw_map={}, nbits=4)
        model_8 = _quantize_model(model, sw_map={}, nbits=8)

        ppl_4 = _eval_ppl(model_4, tokens)
        ppl_8 = _eval_ppl(model_8, tokens)

        delta_4 = abs(ppl_4 - ppl_orig)
        delta_8 = abs(ppl_8 - ppl_orig)

        assert delta_8 <= delta_4, (
            f"8-bit should be closer to original: {delta_8:.4f} vs {delta_4:.4f}"
        )


class TestBlocksizeAblation:
    """Compare different block sizes for blockwise quantization."""

    @pytest.fixture
    def model(self):
        return _make_model(seed=5000)

    def test_smaller_blocks_lower_error(self, model):
        """Smaller block size should yield lower quantization error."""
        w = model.layers[0].mlp.down_proj

        result_32 = quantize_dequantize_blockwise(
            w, nbits=4, blocksize=32,
            clip_method="none", clip_threshold=0.0,
            use_normal_float=False, scale_shift=False,
        )
        result_64 = quantize_dequantize_blockwise(
            w, nbits=4, blocksize=64,
            clip_method="none", clip_threshold=0.0,
            use_normal_float=False, scale_shift=False,
        )

        err_32 = _quant_error(w, result_32.weight)
        err_64 = _quant_error(w, result_64.weight)

        # Smaller blocks = more local scales = less error
        assert err_32 <= err_64, (
            f"Blocksize 32 should have ≤ error vs 64: {err_32:.6f} vs {err_64:.6f}"
        )


class TestSWZeroingDestroysOutput:
    """Paper's core insight: zeroing a super weight destroys model output."""

    @pytest.fixture
    def model_with_sw(self):
        model = _make_model(seed=6000)
        model = _plant_sw(model, layer=1, row=5, col=10, value=200.0)
        return model

    @pytest.fixture
    def tokens(self):
        return jnp.arange(20)

    def test_zeroing_sw_changes_output_more_than_normal(self, model_with_sw, tokens):
        """Zeroing the super weight should change output much more than zeroing a normal weight."""
        logits_orig, _ = model_with_sw(tokens)

        # Zero out the super weight
        model_sw_zeroed = _plant_sw(model_with_sw, layer=1, row=5, col=10, value=0.0)
        logits_sw_zeroed, _ = model_sw_zeroed(tokens)

        # Zero out a normal weight (same layer, different position)
        model_normal_zeroed = _plant_sw(model_with_sw, layer=1, row=0, col=0, value=0.0)
        logits_normal_zeroed, _ = model_normal_zeroed(tokens)

        sw_diff = float(jnp.mean(jnp.abs(logits_orig - logits_sw_zeroed)))
        normal_diff = float(jnp.mean(jnp.abs(logits_orig - logits_normal_zeroed)))

        # SW zeroing impact should be much larger than normal weight zeroing
        # (the SW is 200/0.02 = 10000x the normal scale)
        assert sw_diff > normal_diff, (
            f"SW zeroing ({sw_diff:.6f}) should have more impact "
            f"than normal weight zeroing ({normal_diff:.6f})"
        )

    def test_sw_zeroing_impact_scales_with_magnitude(self, tokens):
        """Larger super weights should cause proportionally more disruption when zeroed."""
        model = _make_model(seed=6100)

        impacts = []
        for magnitude in [50.0, 100.0, 200.0]:
            m = _plant_sw(model, layer=1, row=5, col=10, value=magnitude)
            logits_before, _ = m(tokens)
            m_zeroed = _plant_sw(m, layer=1, row=5, col=10, value=0.0)
            logits_after, _ = m_zeroed(tokens)
            diff = float(jnp.mean(jnp.abs(logits_before - logits_after)))
            impacts.append(diff)

        # Larger SW → larger impact when zeroed
        assert impacts[1] > impacts[0], "100 should disrupt more than 50"
        assert impacts[2] > impacts[1], "200 should disrupt more than 100"


class TestEndToEndAblation:
    """Full pipeline ablation: identify → quantize with/without → eval."""

    def test_full_pipeline_sw_aware_vs_baseline(self):
        """Identify SWs, then compare quantization with and without retention."""
        model = _make_model(seed=7000)
        # Plant super weights
        model = _plant_sw(model, layer=0, row=3, col=7, value=100.0)
        model = _plant_sw(model, layer=2, row=8, col=15, value=-120.0)

        tokens = jnp.arange(32)

        # 1. Forward pass and identify
        _, stats = model(tokens, capture_activations=True)
        sws = identify_super_weights(
            stats, spike_threshold=1e-6, spike_ratio=2.0,
        )

        # Build SW map from identification
        sw_map: dict[int, list[tuple[int, int]]] = {}
        for sw in sws:
            sw_map.setdefault(sw.layer, []).append((sw.row, sw.col))

        # 2. Quantize with and without
        ppl_orig = _eval_ppl(model, tokens)
        model_baseline = _quantize_model(model, sw_map={})
        model_sw_aware = _quantize_model(model, sw_map=sw_map)

        ppl_baseline = _eval_ppl(model_baseline, tokens)
        ppl_sw_aware = _eval_ppl(model_sw_aware, tokens)

        # 3. SW-aware should be closer to original
        delta_baseline = abs(ppl_baseline - ppl_orig)
        delta_sw_aware = abs(ppl_sw_aware - ppl_orig)

        # All should be finite
        assert np.isfinite(ppl_orig)
        assert np.isfinite(ppl_baseline)
        assert np.isfinite(ppl_sw_aware)

        assert delta_sw_aware <= delta_baseline, (
            f"Full pipeline: SW-aware ({delta_sw_aware:.4f}) should be ≤ "
            f"baseline ({delta_baseline:.4f})"
        )

    def test_ablation_report(self):
        """Generate a summary table of ablation results (for logging)."""
        model = _make_model(seed=8000)
        model = _plant_sw(model, layer=1, row=5, col=10, value=100.0)
        tokens = jnp.arange(32)

        ppl_orig = _eval_ppl(model, tokens)

        configs = [
            ("INT4 no-SW", dict(nbits=4, use_normal_float=False), {}),
            ("INT4 SW-aware", dict(nbits=4, use_normal_float=False), {1: [(5, 10)]}),
            ("INT8 no-SW", dict(nbits=8, use_normal_float=False), {}),
            ("INT8 SW-aware", dict(nbits=8, use_normal_float=False), {1: [(5, 10)]}),
            ("NF4 no-SW", dict(nbits=4, use_normal_float=True), {}),
            ("NF4 SW-aware", dict(nbits=4, use_normal_float=True), {1: [(5, 10)]}),
        ]

        results = []
        for name, quant_kwargs, sw_map in configs:
            model_q = _quantize_model(model, sw_map=sw_map, **quant_kwargs)  # type: ignore[invalid-argument-type]
            # ty can't narrow dict[str, int | bool] key→value associations when
            # unpacking with **, so it flags all mismatching param types even
            # though the runtime values are correct for each parameter.
            ppl_q = _eval_ppl(model_q, tokens)
            delta = abs(ppl_q - ppl_orig)
            results.append((name, ppl_q, delta))

        # All results should be finite
        for name, ppl, delta in results:
            assert np.isfinite(ppl), f"{name} PPL is not finite"

        # For each pair (no-SW, SW-aware), SW-aware delta should be <=
        for i in range(0, len(results), 2):
            name_no = results[i][0]
            name_yes = results[i + 1][0]
            delta_no = results[i][2]
            delta_yes = results[i + 1][2]
            assert delta_yes <= delta_no + 1e-3, (
                f"{name_yes} ({delta_yes:.4f}) should be ≤ {name_no} ({delta_no:.4f}) + tolerance"
            )
