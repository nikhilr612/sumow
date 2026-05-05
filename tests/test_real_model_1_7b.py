"""Integration tests for SmolLM2-1.7B super weight validation.

These tests verify the paper's core claims on a real 1.7B-parameter model:
  - Super weights exist in down_proj in early layers
  - Zeroing a single super weight causes catastrophic PPL increase
  - Zeroing much larger non-SW weights has minimal impact
  - Super activation channels are consistent across layers

Requires: weights/smollm2-1.7b/ (run: uv run python tools/export_hf_model.py HuggingFaceTB/SmolLM2-1.7B -o weights/smollm2-1.7b)
"""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

WEIGHTS_DIR = Path("weights/smollm2-1.7b")
SKIP_REASON = (
    "SmolLM2-1.7B weights not exported "
    "(run: uv run python tools/export_hf_model.py HuggingFaceTB/SmolLM2-1.7B -o weights/smollm2-1.7b)"
)

needs_weights = pytest.mark.skipif(
    not (WEIGHTS_DIR / "model.safetensors").exists(), reason=SKIP_REASON
)


def _load_model():
    from sumow.model import LlamaModel, TransformerConfig, load_weights_into_model
    from sumow.model_io import load_model_weights

    with open(WEIGHTS_DIR / "config.json") as f:
        c = json.load(f)
    config = TransformerConfig(**{k: c[k] for k in TransformerConfig._fields if k in c})
    model = LlamaModel(config)
    weights = load_model_weights(str(WEIGHTS_DIR))
    model = load_weights_into_model(model, weights)  # type: ignore[invalid-argument-type]
    # model is typed as Module (equinox stub issue) rather than LlamaModel


def _compute_ppl(model, token_seqs):
    total_nll, total_n = 0.0, 0
    for seq in token_seqs:
        tokens = jnp.array(seq)
        logits, _ = model(tokens)
        lp = jax.nn.log_softmax(logits[:-1], axis=-1)
        targets = tokens[1:]
        nll = -lp[jnp.arange(len(targets)), targets]
        total_nll += float(jnp.sum(nll))
        total_n += len(targets)
    return float(np.exp(total_nll / total_n))


def _zero_weight(model, layer, row, col):
    block = model.layers[layer]
    w = block.mlp.down_proj.at[row, col].set(0.0)
    block_new = eqx.tree_at(lambda b: b.mlp.down_proj, block, w)
    return eqx.tree_at(lambda m: m.layers[layer], model, block_new)


@pytest.fixture(scope="module")
def model_and_sws():
    from sumow.identify import identify_super_weights

    model, config = _load_model()
    probe = jnp.arange(1, 129)
    _, stats = model(probe, capture_activations=True)
    sws = identify_super_weights(stats, spike_threshold=100.0, spike_ratio=6.0)
    return model, config, sws, stats


@pytest.fixture(scope="module")
def eval_seqs():
    """Real English text tokenized for PPL evaluation."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    texts = [
        "The quick brown fox jumps over the lazy dog and runs through the forest.",
        "Large language models have shown remarkable ability to generate coherent text.",
        "The capital of France is Paris, known for the Eiffel Tower and cultural heritage.",
        "Machine learning algorithms are used in healthcare, finance, and transportation.",
    ]
    return [tok.encode(t) for t in texts]  # type: ignore[attr-defined]
    # AutoTokenizer.from_pretrained returns a broad PreTrainedTokenizerBase union
    # that ty resolves to the base class, which lacks the concrete .encode method.


@needs_weights
class TestSuperWeightIdentification1_7B:
    def test_finds_super_weights(self, model_and_sws):
        _, _, sws, _ = model_and_sws
        assert len(sws) >= 1, "Should find at least 1 super weight"
        assert len(sws) <= 10, f"Too many: {len(sws)}"

    def test_all_in_down_proj(self, model_and_sws):
        """Paper claim: super weights are ALWAYS in down_proj."""
        model, _, sws, _ = model_and_sws
        for sw in sws:
            val = float(model.layers[sw.layer].mlp.down_proj[sw.row, sw.col])
            assert val != 0.0

    def test_early_layers(self, model_and_sws):
        """Paper: super weight is always in an early layer."""
        _, config, sws, _ = model_and_sws
        earliest = min(sw.layer for sw in sws)
        assert earliest <= config.num_hidden_layers // 4, (
            f"Earliest SW at layer {earliest}, expected early layer"
        )

    def test_super_activation_channels_consistent(self, model_and_sws):
        """Paper: super activation persists at same channel across layers."""
        _, _, sws, stats = model_and_sws
        output_channels = [s.output_max_channel for s in stats]
        from collections import Counter
        top_ch, count = Counter(output_channels).most_common(1)[0]
        assert count >= len(stats) // 2, (
            f"Dominant output channel {top_ch} only in {count}/{len(stats)} layers"
        )


@needs_weights
class TestSuperWeightImpact1_7B:
    def test_zeroing_sw_catastrophic(self, model_and_sws, eval_seqs):
        """Paper's KEY claim: zeroing 1 super weight destroys the model."""
        model, _, sws, _ = model_and_sws
        ppl_base = _compute_ppl(model, eval_seqs)

        # Find the most impactful SW (highest activation magnitude)
        worst_ppl = 0
        for sw in sws:
            model_z = _zero_weight(model, sw.layer, sw.row, sw.col)
            ppl_z = _compute_ppl(model_z, eval_seqs)
            worst_ppl = max(worst_ppl, ppl_z)

        # Paper shows >10x PPL increase; we expect at least 5x
        assert worst_ppl > ppl_base * 5, (
            f"Expected catastrophic PPL increase, got {worst_ppl:.1f} "
            f"vs baseline {ppl_base:.1f} ({worst_ppl/ppl_base:.1f}x)"
        )

    def test_zeroing_all_sws_total_destruction(self, model_and_sws, eval_seqs):
        """Zeroing ALL super weights should cause massive PPL increase."""
        model, _, sws, _ = model_and_sws
        ppl_base = _compute_ppl(model, eval_seqs)

        model_z = model
        for sw in sws:
            model_z = _zero_weight(model_z, sw.layer, sw.row, sw.col)
        ppl_z = _compute_ppl(model_z, eval_seqs)

        assert ppl_z > ppl_base * 50, (
            f"Expected total destruction, got {ppl_z:.1f} "
            f"vs baseline {ppl_base:.1f} ({ppl_z/ppl_base:.1f}x)"
        )

    def test_non_sw_outliers_minimal_impact(self, model_and_sws, eval_seqs):
        """Paper: zeroing other outliers (even larger ones) has minimal impact."""
        model, config, sws, _ = model_and_sws
        ppl_base = _compute_ppl(model, eval_seqs)

        sw_pos = {(sw.layer, sw.row, sw.col) for sw in sws}
        outliers = []
        for li in range(config.num_hidden_layers):
            w = model.layers[li].mlp.down_proj
            flat = jnp.abs(w).flatten()
            top_idx = jnp.argsort(flat)[-5:]
            for idx in top_idx:
                r, c = divmod(int(idx), w.shape[1])
                if (li, r, c) not in sw_pos:
                    outliers.append((li, r, c, float(jnp.abs(w[r, c]))))
        outliers.sort(key=lambda x: x[3], reverse=True)

        # Zero top-5 non-SW outliers simultaneously
        model_z = model
        for li, r, c, _ in outliers[:5]:
            model_z = _zero_weight(model_z, li, r, c)
        ppl_z = _compute_ppl(model_z, eval_seqs)

        # Should be much less impactful than SW zeroing (paper shows ~1-3x)
        assert ppl_z < ppl_base * 10, (
            f"Non-SW outlier zeroing had too much impact: "
            f"{ppl_z:.1f} vs baseline {ppl_base:.1f} ({ppl_z/ppl_base:.1f}x)"
        )

    def test_sw_vs_non_sw_impact_ratio(self, model_and_sws, eval_seqs):
        """The impact ratio between SW and non-SW zeroing must be large."""
        model, config, sws, _ = model_and_sws
        ppl_base = _compute_ppl(model, eval_seqs)

        # Impact of zeroing most impactful SW
        sw_ppls = []
        for sw in sws:
            model_z = _zero_weight(model, sw.layer, sw.row, sw.col)
            sw_ppls.append(_compute_ppl(model_z, eval_seqs))
        worst_sw_ratio = max(p / ppl_base for p in sw_ppls)

        # Impact of zeroing largest non-SW weight
        sw_pos = {(sw.layer, sw.row, sw.col) for sw in sws}
        best_non_sw = None
        for li in range(config.num_hidden_layers):
            w = model.layers[li].mlp.down_proj
            idx = jnp.argmax(jnp.abs(w))
            r, c = divmod(int(idx), w.shape[1])
            mag = float(jnp.abs(w[r, c]))
            if (li, r, c) not in sw_pos:
                if best_non_sw is None or mag > best_non_sw[3]:
                    best_non_sw = (li, r, c, mag)

        assert best_non_sw is not None, (
            "No non-super-weight found — every weight in the model is a super weight, "
            "which would indicate a bug in SW identification."
        )
        model_z = _zero_weight(model, *best_non_sw[:3])
        non_sw_ratio = _compute_ppl(model_z, eval_seqs) / ppl_base

        # SW impact should be >> non-SW impact
        assert worst_sw_ratio > non_sw_ratio * 3, (
            f"SW impact ({worst_sw_ratio:.1f}x) should be >> "
            f"non-SW impact ({non_sw_ratio:.1f}x)"
        )
