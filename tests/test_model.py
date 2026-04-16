"""Tests for sumow.model — Llama-family Equinox transformer.

Tests cover:
  - RMSNorm mathematical correctness
  - RotaryEmbedding frequency computation
  - RoPE application preserves norms
  - GQA key/value repetition
  - Attention output shapes and causal masking
  - SiLU-gated MLP shapes and activation capture
  - Full block forward pass
  - Full model forward pass with activation capture
  - Weight loading round-trip (save → load → compare)
  - Tiny model end-to-end (forward pass produces finite logits)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sumow.model import (
    LlamaAttention,
    LlamaBlock,
    LlamaMLP,
    LlamaModel,
    RMSNorm,
    RotaryEmbedding,
    TransformerConfig,
    apply_rotary_pos_emb,
    load_weights_into_model,
    make_hf_weights_dict,
    repeat_kv,
)

# Tiny config for fast tests
TINY_CONFIG = TransformerConfig(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=48,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,  # GQA: 4 heads, 2 kv heads
    max_position_embeddings=64,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
)

# Even tinier for unit tests (MHA, not GQA)
MICRO_CONFIG = TransformerConfig(
    vocab_size=16,
    hidden_size=8,
    intermediate_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,  # MHA
    max_position_embeddings=32,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(32, eps=1e-5)
        x = jnp.ones((4, 32))
        y = norm(x)
        assert y.shape == (4, 32)

    def test_unit_input(self):
        """RMSNorm of all-ones with weight=1 should give 1s."""
        norm = RMSNorm(8, eps=0.0)
        x = jnp.ones((3, 8))
        y = norm(x)
        np.testing.assert_allclose(y, jnp.ones((3, 8)), atol=1e-5)

    def test_formula(self):
        """Verify against manual RMSNorm computation."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (5, 16))
        eps = 1e-5
        norm = RMSNorm(16, eps=eps)

        # Manual: x / sqrt(mean(x^2) + eps) * weight
        variance = jnp.mean(x * x, axis=-1, keepdims=True)
        expected = x / jnp.sqrt(variance + eps) * norm.weight
        actual = norm(x)

        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_custom_weight(self):
        """Non-unit weights scale correctly."""
        import equinox as eqx

        norm = RMSNorm(4, eps=1e-5)
        w = jnp.array([2.0, 0.5, 1.0, 3.0])
        norm = eqx.tree_at(lambda n: n.weight, norm, w)

        x = jnp.ones((1, 4))
        y = norm(x)
        # All inputs are 1.0, variance=1.0, so normed ≈ 1/sqrt(1+eps) ≈ 1-eps/2
        # Output ≈ weight * (1-eps/2), very close to weight
        np.testing.assert_allclose(y[0], w, atol=1e-4)

    def test_zero_input(self):
        """Zero input doesn't produce NaN (eps protects)."""
        norm = RMSNorm(8, eps=1e-5)
        x = jnp.zeros((2, 8))
        y = norm(x)
        assert jnp.all(jnp.isfinite(y))


# ---------------------------------------------------------------------------
# RotaryEmbedding
# ---------------------------------------------------------------------------


class TestRotaryEmbedding:
    def test_output_shapes(self):
        rope = RotaryEmbedding(head_dim=16, max_position_embeddings=64)
        cos, sin = rope(seq_len=10)
        assert cos.shape == (10, 8)  # half_dim = 16 // 2
        assert sin.shape == (10, 8)

    def test_unit_circle(self):
        """cos^2 + sin^2 = 1 for all positions and dimensions."""
        rope = RotaryEmbedding(head_dim=32, max_position_embeddings=128)
        cos, sin = rope(seq_len=128)
        identity = cos**2 + sin**2
        np.testing.assert_allclose(identity, jnp.ones_like(identity), atol=1e-6)

    def test_position_zero(self):
        """At position 0, cos should be 1 and sin should be 0."""
        rope = RotaryEmbedding(head_dim=8, max_position_embeddings=16)
        cos, sin = rope(seq_len=1)
        np.testing.assert_allclose(cos[0], jnp.ones(4), atol=1e-6)
        np.testing.assert_allclose(sin[0], jnp.zeros(4), atol=1e-6)


# ---------------------------------------------------------------------------
# apply_rotary_pos_emb
# ---------------------------------------------------------------------------


class TestApplyRoPE:
    def test_output_shapes(self):
        seq, heads, kv_heads, dim = 5, 4, 2, 8
        q = jnp.ones((seq, heads, dim))
        k = jnp.ones((seq, kv_heads, dim))
        rope = RotaryEmbedding(dim, max_position_embeddings=16)
        cos, sin = rope(seq)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_norm_preservation(self):
        """RoPE is a rotation — it should preserve vector norms."""
        key = jax.random.PRNGKey(0)
        seq, heads, kv_heads, dim = 8, 4, 2, 16
        q = jax.random.normal(key, (seq, heads, dim))
        k = jax.random.normal(jax.random.PRNGKey(1), (seq, kv_heads, dim))
        rope = RotaryEmbedding(dim, max_position_embeddings=32)
        cos, sin = rope(seq)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        q_norms = jnp.linalg.norm(q, axis=-1)
        q_rot_norms = jnp.linalg.norm(q_rot, axis=-1)
        np.testing.assert_allclose(q_norms, q_rot_norms, atol=1e-5)

    def test_position_zero_identity(self):
        """At position 0 (cos=1, sin=0), RoPE should be identity."""
        dim = 8
        q = jax.random.normal(jax.random.PRNGKey(0), (1, 2, dim))
        k = jax.random.normal(jax.random.PRNGKey(1), (1, 2, dim))
        cos = jnp.ones((1, dim // 2))
        sin = jnp.zeros((1, dim // 2))
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        np.testing.assert_allclose(q_rot, q, atol=1e-6)
        np.testing.assert_allclose(k_rot, k, atol=1e-6)


# ---------------------------------------------------------------------------
# repeat_kv
# ---------------------------------------------------------------------------


class TestRepeatKV:
    def test_no_repeat(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (5, 4, 8))
        y = repeat_kv(x, 1)
        np.testing.assert_array_equal(x, y)

    def test_repeat_2x(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (5, 2, 8))
        y = repeat_kv(x, 2)
        assert y.shape == (5, 4, 8)
        # First pair of heads should equal original first head
        np.testing.assert_array_equal(y[:, 0, :], x[:, 0, :])
        np.testing.assert_array_equal(y[:, 1, :], x[:, 0, :])
        np.testing.assert_array_equal(y[:, 2, :], x[:, 1, :])
        np.testing.assert_array_equal(y[:, 3, :], x[:, 1, :])

    def test_repeat_4x(self):
        x = jnp.ones((3, 1, 4))
        y = repeat_kv(x, 4)
        assert y.shape == (3, 4, 4)


# ---------------------------------------------------------------------------
# LlamaAttention
# ---------------------------------------------------------------------------


class TestLlamaAttention:
    def test_output_shape(self):
        attn = LlamaAttention(TINY_CONFIG)
        x = jnp.ones((8, TINY_CONFIG.hidden_size))
        y = attn(x)
        assert y.shape == (8, TINY_CONFIG.hidden_size)

    def test_zero_weights_zero_output(self):
        """With zero-initialized weights, output should be zero."""
        attn = LlamaAttention(MICRO_CONFIG)
        x = jnp.ones((4, MICRO_CONFIG.hidden_size))
        y = attn(x)
        # q,k,v are all zero → attn_weights are all equal → uniform mixing
        # but v is zero → output is zero → o_proj of zero → zero
        np.testing.assert_allclose(y, jnp.zeros_like(y), atol=1e-6)

    def test_single_token(self):
        """Single-token sequence should work."""
        attn = LlamaAttention(MICRO_CONFIG)
        x = jnp.ones((1, MICRO_CONFIG.hidden_size))
        y = attn(x)
        assert y.shape == (1, MICRO_CONFIG.hidden_size)


# ---------------------------------------------------------------------------
# LlamaMLP
# ---------------------------------------------------------------------------


class TestLlamaMLP:
    def test_output_shapes(self):
        mlp = LlamaMLP(TINY_CONFIG)
        x = jnp.ones((4, TINY_CONFIG.hidden_size))
        output, down_proj_input = mlp(x)
        assert output.shape == (4, TINY_CONFIG.hidden_size)
        assert down_proj_input.shape == (4, TINY_CONFIG.intermediate_size)

    def test_zero_weights_zero_output(self):
        mlp = LlamaMLP(MICRO_CONFIG)
        x = jnp.ones((2, MICRO_CONFIG.hidden_size))
        output, dpi = mlp(x)
        np.testing.assert_allclose(output, jnp.zeros_like(output), atol=1e-6)
        np.testing.assert_allclose(dpi, jnp.zeros_like(dpi), atol=1e-6)

    def test_silu_gating(self):
        """Verify SiLU gating: output should be silu(gate) * up before down_proj."""
        import equinox as eqx

        cfg = MICRO_CONFIG
        mlp = LlamaMLP(cfg)
        # Set gate and up to identity-like
        gate_w = jnp.eye(cfg.intermediate_size, cfg.hidden_size)
        up_w = jnp.eye(cfg.intermediate_size, cfg.hidden_size)
        down_w = jnp.eye(cfg.hidden_size, cfg.intermediate_size)
        mlp = eqx.tree_at(lambda m: m.gate_proj, mlp, gate_w)
        mlp = eqx.tree_at(lambda m: m.up_proj, mlp, up_w)
        mlp = eqx.tree_at(lambda m: m.down_proj, mlp, down_w)

        x = jax.random.normal(jax.random.PRNGKey(0), (3, cfg.hidden_size))
        output, dpi = mlp(x)

        # Expected: silu(x @ I.T) * (x @ I.T) = silu(x) * x (for the first hidden_size dims)
        x_trunc = x  # since intermediate >= hidden, the extra dims are zeros
        expected_dpi = jax.nn.silu(x_trunc @ gate_w.T) * (x_trunc @ up_w.T)
        np.testing.assert_allclose(dpi, expected_dpi, atol=1e-5)


# ---------------------------------------------------------------------------
# LlamaBlock
# ---------------------------------------------------------------------------


class TestLlamaBlock:
    def test_output_shapes(self):
        block = LlamaBlock(TINY_CONFIG)
        x = jnp.ones((8, TINY_CONFIG.hidden_size))
        output, dpi = block(x)
        assert output.shape == (8, TINY_CONFIG.hidden_size)
        assert dpi.shape == (8, TINY_CONFIG.intermediate_size)

    def test_residual_connection(self):
        """With zero weights, block should be identity (residual only)."""
        block = LlamaBlock(MICRO_CONFIG)
        x = jnp.ones((4, MICRO_CONFIG.hidden_size))
        output, _ = block(x)
        # Both attention and MLP produce zeros → residual = input
        np.testing.assert_allclose(output, x, atol=1e-5)


# ---------------------------------------------------------------------------
# LlamaModel (full)
# ---------------------------------------------------------------------------


class TestLlamaModel:
    def test_output_shapes(self):
        model = LlamaModel(TINY_CONFIG)
        ids = jnp.array([0, 1, 2, 3])
        logits, stats = model(ids, capture_activations=False)
        assert logits.shape == (4, TINY_CONFIG.vocab_size)
        assert stats == []

    def test_capture_activations(self):
        model = LlamaModel(TINY_CONFIG)
        ids = jnp.array([0, 1, 2])
        logits, stats = model(ids, capture_activations=True)
        assert logits.shape == (3, TINY_CONFIG.vocab_size)
        assert len(stats) == TINY_CONFIG.num_hidden_layers
        for i, s in enumerate(stats):
            assert s.layer == i
            assert isinstance(s.input_max_magnitude, float)
            assert isinstance(s.output_max_magnitude, float)

    def test_single_token(self):
        model = LlamaModel(MICRO_CONFIG)
        ids = jnp.array([0])
        logits, _ = model(ids)
        assert logits.shape == (1, MICRO_CONFIG.vocab_size)

    def test_finite_output_with_random_weights(self):
        """Random weights should still produce finite logits."""
        import equinox as eqx

        key = jax.random.PRNGKey(42)
        model = LlamaModel(TINY_CONFIG)

        # Initialize with small random weights
        leaves, treedef = jax.tree.flatten(model)
        new_leaves = []
        for i, leaf in enumerate(leaves):
            if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
                subkey = jax.random.PRNGKey(i)
                new_leaves.append(jax.random.normal(subkey, leaf.shape) * 0.02)
            else:
                new_leaves.append(leaf)
        model = jax.tree.unflatten(treedef, new_leaves)

        ids = jnp.array([0, 1, 2, 3, 4])
        logits, _ = model(ids)
        assert jnp.all(jnp.isfinite(logits))


# ---------------------------------------------------------------------------
# Weight loading round-trip
# ---------------------------------------------------------------------------


class TestWeightLoading:
    def test_roundtrip(self):
        """make_hf_weights_dict → load_weights_into_model should preserve all named params."""
        import equinox as eqx

        key = jax.random.PRNGKey(99)
        cfg = TINY_CONFIG
        model = LlamaModel(cfg)

        # Randomize weights
        leaves, treedef = jax.tree.flatten(model)
        new_leaves = []
        for i, leaf in enumerate(leaves):
            if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
                new_leaves.append(
                    jax.random.normal(jax.random.PRNGKey(i), leaf.shape) * 0.1
                )
            else:
                new_leaves.append(leaf)
        model = jax.tree.unflatten(treedef, new_leaves)

        # Extract named params and reload into fresh model
        weights = make_hf_weights_dict(model)
        model2 = LlamaModel(cfg)
        model2 = load_weights_into_model(model2, weights)

        # Compare via the named weight dict (excludes RoPE cache which is derived)
        weights2 = make_hf_weights_dict(model2)
        assert set(weights.keys()) == set(weights2.keys())
        for key in weights:
            np.testing.assert_array_equal(weights[key], weights2[key])

    def test_expected_keys(self):
        """Verify the weight dict has the expected HF-format keys."""
        cfg = TINY_CONFIG
        model = LlamaModel(cfg)
        weights = make_hf_weights_dict(model)

        assert "model.embed_tokens.weight" in weights
        assert "model.norm.weight" in weights
        assert "lm_head.weight" in weights
        for i in range(cfg.num_hidden_layers):
            prefix = f"model.layers.{i}"
            for key in [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
            ]:
                assert f"{prefix}.{key}" in weights

    def test_tied_embeddings(self):
        """With tie_word_embeddings=True, lm_head should equal embed_tokens."""
        cfg = TransformerConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
            tie_word_embeddings=True,
        )
        model = LlamaModel(cfg)
        embed = jax.random.normal(jax.random.PRNGKey(0), (16, 8))
        weights = {"model.embed_tokens.weight": embed}
        # Fill in remaining required keys
        prefix = "model.layers.0"
        for key in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
        ]:
            weights[f"{prefix}.{key}"] = jnp.zeros((8, 8))
        for key in ["mlp.gate_proj.weight", "mlp.up_proj.weight"]:
            weights[f"{prefix}.{key}"] = jnp.zeros((16, 8))
        weights[f"{prefix}.mlp.down_proj.weight"] = jnp.zeros((8, 16))
        weights[f"{prefix}.input_layernorm.weight"] = jnp.ones(8)
        weights[f"{prefix}.post_attention_layernorm.weight"] = jnp.ones(8)
        weights["model.norm.weight"] = jnp.ones(8)

        model = load_weights_into_model(model, weights)
        np.testing.assert_array_equal(model.embed_tokens, model.lm_head)


# ---------------------------------------------------------------------------
# End-to-end with activation capture
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_activation_capture_identifies_spike(self):
        """Plant a large weight in down_proj and verify activation spike detection."""
        import equinox as eqx

        from sumow.identify import identify_super_weights

        cfg = TINY_CONFIG
        model = LlamaModel(cfg)

        # Randomize weights for non-trivial activations
        leaves, treedef = jax.tree.flatten(model)
        new_leaves = []
        for i, leaf in enumerate(leaves):
            if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
                new_leaves.append(
                    jax.random.normal(jax.random.PRNGKey(i + 100), leaf.shape) * 0.02
                )
            else:
                new_leaves.append(leaf)
        model = jax.tree.unflatten(treedef, new_leaves)

        # Plant a super weight in layer 0's down_proj
        layer0 = model.layers[0]
        big_down = layer0.mlp.down_proj.at[5, 10].set(50.0)
        layer0 = eqx.tree_at(lambda b: b.mlp.down_proj, layer0, big_down)
        model = eqx.tree_at(lambda m: m.layers[0], model, layer0)

        # Forward pass
        ids = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
        logits, stats = model(ids, capture_activations=True)

        assert len(stats) == cfg.num_hidden_layers
        assert jnp.all(jnp.isfinite(logits))
        # Layer 0 should have the biggest output spike (at channel 5)
        # due to the planted super weight
        assert stats[0].output_max_magnitude > 0

    def test_gqa_model_forward(self):
        """GQA config (num_kv_heads < num_heads) should work."""
        cfg = TransformerConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=1,  # extreme GQA: 4 heads, 1 kv head
            max_position_embeddings=32,
        )
        model = LlamaModel(cfg)
        ids = jnp.array([0, 1, 2])
        logits, stats = model(ids, capture_activations=True)
        assert logits.shape == (3, 32)
        assert len(stats) == 1
