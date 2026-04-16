"""Llama-family transformer model in pure JAX/Equinox.

Supports Llama, Llama-2, Llama-3, Mistral, OLMo, and Phi-3 architectures.
All are Llama-style with minor differences in config parameters.

Key design choices:
  - Pure eqx.Module pytree — no mutable state, no side effects
  - Forward pass optionally captures down_proj activations for SW identification
  - Weight loading from HuggingFace safetensors flat dict
  - RoPE precomputed as static array (not a parameter)
  - GQA (grouped query attention) via key/value head repetition
  - Causal attention via additive mask (not triangle matrix multiply)
"""

from __future__ import annotations

from dataclasses import field
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Int, jaxtyped

from sumow.identify import LayerActivationStats, compute_layer_stats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TransformerConfig(NamedTuple):
    """Architecture parameters for Llama-family transformers."""

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32  # = num_attention_heads for MHA, < for GQA
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False


# Some common configs
LLAMA_7B_CONFIG = TransformerConfig(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    max_position_embeddings=2048,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
)

LLAMA3_8B_CONFIG = TransformerConfig(
    vocab_size=128256,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=8192,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
)

MISTRAL_7B_CONFIG = TransformerConfig(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=32768,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class RMSNorm(eqx.Module):
    """Root Mean Square Layer Normalization."""

    weight: Float[Array, "dim"]
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = jnp.ones(dim)
        self.eps = eps

    @jaxtyped(typechecker=beartype)
    def __call__(self, x: Float[Array, "seq dim"]) -> Float[Array, "seq dim"]:
        # RMSNorm: x * w / sqrt(mean(x^2) + eps)
        variance = jnp.mean(x * x, axis=-1, keepdims=True)
        x_normed = x * jax.lax.rsqrt(variance + self.eps)
        return x_normed * self.weight


class RotaryEmbedding(eqx.Module):
    """Precomputed rotary position embeddings (RoPE).

    Stores cos/sin tables as static arrays — not trainable.
    """

    cos_cached: Float[Array, "max_seq half_dim"]
    sin_cached: Float[Array, "max_seq half_dim"]

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
    ):
        half_dim = head_dim // 2
        inv_freq = 1.0 / (
            rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
        )
        positions = jnp.arange(max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.outer(positions, inv_freq)
        self.cos_cached = jnp.cos(freqs)
        self.sin_cached = jnp.sin(freqs)

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, seq_len: int
    ) -> tuple[Float[Array, "seq half_dim"], Float[Array, "seq half_dim"]]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


@jaxtyped(typechecker=beartype)
def apply_rotary_pos_emb(
    q: Float[Array, "seq heads dim"],
    k: Float[Array, "seq kv_heads dim"],
    cos: Float[Array, "seq half_dim"],
    sin: Float[Array, "seq half_dim"],
) -> tuple[Float[Array, "seq heads dim"], Float[Array, "seq kv_heads dim"]]:
    """Apply rotary position embeddings to query and key tensors.

    Splits last dim in half, applies rotation, concatenates back.
    """
    half = q.shape[-1] // 2

    def _rotate(x: Array, cos_: Array, sin_: Array) -> Array:
        x1, x2 = x[..., :half], x[..., half:]
        # Expand cos/sin for broadcasting: [seq, 1, half_dim]
        c = cos_[:, None, :]
        s = sin_[:, None, :]
        return jnp.concatenate(
            [x1 * c - x2 * s, x2 * c + x1 * s],
            axis=-1,
        )

    q_rot = _rotate(q, cos, sin)
    k_rot = _rotate(k, cos, sin)
    return q_rot, k_rot


@jaxtyped(typechecker=beartype)
def repeat_kv(
    x: Float[Array, "seq kv_heads dim"], n_rep: int
) -> Float[Array, "seq heads dim"]:
    """Repeat key/value heads for grouped query attention.

    If n_rep == 1, returns x unchanged.
    """
    if n_rep == 1:
        return x
    seq, kv_heads, dim = x.shape
    # [seq, kv_heads, 1, dim] -> [seq, kv_heads, n_rep, dim] -> [seq, kv_heads*n_rep, dim]
    x = jnp.repeat(x[:, :, None, :], n_rep, axis=2)
    return x.reshape(seq, kv_heads * n_rep, dim)


class LlamaAttention(eqx.Module):
    """Multi-head attention with GQA support and RoPE."""

    q_proj: Float[Array, "num_heads*head_dim hidden"]
    k_proj: Float[Array, "num_kv_heads*head_dim hidden"]
    v_proj: Float[Array, "num_kv_heads*head_dim hidden"]
    o_proj: Float[Array, "hidden num_heads*head_dim"]
    num_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    rotary_emb: RotaryEmbedding

    def __init__(self, config: TransformerConfig):
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = hidden // self.num_heads

        # Initialize with small random values (overwritten on load)
        self.q_proj = jnp.zeros((self.num_heads * self.head_dim, hidden))
        self.k_proj = jnp.zeros((self.num_kv_heads * self.head_dim, hidden))
        self.v_proj = jnp.zeros((self.num_kv_heads * self.head_dim, hidden))
        self.o_proj = jnp.zeros((hidden, self.num_heads * self.head_dim))
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, x: Float[Array, "seq hidden"]
    ) -> Float[Array, "seq hidden"]:
        seq_len, _ = x.shape

        # Project: [seq, hidden] @ [out, hidden].T -> [seq, out]
        q = x @ self.q_proj.T  # [seq, num_heads * head_dim]
        k = x @ self.k_proj.T  # [seq, num_kv_heads * head_dim]
        v = x @ self.v_proj.T  # [seq, num_kv_heads * head_dim]

        # Reshape to [seq, heads, head_dim]
        q = q.reshape(seq_len, self.num_heads, self.head_dim)
        k = k.reshape(seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(seq_len, self.num_kv_heads, self.head_dim)

        # RoPE
        cos, sin = self.rotary_emb(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: repeat k, v to match num_heads
        n_rep = self.num_heads // self.num_kv_heads
        k = repeat_kv(k, n_rep)  # [seq, num_heads, head_dim]
        v = repeat_kv(v, n_rep)  # [seq, num_heads, head_dim]

        # Attention: [heads, seq, seq]
        # Transpose to [heads, seq, dim] for matmul
        q_t = jnp.transpose(q, (1, 0, 2))  # [heads, seq, dim]
        k_t = jnp.transpose(k, (1, 0, 2))  # [heads, seq, dim]
        v_t = jnp.transpose(v, (1, 0, 2))  # [heads, seq, dim]

        scale = 1.0 / jnp.sqrt(jnp.float32(self.head_dim))
        attn_weights = jnp.matmul(q_t, jnp.transpose(k_t, (0, 2, 1))) * scale

        # Causal mask
        causal_mask = jnp.triu(
            jnp.full((seq_len, seq_len), float("-inf")), k=1
        )
        attn_weights = attn_weights + causal_mask[None, :, :]

        # Softmax in float32 for numerical stability
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1)
        attn_weights = attn_weights.astype(q.dtype)

        # Weighted sum: [heads, seq, dim]
        attn_out = jnp.matmul(attn_weights, v_t)

        # [heads, seq, dim] -> [seq, heads, dim] -> [seq, heads*dim]
        attn_out = jnp.transpose(attn_out, (1, 0, 2))
        attn_out = attn_out.reshape(seq_len, -1)

        # Output projection
        return attn_out @ self.o_proj.T


class LlamaMLP(eqx.Module):
    """SiLU-gated MLP (gate_proj, up_proj, down_proj)."""

    gate_proj: Float[Array, "intermediate hidden"]
    up_proj: Float[Array, "intermediate hidden"]
    down_proj: Float[Array, "hidden intermediate"]

    def __init__(self, config: TransformerConfig):
        hidden = config.hidden_size
        inter = config.intermediate_size
        self.gate_proj = jnp.zeros((inter, hidden))
        self.up_proj = jnp.zeros((inter, hidden))
        self.down_proj = jnp.zeros((hidden, inter))

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, x: Float[Array, "seq hidden"]
    ) -> tuple[Float[Array, "seq hidden"], Float[Array, "seq intermediate"]]:
        """Forward pass returning (output, down_proj_input).

        The down_proj_input is returned for super weight identification.
        """
        gate = jax.nn.silu(x @ self.gate_proj.T)  # [seq, inter]
        up = x @ self.up_proj.T  # [seq, inter]
        down_proj_input = gate * up  # [seq, inter]
        output = down_proj_input @ self.down_proj.T  # [seq, hidden]
        return output, down_proj_input


class LlamaBlock(eqx.Module):
    """Single transformer block: attention + MLP with residual connections."""

    self_attn: LlamaAttention
    mlp: LlamaMLP
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm

    def __init__(self, config: TransformerConfig):
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, x: Float[Array, "seq hidden"]
    ) -> tuple[Float[Array, "seq hidden"], Float[Array, "seq intermediate"]]:
        """Forward pass returning (output, down_proj_input for SW detection)."""
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x

        # MLP with residual
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_out, down_proj_input = self.mlp(x)
        x = residual + mlp_out

        return x, down_proj_input


class LlamaModel(eqx.Module):
    """Full Llama-family causal language model.

    Embedding + N transformer blocks + RMSNorm + lm_head.
    """

    embed_tokens: Float[Array, "vocab hidden"]
    layers: list[LlamaBlock]
    norm: RMSNorm
    lm_head: Float[Array, "vocab hidden"]
    config: TransformerConfig = eqx.field(static=True)

    def __init__(self, config: TransformerConfig):
        self.config = config
        self.embed_tokens = jnp.zeros((config.vocab_size, config.hidden_size))
        self.layers = [LlamaBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = jnp.zeros((config.vocab_size, config.hidden_size))

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        input_ids: Int[Array, "seq"],
        capture_activations: bool = False,
    ) -> tuple[
        Float[Array, "seq vocab"],
        list[LayerActivationStats],
    ]:
        """Forward pass with optional activation capture.

        Args:
            input_ids: Token IDs [seq_len].
            capture_activations: If True, record down_proj activation
                statistics for super weight identification.

        Returns:
            Tuple of (logits, activation_stats). activation_stats is empty
            list if capture_activations is False.
        """
        # Embedding lookup
        x = self.embed_tokens[input_ids]  # [seq, hidden]

        activation_stats: list[LayerActivationStats] = []

        for layer_idx, block in enumerate(self.layers):
            x, down_proj_input = block(x)

            if capture_activations:
                # down_proj_input: [seq, intermediate]
                # down_proj output = x contribution from MLP (we need the pre-residual output)
                # Recompute down_proj output for stats (the block already applied residual)
                down_proj_output = down_proj_input @ block.mlp.down_proj.T
                stats = compute_layer_stats(
                    down_proj_input, down_proj_output, layer_idx
                )
                activation_stats.append(stats)

        # Final norm + language model head
        x = self.norm(x)
        logits = x @ self.lm_head.T  # [seq, vocab]

        return logits, activation_stats


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def load_weights_into_model(
    model: LlamaModel,
    weights: dict[str, Array],
) -> LlamaModel:
    """Load HuggingFace weights dict into a LlamaModel.

    Expects keys like:
        model.embed_tokens.weight
        model.layers.0.self_attn.q_proj.weight
        model.layers.0.self_attn.k_proj.weight
        model.layers.0.self_attn.v_proj.weight
        model.layers.0.self_attn.o_proj.weight
        model.layers.0.mlp.gate_proj.weight
        model.layers.0.mlp.up_proj.weight
        model.layers.0.mlp.down_proj.weight
        model.layers.0.input_layernorm.weight
        model.layers.0.post_attention_layernorm.weight
        model.norm.weight
        lm_head.weight

    Args:
        model: A LlamaModel (typically zero-initialized).
        weights: Flat dict of parameter name -> JAX array.

    Returns:
        New LlamaModel with loaded weights (pytree replace, no mutation).
    """

    def _get(name: str) -> Array:
        return weights[name]

    def _maybe_get(name: str) -> Array | None:
        return weights.get(name)

    # Embed tokens
    model = eqx.tree_at(lambda m: m.embed_tokens, model, _get("model.embed_tokens.weight"))

    # Layers
    for i in range(len(model.layers)):
        prefix = f"model.layers.{i}"
        block = model.layers[i]

        # Attention
        block = eqx.tree_at(lambda b: b.self_attn.q_proj, block, _get(f"{prefix}.self_attn.q_proj.weight"))
        block = eqx.tree_at(lambda b: b.self_attn.k_proj, block, _get(f"{prefix}.self_attn.k_proj.weight"))
        block = eqx.tree_at(lambda b: b.self_attn.v_proj, block, _get(f"{prefix}.self_attn.v_proj.weight"))
        block = eqx.tree_at(lambda b: b.self_attn.o_proj, block, _get(f"{prefix}.self_attn.o_proj.weight"))

        # MLP
        block = eqx.tree_at(lambda b: b.mlp.gate_proj, block, _get(f"{prefix}.mlp.gate_proj.weight"))
        block = eqx.tree_at(lambda b: b.mlp.up_proj, block, _get(f"{prefix}.mlp.up_proj.weight"))
        block = eqx.tree_at(lambda b: b.mlp.down_proj, block, _get(f"{prefix}.mlp.down_proj.weight"))

        # Norms
        block = eqx.tree_at(lambda b: b.input_layernorm.weight, block, _get(f"{prefix}.input_layernorm.weight"))
        block = eqx.tree_at(lambda b: b.post_attention_layernorm.weight, block, _get(f"{prefix}.post_attention_layernorm.weight"))

        model = eqx.tree_at(lambda m: m.layers[i], model, block)

    # Final norm
    model = eqx.tree_at(lambda m: m.norm.weight, model, _get("model.norm.weight"))

    # LM head
    lm_head_w = _maybe_get("lm_head.weight")
    if lm_head_w is not None:
        model = eqx.tree_at(lambda m: m.lm_head, model, lm_head_w)
    elif model.config.tie_word_embeddings:
        # Tied embeddings: lm_head = embed_tokens
        model = eqx.tree_at(lambda m: m.lm_head, model, model.embed_tokens)

    return model


@jaxtyped(typechecker=beartype)
def make_hf_weights_dict(model: LlamaModel) -> dict[str, Array]:
    """Extract weights from a LlamaModel into HF-compatible flat dict.

    Inverse of load_weights_into_model. Useful for serialization.
    """
    weights: dict[str, Array] = {}

    weights["model.embed_tokens.weight"] = model.embed_tokens

    for i, block in enumerate(model.layers):
        prefix = f"model.layers.{i}"
        weights[f"{prefix}.self_attn.q_proj.weight"] = block.self_attn.q_proj
        weights[f"{prefix}.self_attn.k_proj.weight"] = block.self_attn.k_proj
        weights[f"{prefix}.self_attn.v_proj.weight"] = block.self_attn.v_proj
        weights[f"{prefix}.self_attn.o_proj.weight"] = block.self_attn.o_proj
        weights[f"{prefix}.mlp.gate_proj.weight"] = block.mlp.gate_proj
        weights[f"{prefix}.mlp.up_proj.weight"] = block.mlp.up_proj
        weights[f"{prefix}.mlp.down_proj.weight"] = block.mlp.down_proj
        weights[f"{prefix}.input_layernorm.weight"] = block.input_layernorm.weight
        weights[f"{prefix}.post_attention_layernorm.weight"] = block.post_attention_layernorm.weight

    weights["model.norm.weight"] = model.norm.weight
    weights["lm_head.weight"] = model.lm_head

    return weights
