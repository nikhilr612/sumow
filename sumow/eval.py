"""Perplexity and evaluation utilities.

Provides lightweight perplexity computation on token sequences using
JAX arrays. Full benchmark evaluation requires a model forward pass;
these utilities handle the numerical parts.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def cross_entropy_loss(
    logits: Float[Array, "seq vocab"],
    targets: Int[Array, "seq"],
) -> Float[Array, ""]:
    """Compute mean cross-entropy loss over a sequence.

    Args:
        logits: Model output logits [seq_len, vocab_size].
        targets: Target token IDs [seq_len].

    Returns:
        Scalar mean cross-entropy loss.
    """
    # Numerically stable log-softmax: log(softmax(x)) = x - log(sum(exp(x)))
    log_probs = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    target_log_probs = log_probs[jnp.arange(targets.shape[0]), targets]
    return -jnp.mean(target_log_probs)


def perplexity_from_loss(loss: float) -> float:
    """Convert cross-entropy loss to perplexity.

    PPL = exp(loss)
    """
    return float(jnp.exp(loss))


def perplexity(
    logits: Float[Array, "seq vocab"],
    targets: Int[Array, "seq"],
) -> float:
    """Compute perplexity from logits and targets.

    Args:
        logits: Model output logits [seq_len, vocab_size].
        targets: Target token IDs [seq_len].

    Returns:
        Perplexity (scalar).
    """
    loss = cross_entropy_loss(logits, targets)
    return perplexity_from_loss(float(loss))
