"""Tests for evaluation utilities."""

import jax.numpy as jnp
import math

from sumow.eval import cross_entropy_loss, perplexity, perplexity_from_loss


def test_perplexity_from_loss_zero():
    assert perplexity_from_loss(0.0) == 1.0


def test_perplexity_from_loss_positive():
    ppl = perplexity_from_loss(1.0)
    assert abs(ppl - math.e) < 0.01


def test_cross_entropy_perfect_prediction():
    """When logits strongly predict the correct token, loss is near zero."""
    # 3 positions, vocab size 4
    logits = jnp.array([
        [-100.0, -100.0, 100.0, -100.0],
        [-100.0, 100.0, -100.0, -100.0],
        [100.0, -100.0, -100.0, -100.0],
    ])
    targets = jnp.array([2, 1, 0])
    loss = cross_entropy_loss(logits, targets)
    assert float(loss) < 0.01


def test_cross_entropy_uniform_prediction():
    """Uniform logits → loss = log(vocab_size)."""
    vocab_size = 8
    logits = jnp.zeros((4, vocab_size))
    targets = jnp.array([0, 1, 2, 3])
    loss = cross_entropy_loss(logits, targets)
    expected = math.log(vocab_size)
    assert abs(float(loss) - expected) < 0.01


def test_perplexity_end_to_end():
    """Perplexity of uniform distribution = vocab_size."""
    vocab_size = 10
    logits = jnp.zeros((5, vocab_size))
    targets = jnp.arange(5)
    ppl = perplexity(logits, targets)
    assert abs(ppl - vocab_size) < 0.5
