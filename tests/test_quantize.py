"""Tests for quantization routines."""

import jax.numpy as jnp
import pytest

from sumow.quantize import (
    clip_percentage,
    clip_zscore,
    dequantize_rtn,
    quantize_activation_sa_aware,
    quantize_dequantize,
    quantize_dequantize_blockwise,
    quantize_rtn,
    quantize_weight_sw_aware,
)


# ---------------------------------------------------------------------------
# RTN quantize / dequantize
# ---------------------------------------------------------------------------


class TestRTN:
    def test_round_trip_fidelity(self):
        """Dequantized values should be close to originals for high bit-width."""
        x = jnp.linspace(-1.0, 1.0, 256)
        q, delta, x_min = quantize_rtn(x, nbits=8)
        x_hat = dequantize_rtn(q, delta, x_min)
        assert jnp.allclose(x, x_hat, atol=0.01)

    def test_quantized_range(self):
        """Quantized values should be in [0, 2^nbits - 1]."""
        x = jnp.array([-10.0, 0.0, 5.0, 100.0])
        for nbits in [2, 4, 8]:
            q, _, _ = quantize_rtn(x, nbits)
            assert jnp.all(q >= 0)
            assert jnp.all(q <= (1 << nbits) - 1)

    def test_constant_tensor(self):
        """Constant tensors should not crash (delta=0 edge case)."""
        x = jnp.ones(16) * 3.14
        q, delta, x_min = quantize_rtn(x, nbits=4)
        x_hat = dequantize_rtn(q, delta, x_min)
        assert jnp.allclose(x_hat, x, atol=1e-5)

    def test_single_element(self):
        """Single-element tensor."""
        x = jnp.array([42.0])
        x_hat = quantize_dequantize(x, nbits=4)
        assert jnp.allclose(x_hat, x, atol=1e-5)


# ---------------------------------------------------------------------------
# Blockwise quantization
# ---------------------------------------------------------------------------


class TestBlockwise:
    def test_shape_preserved(self):
        w = jnp.ones((64, 128))
        w_hat = quantize_dequantize_blockwise(w, nbits=4, blocksize=128)
        assert w_hat.shape == w.shape

    def test_small_blocksize(self):
        w = jnp.linspace(-1.0, 1.0, 256).reshape(16, 16)
        w_hat = quantize_dequantize_blockwise(w, nbits=8, blocksize=16)
        assert jnp.allclose(w, w_hat, atol=0.01)

    def test_indivisible_blocksize_fallback(self):
        """When numel % blocksize != 0, should fall back to per-tensor."""
        w = jnp.ones((10, 10))  # 100 elements, not divisible by 128
        w_hat = quantize_dequantize_blockwise(w, nbits=4, blocksize=128)
        assert w_hat.shape == w.shape

    def test_low_bits_high_error(self):
        """2-bit quantization should have more error than 8-bit."""
        key = jnp.array([0, 1], dtype=jnp.uint32)  # deterministic
        w = jnp.linspace(-5.0, 5.0, 256).reshape(16, 16)
        err_2bit = jnp.mean(
            jnp.abs(w - quantize_dequantize_blockwise(w, nbits=2, blocksize=16))
        )
        err_8bit = jnp.mean(
            jnp.abs(w - quantize_dequantize_blockwise(w, nbits=8, blocksize=16))
        )
        assert err_2bit > err_8bit


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------


class TestClipping:
    def test_zscore_clips_outliers(self):
        w = jnp.zeros(1000).at[0].set(1000.0)
        clipped = clip_zscore(w, z_threshold=3.0)
        assert float(jnp.max(jnp.abs(clipped))) < 1000.0

    def test_zscore_no_change_inliers(self):
        w = jnp.ones(100)
        clipped = clip_zscore(w, z_threshold=3.0)
        assert jnp.allclose(w, clipped)

    def test_percentage_clips_top_k(self):
        w = jnp.arange(1000, dtype=jnp.float32)
        # Put an extreme outlier so clipping is effective
        w = w.at[500].set(5000.0)
        clipped = clip_percentage(w, percentage=0.001)
        assert float(jnp.max(jnp.abs(clipped))) < 5000.0


# ---------------------------------------------------------------------------
# SW-aware weight quantization
# ---------------------------------------------------------------------------


class TestSWAwareWeight:
    def test_super_weight_restored(self):
        """Super weight should be restored exactly after quantization."""
        w = jnp.linspace(-1.0, 1.0, 256).reshape(16, 16)
        # Put a large outlier at (3, 7)
        w = w.at[3, 7].set(50.0)
        sw_coords = [(3, 7)]
        w_hat = quantize_weight_sw_aware(
            w, sw_coords, nbits=4, blocksize=16, z_threshold=3.0
        )
        assert float(w_hat[3, 7]) == pytest.approx(50.0)

    def test_non_sw_quantized(self):
        """Non-super-weight values should be quantized (not identical to original)."""
        # Use random-ish data so quantization has visible effect
        w = jnp.sin(jnp.arange(256, dtype=jnp.float32) * 0.37).reshape(16, 16) * 5.0
        # Add a large outlier as the super weight
        w = w.at[0, 0].set(50.0)
        sw_coords = [(0, 0)]
        w_hat = quantize_weight_sw_aware(
            w, sw_coords, nbits=4, blocksize=16, z_threshold=3.0
        )
        # Super weight should be restored
        assert float(w_hat[0, 0]) == pytest.approx(50.0)
        # Mean error on other values should be nonzero (quantized)
        mask = jnp.ones_like(w, dtype=bool).at[0, 0].set(False)
        mean_err = jnp.mean(jnp.abs(w[mask] - w_hat[mask]))
        assert float(mean_err) > 1e-4

    def test_multiple_super_weights(self):
        """Multiple super weights should all be restored."""
        w = jnp.ones((16, 16))
        w = w.at[1, 2].set(99.0)
        w = w.at[5, 8].set(-77.0)
        sw_coords = [(1, 2), (5, 8)]
        w_hat = quantize_weight_sw_aware(w, sw_coords, nbits=4, blocksize=16)
        assert float(w_hat[1, 2]) == pytest.approx(99.0)
        assert float(w_hat[5, 8]) == pytest.approx(-77.0)

    def test_empty_sw_coords(self):
        """With no super weights, should just be normal quantization."""
        w = jnp.linspace(-1.0, 1.0, 64).reshape(8, 8)
        w_hat = quantize_weight_sw_aware(w, [], nbits=4, blocksize=8)
        assert w_hat.shape == w.shape


# ---------------------------------------------------------------------------
# SA-aware activation quantization
# ---------------------------------------------------------------------------


class TestSAAwareActivation:
    def test_super_activation_restored(self):
        """Super activation value should be preserved exactly."""
        a = jnp.ones((1, 4, 8))
        a = a.at[0, 2, 5].set(500.0)
        sa_positions = [(2, 5)]
        a_hat = quantize_activation_sa_aware(a, sa_positions, nbits=8)
        assert jnp.allclose(a_hat[0, 2, 5], jnp.array(500.0))

    def test_shape_preserved(self):
        a = jnp.ones((2, 4, 8))
        a_hat = quantize_activation_sa_aware(a, [], nbits=8)
        assert a_hat.shape == a.shape

    def test_without_sa_positions(self):
        """No SA positions = normal quantization."""
        a = jnp.linspace(0.0, 1.0, 32).reshape(1, 4, 8)
        a_hat = quantize_activation_sa_aware(a, [], nbits=8)
        assert jnp.allclose(a, a_hat, atol=0.01)
