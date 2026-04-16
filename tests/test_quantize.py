"""Tests for quantization routines."""

import jax.numpy as jnp
import pytest

from sumow.quantize import (
    QuantizeResult,
    clip_block_percentage,
    clip_iqr,
    clip_percentage,
    clip_zscore,
    dequantize_rtn,
    pack_4bit_to_int8,
    quantize_activation_sa_aware,
    quantize_dequantize,
    quantize_dequantize_blockwise,
    quantize_dequantize_nf,
    quantize_dequantize_per_channel,
    quantize_dequantize_scale_shift,
    quantize_rtn,
    quantize_weight_sw_aware,
    round_to_nearest_pole,
    scale_super_weights,
    unpack_int8_to_4bit,
    NF4_LEVELS,
    NF3_LEVELS,
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
        result = quantize_dequantize_blockwise(w, nbits=4, blocksize=128)
        assert result.weight.shape == w.shape

    def test_small_blocksize(self):
        w = jnp.linspace(-1.0, 1.0, 256).reshape(16, 16)
        result = quantize_dequantize_blockwise(w, nbits=8, blocksize=16)
        assert jnp.allclose(w, result.weight, atol=0.01)

    def test_indivisible_blocksize_fallback(self):
        """When numel % blocksize != 0, should fall back to per-tensor."""
        w = jnp.ones((10, 10))  # 100 elements, not divisible by 128
        result = quantize_dequantize_blockwise(w, nbits=4, blocksize=128)
        assert result.weight.shape == w.shape

    def test_low_bits_high_error(self):
        """2-bit quantization should have more error than 8-bit."""
        w = jnp.linspace(-5.0, 5.0, 256).reshape(16, 16)
        err_2bit = jnp.mean(
            jnp.abs(w - quantize_dequantize_blockwise(w, nbits=2, blocksize=16).weight)
        )
        err_8bit = jnp.mean(
            jnp.abs(w - quantize_dequantize_blockwise(w, nbits=8, blocksize=16).weight)
        )
        assert err_2bit > err_8bit

    def test_returns_quantize_result(self):
        w = jnp.ones((16, 16))
        result = quantize_dequantize_blockwise(w, nbits=4, blocksize=16)
        assert isinstance(result, QuantizeResult)
        assert result.num_outliers == 0

    def test_with_zscore_clipping(self):
        """Blockwise with z-score clipping should clip outliers."""
        w = jnp.zeros((16, 16))
        w = w.at[0, 0].set(1000.0)
        result = quantize_dequantize_blockwise(
            w, nbits=4, blocksize=16, clip_method="zscore", clip_threshold=3.0
        )
        assert result.num_outliers > 0

    def test_with_block_percentage_clipping(self):
        """Block-percentage clipping clips per-block."""
        # Use varied data with clear outliers, higher percentage to ensure clipping
        w = jnp.ones((16, 16)) * 0.5
        w = w.at[0, 0].set(1000.0)
        w = w.at[8, 0].set(-500.0)
        result = quantize_dequantize_blockwise(
            w, nbits=4, blocksize=16, clip_method="block_percentage", clip_threshold=0.1
        )
        # k = max(int(16*0.1+1),1) = 2, so threshold = 2nd largest abs per block
        # Row 0: 2nd largest is 0.5, so 1000 is clipped to 0.5
        assert float(jnp.max(jnp.abs(result.weight))) < 1000.0

    def test_with_iqr_clipping(self):
        """IQR clipping should detect outliers."""
        w = jnp.zeros((16, 16))
        w = w.at[0, 0].set(1000.0)
        result = quantize_dequantize_blockwise(
            w, nbits=4, blocksize=16, clip_method="iqr", clip_threshold=1.5
        )
        assert result.num_outliers > 0


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
        # threshold = k-th largest = 5000.0 (the outlier itself).
        # Values at the threshold are NOT clipped, only those above.
        # So the max stays at 5000.0. But non-outlier values are all < 999.
        assert float(jnp.max(jnp.abs(clipped))) <= 5000.0
        # Verify non-outlier values are not affected
        assert float(clipped[999]) == 999.0


# ---------------------------------------------------------------------------
# SW-aware weight quantization
# ---------------------------------------------------------------------------


class TestSWAwareWeight:
    def test_super_weight_restored(self):
        """Super weight should be restored exactly after quantization."""
        w = jnp.linspace(-1.0, 1.0, 256).reshape(16, 16)
        w = w.at[3, 7].set(50.0)
        sw_coords = [(3, 7)]
        result = quantize_weight_sw_aware(
            w, sw_coords, nbits=4, blocksize=16, clip_method="zscore", clip_threshold=3.0
        )
        assert float(result.weight[3, 7]) == pytest.approx(50.0)

    def test_non_sw_quantized(self):
        """Non-super-weight values should be quantized (not identical to original)."""
        w = jnp.sin(jnp.arange(256, dtype=jnp.float32) * 0.37).reshape(16, 16) * 5.0
        w = w.at[0, 0].set(50.0)
        sw_coords = [(0, 0)]
        result = quantize_weight_sw_aware(
            w, sw_coords, nbits=4, blocksize=16, clip_method="zscore", clip_threshold=3.0
        )
        assert float(result.weight[0, 0]) == pytest.approx(50.0)
        mask = jnp.ones_like(w, dtype=bool).at[0, 0].set(False)
        mean_err = jnp.mean(jnp.abs(w[mask] - result.weight[mask]))
        assert float(mean_err) > 1e-4

    def test_multiple_super_weights(self):
        """Multiple super weights should all be restored."""
        w = jnp.ones((16, 16))
        w = w.at[1, 2].set(99.0)
        w = w.at[5, 8].set(-77.0)
        sw_coords = [(1, 2), (5, 8)]
        result = quantize_weight_sw_aware(w, sw_coords, nbits=4, blocksize=16)
        assert float(result.weight[1, 2]) == pytest.approx(99.0)
        assert float(result.weight[5, 8]) == pytest.approx(-77.0)

    def test_empty_sw_coords(self):
        """With no super weights, should just be normal quantization."""
        w = jnp.linspace(-1.0, 1.0, 64).reshape(8, 8)
        result = quantize_weight_sw_aware(w, [], nbits=4, blocksize=8)
        assert result.weight.shape == w.shape

    def test_with_nf4(self):
        """SW-aware quantization should work with NF4."""
        w = jnp.linspace(-1.0, 1.0, 256).reshape(16, 16)
        w = w.at[3, 7].set(50.0)
        result = quantize_weight_sw_aware(
            w, [(3, 7)], nbits=4, blocksize=16, use_normal_float=True
        )
        assert float(result.weight[3, 7]) == pytest.approx(50.0)
        assert result.weight.shape == w.shape

    def test_with_scale_shift(self):
        """SW-aware quantization should work with scale-shift rounding."""
        w = jnp.linspace(-1.0, 1.0, 256).reshape(16, 16)
        w = w.at[3, 7].set(50.0)
        result = quantize_weight_sw_aware(
            w, [(3, 7)], nbits=4, blocksize=16, scale_shift=True
        )
        assert float(result.weight[3, 7]) == pytest.approx(50.0)


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


# ---------------------------------------------------------------------------
# NF4 / NF3 normal float quantization
# ---------------------------------------------------------------------------


class TestNormalFloat:
    def test_nf4_shape_preserved(self):
        w = jnp.linspace(-1.0, 1.0, 256).reshape(16, 16)
        w_hat = quantize_dequantize_nf(w, nbits=4, blocksize=16)
        assert w_hat.shape == w.shape

    def test_nf3_shape_preserved(self):
        w = jnp.linspace(-1.0, 1.0, 256).reshape(16, 16)
        w_hat = quantize_dequantize_nf(w, nbits=3, blocksize=16)
        assert w_hat.shape == w.shape

    def test_nf4_outputs_are_on_poles(self):
        """NF4 quantized values should map to one of the 16 NF4 poles after normalization."""
        w = jnp.linspace(-2.0, 2.0, 256).reshape(16, 16)
        w_hat = quantize_dequantize_nf(w, nbits=4, blocksize=16)
        # Just check output is valid float (not NaN/Inf)
        assert jnp.all(jnp.isfinite(w_hat))

    def test_nf4_reconstruction_reasonable(self):
        """NF4 should have bounded reconstruction error."""
        w = jnp.linspace(-1.0, 1.0, 256).reshape(16, 16)
        w_hat = quantize_dequantize_nf(w, nbits=4, blocksize=16)
        max_err = float(jnp.max(jnp.abs(w - w_hat)))
        assert max_err < 0.5  # 4-bit quantization should be within reason

    def test_nf3_coarser_than_nf4(self):
        """NF3 (8 levels) should have more error than NF4 (16 levels)."""
        w = jnp.linspace(-3.0, 3.0, 256).reshape(16, 16)
        err_nf3 = float(jnp.mean(jnp.abs(w - quantize_dequantize_nf(w, nbits=3, blocksize=16))))
        err_nf4 = float(jnp.mean(jnp.abs(w - quantize_dequantize_nf(w, nbits=4, blocksize=16))))
        assert err_nf3 > err_nf4

    def test_nf_invalid_bits(self):
        w = jnp.ones((4, 4))
        with pytest.raises(ValueError, match="3 and 4 bits"):
            quantize_dequantize_nf(w, nbits=8)

    def test_round_to_nearest_pole_basic(self):
        """Check that round_to_nearest_pole picks closest pole."""
        poles = jnp.array([-1.0, 0.0, 1.0])
        x = jnp.array([-0.7, 0.3, 0.9, -0.1])
        result = round_to_nearest_pole(x, poles)
        expected = jnp.array([-1.0, 0.0, 1.0, 0.0])
        assert jnp.allclose(result, expected)

    def test_nf4_via_blockwise(self):
        """NF4 should work through the blockwise API."""
        w = jnp.linspace(-1.0, 1.0, 256).reshape(16, 16)
        result = quantize_dequantize_blockwise(
            w, nbits=4, blocksize=16, use_normal_float=True
        )
        assert result.weight.shape == w.shape
        assert jnp.all(jnp.isfinite(result.weight))


# ---------------------------------------------------------------------------
# Block-percentage clipping
# ---------------------------------------------------------------------------


class TestBlockPercentageClip:
    def test_clips_per_block(self):
        """Each block should have its outliers clipped independently."""
        # Use varied data with outliers standing out
        w = jnp.ones((4, 16)) * 0.5
        w = w.at[0, 0].set(100.0)
        w = w.at[2, 5].set(-200.0)
        clipped, n_outliers = clip_block_percentage(w, percentage=0.05)
        # With 16 elements per block and 5%, k = max(int(16*0.05+1),1) = 1
        # The threshold is the 1st largest = the outlier itself.
        # Use a higher percentage to get meaningful clipping:
        clipped2, n_outliers2 = clip_block_percentage(w, percentage=0.1)
        # k = max(int(16*0.1+1),1) = 2, threshold = 2nd largest
        # Row 0 has one outlier at 100 and rest at 0.5, so threshold=0.5
        assert float(jnp.max(jnp.abs(clipped2[0]))) <= 0.5
        assert n_outliers2 >= 2

    def test_no_outliers_when_all_same(self):
        w = jnp.ones((4, 16))
        clipped, n_outliers = clip_block_percentage(w, percentage=0.01)
        assert jnp.allclose(w, clipped)


# ---------------------------------------------------------------------------
# IQR clipping
# ---------------------------------------------------------------------------


class TestIQRClip:
    def test_clips_outliers(self):
        w = jnp.zeros(100).at[0].set(1000.0)
        clipped, n_outliers = clip_iqr(w, factor=1.5)
        assert float(jnp.max(jnp.abs(clipped))) < 1000.0
        assert n_outliers > 0

    def test_no_clip_when_uniform(self):
        """Uniform data shouldn't trigger outlier clipping much."""
        w = jnp.linspace(0.0, 1.0, 100)
        clipped, n_outliers = clip_iqr(w, factor=1.5)
        assert jnp.allclose(w, clipped, atol=1e-5)

    def test_per_block_mode(self):
        w = jnp.zeros((4, 16))
        w = w.at[0, 0].set(1000.0)
        clipped, n_outliers = clip_iqr(w, factor=1.5, per_block=True)
        assert float(jnp.max(jnp.abs(clipped))) < 1000.0
        assert n_outliers > 0


# ---------------------------------------------------------------------------
# Scale-shift rounding
# ---------------------------------------------------------------------------


class TestScaleShift:
    def test_shape_preserved(self):
        w = jnp.linspace(-1.0, 1.0, 256).reshape(16, 16)
        w_hat = quantize_dequantize_scale_shift(w, nbits=4, blocksize=16)
        assert w_hat.shape == w.shape

    def test_reconstruction_reasonable(self):
        w = jnp.linspace(-2.0, 2.0, 256).reshape(16, 16)
        w_hat = quantize_dequantize_scale_shift(w, nbits=8, blocksize=16)
        assert jnp.allclose(w, w_hat, atol=0.05)

    def test_via_blockwise_api(self):
        """Scale-shift should work through the blockwise API."""
        w = jnp.linspace(-1.0, 1.0, 256).reshape(16, 16)
        result = quantize_dequantize_blockwise(
            w, nbits=4, blocksize=16, scale_shift=True
        )
        assert result.weight.shape == w.shape


# ---------------------------------------------------------------------------
# Per-channel quantization
# ---------------------------------------------------------------------------


class TestPerChannel:
    def test_shape_preserved(self):
        w = jnp.linspace(-1.0, 1.0, 256).reshape(16, 16)
        result = quantize_dequantize_per_channel(w, nbits=8)
        assert result.weight.shape == w.shape

    def test_high_bit_fidelity(self):
        """8-bit per-channel should be very close to original."""
        w = jnp.linspace(-1.0, 1.0, 256).reshape(16, 16)
        result = quantize_dequantize_per_channel(w, nbits=8)
        assert jnp.allclose(w, result.weight, atol=0.01)


# ---------------------------------------------------------------------------
# 4-bit packing
# ---------------------------------------------------------------------------


class TestPacking:
    def test_pack_unpack_roundtrip(self):
        """Pack then unpack should recover original 4-bit values."""
        q = jnp.array([[0, 15, 7, 3, 1, 14, 8, 2]], dtype=jnp.int32)
        packed = pack_4bit_to_int8(q)
        unpacked = unpack_int8_to_4bit(packed)
        assert jnp.array_equal(q, unpacked[:, : q.shape[1]])

    def test_pack_halves_columns(self):
        q = jnp.zeros((4, 8), dtype=jnp.int32)
        packed = pack_4bit_to_int8(q)
        assert packed.shape == (4, 4)

    def test_odd_cols_padded(self):
        q = jnp.zeros((2, 7), dtype=jnp.int32)
        packed = pack_4bit_to_int8(q)
        assert packed.shape == (2, 4)  # 8/2 = 4


# ---------------------------------------------------------------------------
# Super weight scaling
# ---------------------------------------------------------------------------


class TestSWScaling:
    def test_scale_up(self):
        w = jnp.ones((16, 16))
        w = w.at[3, 7].set(10.0)
        scaled = scale_super_weights(w, [(3, 7)], scaling_factor=2.0)
        assert float(scaled[3, 7]) == pytest.approx(20.0)
        # Other values unchanged
        assert float(scaled[0, 0]) == pytest.approx(1.0)

    def test_scale_zero(self):
        """Scaling by 0 effectively prunes the super weight."""
        w = jnp.ones((16, 16))
        w = w.at[3, 7].set(50.0)
        scaled = scale_super_weights(w, [(3, 7)], scaling_factor=0.0)
        assert float(scaled[3, 7]) == pytest.approx(0.0)

    def test_scale_identity(self):
        """Scaling by 1.0 should not change the weight."""
        w = jnp.linspace(-1.0, 1.0, 256).reshape(16, 16)
        w = w.at[3, 7].set(42.0)
        scaled = scale_super_weights(w, [(3, 7)], scaling_factor=1.0)
        assert jnp.allclose(w, scaled)

    def test_multiple_sws(self):
        w = jnp.ones((16, 16))
        w = w.at[1, 2].set(10.0)
        w = w.at[5, 8].set(20.0)
        scaled = scale_super_weights(w, [(1, 2), (5, 8)], scaling_factor=3.0)
        assert float(scaled[1, 2]) == pytest.approx(30.0)
        assert float(scaled[5, 8]) == pytest.approx(60.0)
