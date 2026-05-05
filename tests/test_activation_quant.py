"""Tests for activation quantization (SA-aware).

Tests the paper's Equation 1:
  Â = Restore(Q⁻¹(Q(Replace(A))))

Verifies:
- Super activations are preserved exactly
- Non-SA elements are quantized
- Replacement with median before quantization works correctly
"""

import jax.numpy as jnp
import numpy as np

from sumow.quantize import quantize_activation_sa_aware


class TestActivationQuantBasic:
    """Basic SA-aware activation quantization tests."""

    def test_no_sa_positions(self):
        """With no SAs, all activations are quantized."""
        a = jnp.array([[[1.0, 2.0, 3.0, 4.0]]]).astype(jnp.float32)  # [1, 1, 4]
        result = quantize_activation_sa_aware(a, sa_positions=[], nbits=8)
        assert result.shape == a.shape
        assert jnp.all(jnp.isfinite(result))

    def test_sa_preserved_exactly(self):
        """Super activations should be restored to original values."""
        rng = jnp.array([1.0, 2.0, 100.0, 3.0, 4.0])
        a = rng.reshape(1, 1, 5)  # [1, 1, 5]
        sa_positions = [(0, 2)]  # (seq_idx=0, hidden_idx=2) — the outlier at 100.0

        result = quantize_activation_sa_aware(a, sa_positions, nbits=8)

        # The super activation must be exactly preserved
        np.testing.assert_equal(float(result[0, 0, 2]), 100.0)

    def test_non_sa_elements_quantized(self):
        """Non-SA elements should differ from original (quantization error)."""
        a = jnp.linspace(-10, 10, 20).reshape(1, 4, 5)  # [1, 4, 5]
        sa_positions = [(0, 0)]  # protect seq=0, hidden=0

        result = quantize_activation_sa_aware(a, sa_positions, nbits=4)

        # SA element preserved
        np.testing.assert_equal(float(result[0, 0, 0]), float(a[0, 0, 0]))

        # At 4-bit with wide range, some elements should differ
        non_sa_orig = np.array(a[0, 1:, :])
        non_sa_quant = np.array(result[0, 1:, :])
        assert not np.allclose(non_sa_orig, non_sa_quant, atol=1e-6), \
            "Non-SA elements should have quantization error at 4-bit"

    def test_multiple_sa_positions(self):
        """Multiple super activations all preserved."""
        a = jnp.array([
            [[1.0, 200.0, 3.0, 4.0],
             [5.0, 6.0, -150.0, 8.0],
             [9.0, 10.0, 11.0, 12.0]]
        ])  # [1, 3, 4]

        sa_positions = [(0, 1), (1, 2)]  # 200.0 and -150.0

        result = quantize_activation_sa_aware(a, sa_positions, nbits=8)

        np.testing.assert_equal(float(result[0, 0, 1]), 200.0)
        np.testing.assert_equal(float(result[0, 1, 2]), -150.0)

    def test_batch_dimension(self):
        """Works with batch > 1."""
        a = jnp.ones((2, 3, 4)) * jnp.arange(4).reshape(1, 1, 4)
        a = a.at[0, 0, 0].set(500.0)
        a = a.at[1, 0, 0].set(-500.0)

        sa_positions = [(0, 0)]
        result = quantize_activation_sa_aware(a, sa_positions, nbits=8)

        # SA is per-batch, so both batch elements at (seq=0, hidden=0) restored
        np.testing.assert_equal(float(result[0, 0, 0]), 500.0)
        np.testing.assert_equal(float(result[1, 0, 0]), -500.0)


class TestActivationQuantEdgeCases:
    """Edge cases for activation quantization."""

    def test_all_positions_sa(self):
        """If all positions are SAs, output == input."""
        a = jnp.array([[[10.0, 20.0, 30.0]]]).astype(jnp.float32)
        sa_positions = [(0, 0), (0, 1), (0, 2)]

        result = quantize_activation_sa_aware(a, sa_positions, nbits=8)
        np.testing.assert_array_equal(np.array(result), np.array(a))

    def test_constant_activation(self):
        """Constant tensor → quantize/dequantize is identity."""
        a = jnp.ones((1, 4, 8)) * 5.0
        result = quantize_activation_sa_aware(a, sa_positions=[], nbits=8)
        np.testing.assert_allclose(np.array(result), np.array(a), atol=1e-4)

    def test_high_bitwidth_near_identity(self):
        """At 8-bit, error should be very small."""
        a = jnp.linspace(-1, 1, 32).reshape(1, 4, 8)
        result = quantize_activation_sa_aware(a, sa_positions=[], nbits=8)
        np.testing.assert_allclose(np.array(result), np.array(a), atol=0.02)

    def test_sa_with_outlier_much_larger(self):
        """SA is an extreme outlier — after replacement + quant, others should be closer."""
        normal = jnp.ones((1, 1, 8)) * 2.0
        a = normal.at[0, 0, 3].set(1000.0)  # huge outlier at hidden=3

        # Without SA-awareness (quantize with outlier present)
        result_no_sa = quantize_activation_sa_aware(a, sa_positions=[], nbits=8)

        # With SA-awareness (outlier replaced with median before quantization)
        result_with_sa = quantize_activation_sa_aware(a, sa_positions=[(0, 3)], nbits=8)

        # The normal elements should be quantized more accurately when SA is replaced
        normal_err_no = float(jnp.mean(jnp.abs(a[0, 0, :3] - result_no_sa[0, 0, :3])))
        normal_err_yes = float(jnp.mean(jnp.abs(a[0, 0, :3] - result_with_sa[0, 0, :3])))

        assert normal_err_yes <= normal_err_no, (
            f"SA replacement should help normal elements: {normal_err_yes:.6f} vs {normal_err_no:.6f}"
        )

        # And the outlier itself is preserved
        np.testing.assert_equal(float(result_with_sa[0, 0, 3]), 1000.0)


class TestActivationQuantNumerics:
    """Numerical properties of activation quantization."""

    def test_quantization_is_deterministic(self):
        """Same input → same output."""
        a = jnp.linspace(-5, 5, 24).reshape(1, 3, 8)
        r1 = quantize_activation_sa_aware(a, [(0, 0)], nbits=8)
        r2 = quantize_activation_sa_aware(a, [(0, 0)], nbits=8)
        np.testing.assert_array_equal(np.array(r1), np.array(r2))

    def test_lower_bits_more_error(self):
        """4-bit should have more error than 8-bit."""
        a = jnp.linspace(-10, 10, 40).reshape(1, 5, 8)

        r4 = quantize_activation_sa_aware(a, [], nbits=4)
        r8 = quantize_activation_sa_aware(a, [], nbits=8)

        err4 = float(jnp.mean(jnp.abs(a - r4)))
        err8 = float(jnp.mean(jnp.abs(a - r8)))

        assert err4 >= err8, f"4-bit err ({err4:.6f}) should be >= 8-bit ({err8:.6f})"

    def test_output_shape_preserved(self):
        """Output shape matches input for various shapes."""
        for shape in [(1, 1, 4), (2, 3, 8), (1, 10, 16)]:
            a = jnp.ones(shape)
            result = quantize_activation_sa_aware(a, [], nbits=8)
            assert result.shape == shape
