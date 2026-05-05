"""Reference equivalence tests.

Generates random inputs and verifies that sumow (JAX) produces numerically
equivalent results to the reference implementation (PyTorch) from
llmsuperweight/outliers/functional/quantization.py.

These tests serve as the ground truth: if the reference and sumow disagree
on arbitrary inputs, the JAX code is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

# Add reference implementation to path
_REF_ROOT = Path(__file__).resolve().parent.parent / "llmsuperweight" / "outliers"
sys.path.insert(0, str(_REF_ROOT))

from functional.quantization import (  # noqa: E402  # sys.path must be modified before importing the llmsuperweight submodule reference implementation
    # type: ignore[import-unresolved]  # path is added dynamically above; ty can't resolve it statically
    pack_4bit_to_int8 as ref_pack_4bit_to_int8,
    quantize_blockwise as ref_quantize_blockwise,
    round_to_nearest_pole as ref_round_to_nearest_pole,
    unpack_int8_to_4bit as ref_unpack_int8_to_4bit,
)

from sumow.quantize import (  # noqa: E402  # comes after sys.path.insert; grouped with other post-path imports for readability
    NF3_LEVELS,
    NF4_LEVELS,
    pack_4bit_to_int8 as jax_pack_4bit_to_int8,
    quantize_dequantize_blockwise as jax_quantize_blockwise,
    round_to_nearest_pole as jax_round_to_nearest_pole,
    unpack_int8_to_4bit as jax_unpack_int8_to_4bit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.RandomState(42)


def random_weight(rows: int, cols: int, scale: float = 1.0) -> np.ndarray:
    """Generate a random weight matrix with some outliers."""
    w = RNG.randn(rows, cols).astype(np.float32) * scale
    # Inject a few outliers
    n_outliers = max(1, rows * cols // 100)
    idxs = RNG.choice(rows * cols, n_outliers, replace=False)
    w.flat[idxs] = RNG.randn(n_outliers).astype(np.float32) * scale * 20
    return w


def to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.copy())


def to_jax(arr: np.ndarray) -> jnp.ndarray:
    return jnp.array(arr)


# ---------------------------------------------------------------------------
# round_to_nearest_pole
# ---------------------------------------------------------------------------


class TestRoundToNearestPole:
    """Verify JAX round_to_nearest_pole matches reference exactly."""

    @pytest.mark.parametrize("n", [16, 100, 1000])
    def test_nf4_poles(self, n: int):
        x_np = RNG.uniform(-1, 1, size=n).astype(np.float32)
        poles_np = np.array(NF4_LEVELS, dtype=np.float32)

        ref_result = ref_round_to_nearest_pole(
            to_torch(x_np), to_torch(poles_np)
        ).numpy()
        jax_result = np.array(jax_round_to_nearest_pole(to_jax(x_np), to_jax(poles_np)))

        np.testing.assert_array_equal(ref_result, jax_result)

    @pytest.mark.parametrize("n", [16, 100, 1000])
    def test_nf3_poles(self, n: int):
        x_np = RNG.uniform(-1, 1, size=n).astype(np.float32)
        poles_np = np.array(NF3_LEVELS, dtype=np.float32)

        ref_result = ref_round_to_nearest_pole(
            to_torch(x_np), to_torch(poles_np)
        ).numpy()
        jax_result = np.array(jax_round_to_nearest_pole(to_jax(x_np), to_jax(poles_np)))

        np.testing.assert_array_equal(ref_result, jax_result)

    def test_edge_values(self):
        """Test boundary values: exactly on poles, exactly between poles."""
        poles_np = np.array(NF4_LEVELS, dtype=np.float32)
        # Values exactly on poles
        x_np = poles_np.copy()

        ref_result = ref_round_to_nearest_pole(
            to_torch(x_np), to_torch(poles_np)
        ).numpy()
        jax_result = np.array(jax_round_to_nearest_pole(to_jax(x_np), to_jax(poles_np)))

        np.testing.assert_array_equal(ref_result, jax_result)


# ---------------------------------------------------------------------------
# pack / unpack 4-bit
# ---------------------------------------------------------------------------


class TestPackUnpack:
    """Verify JAX pack/unpack matches reference exactly."""

    @pytest.mark.parametrize("rows,cols", [(4, 8), (1, 16), (8, 7), (16, 32)])
    def test_pack_equivalence(self, rows: int, cols: int):
        q_np = RNG.randint(0, 16, size=(rows, cols)).astype(np.uint8)

        ref_packed = ref_pack_4bit_to_int8(to_torch(q_np)).numpy()
        jax_packed = np.array(jax_pack_4bit_to_int8(to_jax(q_np.astype(np.int32))))

        # Compare as uint8
        np.testing.assert_array_equal(
            ref_packed.astype(np.uint8), jax_packed.astype(np.uint8)
        )

    @pytest.mark.parametrize("rows,cols", [(4, 4), (1, 8), (16, 16)])
    def test_unpack_equivalence(self, rows: int, cols: int):
        packed_np = RNG.randint(0, 256, size=(rows, cols)).astype(np.uint8)

        ref_unpacked = ref_unpack_int8_to_4bit(to_torch(packed_np)).numpy()
        jax_unpacked = np.array(jax_unpack_int8_to_4bit(to_jax(packed_np.astype(np.int8))))

        np.testing.assert_array_equal(ref_unpacked, jax_unpacked)

    @pytest.mark.parametrize("rows,cols", [(4, 8), (8, 16), (2, 32)])
    def test_roundtrip_equivalence(self, rows: int, cols: int):
        """Pack then unpack should be identical in both frameworks."""
        q_np = RNG.randint(0, 16, size=(rows, cols)).astype(np.uint8)

        # Reference roundtrip
        ref_packed = ref_pack_4bit_to_int8(to_torch(q_np))
        ref_rt = ref_unpack_int8_to_4bit(ref_packed).numpy()

        # JAX roundtrip
        jax_packed = jax_pack_4bit_to_int8(to_jax(q_np.astype(np.int32)))
        jax_rt = np.array(jax_unpack_int8_to_4bit(jax_packed))

        # Both should recover the original (for even cols)
        np.testing.assert_array_equal(ref_rt[:, :cols], jax_rt[:, :cols])


# ---------------------------------------------------------------------------
# Blockwise INT quantization (no clipping)
# ---------------------------------------------------------------------------


class TestBlockwiseINT:
    """Verify JAX blockwise INT quantization matches reference."""

    @pytest.mark.parametrize(
        "shape,nbits,blocksize",
        [
            ((16, 128), 4, 128),
            ((32, 64), 4, 64),
            ((64, 256), 8, 128),
            ((16, 16), 4, 16),
            ((8, 32), 3, 32),
            ((16, 128), 4, 64),
        ],
    )
    def test_no_clip(self, shape: tuple, nbits: int, blocksize: int):
        w_np = random_weight(*shape)

        ref_result, ref_outliers = ref_quantize_blockwise(
            to_torch(w_np), nbits, blocksize, clip_method="no", clip_threshold=0
        )
        ref_result = ref_result.numpy()

        jax_result = jax_quantize_blockwise(
            to_jax(w_np), nbits=nbits, blocksize=blocksize, clip_method="none"
        )

        np.testing.assert_allclose(
            ref_result, np.array(jax_result.weight), rtol=1e-5, atol=1e-5
        )
        assert ref_outliers == jax_result.num_outliers == 0

    @pytest.mark.parametrize(
        "shape,nbits,blocksize",
        [
            ((16, 128), 4, 128),
            ((32, 64), 4, 64),
            ((64, 128), 8, 128),
        ],
    )
    def test_scale_shift(self, shape: tuple, nbits: int, blocksize: int):
        w_np = random_weight(*shape)

        ref_result, _ = ref_quantize_blockwise(
            to_torch(w_np), nbits, blocksize,
            clip_method="no", clip_threshold=0, scale_shift=True,
        )

        jax_result = jax_quantize_blockwise(
            to_jax(w_np), nbits=nbits, blocksize=blocksize,
            clip_method="none", scale_shift=True,
        )

        ref_np = ref_result.numpy()
        jax_np = np.array(jax_result.weight)

        # Scale-shift: float32 mul/sub chains can produce ±1 ULP differences
        # at rounding boundaries (x.5), causing ±1 quantization level error.
        # Assert: at most 1% of elements may differ by more than 1e-4,
        # and those must be within one quantization step.
        diff = np.abs(ref_np - jax_np)
        flat_w = w_np.reshape(-1, blocksize)
        block_range = flat_w.max(axis=1) - flat_w.min(axis=1)
        step_per_block = block_range / ((1 << nbits) - 0.01)
        # Expand step to match each element
        step = np.repeat(step_per_block, blocksize).reshape(ref_np.shape)

        n_boundary = int(np.sum(diff > 1e-4))
        assert n_boundary <= 0.01 * ref_np.size, (
            f"Too many mismatches: {n_boundary}/{ref_np.size}"
        )
        # Every mismatch must be at most one quantization step
        np.testing.assert_array_less(diff, step + 1e-4)


# ---------------------------------------------------------------------------
# Blockwise NF4 quantization
# ---------------------------------------------------------------------------


class TestBlockwiseNF4:
    """Verify JAX NF4 quantization matches reference."""

    @pytest.mark.parametrize(
        "shape,nbits,blocksize",
        [
            ((16, 128), 4, 128),
            ((32, 64), 4, 64),
            ((16, 16), 4, 16),
        ],
    )
    def test_nf4_no_clip(self, shape: tuple, nbits: int, blocksize: int):
        w_np = random_weight(*shape, scale=0.5)

        ref_result, _ = ref_quantize_blockwise(
            to_torch(w_np), nbits, blocksize,
            clip_method="no", clip_threshold=0, use_normal_float=True,
        )

        jax_result = jax_quantize_blockwise(
            to_jax(w_np), nbits=nbits, blocksize=blocksize,
            clip_method="none", use_normal_float=True,
        )

        np.testing.assert_allclose(
            ref_result.numpy(), np.array(jax_result.weight), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("blocksize", [16, 64, 128])
    def test_nf3_no_clip(self, blocksize: int):
        rows = 16
        cols = blocksize
        w_np = random_weight(rows, cols, scale=0.5)

        ref_result, _ = ref_quantize_blockwise(
            to_torch(w_np), 3, blocksize,
            clip_method="no", clip_threshold=0, use_normal_float=True,
        )

        jax_result = jax_quantize_blockwise(
            to_jax(w_np), nbits=3, blocksize=blocksize,
            clip_method="none", use_normal_float=True,
        )

        np.testing.assert_allclose(
            ref_result.numpy(), np.array(jax_result.weight), rtol=1e-5, atol=1e-5
        )


# ---------------------------------------------------------------------------
# Blockwise with clipping
# ---------------------------------------------------------------------------


class TestBlockwiseClipping:
    """Verify clipping methods produce equivalent results."""

    @pytest.mark.parametrize("blocksize", [32, 64, 128])
    def test_zscore_clip(self, blocksize: int):
        w_np = random_weight(32, blocksize, scale=2.0)
        z_threshold = 3.0

        ref_result, ref_outliers = ref_quantize_blockwise(
            to_torch(w_np), 4, blocksize,
            clip_method="zscore", clip_threshold=z_threshold,
        )

        jax_result = jax_quantize_blockwise(
            to_jax(w_np), nbits=4, blocksize=blocksize,
            clip_method="zscore", clip_threshold=z_threshold,
        )

        np.testing.assert_allclose(
            ref_result.numpy(), np.array(jax_result.weight), rtol=1e-5, atol=1e-5
        )
        assert ref_outliers == jax_result.num_outliers

    @pytest.mark.parametrize("blocksize", [32, 64, 128])
    def test_block_percentage_clip(self, blocksize: int):
        w_np = random_weight(32, blocksize, scale=2.0)
        pct = 0.01

        ref_result, ref_outliers = ref_quantize_blockwise(
            to_torch(w_np), 4, blocksize,
            clip_method="block_percentage", clip_threshold=pct,
        )

        jax_result = jax_quantize_blockwise(
            to_jax(w_np), nbits=4, blocksize=blocksize,
            clip_method="block_percentage", clip_threshold=pct,
        )

        np.testing.assert_allclose(
            ref_result.numpy(), np.array(jax_result.weight), rtol=1e-5, atol=1e-5
        )
        assert ref_outliers == jax_result.num_outliers

    @pytest.mark.parametrize("blocksize", [32, 64, 128])
    def test_iqr_clip(self, blocksize: int):
        w_np = random_weight(32, blocksize, scale=2.0)
        iqr_factor = 1.5

        ref_result, ref_outliers = ref_quantize_blockwise(
            to_torch(w_np), 4, blocksize,
            clip_method="iqr", clip_threshold=iqr_factor,
        )

        jax_result = jax_quantize_blockwise(
            to_jax(w_np), nbits=4, blocksize=blocksize,
            clip_method="iqr", clip_threshold=iqr_factor,
        )

        np.testing.assert_allclose(
            ref_result.numpy(), np.array(jax_result.weight), rtol=1e-5, atol=1e-5
        )
        assert ref_outliers == jax_result.num_outliers

    @pytest.mark.parametrize("blocksize", [32, 64, 128])
    def test_tensor_percentage_clip(self, blocksize: int):
        w_np = random_weight(32, blocksize, scale=2.0)
        pct = 1e-3

        ref_result, ref_outliers = ref_quantize_blockwise(
            to_torch(w_np), 4, blocksize,
            clip_method="tensor_percentage", clip_threshold=pct,
        )

        jax_result = jax_quantize_blockwise(
            to_jax(w_np), nbits=4, blocksize=blocksize,
            clip_method="tensor_percentage", clip_threshold=pct,
        )

        np.testing.assert_allclose(
            ref_result.numpy(), np.array(jax_result.weight), rtol=1e-5, atol=1e-5
        )
        assert ref_outliers == jax_result.num_outliers


# ---------------------------------------------------------------------------
# Combined: clip + NF4
# ---------------------------------------------------------------------------


class TestClipPlusNF:
    """Clipping followed by NF4 quantization."""

    @pytest.mark.parametrize("clip_method,clip_threshold", [
        ("zscore", 3.0),
        ("block_percentage", 0.01),
        ("iqr", 1.5),
    ])
    def test_clip_then_nf4(self, clip_method: str, clip_threshold: float):
        w_np = random_weight(16, 128, scale=2.0)

        ref_result, ref_outliers = ref_quantize_blockwise(
            to_torch(w_np), 4, 128,
            clip_method=clip_method, clip_threshold=clip_threshold,
            use_normal_float=True,
        )

        jax_result = jax_quantize_blockwise(
            to_jax(w_np), nbits=4, blocksize=128,
            clip_method=clip_method, clip_threshold=clip_threshold,
            use_normal_float=True,
        )

        np.testing.assert_allclose(
            ref_result.numpy(), np.array(jax_result.weight), rtol=1e-5, atol=1e-5
        )
        assert ref_outliers == jax_result.num_outliers


# ---------------------------------------------------------------------------
# Stress: random parameters
# ---------------------------------------------------------------------------


class TestRandomStress:
    """Run many random configurations and check equivalence."""

    @pytest.mark.parametrize("seed", range(10))
    def test_random_int_config(self, seed: int):
        rng = np.random.RandomState(seed + 1000)
        rows = int(rng.choice([8, 16, 32, 64]))
        blocksize = int(rng.choice([16, 32, 64, 128]))
        cols = blocksize * int(rng.randint(1, 5))
        nbits = int(rng.choice([3, 4, 8]))
        scale = float(rng.uniform(0.1, 10.0))

        w_np = (rng.randn(rows, cols) * scale).astype(np.float32)
        # Inject outliers
        n_out = max(1, w_np.size // 50)
        idxs = rng.choice(w_np.size, n_out, replace=False)
        w_np.flat[idxs] *= 20

        ref_result, _ = ref_quantize_blockwise(
            to_torch(w_np), nbits, blocksize,
            clip_method="no", clip_threshold=0,
        )

        jax_result = jax_quantize_blockwise(
            to_jax(w_np), nbits=nbits, blocksize=blocksize,
            clip_method="none",
        )

        np.testing.assert_allclose(
            ref_result.numpy(), np.array(jax_result.weight), rtol=1e-4, atol=5e-5
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_random_nf4_config(self, seed: int):
        rng = np.random.RandomState(seed + 2000)
        blocksize = int(rng.choice([16, 32, 64, 128]))
        rows = int(rng.choice([8, 16, 32]))
        cols = blocksize * int(rng.randint(1, 4))

        w_np = (rng.randn(rows, cols) * 0.5).astype(np.float32)

        ref_result, _ = ref_quantize_blockwise(
            to_torch(w_np), 4, blocksize,
            clip_method="no", clip_threshold=0, use_normal_float=True,
        )

        jax_result = jax_quantize_blockwise(
            to_jax(w_np), nbits=4, blocksize=blocksize,
            clip_method="none", use_normal_float=True,
        )

        np.testing.assert_allclose(
            ref_result.numpy(), np.array(jax_result.weight), rtol=1e-5, atol=1e-5
        )
