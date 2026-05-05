"""Tests for model I/O utilities."""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from safetensors.numpy import save_file

from sumow.model_io import (
    extract_down_proj_weights,
    get_super_weight_values,
    load_safetensors,
    load_model_weights,
)


def _create_mock_safetensors(
    path: Path,
    tensors: dict[str, np.ndarray],
) -> None:
    """Write a mock safetensors file."""
    save_file(tensors, str(path))


class TestLoadSafetensors:
    def test_basic_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.safetensors"
            _create_mock_safetensors(path, {
                "model.layers.0.mlp.down_proj.weight": np.ones((4, 8), dtype=np.float32),
                "model.layers.1.mlp.down_proj.weight": np.zeros((4, 8), dtype=np.float32),
            })
            tensors = load_safetensors(path)
            assert len(tensors) == 2
            assert tensors["model.layers.0.mlp.down_proj.weight"].shape == (4, 8)


class TestLoadModelWeights:
    def test_multi_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_mock_safetensors(
                Path(tmpdir) / "model-00001.safetensors",
                {"model.layers.0.mlp.down_proj.weight": np.ones((4, 8), dtype=np.float32)},
            )
            _create_mock_safetensors(
                Path(tmpdir) / "model-00002.safetensors",
                {"model.layers.1.mlp.down_proj.weight": np.zeros((4, 8), dtype=np.float32)},
            )
            tensors = load_model_weights(tmpdir)
            assert len(tensors) == 2

    def test_no_files_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                load_model_weights(tmpdir)
                assert False, "Should have raised"
            except FileNotFoundError:
                pass


class TestExtractDownProj:
    def test_extracts_correct_layers(self):
        weights = {
            "model.layers.0.mlp.down_proj.weight": jnp.ones((4, 8)),
            "model.layers.0.mlp.up_proj.weight": jnp.ones((8, 4)),
            "model.layers.2.mlp.down_proj.weight": jnp.ones((4, 8)),
            "model.embed_tokens.weight": jnp.ones((100, 4)),
        }
        down_proj = extract_down_proj_weights(weights)
        assert set(down_proj.keys()) == {0, 2}

    def test_empty_weights(self):
        assert extract_down_proj_weights({}) == {}


class TestGetSuperWeightValues:
    def test_known_model(self):
        # Create mock weights matching llama-7B structure
        weights = {
            "model.layers.2.mlp.down_proj.weight": jnp.zeros((4096, 11008)).at[3968, 7003].set(42.5),
        }
        values = get_super_weight_values(weights, "huggyllama/llama-7B")
        assert (2, 3968, 7003) in values
        assert abs(values[(2, 3968, 7003)] - 42.5) < 1e-5

    def test_unknown_model(self):
        values = get_super_weight_values({}, "unknown/model")
        assert values == {}
