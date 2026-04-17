"""Load HuggingFace model weights into JAX arrays.

Provides utilities to load safetensors-format pretrained weights into
flat dictionaries of JAX arrays, with helpers for extracting down_proj
weight matrices and super weight values.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import jaxtyped,  Array, Float
from safetensors import safe_open

from sumow.config import SUPER_WEIGHT_DIRECTORY


@jaxtyped(typechecker=beartype)
def load_safetensors(path: str | Path) -> dict[str, Array]:
    """Load all tensors from a safetensors file into JAX arrays.

    Args:
        path: Path to a .safetensors file.

    Returns:
        Dictionary mapping tensor names to JAX arrays.
    """
    tensors = {}
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            tensors[key] = jnp.array(f.get_tensor(key))
    return tensors


@jaxtyped(typechecker=beartype)
def load_model_weights(model_dir: str | Path) -> dict[str, Array]:
    """Load all safetensors files from a model directory.

    Handles both single-file and sharded models.

    Args:
        model_dir: Path to the directory containing .safetensors files.

    Returns:
        Dictionary mapping parameter names to JAX arrays.
    """
    model_dir = Path(model_dir)
    tensors: dict[str, Array] = {}
    for sf_path in sorted(model_dir.glob("*.safetensors")):
        tensors.update(load_safetensors(sf_path))
    if not tensors:
        raise FileNotFoundError(
            f"No .safetensors files found in {model_dir}"
        )
    return tensors


@jaxtyped(typechecker=beartype)
def extract_down_proj_weights(
    weights: dict[str, Array],
    down_proj_pattern: str = "down_proj.weight",
) -> dict[int, Float[Array, "out_dim in_dim"]]:
    """Extract down_proj weight matrices indexed by layer number.

    Scans the weight dictionary for keys matching the pattern and extracts
    the layer number from the key.

    Args:
        weights: Full model weight dictionary.
        down_proj_pattern: Substring to match for down_proj weights.

    Returns:
        Dictionary mapping layer index to weight matrix.
    """
    import re

    down_proj_weights: dict[int, Array] = {}
    for key, tensor in weights.items():
        if down_proj_pattern in key:
            match = re.search(r"layers?\.(\d+)\.", key)
            if match:
                layer_idx = int(match.group(1))
                down_proj_weights[layer_idx] = tensor
    return down_proj_weights


@jaxtyped(typechecker=beartype)
def get_super_weight_values(
    weights: dict[str, Array],
    model_id: str,
) -> dict[tuple[int, int, int], float]:
    """Look up known super weight values from the loaded weights.

    Args:
        weights: Full model weight dictionary.
        model_id: HuggingFace model identifier.

    Returns:
        Dictionary mapping (layer, row, col) to the scalar value.
    """
    coords = SUPER_WEIGHT_DIRECTORY.get(model_id, [])
    if not coords:
        return {}

    down_proj = extract_down_proj_weights(weights)
    values: dict[tuple[int, int, int], float] = {}
    for layer, row, col in coords:
        if layer in down_proj:
            values[(layer, row, col)] = float(down_proj[layer][row, col])
    return values
