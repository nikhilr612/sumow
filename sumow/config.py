"""Configuration models for sumow.

Pydantic models for model configuration, quantization parameters,
and the known super weight coordinate directory from the paper.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ClipMethod(str, Enum):
    """Outlier clipping strategy for quantization."""

    NONE = "none"
    ZSCORE = "zscore"
    TENSOR_PERCENTAGE = "tensor_percentage"
    BLOCK_PERCENTAGE = "block_percentage"


# ---------------------------------------------------------------------------
# Super Weight Directory (Table 2 from the paper)
# Coordinates are (layer, row, col) into `layers[layer].mlp.down_proj.weight`.
# ---------------------------------------------------------------------------

SUPER_WEIGHT_DIRECTORY: dict[str, list[tuple[int, int, int]]] = {
    "huggyllama/llama-7B": [(2, 3968, 7003)],
    "huggyllama/llama-13B": [(2, 2231, 2278), (2, 2231, 6939)],
    "huggyllama/llama-30B": [
        (3, 5633, 12817),
        (3, 5633, 17439),
        (10, 5633, 14386),
    ],
    "meta-llama/Llama-2-7b-hf": [(1, 2533, 7890)],
    "meta-llama/Llama-2-13b-hf": [(3, 4743, 7678)],
    "meta-llama/Meta-Llama-3-8B": [
        (1, 788, 2427),
        (1, 1384, 2427),
        (1, 4062, 2427),
    ],
    "mistralai/Mistral-7B-v0.1": [(1, 2070, 7310)],
    "allenai/OLMo-1B-0724-hf": [(1, 1764, 1710), (2, 1764, 8041)],
    "allenai/OLMo-7B-0724-hf": [
        (1, 269, 7467),
        (2, 269, 8275),
        (7, 269, 453),
        (24, 269, 2300),
    ],
    "microsoft/Phi-3-mini-4k-instruct": [
        (2, 525, 808),
        (2, 1693, 808),
        (2, 1113, 808),
        (4, 525, 2723),
        (4, 1113, 2723),
        (4, 1693, 2723),
    ],
}


class QuantizationConfig(BaseModel):
    """Parameters for weight or activation quantization."""

    nbits: int = Field(default=4, ge=2, le=16, description="Quantization bit-width")
    blocksize: int = Field(
        default=128,
        ge=1,
        description="Block size for blockwise quantization. Use -1 for per-tensor.",
    )
    clip_method: ClipMethod = Field(
        default=ClipMethod.NONE,
        description="Outlier clipping strategy before quantization",
    )
    clip_threshold: float = Field(
        default=9.0,
        description="Threshold for clipping (z-score value or top-k percentage)",
    )
    restore_super_weight: bool = Field(
        default=True,
        description="Restore super weight in fp16 after quantization",
    )


class ModelConfig(BaseModel):
    """Configuration for a model to be analyzed."""

    pretrained: str = Field(
        description="HuggingFace model identifier (e.g. 'huggyllama/llama-7B')"
    )
    dtype: Literal["float16", "bfloat16", "float32"] = Field(default="float16")
    down_proj_key: str = Field(
        default="mlp.down_proj",
        description="Key path to the down projection module in each layer",
    )

    @property
    def known_super_weights(self) -> list[tuple[int, int, int]] | None:
        """Return known super weight coordinates if this model is in the directory."""
        return SUPER_WEIGHT_DIRECTORY.get(self.pretrained)


class IdentifyConfig(BaseModel):
    """Parameters for super weight identification."""

    spike_threshold: float = Field(
        default=100.0,
        description="Minimum absolute activation magnitude to count as a spike",
    )
    spike_ratio: float = Field(
        default=6.0,
        description="Minimum ratio of spike to median magnitude across layers",
    )
    prompt: str = Field(
        default="Apple Inc. is a worldwide tech company.",
        description="Prompt for the single-pass identification forward pass",
    )


class EvalConfig(BaseModel):
    """Parameters for model evaluation."""

    batch_size: int = Field(default=4, ge=1)
    max_samples: int | None = Field(
        default=None,
        description="Cap the number of evaluation samples (None = full dataset)",
    )
