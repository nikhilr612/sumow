"""Tests for the config module."""

from sumow.config import (
    SUPER_WEIGHT_DIRECTORY,
    ClipMethod,
    EvalConfig,
    IdentifyConfig,
    ModelConfig,
    QuantizationConfig,
)


def test_super_weight_directory_format():
    """Every entry is a list of (layer, row, col) integer tuples."""
    for model_id, coords in SUPER_WEIGHT_DIRECTORY.items():
        assert isinstance(model_id, str)
        assert len(coords) >= 1
        for layer, row, col in coords:
            assert isinstance(layer, int) and layer >= 0
            assert isinstance(row, int) and row >= 0
            assert isinstance(col, int) and col >= 0


def test_model_config_known_super_weights():
    cfg = ModelConfig(pretrained="huggyllama/llama-7B")
    sw = cfg.known_super_weights
    assert sw is not None
    assert sw == [(2, 3968, 7003)]


def test_model_config_unknown_model():
    cfg = ModelConfig(pretrained="unknown/model-1B")
    assert cfg.known_super_weights is None


def test_quantization_config_defaults():
    cfg = QuantizationConfig()
    assert cfg.nbits == 4
    assert cfg.blocksize == 128
    assert cfg.clip_method == ClipMethod.NONE
    assert cfg.restore_super_weight is True


def test_identify_config_defaults():
    cfg = IdentifyConfig()
    assert cfg.spike_threshold == 100.0
    assert isinstance(cfg.prompt, str) and len(cfg.prompt) > 0


def test_eval_config_defaults():
    cfg = EvalConfig()
    assert cfg.batch_size == 4
    assert cfg.max_samples is None
