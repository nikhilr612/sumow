"""Tests for the benchmarking infrastructure."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sumow.benchmark import (
    BenchmarkReport,
    BenchmarkResult,
    PAPER_CONFIGS,
    QuantConfig,
    run_benchmark,
    run_benchmark_with_identification,
)
from sumow.model import LlamaModel, TransformerConfig

import equinox as eqx

BENCH_CONFIG = TransformerConfig(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
)


def _make_model(seed: int = 42) -> LlamaModel:
    model = LlamaModel(BENCH_CONFIG)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key = jax.random.PRNGKey(seed + i)
            new_leaves.append(jax.random.normal(key, leaf.shape) * 0.02)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


def _plant_sw(model, layer, row, col, value):
    block = model.layers[layer]
    down = block.mlp.down_proj.at[row, col].set(value)
    block = eqx.tree_at(lambda b: b.mlp.down_proj, block, down)
    return eqx.tree_at(lambda m: m.layers[layer], model, block)


class TestQuantConfig:
    """Test QuantConfig dataclass."""

    def test_defaults(self):
        cfg = QuantConfig(name="test")
        assert cfg.nbits == 4
        assert cfg.blocksize == 128
        assert cfg.retain_sw is True

    def test_paper_configs_count(self):
        assert len(PAPER_CONFIGS) >= 8

    def test_paper_configs_have_unique_names(self):
        names = [c.name for c in PAPER_CONFIGS]
        assert len(names) == len(set(names))


class TestBenchmarkReport:
    """Test report formatting."""

    def test_format_table(self):
        report = BenchmarkReport(model_name="test", baseline_ppl=10.0)
        report.results.append(
            BenchmarkResult("INT4", ppl=12.0, ppl_delta=2.0,
                            mean_weight_error=0.01, max_weight_error=0.1)
        )
        table = report.format_table()
        assert "test" in table
        assert "INT4" in table
        assert "10.0000" in table
        assert "+2.0000" in table

    def test_empty_report(self):
        report = BenchmarkReport(model_name="empty", baseline_ppl=5.0)
        table = report.format_table()
        assert "empty" in table


class TestRunBenchmark:
    """Test the main benchmark runner."""

    @pytest.fixture
    def model(self):
        m = _make_model(seed=100)
        return _plant_sw(m, layer=1, row=5, col=10, value=100.0)

    @pytest.fixture
    def tokens(self):
        return jnp.arange(32)

    def test_basic_benchmark(self, model, tokens):
        """Run benchmark with a few configs."""
        configs = [
            QuantConfig("INT4", nbits=4, clip_method="none", clip_threshold=0.0,
                        use_normal_float=False, retain_sw=False),
            QuantConfig("INT4+SW", nbits=4, clip_method="none", clip_threshold=0.0,
                        use_normal_float=False, retain_sw=True),
        ]
        sw_map = {1: [(5, 10)]}
        report = run_benchmark(model, BENCH_CONFIG, tokens, sw_map,
                               quant_configs=configs, model_name="tiny")

        assert report.model_name == "tiny"
        assert np.isfinite(report.baseline_ppl)
        assert len(report.results) == 2
        for r in report.results:
            assert np.isfinite(r.ppl)
            assert r.mean_weight_error >= 0
            assert r.max_weight_error >= 0

    def test_sw_retention_reduces_delta(self, model, tokens):
        """SW-aware config should have smaller PPL delta."""
        configs = [
            QuantConfig("no-SW", nbits=4, clip_method="zscore",
                        clip_threshold=3.0, retain_sw=False),
            QuantConfig("with-SW", nbits=4, clip_method="zscore",
                        clip_threshold=3.0, retain_sw=True),
        ]
        sw_map = {1: [(5, 10)]}
        report = run_benchmark(model, BENCH_CONFIG, tokens, sw_map,
                               quant_configs=configs)

        r_no = report.results[0]
        r_yes = report.results[1]
        assert abs(r_yes.ppl_delta) <= abs(r_no.ppl_delta)

    def test_paper_configs_all_produce_results(self, model, tokens):
        """All paper configs produce finite results."""
        # Use smaller blocksize for tiny model
        configs = [
            QuantConfig(c.name, nbits=c.nbits, blocksize=32,
                        clip_method=c.clip_method, clip_threshold=c.clip_threshold,
                        use_normal_float=c.use_normal_float, retain_sw=c.retain_sw)
            for c in PAPER_CONFIGS
        ]
        sw_map = {1: [(5, 10)]}
        report = run_benchmark(model, BENCH_CONFIG, tokens, sw_map,
                               quant_configs=configs)

        assert len(report.results) == len(PAPER_CONFIGS)
        for r in report.results:
            assert np.isfinite(r.ppl), f"{r.config_name} PPL not finite"

    def test_format_produces_table(self, model, tokens):
        """Formatted table has correct structure."""
        configs = [
            QuantConfig("INT4", nbits=4, clip_method="none", clip_threshold=0.0,
                        use_normal_float=False, retain_sw=False),
        ]
        report = run_benchmark(model, BENCH_CONFIG, tokens, {}, quant_configs=configs)
        table = report.format_table()
        assert "INT4" in table
        assert "Baseline" in table


class TestRunBenchmarkWithIdentification:
    """Test auto-identification + benchmark."""

    def test_auto_identify_and_benchmark(self):
        """Full: forward pass → identify → benchmark."""
        model = _make_model(seed=200)
        model = _plant_sw(model, layer=1, row=5, col=10, value=100.0)
        tokens = jnp.arange(32)

        configs = [
            QuantConfig("INT4", nbits=4, blocksize=32, clip_method="none",
                        clip_threshold=0.0, retain_sw=False),
            QuantConfig("INT4+SW", nbits=4, blocksize=32, clip_method="none",
                        clip_threshold=0.0, retain_sw=True),
        ]

        report = run_benchmark_with_identification(
            model, BENCH_CONFIG, tokens,
            quant_configs=configs,
            model_name="auto-id",
            spike_threshold=1e-6,
            spike_ratio=2.0,
        )

        assert report.model_name == "auto-id"
        assert len(report.results) == 2
        for r in report.results:
            assert np.isfinite(r.ppl)
