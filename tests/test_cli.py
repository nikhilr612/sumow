"""Tests for the Typer/OmegaConf CLI."""

import subprocess
import sys
from pathlib import Path

MAIN = Path(__file__).parent.parent / "main.py"


def _run(*args: str, expect_ok: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(MAIN), *args],
        capture_output=True, text=True, timeout=120,
    )
    if expect_ok:
        assert result.returncode == 0, (
            f"CLI failed: {' '.join(args)}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


class TestCLIHelp:
    """CLI help and basic commands."""

    def test_main_help(self):
        r = _run("--help")
        assert "identify" in r.stdout
        assert "quantize" in r.stdout
        assert "benchmark" in r.stdout

    def test_identify_help(self):
        r = _run("identify", "--help")
        assert "model" in r.stdout.lower()
        assert "--detect" in r.stdout

    def test_quantize_help(self):
        r = _run("quantize", "--help")
        assert "--nbits" in r.stdout
        assert "--clip-method" in r.stdout
        assert "--nf" in r.stdout

    def test_benchmark_help(self):
        r = _run("benchmark", "--help")
        assert "--hidden-size" in r.stdout
        assert "--num-layers" in r.stdout
        assert "--config" in r.stdout


class TestIdentifyCommand:
    """Test the identify subcommand."""

    def test_known_model(self):
        r = _run("identify", "meta-llama/Llama-2-7b-hf")
        assert "layers[1]" in r.stdout
        assert "down_proj" in r.stdout

    def test_unknown_model_fails(self):
        r = _run("identify", "nonexistent/model", expect_ok=False)
        assert r.returncode != 0
        assert "No known super weights" in r.stdout

    def test_phi3_has_6_sws(self):
        r = _run("identify", "microsoft/Phi-3-mini-4k-instruct")
        lines = [line for line in r.stdout.strip().split("\n") if "layers[" in line]
        assert len(lines) == 6


class TestListModels:
    """Test list-models subcommand."""

    def test_lists_all_models(self):
        r = _run("list-models")
        assert "meta-llama" in r.stdout
        assert "microsoft/Phi-3" in r.stdout
        assert "allenai/OLMo" in r.stdout

    def test_shows_sw_counts(self):
        r = _run("list-models")
        assert "super weight" in r.stdout


class TestEvalCommand:
    """Test the eval subcommand."""

    def test_eval_shows_usage(self):
        r = _run("eval", "test-model")
        assert "perplexity" in r.stdout.lower() or "Python API" in r.stdout


class TestBenchmarkCommand:
    """Test the benchmark subcommand."""

    def test_benchmark_default(self):
        r = _run("benchmark", "--num-layers", "2", "--hidden-size", "32", "--seq-len", "16")
        assert "Baseline PPL" in r.stdout
        assert "INT4" in r.stdout

    def test_benchmark_no_plant(self):
        r = _run("benchmark", "--no-plant-sw", "--num-layers", "2",
                 "--hidden-size", "32", "--seq-len", "16")
        assert "Baseline PPL" in r.stdout

    def test_benchmark_with_yaml(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text(
            "benchmark:\n"
            "  hidden_size: 32\n"
            "  num_layers: 2\n"
            "  seq_len: 16\n"
            "  blocksize: 16\n"
        )
        r = _run("benchmark", "--config", str(cfg))
        assert "Baseline PPL" in r.stdout
