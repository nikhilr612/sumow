# Copilot Instructions

## Project Overview

**sumow** is a JAX/Equinox reimplementation of "The Super Weight in Large Language Models" (ICLR 2025, arXiv 2411.07191). It provides tools for identifying, analyzing, and preserving super weights during LLM quantization.

### Repository Structure

- **`sumow/`**: Main JAX package with config, quantization, identification, model I/O, and evaluation modules.
- **`llmsuperweight/`**: Git submodule with the original paper's PyTorch/HuggingFace reference implementation (read-only).
- **`paper/`**: LaTeX source for the ICLR 2025 paper (read-only reference).
- **`tests/`**: pytest test suite mirroring the `sumow/` package structure.

## Build, Test, and Run

This project uses [uv](https://docs.astral.sh/uv/) for dependency management (Python 3.12+).

```sh
uv sync --extra dev          # install all dependencies including test deps
uv run python -m pytest tests/ -v          # run full test suite
uv run python -m pytest tests/test_quantize.py -k "test_super_weight_restored"  # single test
uv run python main.py identify "huggyllama/llama-7B"   # CLI: list known super weights
uv run python main.py quantize MODEL WEIGHTS_DIR       # CLI: SW-aware quantization
```

## Key Conventions

- **JAX stack, not PyTorch**: Uses JAX + jaxtyping + beartype. Use `jax.numpy` instead of `torch`, Equinox modules for neural network components. The `llmsuperweight/` submodule is PyTorch — don't mix.
- **Type annotations**: Use `jaxtyping` for array shape annotations (e.g., `Float[Array, "batch seq hidden"]`) and `beartype` for runtime checking.
- **Config**: `pydantic` models for configuration validation (`sumow/config.py`). `omegaconf` for YAML config loading.
- **Pure functions**: Quantization and identification routines are pure functions operating on JAX arrays. No global state.
- **Testing**: Every module has a corresponding `tests/test_*.py`. Tests use small synthetic data, not full models.

## Domain Context

- A **super weight** is a single scalar in an LLM's `mlp.down_proj` weight matrix that, when zeroed, catastrophically destroys text generation (perplexity increases 3 orders of magnitude).
- A **super activation** is the outsized hidden-state activation induced by a super weight.
- Super weights are identified via activation spike detection in a single forward pass (no training data needed).
- Known super weight coordinates for common models are in `sumow/config.py::SUPER_WEIGHT_DIRECTORY`.
- The quantization approach: clip outliers → round-to-nearest quantize → restore super weights in original precision.
