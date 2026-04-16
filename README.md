# sumow

Pure JAX/Equinox implementation of **"The Super Weight in Large Language Models"** (ICLR 2025, [arXiv 2411.07191](https://arxiv.org/abs/2411.07191)).

## What is sumow?

Large language models contain a handful of **super weights** — individual scalar parameters (typically 1–6 per model) that, when zeroed, catastrophically degrade model output. These weights always appear in `mlp.down_proj` matrices of early layers and induce **super activations**: outlier channels with magnitudes orders of magnitude above the mean.

`sumow` provides:
- **Identification**: Detect super weights via a single data-free forward pass
- **Quantization**: SW-aware INT4/INT8/NF4 quantization that retains super weights in full precision
- **Evaluation**: Perplexity measurement and benchmark infrastructure matching the paper's Table 1
- **Model**: Equinox Llama-family transformer (GQA, RoPE) for end-to-end experiments

## Installation

```bash
pip install -e .
# or with uv:
uv pip install -e .
```

Requires: JAX, Equinox, jaxtyping, beartype, pydantic, omegaconf.

## Quick Start

### Identify Super Weights

```python
from sumow import LlamaModel, TransformerConfig, identify_super_weights

# Create/load a model
config = TransformerConfig(vocab_size=32000, hidden_size=4096, ...)
model = LlamaModel(config)

# Forward pass with activation capture
tokens = jnp.array([1, 2, 3, 4, 5])
logits, stats = model(tokens, capture_activations=True)

# Identify super weights from activation spikes
sws = identify_super_weights(stats, spike_threshold=100.0, spike_ratio=6.0)
for sw in sws:
    print(f"Layer {sw.layer}: down_proj[{sw.row}, {sw.col}]")
```

### SW-Aware Quantization

```python
from sumow import quantize_weight_sw_aware

# Quantize a weight matrix, preserving super weights in full precision
result = quantize_weight_sw_aware(
    weight_matrix,
    sw_coords=[(sw.row, sw.col) for sw in sws],
    nbits=4,
    blocksize=128,
    clip_method="zscore",
    clip_threshold=3.0,
)
# result.weight has super weights restored exactly
```

### Benchmarking

```python
from sumow import run_benchmark, PAPER_CONFIGS, QuantConfig

report = run_benchmark(
    model, config, tokens,
    sw_map={1: [(5, 10)]},
    quant_configs=PAPER_CONFIGS,
    model_name="my-model",
)
print(report.format_table())
```

### CLI

```bash
# List known super weights for a model
python main.py identify meta-llama/Llama-2-7b-hf

# Quantize model weights
python main.py quantize meta-llama/Llama-2-7b-hf /path/to/weights --nbits 4
```

## Modules

| Module | Description |
|--------|-------------|
| `sumow.model` | Equinox Llama transformer (RMSNorm, RoPE, GQA, MLP) |
| `sumow.identify` | Super weight identification via activation spike detection |
| `sumow.quantize` | INT4/INT8/NF4 blockwise quantization with SW/SA retention |
| `sumow.benchmark` | Benchmark infrastructure matching paper Table 1 |
| `sumow.eval` | Cross-entropy loss and perplexity computation |
| `sumow.config` | Pydantic configs + SUPER_WEIGHT_DIRECTORY (paper Table 2) |
| `sumow.model_io` | Safetensors → JAX weight loading |

## Quantization Methods

- **INT4/INT8**: Round-to-nearest (RTN) blockwise quantization
- **NF4/NF3**: Normal float quantization (optimal for normally distributed weights)
- **Clipping**: Z-score, tensor/block percentage, IQR outlier clipping
- **Scale-shift**: Alternative rounding scheme
- **SW-aware**: Super weights excluded from quantization grid, restored in fp32
- **SA-aware**: Super activations replaced with median before activation quantization

## Paper Reference

```bibtex
@inproceedings{superweight2025,
  title={The Super Weight in Large Language Models},
  author={Mengxia Yu and De Wang and Qi Shan and Colorado Reed and Alvin Wan},
  booktitle={ICLR},
  year={2025}
}
```

## Testing

```bash
# Run all 246 tests
uv run python -m pytest tests/ -v

# Specific test suites
uv run python -m pytest tests/test_model.py           # Transformer model
uv run python -m pytest tests/test_quantize.py         # Quantization
uv run python -m pytest tests/test_reference_equiv.py  # JAX vs PyTorch reference
uv run python -m pytest tests/test_ablation.py         # Ablation studies
uv run python -m pytest tests/test_benchmark.py        # Benchmark infra
uv run python -m pytest tests/test_smoke.py            # End-to-end pipeline
uv run python -m pytest tests/test_harden.py           # Edge cases
```

## License

See [LICENSE](LICENSE).
