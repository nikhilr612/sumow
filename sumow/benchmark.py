"""Benchmarking infrastructure for reproducing paper results.

Generates tables matching the format of Table 1 in arXiv 2411.07191:
  Model × QuantMethod × ClipMethod → Perplexity
"""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import jaxtyped

from sumow.eval import perplexity
from sumow.identify import identify_super_weights
from sumow.model import LlamaModel, TransformerConfig
from sumow.quantize import quantize_weight_sw_aware


@dataclass(frozen=True)
class QuantConfig:
    """A single quantization configuration to benchmark."""

    name: str
    nbits: int = 4
    blocksize: int = 128
    clip_method: str = "zscore"
    clip_threshold: float = 3.0
    use_normal_float: bool = False
    scale_shift: bool = False
    retain_sw: bool = True


@dataclass(frozen=True)
class BenchmarkResult:
    """Result for a single quantization configuration."""

    config_name: str
    ppl: float
    ppl_delta: float  # difference from FP baseline
    mean_weight_error: float
    max_weight_error: float


@dataclass
class BenchmarkReport:
    """Full benchmark report across multiple configurations."""

    model_name: str
    baseline_ppl: float
    results: list[BenchmarkResult] = field(default_factory=list)

    def format_table(self) -> str:
        """Format results as an ASCII table matching paper Table 1 style."""
        lines = []
        lines.append(f"Model: {self.model_name}")
        lines.append(f"FP Baseline PPL: {self.baseline_ppl:.4f}")
        lines.append("")
        header = f"{'Config':<30} {'PPL':>10} {'ΔPPL':>10} {'MeanErr':>10} {'MaxErr':>10}"
        lines.append(header)
        lines.append("-" * len(header))
        for r in self.results:
            lines.append(
                f"{r.config_name:<30} {r.ppl:>10.4f} {r.ppl_delta:>+10.4f} "
                f"{r.mean_weight_error:>10.6f} {r.max_weight_error:>10.6f}"
            )
        return "\n".join(lines)


# Standard configurations matching paper's evaluation matrix
PAPER_CONFIGS = [
    QuantConfig("INT4", nbits=4, clip_method="none", clip_threshold=0.0,
                use_normal_float=False, retain_sw=False),
    QuantConfig("INT4+zscore", nbits=4, clip_method="zscore", clip_threshold=3.0,
                use_normal_float=False, retain_sw=False),
    QuantConfig("INT4+zscore+SW", nbits=4, clip_method="zscore", clip_threshold=3.0,
                use_normal_float=False, retain_sw=True),
    QuantConfig("INT8", nbits=8, clip_method="none", clip_threshold=0.0,
                use_normal_float=False, retain_sw=False),
    QuantConfig("INT8+zscore", nbits=8, clip_method="zscore", clip_threshold=3.0,
                use_normal_float=False, retain_sw=False),
    QuantConfig("INT8+zscore+SW", nbits=8, clip_method="zscore", clip_threshold=3.0,
                use_normal_float=False, retain_sw=True),
    QuantConfig("NF4", nbits=4, clip_method="none", clip_threshold=0.0,
                use_normal_float=True, retain_sw=False),
    QuantConfig("NF4+SW", nbits=4, clip_method="none", clip_threshold=0.0,
                use_normal_float=True, retain_sw=True),
]


@jaxtyped(typechecker=beartype)
def _eval_ppl(
    model: LlamaModel, tokens: jnp.ndarray
) -> float:
    """Forward pass → perplexity."""
    logits, _ = model(tokens)
    return perplexity(logits[:-1], tokens[1:])


def _quantize_down_projs(
    model: LlamaModel,
    config: TransformerConfig,
    qcfg: QuantConfig,
    sw_map: dict[int, list[tuple[int, int]]],
) -> tuple[LlamaModel, float, float]:
    """Quantize all down_proj weights. Returns (model, mean_err, max_err)."""
    total_err = 0.0
    max_err = 0.0
    count = 0

    model_q = model
    for layer_idx in range(config.num_hidden_layers):
        block = model_q.layers[layer_idx]
        w = block.mlp.down_proj
        coords = sw_map.get(layer_idx, []) if qcfg.retain_sw else []

        result = quantize_weight_sw_aware(
            w,
            sw_coords=coords,
            nbits=qcfg.nbits,
            blocksize=qcfg.blocksize,
            clip_method=qcfg.clip_method,
            clip_threshold=qcfg.clip_threshold,
            use_normal_float=qcfg.use_normal_float,
            scale_shift=qcfg.scale_shift,
        )

        err = jnp.abs(w - result.weight)
        total_err += float(jnp.mean(err))
        max_err = max(max_err, float(jnp.max(err)))
        count += 1

        block = eqx.tree_at(lambda b: b.mlp.down_proj, block, result.weight)
        model_q = eqx.tree_at(lambda m: m.layers[layer_idx], model_q, block)

    mean_err = total_err / max(count, 1)
    return model_q, mean_err, max_err


@beartype
def run_benchmark(
    model: LlamaModel,
    config: TransformerConfig,
    tokens: jnp.ndarray,
    sw_map: dict[int, list[tuple[int, int]]],
    quant_configs: list[QuantConfig] | None = None,
    model_name: str = "model",
) -> BenchmarkReport:
    """Run a full benchmark across multiple quantization configurations.

    Args:
        model: The Equinox LlamaModel to benchmark.
        config: Transformer config for the model.
        tokens: Token IDs for perplexity evaluation.
        sw_map: Super weight map {layer_idx: [(row, col), ...]}.
        quant_configs: List of QuantConfigs to evaluate. Defaults to PAPER_CONFIGS.
        model_name: Name for the report.

    Returns:
        BenchmarkReport with results for each configuration.
    """
    if quant_configs is None:
        quant_configs = PAPER_CONFIGS

    # Baseline FP perplexity
    baseline_ppl = _eval_ppl(model, tokens)
    report = BenchmarkReport(model_name=model_name, baseline_ppl=baseline_ppl)

    for qcfg in quant_configs:
        model_q, mean_err, max_err = _quantize_down_projs(
            model, config, qcfg, sw_map
        )
        ppl = _eval_ppl(model_q, tokens)
        result = BenchmarkResult(
            config_name=qcfg.name,
            ppl=ppl,
            ppl_delta=ppl - baseline_ppl,
            mean_weight_error=mean_err,
            max_weight_error=max_err,
        )
        report.results.append(result)

    return report


@beartype
def run_benchmark_with_identification(
    model: LlamaModel,
    config: TransformerConfig,
    tokens: jnp.ndarray,
    quant_configs: list[QuantConfig] | None = None,
    model_name: str = "model",
    spike_threshold: float = 100.0,
    spike_ratio: float = 6.0,
) -> BenchmarkReport:
    """Run benchmark with automatic SW identification.

    Performs a forward pass with activation capture, identifies super weights,
    then runs the benchmark.
    """
    # Identify super weights
    _, stats = model(tokens, capture_activations=True)
    sws = identify_super_weights(stats, spike_threshold, spike_ratio)

    sw_map: dict[int, list[tuple[int, int]]] = {}
    for sw in sws:
        sw_map.setdefault(sw.layer, []).append((sw.row, sw.col))

    return run_benchmark(
        model, config, tokens, sw_map,
        quant_configs=quant_configs,
        model_name=model_name,
    )
