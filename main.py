"""sumow CLI — Super Weight Analysis for Large Language Models.

Built with Typer + OmegaConf. Subcommands:
    identify   — Identify super weights in a model
    quantize   — Apply SW-aware quantization to model weights
    eval       — Evaluate perplexity on quantized vs. original weights
    benchmark  — Run quantization benchmark (Paper Table 1 format)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from omegaconf import OmegaConf

app = typer.Typer(
    name="sumow",
    help="Super Weight Analysis for Large Language Models (arXiv 2411.07191)",
    add_completion=False,
    no_args_is_help=True,
)


def _load_yaml_overrides(config_path: Path | None) -> dict:
    """Load OmegaConf YAML config file and return as dict."""
    if config_path is None:
        return {}
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _merge_config(yaml_overrides: dict, **cli_kwargs) -> dict:
    """Merge YAML config with CLI overrides. CLI takes precedence."""
    base = OmegaConf.create(yaml_overrides)
    cli = OmegaConf.create({k: v for k, v in cli_kwargs.items() if v is not None})
    merged = OmegaConf.merge(base, cli)
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]


@app.command()
def identify(
    model: Annotated[str, typer.Argument(help="HuggingFace model ID")],
    detect: Annotated[bool, typer.Option(
        "--detect", help="Run forward-pass detection instead of table lookup"
    )] = False,
    weights_dir: Annotated[Optional[Path], typer.Option(
        "--weights-dir", help="Directory with .safetensors (required for --detect)"
    )] = None,
    spike_threshold: Annotated[float, typer.Option(
        help="Minimum magnitude for spike detection"
    )] = 100.0,
    spike_ratio: Annotated[float, typer.Option(
        help="Minimum ratio vs. cross-layer median"
    )] = 6.0,
    config: Annotated[Optional[Path], typer.Option(
        "--config", "-c", help="OmegaConf YAML config file"
    )] = None,
) -> None:
    """Identify super weights in a model (table lookup or detection)."""
    from sumow.config import SUPER_WEIGHT_DIRECTORY

    yaml = _load_yaml_overrides(config)
    params = _merge_config(
        yaml.get("identify", {}),
        spike_threshold=spike_threshold,
        spike_ratio=spike_ratio,
    )

    if not detect:
        coords = SUPER_WEIGHT_DIRECTORY.get(model)
        if coords is None:
            typer.echo(f"No known super weights for '{model}'.")
            typer.echo(
                "Known models: "
                + ", ".join(sorted(SUPER_WEIGHT_DIRECTORY.keys()))
            )
            typer.echo("\nUse --detect with --weights-dir to run identification.")
            raise typer.Exit(1)

        typer.echo(f"Super weights for {model}:")
        for layer, row, col in coords:
            typer.echo(
                f"  layers[{layer}].mlp.down_proj.weight[{row}, {col}]"
            )
        return

    # Detection mode
    if weights_dir is None:
        typer.echo("--weights-dir is required when using --detect")
        raise typer.Exit(1)

    import jax.numpy as jnp

    from sumow.identify import identify_super_weights
    from sumow.model_io import extract_down_proj_weights, load_model_weights

    typer.echo(f"Loading weights from {weights_dir} ...")
    weights = load_model_weights(str(weights_dir))
    typer.echo(f"Loaded {len(weights)} weight tensors.")
    typer.echo("Detection via forward pass requires a full model. "
               "Use the Python API for dynamic identification.")


@app.command()
def quantize(
    model: Annotated[str, typer.Argument(help="HuggingFace model ID")],
    weights_dir: Annotated[Path, typer.Argument(help="Directory with .safetensors")],
    nbits: Annotated[int, typer.Option(help="Quantization bit-width")] = 4,
    blocksize: Annotated[int, typer.Option(help="Block size for blockwise quant")] = 128,
    clip_method: Annotated[str, typer.Option(
        help="Clipping method: none, zscore, tensor_percentage, block_percentage, iqr"
    )] = "zscore",
    clip_threshold: Annotated[float, typer.Option(
        help="Clipping threshold (z-score or percentage)"
    )] = 9.0,
    use_normal_float: Annotated[bool, typer.Option(
        "--nf/--int", help="Use NormalFloat (NF4/NF3) instead of INT"
    )] = False,
    retain_sw: Annotated[bool, typer.Option(
        "--retain-sw/--no-retain-sw", help="Restore super weights after quantization"
    )] = True,
    output_dir: Annotated[Optional[Path], typer.Option(
        "--output-dir", "-o", help="Save quantized weights to this directory"
    )] = None,
    config: Annotated[Optional[Path], typer.Option(
        "--config", "-c", help="OmegaConf YAML config file"
    )] = None,
) -> None:
    """Apply SW-aware quantization to model weights."""
    import jax.numpy as jnp

    from sumow.config import SUPER_WEIGHT_DIRECTORY, ModelConfig
    from sumow.model_io import extract_down_proj_weights, load_model_weights
    from sumow.quantize import quantize_weight_sw_aware

    yaml = _load_yaml_overrides(config)
    params = _merge_config(
        yaml.get("quantize", {}),
        nbits=nbits,
        blocksize=blocksize,
        clip_method=clip_method,
        clip_threshold=clip_threshold,
    )

    typer.echo(f"Loading weights from {weights_dir} ...")
    weights = load_model_weights(str(weights_dir))
    down_proj = extract_down_proj_weights(weights)

    # Get SW coordinates
    sw_coords = SUPER_WEIGHT_DIRECTORY.get(model, []) if retain_sw else []
    sw_by_layer: dict[int, list[tuple[int, int]]] = {}
    for layer, row, col in sw_coords:
        sw_by_layer.setdefault(layer, []).append((row, col))

    total_layers = len(down_proj)
    quantized_count = 0
    for layer_idx, w in sorted(down_proj.items()):
        layer_sw = sw_by_layer.get(layer_idx, [])
        result = quantize_weight_sw_aware(
            w,
            layer_sw,
            nbits=int(params.get("nbits", nbits)),
            blocksize=int(params.get("blocksize", blocksize)),
            clip_method=str(params.get("clip_method", clip_method)),
            clip_threshold=float(params.get("clip_threshold", clip_threshold)),
            use_normal_float=use_normal_float,
        )
        err = float(jnp.mean(jnp.abs(w - result.weight)))
        sw_info = f" (SW restored: {layer_sw})" if layer_sw else ""
        outlier_info = (
            f" ({result.num_outliers} outliers clipped)"
            if result.num_outliers
            else ""
        )
        typer.echo(
            f"  Layer {layer_idx}: mean abs error = {err:.6f}{sw_info}{outlier_info}"
        )
        quantized_count += 1

    typer.echo(
        f"\nQuantized {quantized_count}/{total_layers} down_proj layers "
        f"at {params.get('nbits', nbits)}-bit, "
        f"blocksize={params.get('blocksize', blocksize)}, "
        f"clip={params.get('clip_method', clip_method)}"
    )

    if output_dir:
        typer.echo(f"\nSaving quantized weights to {output_dir} ...")
        output_dir.mkdir(parents=True, exist_ok=True)
        typer.echo("(Serialization not yet implemented — coming soon)")


@app.command(name="eval")
def evaluate(
    model: Annotated[str, typer.Argument(help="HuggingFace model ID")],
    weights_dir: Annotated[Optional[Path], typer.Option(
        "--weights-dir", help="Directory with .safetensors"
    )] = None,
    config: Annotated[Optional[Path], typer.Option(
        "--config", "-c", help="OmegaConf YAML config file"
    )] = None,
) -> None:
    """Evaluate model quality (perplexity)."""
    typer.echo("Evaluation requires a full model forward pass.")
    typer.echo("Use the Python API for end-to-end evaluation:")
    typer.echo("")
    typer.echo("  from sumow import LlamaModel, perplexity")
    typer.echo("  logits, _ = model(tokens)")
    typer.echo("  ppl = perplexity(logits[:-1], tokens[1:])")
    typer.echo("")
    typer.echo("Or use the benchmark API:")
    typer.echo("  from sumow import run_benchmark, PAPER_CONFIGS")


@app.command()
def benchmark(
    config: Annotated[Optional[Path], typer.Option(
        "--config", "-c", help="OmegaConf YAML config file with benchmark settings"
    )] = None,
    nbits: Annotated[int, typer.Option(help="Quantization bit-width")] = 4,
    blocksize: Annotated[int, typer.Option(help="Block size")] = 32,
    hidden_size: Annotated[int, typer.Option(help="Model hidden size")] = 64,
    num_layers: Annotated[int, typer.Option(help="Number of transformer layers")] = 4,
    seq_len: Annotated[int, typer.Option(help="Sequence length for eval")] = 48,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    plant_sw: Annotated[bool, typer.Option(
        "--plant-sw/--no-plant-sw",
        help="Plant synthetic super weights for demo"
    )] = True,
) -> None:
    """Run quantization benchmark on a synthetic model (Paper Table 1 format)."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from sumow.benchmark import QuantConfig, run_benchmark as _run_bench
    from sumow.identify import identify_super_weights
    from sumow.model import LlamaModel, TransformerConfig

    yaml = _load_yaml_overrides(config)
    params = _merge_config(
        yaml.get("benchmark", {}),
        nbits=nbits,
        blocksize=blocksize,
        hidden_size=hidden_size,
        num_layers=num_layers,
        seq_len=seq_len,
        seed=seed,
    )

    hs = int(params.get("hidden_size", hidden_size))
    nl = int(params.get("num_layers", num_layers))
    sl = int(params.get("seq_len", seq_len))
    bs = int(params.get("blocksize", blocksize))
    sd = int(params.get("seed", seed))

    cfg = TransformerConfig(
        vocab_size=max(256, hs * 2),
        hidden_size=hs,
        intermediate_size=hs * 3 // 2,
        num_hidden_layers=nl,
        num_attention_heads=max(4, hs // 16),
        num_key_value_heads=max(2, hs // 32),
        max_position_embeddings=max(256, sl * 2),
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )

    typer.echo(f"Building synthetic model: {nl} layers, hidden={hs}")
    model = LlamaModel(cfg)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key = jax.random.PRNGKey(sd + i)
            new_leaves.append(jax.random.normal(key, leaf.shape) * 0.5)
        else:
            new_leaves.append(leaf)
    model = jax.tree.unflatten(treedef, new_leaves)

    sw_map: dict[int, list[tuple[int, int]]] = {}
    if plant_sw and nl >= 2:
        typer.echo("Planting synthetic super weights...")
        for layer_idx, val in [(0, 200.0), (min(2, nl - 1), -180.0)]:
            row, col = 5, 10
            block = model.layers[layer_idx]
            down = block.mlp.down_proj.at[row, col].set(val)
            block = eqx.tree_at(lambda b: b.mlp.down_proj, block, down)
            model = eqx.tree_at(
                lambda m, li=layer_idx: m.layers[li], model, block
            )
            sw_map.setdefault(layer_idx, []).append((row, col))
            typer.echo(f"  Layer {layer_idx}: down_proj[{row},{col}] = {val}")

    tokens = jnp.arange(sl)

    bench_configs = [
        QuantConfig("INT4 baseline", nbits=4, blocksize=bs,
                    clip_method="none", clip_threshold=0.0,
                    use_normal_float=False, retain_sw=False),
        QuantConfig("INT4 + zscore", nbits=4, blocksize=bs,
                    clip_method="zscore", clip_threshold=3.0,
                    use_normal_float=False, retain_sw=False),
        QuantConfig("INT4 + zscore + SW", nbits=4, blocksize=bs,
                    clip_method="zscore", clip_threshold=3.0,
                    use_normal_float=False, retain_sw=True),
        QuantConfig("NF4 baseline", nbits=4, blocksize=bs,
                    clip_method="none", clip_threshold=0.0,
                    use_normal_float=True, retain_sw=False),
        QuantConfig("NF4 + SW", nbits=4, blocksize=bs,
                    clip_method="none", clip_threshold=0.0,
                    use_normal_float=True, retain_sw=True),
        QuantConfig("INT8 baseline", nbits=8, blocksize=bs,
                    clip_method="none", clip_threshold=0.0,
                    use_normal_float=False, retain_sw=False),
        QuantConfig("INT8 + SW", nbits=8, blocksize=bs,
                    clip_method="none", clip_threshold=0.0,
                    use_normal_float=False, retain_sw=True),
    ]

    typer.echo(f"\nRunning benchmark ({len(bench_configs)} configs, seq_len={sl})...")
    report = _run_bench(
        model, cfg, tokens, sw_map,
        quant_configs=bench_configs,
        model_name=f"Synthetic-{nl}L-h{hs}",
    )

    typer.echo(f"\n{report.format_table()}")


@app.command()
def list_models() -> None:
    """List all models with known super weights."""
    from sumow.config import SUPER_WEIGHT_DIRECTORY

    typer.echo("Models with known super weights (Paper Table 2):\n")
    for model_id, coords in sorted(SUPER_WEIGHT_DIRECTORY.items()):
        typer.echo(f"  {model_id} ({len(coords)} super weight{'s' if len(coords) != 1 else ''})")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

