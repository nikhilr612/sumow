"""Export HuggingFace model weights to safetensors for torch-free loading.

Downloads a pretrained model via transformers, extracts its state dict,
saves as safetensors files, and prints the TransformerConfig needed to
reconstruct the model in Equinox.

Usage:
    python tools/export_hf_model.py HuggingFaceTB/SmolLM2-135M --output-dir weights/smollm2-135m
    python tools/export_hf_model.py tiiuae/Falcon3-1B-Base --output-dir weights/falcon3-1b
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from omegaconf import OmegaConf

app = typer.Typer(add_completion=False)


@app.command()
def export(
    model_id: str = typer.Argument(help="HuggingFace model ID"),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Directory to save safetensors + config"
    ),
    dtype: str = typer.Option("float32", help="Weight dtype: float32, float16, bfloat16"),
) -> None:
    """Download a HF model and export weights as safetensors."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM

    typer.echo(f"Downloading {model_id} ...")
    hf_cfg = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, dtype),
    )
    state_dict = model.state_dict()

    typer.echo(f"Model type: {hf_cfg.model_type}")
    typer.echo(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    typer.echo(f"Weight keys: {len(state_dict)}")

    # Build our TransformerConfig
    rope_theta = getattr(hf_cfg, "rope_theta", 10000.0)
    rope_scaling = getattr(hf_cfg, "rope_scaling", None)
    if rope_scaling and isinstance(rope_scaling, dict):
        rope_theta = rope_scaling.get("rope_theta", rope_theta)

    eqx_config = {
        "model_id": model_id,
        "model_type": hf_cfg.model_type,
        "vocab_size": hf_cfg.vocab_size,
        "hidden_size": hf_cfg.hidden_size,
        "intermediate_size": hf_cfg.intermediate_size,
        "num_hidden_layers": hf_cfg.num_hidden_layers,
        "num_attention_heads": hf_cfg.num_attention_heads,
        "num_key_value_heads": getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads),
        "max_position_embeddings": hf_cfg.max_position_embeddings,
        "rms_norm_eps": getattr(hf_cfg, "rms_norm_eps", 1e-5),
        "rope_theta": rope_theta,
        "tie_word_embeddings": getattr(hf_cfg, "tie_word_embeddings", False),
    }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to float32 for safetensors (so JAX loads without dtype issues)
    # Clone all tensors to break shared memory (e.g. tied embeddings)
    st_dict = {}
    for k, v in state_dict.items():
        if dtype == "float32":
            st_dict[k] = v.float().clone().contiguous()
        else:
            st_dict[k] = v.clone().contiguous()

    safetensors_path = output_dir / "model.safetensors"
    save_file(st_dict, str(safetensors_path))
    typer.echo(f"Saved weights: {safetensors_path} ({safetensors_path.stat().st_size / 1e6:.1f} MB)")

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(eqx_config, indent=2))
    typer.echo(f"Saved config: {config_path}")

    # Also save as OmegaConf YAML
    yaml_path = output_dir / "config.yaml"
    cfg = OmegaConf.create(eqx_config)
    OmegaConf.save(cfg, str(yaml_path))
    typer.echo(f"Saved YAML:   {yaml_path}")

    typer.echo(f"\nTransformerConfig:")
    for k, v in eqx_config.items():
        typer.echo(f"  {k}: {v}")

    # Clean up torch model to free memory
    del model, state_dict, st_dict
    typer.echo("\nDone. Load in sumow with:")
    typer.echo(f"  from sumow.model import LlamaModel, TransformerConfig, load_weights_into_model")
    typer.echo(f"  from sumow.model_io import load_model_weights")
    typer.echo(f"  weights = load_model_weights('{output_dir}')")
    typer.echo(f"  model = LlamaModel(TransformerConfig(...))")
    typer.echo(f"  model = load_weights_into_model(model, weights)")


if __name__ == "__main__":
    app()
