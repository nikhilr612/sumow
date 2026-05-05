"""Load exported weights into Equinox and validate against PyTorch reference.

Compares forward-pass outputs between HF transformers (PyTorch) and our
Equinox LlamaModel to ensure weight loading produces identical results.

Usage:
    python tools/validate_model.py HuggingFaceTB/SmolLM2-135M --weights-dir weights/smollm2-135m
    python tools/validate_model.py HuggingFaceTB/SmolLM2-135M  # auto-export if needed
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


def _load_eqx_model(weights_dir: Path):
    """Load safetensors into our Equinox model."""
    import jax.numpy as jnp

    from sumow.model import LlamaModel, TransformerConfig, load_weights_into_model
    from sumow.model_io import load_model_weights

    config_path = weights_dir / "config.json"
    with open(config_path) as f:
        cfg_dict = json.load(f)

    config = TransformerConfig(
        vocab_size=cfg_dict["vocab_size"],
        hidden_size=cfg_dict["hidden_size"],
        intermediate_size=cfg_dict["intermediate_size"],
        num_hidden_layers=cfg_dict["num_hidden_layers"],
        num_attention_heads=cfg_dict["num_attention_heads"],
        num_key_value_heads=cfg_dict["num_key_value_heads"],
        max_position_embeddings=cfg_dict["max_position_embeddings"],
        rms_norm_eps=cfg_dict["rms_norm_eps"],
        rope_theta=cfg_dict["rope_theta"],
        tie_word_embeddings=cfg_dict.get("tie_word_embeddings", False),
    )

    typer.echo(f"Building Equinox model: {config.num_hidden_layers} layers, "
               f"hidden={config.hidden_size}")
    model = LlamaModel(config)

    typer.echo(f"Loading weights from {weights_dir} ...")
    weights = load_model_weights(str(weights_dir))
    typer.echo(f"  Loaded {len(weights)} tensors")
    model = load_weights_into_model(model, weights)

    return model, config


@app.command()
def validate(
    model_id: str = typer.Argument(help="HuggingFace model ID"),
    weights_dir: Path = typer.Option(
        None, "--weights-dir", "-w",
        help="Directory with exported safetensors (auto-exports if missing)"
    ),
    seq_len: int = typer.Option(32, help="Sequence length for comparison"),
) -> None:
    """Load weights into Equinox and compare forward pass with PyTorch."""
    import numpy as np

    # Auto-export if needed
    if weights_dir is None:
        safe_name = model_id.replace("/", "--")
        weights_dir = Path(f"weights/{safe_name}")

    if not (weights_dir / "model.safetensors").exists():
        typer.echo(f"Weights not found at {weights_dir}, exporting...")
        import subprocess, sys
        subprocess.run([
            sys.executable, "tools/export_hf_model.py",
            model_id, "--output-dir", str(weights_dir),
        ], check=True)

    # Load into Equinox
    eqx_model, config = _load_eqx_model(weights_dir)

    # Get PyTorch reference output
    typer.echo("\n--- PyTorch reference forward pass ---")
    import torch
    from transformers import AutoModelForCausalLM

    torch_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    torch_model.eval()

    input_ids = list(range(1, seq_len + 1))
    with torch.no_grad():
        torch_out = torch_model(torch.tensor([input_ids]))
        torch_logits = torch_out.logits[0].numpy()  # [seq, vocab]

    typer.echo(f"  PyTorch logits shape: {torch_logits.shape}")
    typer.echo(f"  PyTorch logits[0,:5]: {torch_logits[0, :5]}")

    # Equinox forward pass
    typer.echo("\n--- Equinox forward pass ---")
    import jax.numpy as jnp
    jax_ids = jnp.array(input_ids)
    jax_logits, _ = eqx_model(jax_ids)
    jax_logits_np = np.array(jax_logits)

    typer.echo(f"  Equinox logits shape: {jax_logits_np.shape}")
    typer.echo(f"  Equinox logits[0,:5]: {jax_logits_np[0, :5]}")

    # Compare
    typer.echo("\n--- Comparison ---")
    abs_diff = np.abs(torch_logits - jax_logits_np)
    typer.echo(f"  Max abs diff:  {abs_diff.max():.6e}")
    typer.echo(f"  Mean abs diff: {abs_diff.mean():.6e}")
    typer.echo(f"  Rel diff (L2): {np.linalg.norm(abs_diff) / np.linalg.norm(torch_logits):.6e}")

    # Check top-1 token agreement
    torch_top1 = np.argmax(torch_logits, axis=-1)
    jax_top1 = np.argmax(jax_logits_np, axis=-1)
    agreement = np.mean(torch_top1 == jax_top1)
    typer.echo(f"  Top-1 agreement: {agreement * 100:.1f}%")

    # Check KL divergence (more meaningful than raw abs diff for logits)
    from scipy.special import log_softmax as _log_softmax
    pt_lp = _log_softmax(torch_logits, axis=-1)
    jax_lp = _log_softmax(jax_logits_np, axis=-1)
    kl = np.sum(np.exp(pt_lp) * (pt_lp - jax_lp), axis=-1).mean()
    typer.echo(f"  Mean KL div:   {kl:.6e}")

    if agreement == 1.0 and kl < 0.01:
        typer.echo("\n✓ Model verified: 100% top-1 agreement, KL < 0.01.")
    elif agreement >= 0.95 and kl < 0.1:
        typer.echo("\n~ Model close: minor float precision differences.")
    else:
        typer.echo("\n✗ Model outputs differ significantly. Check weight loading.")
        raise typer.Exit(1)

    # Cleanup
    del torch_model, eqx_model
    typer.echo("\nDone.")


@app.command()
def identify_sw(
    model_id: str = typer.Argument(help="HuggingFace model ID"),
    weights_dir: Path = typer.Option(
        None, "--weights-dir", "-w",
        help="Directory with exported safetensors"
    ),
    seq_len: int = typer.Option(64, help="Sequence length for identification"),
    spike_threshold: float = typer.Option(100.0, help="Spike threshold"),
    spike_ratio: float = typer.Option(6.0, help="Spike ratio"),
) -> None:
    """Load real model and identify super weights."""
    import jax.numpy as jnp

    from sumow.identify import identify_super_weights

    if weights_dir is None:
        safe_name = model_id.replace("/", "--")
        weights_dir = Path(f"weights/{safe_name}")

    if not (weights_dir / "model.safetensors").exists():
        typer.echo(f"Weights not found. Run: python tools/export_hf_model.py {model_id} -o {weights_dir}")
        raise typer.Exit(1)

    eqx_model, config = _load_eqx_model(weights_dir)

    typer.echo(f"\nRunning forward pass (seq_len={seq_len}) with activation capture...")
    tokens = jnp.arange(1, seq_len + 1)
    logits, stats = eqx_model(tokens, capture_activations=True)

    typer.echo(f"\nActivation statistics per layer:")
    for s in stats:
        inp = f"in={s.input_max_magnitude:.2f} (ch {s.input_max_channel})"
        out = f"out={s.output_max_magnitude:.2f} (ch {s.output_max_channel})"
        typer.echo(f"  Layer {s.layer:>2}: {inp}, {out}")

    sws = identify_super_weights(stats, spike_threshold, spike_ratio)

    if sws:
        typer.echo(f"\nIdentified {len(sws)} super weight(s):")
        for sw in sws:
            val = float(eqx_model.layers[sw.layer].mlp.down_proj[sw.row, sw.col])
            typer.echo(
                f"  Layer {sw.layer}: down_proj[{sw.row}, {sw.col}] = {val:.4f}"
                f"  (in_mag={sw.input_magnitude:.2f}, out_mag={sw.output_magnitude:.2f})"
            )
    else:
        typer.echo(f"\nNo super weights detected with threshold={spike_threshold}, ratio={spike_ratio}")
        typer.echo("Try lowering --spike-threshold or --spike-ratio")


if __name__ == "__main__":
    app()
