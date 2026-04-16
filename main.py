"""sumow CLI — Super Weight Analysis for Large Language Models.

Subcommands:
    identify  — Identify super weights in a model (from known directory or via detection)
    quantize  — Apply SW-aware quantization to model weights
    eval      — Evaluate perplexity on quantized vs. original weights
"""

from __future__ import annotations

import argparse
import json
import sys

import jax.numpy as jnp

from sumow.config import (
    SUPER_WEIGHT_DIRECTORY,
    IdentifyConfig,
    ModelConfig,
    QuantizationConfig,
)
from sumow.identify import (
    LayerActivationStats,
    SuperWeight,
    identify_super_weights,
)
from sumow.model_io import extract_down_proj_weights, load_model_weights
from sumow.quantize import quantize_weight_sw_aware


def cmd_identify(args: argparse.Namespace) -> None:
    """List known super weights for a model."""
    model_id = args.model
    coords = SUPER_WEIGHT_DIRECTORY.get(model_id)
    if coords is None:
        print(f"No known super weights for '{model_id}'.")
        print("Known models:", ", ".join(sorted(SUPER_WEIGHT_DIRECTORY.keys())))
        sys.exit(1)

    print(f"Super weights for {model_id}:")
    for layer, row, col in coords:
        print(f"  layers[{layer}].mlp.down_proj.weight[{row}, {col}]")


def cmd_quantize(args: argparse.Namespace) -> None:
    """Apply SW-aware quantization to model weights on disk."""
    model_cfg = ModelConfig(pretrained=args.model)
    quant_cfg = QuantizationConfig(
        nbits=args.nbits,
        blocksize=args.blocksize,
        clip_threshold=args.z_threshold,
    )

    print(f"Loading weights from {args.weights_dir} ...")
    weights = load_model_weights(args.weights_dir)
    down_proj = extract_down_proj_weights(weights)

    sw_coords = model_cfg.known_super_weights or []
    # Group SW coords by layer
    sw_by_layer: dict[int, list[tuple[int, int]]] = {}
    for layer, row, col in sw_coords:
        sw_by_layer.setdefault(layer, []).append((row, col))

    total_layers = len(down_proj)
    quantized_count = 0
    for layer_idx, w in sorted(down_proj.items()):
        layer_sw = sw_by_layer.get(layer_idx, [])
        q_w = quantize_weight_sw_aware(
            w,
            layer_sw,
            nbits=quant_cfg.nbits,
            blocksize=quant_cfg.blocksize,
            z_threshold=quant_cfg.clip_threshold,
        )
        err = float(jnp.mean(jnp.abs(w - q_w)))
        print(f"  Layer {layer_idx}: mean abs error = {err:.6f}" +
              (f" (SW restored: {layer_sw})" if layer_sw else ""))
        quantized_count += 1

    print(f"\nQuantized {quantized_count}/{total_layers} down_proj layers "
          f"at {quant_cfg.nbits}-bit, blocksize={quant_cfg.blocksize}")


def cmd_eval(args: argparse.Namespace) -> None:
    """Placeholder for evaluation (requires full model forward pass)."""
    print("Evaluation requires a full model forward pass infrastructure.")
    print("Use the perplexity utilities in sumow.eval for custom pipelines:")
    print("  from sumow.eval import perplexity, cross_entropy_loss")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sumow",
        description="Super Weight Analysis for Large Language Models",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # identify
    p_id = subparsers.add_parser("identify", help="List known super weights")
    p_id.add_argument("model", help="HuggingFace model ID")
    p_id.set_defaults(func=cmd_identify)

    # quantize
    p_q = subparsers.add_parser("quantize", help="SW-aware quantization")
    p_q.add_argument("model", help="HuggingFace model ID")
    p_q.add_argument("weights_dir", help="Directory with .safetensors files")
    p_q.add_argument("--nbits", type=int, default=4)
    p_q.add_argument("--blocksize", type=int, default=128)
    p_q.add_argument("--z-threshold", type=float, default=9.0)
    p_q.set_defaults(func=cmd_quantize)

    # eval
    p_e = subparsers.add_parser("eval", help="Evaluate model quality")
    p_e.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

