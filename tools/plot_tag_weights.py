#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import get_timestep_embedding
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.tag import TimeAdaptiveGating


def extract_tag_state(state_dict):
    tag_state = {}
    prefix = "tag_module."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            tag_state[k[len(prefix) :]] = v
    if not tag_state:
        raise KeyError("No tag_module.* weights found in checkpoint.")
    return tag_state


def infer_tag_dims(tag_state):
    in_dim = int(tag_state["mlp.0.weight"].shape[1])
    num_blocks = int(tag_state["mlp.2.weight"].shape[0])
    return in_dim, num_blocks


def map_sd320_to_tag_input(sd320_emb, full_state_dict, target_dim):
    if target_dim == 320:
        return sd320_emb

    w1 = full_state_dict.get("time_embedding.linear_1.weight")
    b1 = full_state_dict.get("time_embedding.linear_1.bias")
    w2 = full_state_dict.get("time_embedding.linear_2.weight")
    b2 = full_state_dict.get("time_embedding.linear_2.bias")

    can_use_time_mlp = (
        w1 is not None
        and b1 is not None
        and w2 is not None
        and b2 is not None
        and w1.shape[1] == 320
        and w2.shape[0] == target_dim
    )
    if can_use_time_mlp:
        x = F.linear(sd320_emb, w1, b1)
        x = F.silu(x)
        x = F.linear(x, w2, b2)
        return x

    if target_dim > 320:
        repeat = (target_dim + 319) // 320
        x = sd320_emb.repeat(1, repeat)
        return x[:, :target_dim]
    return sd320_emb[:, :target_dim]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="/work/outputs/exp_DT_CAFE_TAG_patch512/controlnet/diffusion_pytorch_model.safetensors",
        help="Path to controlnet safetensors checkpoint.",
    )
    ap.add_argument("--num-steps", type=int, default=1000, help="Number of sampled timesteps.")
    ap.add_argument("--output", type=str, default="tag_scales_plot.png", help="Output plot path.")
    ap.add_argument(
        "--x-scale",
        type=str,
        choices=["linear", "log", "symlog"],
        default="linear",
        help="X-axis scale for timestep visualization.",
    )
    ap.add_argument(
        "--zoom-steps",
        type=int,
        default=50,
        help="Right subplot zoom range near small timesteps.",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = load_file(str(ckpt_path))
    tag_state = extract_tag_state(state)
    time_embed_dim, num_blocks = infer_tag_dims(tag_state)

    tag = TimeAdaptiveGating(time_embed_dim=time_embed_dim, num_blocks=num_blocks)
    tag.load_state_dict(tag_state, strict=True)
    tag.eval()

    timesteps = torch.linspace(0, 1000, steps=args.num_steps, dtype=torch.float32)
    sd320 = get_timestep_embedding(
        timesteps,
        embedding_dim=320,
        flip_sin_to_cos=True,
        downscale_freq_shift=1,
    )
    tag_input = map_sd320_to_tag_input(sd320, state, target_dim=time_embed_dim)

    with torch.no_grad():
        scales = tag(tag_input).cpu()

    # X axis is shown from 1000 -> 0 as requested.
    x = timesteps.flip(0).numpy()
    y = scales.flip(0).numpy()

    # Log scale cannot include zero directly, so we shift to (t + 1).
    if args.x_scale == "log":
        x_plot = x + 1.0
        x_label = "Diffusion Timestep (log, t+1)"
        left_xlim = (1001, 1)
        right_high = max(2, min(args.zoom_steps, 1000) + 1)
        right_xlim = (right_high, 1)
    else:
        x_plot = x
        x_label = "Diffusion Timestep"
        left_xlim = (1000, 0)
        right_high = max(1, min(args.zoom_steps, 1000))
        right_xlim = (right_high, 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax in axes:
        for i in range(y.shape[1]):
            ax.plot(x_plot, y[:, i], linewidth=1.2, label=f"gate_{i}")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        if args.x_scale == "log":
            ax.set_xscale("log")
        elif args.x_scale == "symlog":
            ax.set_xscale("symlog", linthresh=1.0)

    axes[0].set_xlim(*left_xlim)
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Gate Scale")
    axes[0].set_title("TAG Gate Scales (Full Range)")
    axes[0].legend(ncol=3, fontsize=8)

    axes[1].set_xlim(*right_xlim)
    axes[1].set_xlabel(x_label)
    axes[1].set_title(f"TAG Gate Scales (Zoom <= {max(1, min(args.zoom_steps, 1000))})")

    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
