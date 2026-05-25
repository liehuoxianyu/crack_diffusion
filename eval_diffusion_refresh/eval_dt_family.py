#!/usr/bin/env python3
"""Evaluate the DT-family and first-ablation diffusion runs.

Outputs are intentionally compact:
  OUTDIR/id{id}/sd_only/gs7.5.png
  OUTDIR/id{id}/controlnet_dt/step{1000,2000}/gs7.5_cs1.0.png
  OUTDIR/id{id}/controlnet_dt_tag/step{1000,2000}/gs7.5_cs1.0.png
  OUTDIR/id{id}/controlnet_dt_cafe_tag/step{1000,2000}/gs7.5_cs1.0.png
  OUTDIR/id{id}/controlnet_topology/step{1000,2000}/gs7.5_cs1.0.png
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionPipeline, UniPCMultistepScheduler
from PIL import Image
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.cafe import CAFEEmbedding
from models.tag import inject_tag_into_controlnet


@dataclass(frozen=True)
class EvalRun:
    mode: str
    run_dir: str
    use_cafe: bool
    use_tag: bool
    cond_dir: str


AVAILABLE_RUNS = (
    EvalRun("dt", "/work/outputs/exp_dt_patch512", use_cafe=False, use_tag=False, cond_dir="/CrackTree260/cond_dt"),
    EvalRun(
        "dt_updated_prompt",
        "/work/outputs/exp_DT_updated_prompt_patch512",
        use_cafe=False,
        use_tag=False,
        cond_dir="/CrackTree260/cond_dt",
    ),
    EvalRun("dt_cafe", "/work/outputs/exp_DT_CAFE_patch512", use_cafe=True, use_tag=False, cond_dir="/CrackTree260/cond_dt"),
    EvalRun("dt_tag", "/work/outputs/exp_DT_TAG_patch512", use_cafe=False, use_tag=True, cond_dir="/CrackTree260/cond_dt"),
    EvalRun(
        "dt_cafe_tag",
        "/work/outputs/exp_DT_CAFE_TAG_patch512",
        use_cafe=True,
        use_tag=True,
        cond_dir="/CrackTree260/cond_dt",
    ),
    EvalRun(
        "topology",
        "/work/outputs/exp_TOPOLOGY_patch512",
        use_cafe=False,
        use_tag=False,
        cond_dir="/CrackTree260/cond_topology",
    ),
    EvalRun(
        "appearance",
        "/work/outputs/exp_APPEARANCE_patch512",
        use_cafe=False,
        use_tag=False,
        cond_dir="/CrackTree260/cond_appearance",
    ),
    EvalRun(
        "topology_weighted",
        "/work/outputs/exp_TOPOLOGY_WEIGHTED_patch512",
        use_cafe=False,
        use_tag=False,
        cond_dir="/CrackTree260/cond_topology",
    ),
)

DEFAULT_MODES = ("dt", "dt_tag", "dt_cafe_tag")


def parse_int_list(raw: list[str] | str) -> list[int]:
    if isinstance(raw, str):
        parts = raw.replace(",", " ").split()
    else:
        parts = []
        for item in raw:
            parts.extend(item.replace(",", " ").split())
    return [int(x.strip()) for x in parts if x.strip()]


def read_ids(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def selected_runs(modes: list[str] | None) -> tuple[EvalRun, ...]:
    if not modes:
        modes = list(DEFAULT_MODES)
    by_mode = {run.mode: run for run in AVAILABLE_RUNS}
    unknown = [mode for mode in modes if mode not in by_mode]
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}. Available modes: {sorted(by_mode)}")
    return tuple(by_mode[mode] for mode in modes)


def save_settings(path: Path, args: argparse.Namespace, ids: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"base_model: {args.base_model}",
        f"controlnet_base_model: {args.controlnet_base_model}",
        f"eval_ids: {args.eval_ids}",
        f"ids: {ids}",
        f"ckpts: {args.ckpts}",
        f"prompt: {args.prompt}",
        f"negative_prompt: {args.negative_prompt}",
        f"num_inference_steps: {args.num_inference_steps}",
        f"guidance_scale: {args.guidance_scale}",
        f"control_scale: {args.control_scale}",
        f"seed: {args.seed}",
        "runs:",
    ]
    for run in selected_runs(args.modes):
        lines.append(
            f"  - {run.mode}: {run.run_dir}, cafe={run.use_cafe}, tag={run.use_tag}, cond_dir={run.cond_dir}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def device_and_dtype(use_fp16: bool) -> tuple[str, torch.dtype]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if use_fp16 and device == "cuda" else torch.float32
    return device, dtype


def maybe_enable_xformers(pipe):
    if pipe.device.type == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            print(f"xformers unavailable, continuing without it: {exc}")


def controlnet_dir(run_dir: str, step: int) -> Path:
    p = Path(run_dir) / f"checkpoint-{step}" / "controlnet"
    if not (p / "config.json").exists() or not (p / "diffusion_pytorch_model.safetensors").exists():
        raise FileNotFoundError(f"invalid controlnet checkpoint: {p}")
    return p


def load_controlnet(run: EvalRun, step: int, args: argparse.Namespace, device: str, dtype: torch.dtype):
    ckpt_dir = controlnet_dir(run.run_dir, step)
    controlnet = ControlNetModel.from_pretrained(args.controlnet_base_model, torch_dtype=dtype)

    if run.use_cafe:
        controlnet.controlnet_cond_embedding = CAFEEmbedding()
        controlnet.controlnet_cond_embedding.to(device=next(controlnet.parameters()).device, dtype=dtype)
    if run.use_tag:
        controlnet = inject_tag_into_controlnet(controlnet)

    state = load_file(str(ckpt_dir / "diffusion_pytorch_model.safetensors"))
    missing, unexpected = controlnet.load_state_dict(state, strict=False)
    print(
        f"loaded {run.mode} step{step}: missing={len(missing)} unexpected={len(unexpected)} "
        f"from {ckpt_dir}"
    )
    controlnet = controlnet.to(device=device, dtype=dtype)
    return controlnet


def load_sd_pipe(args: argparse.Namespace, device: str, dtype: torch.dtype):
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    maybe_enable_xformers(pipe)
    return pipe


def load_cn_pipe(run: EvalRun, step: int, args: argparse.Namespace, device: str, dtype: torch.dtype):
    controlnet = load_controlnet(run, step, args, device, dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    maybe_enable_xformers(pipe)
    return pipe


@torch.inference_mode()
def infer_sd(pipe, args: argparse.Namespace, seed: int, device: str):
    generator = torch.Generator(device=device).manual_seed(seed)
    return pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]


@torch.inference_mode()
def infer_cn(pipe, cond_img: Image.Image, args: argparse.Namespace, seed: int, device: str):
    generator = torch.Generator(device=device).manual_seed(seed)
    return pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        image=cond_img,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.control_scale,
        generator=generator,
    ).images[0]


def prepare_id_dirs(ids: list[str], args: argparse.Namespace) -> dict[str, Path]:
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)
    id_dirs = {}
    for image_id in ids:
        id_dir = out_root / f"id{image_id}"
        id_dir.mkdir(parents=True, exist_ok=True)
        id_dirs[image_id] = id_dir
    return id_dirs


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--controlnet-base-model", default="lllyasviel/sd-controlnet-seg")
    ap.add_argument("--eval-ids", default="/CrackTree260/eval_ids.txt")
    ap.add_argument("--outdir", default="/work/outputs/diffusion_eval_dt_family")
    ap.add_argument("--ckpts", nargs="+", default=["1000", "2000"])
    ap.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Subset of modes: dt dt_updated_prompt dt_cafe dt_tag dt_cafe_tag topology appearance topology_weighted",
    )
    ap.add_argument("--ids", nargs="+", default=None)
    ap.add_argument("--limit-ids", type=int, default=None)
    ap.add_argument(
        "--prompt",
        default="a photo of pavement crack, realistic texture, high detail, natural shadowing, realistic lighting",
    )
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--num-inference-steps", type=int, default=25)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--control-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp32", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    args.ckpts = parse_int_list(args.ckpts)
    ids = args.ids if args.ids else read_ids(args.eval_ids)
    if args.limit_ids is not None:
        ids = ids[: args.limit_ids]
    if not ids:
        raise RuntimeError("no eval ids selected")
    runs = selected_runs(args.modes)

    device, dtype = device_and_dtype(use_fp16=not args.fp32)
    print(f"device={device} dtype={dtype} ids={len(ids)} ckpts={args.ckpts} modes={[run.mode for run in runs]}")
    id_dirs = prepare_id_dirs(ids, args)
    save_settings(Path(args.outdir) / "settings.txt", args, ids)

    sd_pipe = load_sd_pipe(args, device, dtype)
    for idx, image_id in enumerate(ids):
        out_dir = id_dirs[image_id] / "sd_only"
        out_dir.mkdir(parents=True, exist_ok=True)
        im = infer_sd(sd_pipe, args, args.seed + idx, device)
        im.save(out_dir / f"gs{args.guidance_scale}.png")
    del sd_pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    for run in runs:
        for step in args.ckpts:
            pipe = load_cn_pipe(run, step, args, device, dtype)
            for idx, image_id in enumerate(ids):
                cond_path = Path(run.cond_dir) / f"{image_id}.png"
                if not cond_path.exists():
                    raise FileNotFoundError(f"missing condition image for {run.mode}: {cond_path}")
                cond_img = Image.open(cond_path).convert("RGB")
                if step == args.ckpts[0]:
                    cond_img.save(id_dirs[image_id] / f"cond_{run.mode}.png")
                out_dir = id_dirs[image_id] / f"controlnet_{run.mode}" / f"step{step}"
                out_dir.mkdir(parents=True, exist_ok=True)
                im = infer_cn(pipe, cond_img, args, args.seed + idx, device)
                im.save(out_dir / f"gs{args.guidance_scale}_cs{args.control_scale}.png")
            del pipe
            if device == "cuda":
                torch.cuda.empty_cache()

    print(f"DONE. results saved to {args.outdir}")


if __name__ == "__main__":
    main()
