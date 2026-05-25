#!/usr/bin/env python3
"""Export baseline images to the metric-compatible layout.

The metric scripts still expect the diffusion-style path:
  <eval-root>/<baseline>/id{id}/controlnet_binary/step<label>/gs7.5_cs1.0.png

For pix2pix/CycleGAN, ``label`` is a real training epoch. For VQGAN it is a
real optimizer step. Missing labels are not silently replaced by ``latest``.
"""

import argparse
import os
import shutil
from pathlib import Path

from PIL import Image


BASELINES = ("pix2pix", "cyclegan", "vqgan")


def read_ids(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def find_generated(
    raw_root: Path,
    baseline: str,
    label: int,
    image_id: str,
    allow_latest_fallback: bool = False,
) -> Path | None:
    candidates = []
    if baseline in ("pix2pix", "cyclegan"):
        # pytorch-CycleGAN-and-pix2pix results layout:
        # raw_results/<name>/test_<epoch>/images/<id>_fake_B.png
        epochs = [f"test_{label}"]
        if allow_latest_fallback:
            epochs.append("test_latest")
        for epoch in epochs:
            images = raw_root / f"{baseline}_binary" / epoch / "images"
            candidates.extend([images / f"{image_id}_fake_B.png", images / f"{image_id}_fake.png"])
    else:
        images = raw_root / "vqgan_binary" / f"step{label}" / "images"
        candidates.append(images / f"{image_id}_fake_B.png")
        if allow_latest_fallback:
            candidates.append(raw_root / "vqgan_binary" / "latest" / "images" / f"{image_id}_fake_B.png")

    for p in candidates:
        if p.exists():
            return p
    return None


def copy_image(src: Path, dst: Path, size: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    im.save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=BASELINES, required=True)
    ap.add_argument("--raw-root", default="/work/outputs/GAN/raw_results")
    ap.add_argument("--eval-root", default="/work/outputs/GAN/eval")
    ap.add_argument("--eval-ids", default="/CrackTree260/eval_ids.txt")
    ap.add_argument("--cond-dir", default="/CrackTree260/cond_mask")
    ap.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=[200],
        help="Real epoch labels for pix2pix/cyclegan, real optimizer steps for vqgan.",
    )
    ap.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Deprecated alias for --labels, kept for old command lines.",
    )
    ap.add_argument(
        "--allow-latest-fallback",
        action="store_true",
        help="Explicitly allow missing labels to reuse latest output. Off by default to protect metrics.",
    )
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--strict", action="store_true", help="Exit with an error if any expected image is missing.")
    args = ap.parse_args()

    labels = args.steps if args.steps is not None else args.labels
    ids = read_ids(args.eval_ids)
    raw_root = Path(args.raw_root)
    out_root = Path(args.eval_root) / args.baseline
    n = 0
    missing = 0
    missing_examples: list[str] = []

    for image_id in ids:
        id_root = out_root / f"id{image_id}"
        cond_path = Path(args.cond_dir) / f"{image_id}.png"
        if cond_path.exists():
            copy_image(cond_path, id_root / "cond_binary.png", args.size)

        for label in labels:
            gen = find_generated(
                raw_root,
                args.baseline,
                label,
                image_id,
                allow_latest_fallback=args.allow_latest_fallback,
            )
            if gen is None:
                missing += 1
                if len(missing_examples) < 10:
                    missing_examples.append(f"id{image_id}: label={label}")
                continue
            dst = id_root / "controlnet_binary" / f"step{label}" / "gs7.5_cs1.0.png"
            copy_image(gen, dst, args.size)
            n += 1

    print(
        f"exported baseline={args.baseline} labels={labels} images={n} "
        f"missing={missing} out={out_root}"
    )
    if missing_examples:
        print("missing examples:")
        for item in missing_examples:
            print(f"  {item}")
    if args.strict and missing:
        raise SystemExit(f"Missing {missing} expected generated images")


if __name__ == "__main__":
    main()
