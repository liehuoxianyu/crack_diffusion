#!/usr/bin/env python3
"""Prepare Binary-condition paired data for image-to-image baselines.

The output layout serves both pix2pix and CycleGAN-style loaders:

  <out_root>/paired/
    train/
      A/   binary condition images
      B/   real crack images
      AB/  side-by-side A|B pairs for pix2pix aligned mode
    test/
      A/
      B/
      AB/
    eval_ids.txt
  <out_root>/pix2pix_aligned/{train,test}/*.png
  <out_root>/cyclegan_unaligned/{trainA,trainB,testA,testB}/*.png
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from PIL import Image


def read_ids(path: str) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def ensure_dirs(root: Path):
    root = root / "paired"
    for split in ("train", "test"):
        for sub in ("A", "B", "AB"):
            (root / split / sub).mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        (root.parent / "pix2pix_aligned" / split).mkdir(parents=True, exist_ok=True)
    for sub in ("trainA", "trainB", "testA", "testB"):
        (root.parent / "cyclegan_unaligned" / sub).mkdir(parents=True, exist_ok=True)


def save_pair(cond_path: Path, real_path: Path, out_root: Path, split: str, image_id: str, size: int):
    cond = Image.open(cond_path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    real = Image.open(real_path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)

    paired_root = out_root / "paired" / split
    cond_out = paired_root / "A" / f"{image_id}.png"
    real_out = paired_root / "B" / f"{image_id}.png"
    ab_out = paired_root / "AB" / f"{image_id}.png"

    cond.save(cond_out)
    real.save(real_out)

    ab = Image.new("RGB", (size * 2, size))
    ab.paste(cond, (0, 0))
    ab.paste(real, (size, 0))
    ab.save(ab_out)
    ab.save(out_root / "pix2pix_aligned" / split / f"{image_id}.png")

    phase = "test" if split == "test" else "train"
    cond.save(out_root / "cyclegan_unaligned" / f"{phase}A" / f"{image_id}.png")
    real.save(out_root / "cyclegan_unaligned" / f"{phase}B" / f"{image_id}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/CrackTree260/train_linux.jsonl")
    ap.add_argument("--eval-ids", default="/CrackTree260/eval_ids.txt")
    ap.add_argument("--cond-dir", default="/CrackTree260/cond_mask")
    ap.add_argument("--out-root", default="/work/outputs/GAN/data")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dirs(out_root)
    eval_ids = read_ids(args.eval_ids)

    n_train = 0
    n_test = 0
    for item in iter_jsonl(args.jsonl):
        real_path = Path(item["image"])
        image_id = real_path.stem
        cond_path = Path(args.cond_dir) / f"{image_id}.png"
        if not real_path.exists():
            raise FileNotFoundError(f"missing image: {real_path}")
        if not cond_path.exists():
            raise FileNotFoundError(f"missing condition: {cond_path}")

        split = "test" if image_id in eval_ids else "train"
        save_pair(cond_path, real_path, out_root, split, image_id, args.size)
        if split == "test":
            n_test += 1
        else:
            n_train += 1

    shutil.copy2(args.eval_ids, out_root / "eval_ids.txt")
    print(f"prepared binary paired dataset: train={n_train}, test={n_test}, out={out_root}")


if __name__ == "__main__":
    main()
