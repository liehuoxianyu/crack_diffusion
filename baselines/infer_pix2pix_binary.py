#!/usr/bin/env python3
"""Run pix2pix inference for Binary baseline snapshots.

This is a thin wrapper around the cloned CycleGAN/pix2pix `test.py`.
It does not train or evaluate metrics.
"""

import argparse
import subprocess
from pathlib import Path


def run_epoch(args, epoch: str):
    cmd = [
        "python",
        "test.py",
        "--dataroot",
        args.data_root,
        "--name",
        args.name,
        "--model",
        "pix2pix",
        "--direction",
        "AtoB",
        "--dataset_mode",
        "aligned",
        "--preprocess",
        "resize_and_crop",
        "--load_size",
        str(args.image_size),
        "--crop_size",
        str(args.image_size),
        "--input_nc",
        "3",
        "--output_nc",
        "3",
        "--checkpoints_dir",
        args.checkpoints_dir,
        "--results_dir",
        args.results_dir,
        "--phase",
        "test",
        "--epoch",
        str(epoch),
        "--num_test",
        str(args.num_test),
        "--no_dropout",
        "--eval",
    ]
    subprocess.run(cmd, cwd=Path(args.repo_root), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/work/third_party/pytorch-CycleGAN-and-pix2pix")
    ap.add_argument("--data-root", default="/work/outputs/GAN/data/pix2pix_aligned")
    ap.add_argument("--checkpoints-dir", default="/work/outputs/GAN/checkpoints")
    ap.add_argument("--results-dir", default="/work/outputs/GAN/raw_results")
    ap.add_argument("--name", default="pix2pix_binary")
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--epochs", nargs="+", default=None, help="Run multiple saved epochs, e.g. 25 50 ... 200.")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--num-test", type=int, default=100000)
    args = ap.parse_args()

    for epoch in (args.epochs or [args.epoch]):
        run_epoch(args, epoch)


if __name__ == "__main__":
    main()
