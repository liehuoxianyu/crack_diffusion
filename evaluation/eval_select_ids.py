#!/usr/bin/env python3
"""Select evaluation IDs from DT condition maps.

This keeps the original sampling behavior from /work/select_eval_ids.py but
provides a standardized eval_* entrypoint under /work/evaluation.
"""

import argparse
import glob
import os
import random

import numpy as np
from PIL import Image


def score_dt(path: str, threshold: int) -> float:
    image = Image.open(path).convert("L")
    arr = np.array(image, dtype=np.uint8)
    return float((arr > threshold).mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select evaluation ids from DT maps.")
    parser.add_argument("--dt-dir", default="/CrackTree260/cond_dt")
    parser.add_argument("--out-txt", default="/CrackTree260/eval_ids.txt")
    parser.add_argument("--n-top", type=int, default=4)
    parser.add_argument("--n-bottom", type=int, default=4)
    parser.add_argument("--n-random", type=int, default=2)
    parser.add_argument("--n-random-extra", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def choose_ids(args: argparse.Namespace) -> list[str]:
    paths = sorted(glob.glob(os.path.join(args.dt_dir, "*.png")))
    if not paths:
        raise RuntimeError(f"No png found in {args.dt_dir}")

    scores: list[tuple[str, float]] = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        scores.append((stem, score_dt(path, args.threshold)))

    scores_nonempty = [(idx, score) for idx, score in scores if score > 0.0001]
    min_required = args.n_top + args.n_bottom + args.n_random
    if len(scores_nonempty) < min_required:
        scores_nonempty = scores

    scores_sorted = sorted(scores_nonempty, key=lambda item: item[1])
    bottom = [idx for idx, _ in scores_sorted[: args.n_bottom]]
    top = [idx for idx, _ in scores_sorted[-args.n_top :]]

    pool = [idx for idx, _ in scores_sorted]
    random.seed(args.seed)

    selected = top + bottom
    used = set(selected)
    for _ in range(args.n_random):
        candidate = random.choice(pool)
        while candidate in used:
            candidate = random.choice(pool)
        selected.append(candidate)
        used.add(candidate)

    for _ in range(args.n_random_extra):
        candidate = random.choice(pool)
        while candidate in used:
            candidate = random.choice(pool)
        selected.append(candidate)
        used.add(candidate)

    return selected


def main() -> None:
    args = parse_args()
    ids = choose_ids(args)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        for item in ids:
            f.write(f"{item}\n")
    print("wrote", args.out_txt)
    print("ids:", ids)


if __name__ == "__main__":
    main()
