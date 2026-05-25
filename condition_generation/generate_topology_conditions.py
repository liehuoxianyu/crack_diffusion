#!/usr/bin/env python3
"""Generate topology-aware ControlNet conditions from binary crack masks.

Default RGB layout:
  R = skeleton centerline
  G = distance-transform heatmap around crack pixels
  B = crack width map sampled on the skeleton

Alternative layout:
  R = skeleton centerline
  G = distance-transform heatmap
  B = endpoint/branchpoint heatmap
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def normalize_u8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    max_val = float(arr.max())
    if max_val <= 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip(arr / max_val * 255.0, 0, 255).astype(np.uint8)


def dt_heat_from_binary(crack: np.ndarray, sigma: float) -> np.ndarray:
    bg = 1 - crack.astype(np.uint8)
    dist_to_crack = cv2.distanceTransform(bg, cv2.DIST_L2, 3).astype(np.float32)
    heat = np.exp(-(dist_to_crack**2) / (2.0 * sigma**2))
    return np.clip(heat * 255.0, 0, 255).astype(np.uint8)


def skeletonize_binary(crack: np.ndarray) -> np.ndarray:
    """Morphological skeletonization without requiring scikit-image."""
    img = (crack > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while cv2.countNonZero(img) > 0:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded

    return (skel > 0).astype(np.uint8)


def width_map_from_binary(crack: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    dist_inside = cv2.distanceTransform(crack.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    width = dist_inside * 2.0 * skeleton.astype(np.float32)
    return normalize_u8(width)


def topology_keypoint_heat(skeleton: np.ndarray, radius: int) -> np.ndarray:
    skel = (skeleton > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, cv2.CV_16S, kernel, borderType=cv2.BORDER_CONSTANT) - skel
    endpoints = ((skel == 1) & (neighbor_count == 1)).astype(np.uint8)
    branchpoints = ((skel == 1) & (neighbor_count >= 3)).astype(np.uint8)
    keypoints = np.maximum(endpoints, branchpoints)

    if radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        keypoints = cv2.dilate(keypoints, k, iterations=1)

    return (keypoints * 255).astype(np.uint8)


def make_topology_condition(mask_u8: np.ndarray, sigma: float, keypoint_radius: int, layout: str) -> np.ndarray:
    crack = (mask_u8 > 127).astype(np.uint8)
    skeleton = skeletonize_binary(crack)
    skeleton_u8 = (skeleton * 255).astype(np.uint8)
    dt_heat = dt_heat_from_binary(crack, sigma=sigma)

    if layout == "skeleton_dt_keypoints":
        third = topology_keypoint_heat(skeleton, radius=keypoint_radius)
    elif layout == "skeleton_dt_width":
        third = width_map_from_binary(crack, skeleton)
    else:
        raise ValueError(f"unknown layout: {layout}")

    return cv2.merge([skeleton_u8, dt_heat, third])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", default="/CrackTree260/cond_mask")
    parser.add_argument("--out-dir", default="/CrackTree260/cond_topology")
    parser.add_argument("--sigma", type=float, default=15.0)
    parser.add_argument("--keypoint-radius", type=int, default=3)
    parser.add_argument(
        "--layout",
        choices=("skeleton_dt_width", "skeleton_dt_keypoints"),
        default="skeleton_dt_width",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_images(mask_dir)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(f"no masks found in {mask_dir}")

    for idx, path in enumerate(files, start=1):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read {path}")
        cond = make_topology_condition(mask, sigma=args.sigma, keypoint_radius=args.keypoint_radius, layout=args.layout)
        out_path = out_dir / f"{path.stem}.png"
        cv2.imwrite(str(out_path), cond)
        if idx % 100 == 0:
            print(f"processed {idx}/{len(files)}")

    print(f"done. out_dir={out_dir} num={len(files)} layout={args.layout}")


if __name__ == "__main__":
    main()
