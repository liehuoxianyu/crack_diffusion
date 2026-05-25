#!/usr/bin/env python3
"""Crack-specific topology metrics for generated images.

The script compares generated crack-like edges against the binary crack mask.
It complements generic FID/PSNR/LPIPS and the older structural-alignment metric
with connectivity, skeleton, endpoint, branchpoint, and width-distribution cues.
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.replace(" ", ",").split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in parse_csv_list(raw)]


def read_ids(path: str | None) -> list[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def disk_kernel(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    img = (mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while cv2.countNonZero(img) > 0:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded

    return (skel > 0).astype(np.uint8)


def keypoints_from_skeleton(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    skel = (skeleton > 0).astype(np.uint8)
    neighbors = cv2.filter2D(skel, cv2.CV_16S, np.ones((3, 3), dtype=np.uint8)) - skel
    endpoints = ((skel == 1) & (neighbors == 1)).astype(np.uint8)
    branchpoints = ((skel == 1) & (neighbors >= 3)).astype(np.uint8)
    return endpoints, branchpoints


def generated_edge_skeleton(gen_gray: np.ndarray, canny1: int, canny2: int) -> np.ndarray:
    edges = cv2.Canny(gen_gray, canny1, canny2)
    return skeletonize_binary((edges > 0).astype(np.uint8))


def dark_crack_mask(gen_gray: np.ndarray, percentile: float, min_threshold: int, allow_region: np.ndarray) -> np.ndarray:
    threshold = max(min_threshold, int(np.percentile(gen_gray, percentile)))
    pred = (gen_gray <= threshold).astype(np.uint8)
    pred = pred & allow_region.astype(np.uint8)
    return pred


def resize_to(reference: np.ndarray, image: np.ndarray, nearest: bool = True) -> np.ndarray:
    h, w = reference.shape[:2]
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(image, (w, h), interpolation=interp)


def symmetric_chamfer(a: np.ndarray, b: np.ndarray, max_dist: float) -> float:
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    if a.sum() == 0 or b.sum() == 0:
        return float(max_dist)

    dt_to_b = cv2.distanceTransform(1 - b, cv2.DIST_L2, 3).astype(np.float32)
    dt_to_a = cv2.distanceTransform(1 - a, cv2.DIST_L2, 3).astype(np.float32)
    d_ab = np.clip(dt_to_b[a > 0], 0, max_dist)
    d_ba = np.clip(dt_to_a[b > 0], 0, max_dist)
    return float((d_ab.mean() + d_ba.mean()) * 0.5)


def point_f1(pred_points: np.ndarray, gt_points: np.ndarray, radius: int) -> tuple[float, float, float]:
    pred = (pred_points > 0).astype(np.uint8)
    gt = (gt_points > 0).astype(np.uint8)
    pred_count = int(pred.sum())
    gt_count = int(gt.sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0, 1.0, 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0, 0.0, 0.0

    kernel = disk_kernel(radius)
    gt_dil = cv2.dilate(gt, kernel)
    pred_dil = cv2.dilate(pred, kernel)
    tp_pred = int(((pred > 0) & (gt_dil > 0)).sum())
    tp_gt = int(((gt > 0) & (pred_dil > 0)).sum())
    precision = tp_pred / pred_count if pred_count else 0.0
    recall = tp_gt / gt_count if gt_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return float(precision), float(recall), float(f1)


def component_count(mask: np.ndarray) -> int:
    num_labels, _ = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return max(0, int(num_labels) - 1)


def width_values(mask: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    vals = dist[skeleton > 0] * 2.0
    return vals.astype(np.float32)


def width_mae(pred_mask: np.ndarray, pred_skel: np.ndarray, gt_mask: np.ndarray, gt_skel: np.ndarray) -> float:
    pred_vals = width_values(pred_mask, pred_skel)
    gt_vals = width_values(gt_mask, gt_skel)
    if len(pred_vals) == 0 or len(gt_vals) == 0:
        return float("nan")
    pred_q = np.percentile(pred_vals, [25, 50, 75])
    gt_q = np.percentile(gt_vals, [25, 50, 75])
    return float(np.mean(np.abs(pred_q - gt_q)))


def generated_path(eval_root: Path, image_id: str, mode: str, step: int, gs: float, cs: float) -> Path:
    return eval_root / f"id{image_id}" / f"controlnet_{mode}" / f"step{step}" / f"gs{gs}_cs{cs}.png"


def summarize(vals: list[float]) -> dict[str, float]:
    arr = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float32)
    if len(arr) == 0:
        return {"mean": float("nan"), "median": float("nan")}
    return {"mean": float(arr.mean()), "median": float(np.median(arr))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default="/work/outputs/diffusion_eval_fairness")
    parser.add_argument("--eval-ids", default="/CrackTree260/eval_ids.txt")
    parser.add_argument("--mask-dir", default="/CrackTree260/cond_mask")
    parser.add_argument("--modes", default="dt,dt_cafe,dt_tag,dt_cafe_tag")
    parser.add_argument("--steps", default="500,1000,1500,2000")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--canny1", type=int, default=50)
    parser.add_argument("--canny2", type=int, default=150)
    parser.add_argument("--match-radius", type=int, default=5)
    parser.add_argument("--allow-dilate", type=int, default=7)
    parser.add_argument("--dark-percentile", type=float, default=25.0)
    parser.add_argument("--dark-min-threshold", type=int, default=80)
    parser.add_argument("--max-chamfer", type=float, default=30.0)
    parser.add_argument("--out-rows", default=None)
    parser.add_argument("--out-summary", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    mask_dir = Path(args.mask_dir)
    modes = parse_csv_list(args.modes)
    steps = parse_int_list(args.steps)
    ids = read_ids(args.eval_ids)
    if not ids:
        raise RuntimeError(f"no eval ids found: {args.eval_ids}")

    out_rows = Path(args.out_rows or eval_root / "metric_crack_topology_rows.csv")
    out_summary = Path(args.out_summary or eval_root / "metric_crack_topology_summary.csv")
    out_rows.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_id in ids:
        mask_path = mask_dir / f"{image_id}.png"
        if not mask_path.exists():
            continue
        gt_mask = (load_gray(mask_path) > 127).astype(np.uint8)
        gt_skel = skeletonize_binary(gt_mask)
        gt_endpoints, gt_branchpoints = keypoints_from_skeleton(gt_skel)
        allow = gt_mask
        if args.allow_dilate > 0:
            allow = cv2.dilate(allow, disk_kernel(args.allow_dilate), iterations=1)

        for mode in modes:
            for step in steps:
                gen_path = generated_path(eval_root, image_id, mode, step, args.guidance_scale, args.control_scale)
                if not gen_path.exists():
                    continue
                gen_gray = load_gray(gen_path)
                if gen_gray.shape != gt_mask.shape:
                    gen_gray = resize_to(gt_mask, gen_gray, nearest=False)

                pred_skel = generated_edge_skeleton(gen_gray, args.canny1, args.canny2)
                pred_mask = dark_crack_mask(gen_gray, args.dark_percentile, args.dark_min_threshold, allow)
                pred_endpoints, pred_branchpoints = keypoints_from_skeleton(pred_skel)

                endpoint_p, endpoint_r, endpoint_f1 = point_f1(
                    pred_endpoints, gt_endpoints, radius=args.match_radius
                )
                branch_p, branch_r, branch_f1 = point_f1(
                    pred_branchpoints, gt_branchpoints, radius=args.match_radius
                )
                pred_components = component_count(pred_skel)
                gt_components = component_count(gt_skel)

                rows.append(
                    {
                        "id": image_id,
                        "mode": mode,
                        "step": step,
                        "gen_path": str(gen_path),
                        "skeleton_chamfer": symmetric_chamfer(pred_skel, gt_skel, args.max_chamfer),
                        "component_abs_error": abs(pred_components - gt_components),
                        "pred_components": pred_components,
                        "gt_components": gt_components,
                        "endpoint_precision": endpoint_p,
                        "endpoint_recall": endpoint_r,
                        "endpoint_f1": endpoint_f1,
                        "branch_precision": branch_p,
                        "branch_recall": branch_r,
                        "branch_f1": branch_f1,
                        "width_quantile_mae": width_mae(pred_mask, skeletonize_binary(pred_mask), gt_mask, gt_skel),
                    }
                )

    if not rows:
        raise RuntimeError("no topology metric rows collected")

    with out_rows.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["step"])].append(row)

    metric_keys = [
        "skeleton_chamfer",
        "component_abs_error",
        "endpoint_f1",
        "branch_f1",
        "width_quantile_mae",
    ]
    summary_rows = []
    for (mode, step), group_rows in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        summary = {"mode": mode, "step": step, "n_images": len(group_rows)}
        for key in metric_keys:
            stats = summarize([float(row[key]) for row in group_rows])
            summary[f"{key}_mean"] = stats["mean"]
            summary[f"{key}_median"] = stats["median"]
        summary_rows.append(summary)

    with out_summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"wrote: {out_rows} rows={len(rows)}")
    print(f"wrote: {out_summary} groups={len(summary_rows)}")


if __name__ == "__main__":
    main()
