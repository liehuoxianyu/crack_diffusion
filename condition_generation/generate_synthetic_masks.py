#!/usr/bin/env python3
"""Generate synthetic crack masks for segmentation dataset expansion.

The generator learns lightweight statistics from real binary crack masks and
uses them as priors for a stochastic crack-structure sampler. The synthetic
binary masks are intended to become segmentation labels, while optional DT and
topology outputs can be fed into the existing SD1.5 + ControlNet pipeline.
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from generate_topology_conditions import dt_heat_from_binary, make_topology_condition, skeletonize_binary


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class MaskStats:
    image_id: str
    height: int
    width: int
    area_ratio: float
    component_count: int
    skeleton_length: int
    skeleton_length_norm: float
    endpoint_count: int
    branchpoint_count: int
    orientation_deg: float
    width_mean: float
    width_median: float
    width_p90: float


@dataclass
class GeneratedMask:
    mask: np.ndarray
    skeleton: np.ndarray
    params: dict[str, Any]
    stats: MaskStats


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {path}")
    return ((mask > 127).astype(np.uint8) * 255)


def component_count(mask: np.ndarray) -> int:
    num_labels, _labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return max(0, int(num_labels) - 1)


def keypoints_from_skeleton(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    skel = (skeleton > 0).astype(np.uint8)
    neighbors = cv2.filter2D(skel, cv2.CV_16S, np.ones((3, 3), dtype=np.uint8)) - skel
    endpoints = ((skel == 1) & (neighbors == 1)).astype(np.uint8)
    branchpoints = ((skel == 1) & (neighbors >= 3)).astype(np.uint8)
    return endpoints, branchpoints


def skeleton_orientation_deg(skeleton: np.ndarray) -> float:
    ys, xs = np.nonzero(skeleton > 0)
    if len(xs) < 2:
        return 0.0
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    vec = vecs[:, int(np.argmax(vals))]
    angle = math.degrees(math.atan2(float(vec[1]), float(vec[0])))
    # Normalize to [-90, 90), because a line has no arrow direction.
    while angle < -90.0:
        angle += 180.0
    while angle >= 90.0:
        angle -= 180.0
    return float(angle)


def width_values(mask: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    crack = (mask > 0).astype(np.uint8)
    skel = skeleton > 0
    if not np.any(crack) or not np.any(skel):
        return np.asarray([], dtype=np.float32)
    dist = cv2.distanceTransform(crack, cv2.DIST_L2, 3).astype(np.float32)
    return (dist[skel] * 2.0).astype(np.float32)


def compute_mask_stats(mask: np.ndarray, image_id: str) -> MaskStats:
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    skeleton = skeletonize_binary(binary)
    endpoints, branchpoints = keypoints_from_skeleton(skeleton)
    widths = width_values(binary, skeleton)
    if len(widths) == 0:
        width_mean = width_median = width_p90 = 1.0
    else:
        width_mean = float(np.mean(widths))
        width_median = float(np.median(widths))
        width_p90 = float(np.percentile(widths, 90))

    return MaskStats(
        image_id=image_id,
        height=int(h),
        width=int(w),
        area_ratio=float(binary.mean()),
        component_count=component_count(binary),
        skeleton_length=int(skeleton.sum()),
        skeleton_length_norm=float(skeleton.sum() / max(1.0, math.sqrt(h * w))),
        endpoint_count=int(endpoints.sum()),
        branchpoint_count=int(branchpoints.sum()),
        orientation_deg=skeleton_orientation_deg(skeleton),
        width_mean=width_mean,
        width_median=width_median,
        width_p90=width_p90,
    )


def learn_mask_priors(real_mask_dir: Path, limit: int | None) -> list[MaskStats]:
    paths = list_images(real_mask_dir)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise RuntimeError(f"no masks found in {real_mask_dir}")

    stats: list[MaskStats] = []
    for path in paths:
        mask = load_binary_mask(path)
        item = compute_mask_stats(mask, path.stem)
        if item.area_ratio > 0.0 and item.skeleton_length > 0:
            stats.append(item)
    if not stats:
        raise RuntimeError(f"no non-empty masks found in {real_mask_dir}")
    return stats


def finite_values(stats: list[MaskStats], field: str) -> np.ndarray:
    vals = np.asarray([float(getattr(s, field)) for s in stats], dtype=np.float32)
    return vals[np.isfinite(vals)]


def quantile_bounds(stats: list[MaskStats], field: str, lo: float, hi: float) -> tuple[float, float]:
    vals = finite_values(stats, field)
    if len(vals) == 0:
        return 0.0, 0.0
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def summarize_stats(stats: list[MaskStats]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    fields = [
        "area_ratio",
        "component_count",
        "skeleton_length_norm",
        "endpoint_count",
        "branchpoint_count",
        "orientation_deg",
        "width_median",
        "width_p90",
    ]
    for field in fields:
        vals = finite_values(stats, field)
        if len(vals) == 0:
            summary[field] = {"mean": float("nan"), "median": float("nan")}
            continue
        summary[field] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p05": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95)),
        }
    return summary


def sample_template(stats: list[MaskStats], rng: np.random.Generator) -> MaskStats:
    return stats[int(rng.integers(0, len(stats)))]


def jitter_positive(value: float, rng: np.random.Generator, sigma: float, floor: float) -> float:
    if value <= 0:
        return floor
    return max(floor, float(value * math.exp(float(rng.normal(0.0, sigma)))))


def start_point(size: int, angle_rad: float, rng: np.random.Generator, margin: int) -> tuple[float, float]:
    # Often start near a border so long cracks cross the image, but keep a
    # center-biased fallback for shorter isolated cracks.
    if float(rng.random()) < 0.65:
        side = int(rng.integers(0, 4))
        if side == 0:
            return float(rng.uniform(margin, size - margin)), float(margin)
        if side == 1:
            return float(size - margin), float(rng.uniform(margin, size - margin))
        if side == 2:
            return float(rng.uniform(margin, size - margin)), float(size - margin)
        return float(margin), float(rng.uniform(margin, size - margin))

    center = size * 0.5
    spread = size * 0.25
    return (
        float(np.clip(rng.normal(center, spread), margin, size - margin)),
        float(np.clip(rng.normal(center, spread), margin, size - margin)),
    )


def reflect_angle_at_bounds(x: float, y: float, angle: float, size: int, margin: int) -> float:
    if x <= margin or x >= size - margin:
        angle = math.pi - angle
    if y <= margin or y >= size - margin:
        angle = -angle
    return angle


def random_walk_points(
    *,
    size: int,
    length: float,
    angle_deg: float,
    turn_std: float,
    rng: np.random.Generator,
    start: tuple[float, float] | None = None,
    step: float = 3.0,
) -> list[tuple[int, int]]:
    margin = max(2, int(size * 0.02))
    angle = math.radians(angle_deg)
    if start is None:
        x, y = start_point(size, angle, rng, margin)
    else:
        x, y = float(start[0]), float(start[1])

    points: list[tuple[int, int]] = [(int(round(x)), int(round(y)))]
    steps = max(2, int(length / step))
    for _ in range(steps):
        angle += float(rng.normal(0.0, turn_std))
        x += math.cos(angle) * step
        y += math.sin(angle) * step
        angle = reflect_angle_at_bounds(x, y, angle, size, margin)
        x = float(np.clip(x, margin, size - margin))
        y = float(np.clip(y, margin, size - margin))
        pt = (int(round(x)), int(round(y)))
        if pt != points[-1]:
            points.append(pt)
    return points


def draw_polyline(mask: np.ndarray, points: list[tuple[int, int]], thickness: int) -> None:
    if len(points) < 2:
        return
    for p0, p1 in zip(points[:-1], points[1:]):
        cv2.line(mask, p0, p1, 255, thickness=thickness, lineType=cv2.LINE_8)


def draw_variable_width_polyline(
    mask: np.ndarray,
    points: list[tuple[int, int]],
    base_width: float,
    rng: np.random.Generator,
    width_jitter: float,
) -> None:
    if len(points) < 2:
        return
    current_width = max(1.0, base_width)
    for idx, (p0, p1) in enumerate(zip(points[:-1], points[1:])):
        if idx % 5 == 0:
            current_width = jitter_positive(base_width, rng, width_jitter, 1.0)
        thickness = max(1, int(round(current_width)))
        cv2.line(mask, p0, p1, 255, thickness=thickness, lineType=cv2.LINE_8)


def add_gaps(mask: np.ndarray, points: list[tuple[int, int]], rng: np.random.Generator, gap_prob: float, width: float) -> int:
    if len(points) < 2 or gap_prob <= 0:
        return 0
    gap_count = int(rng.poisson(max(0.0, gap_prob * len(points) / 60.0)))
    if gap_count <= 0:
        return 0
    radius_base = max(2, int(round(width * 1.8)))
    for _ in range(gap_count):
        x, y = points[int(rng.integers(0, len(points)))]
        rx = int(rng.integers(radius_base, radius_base * 3 + 1))
        ry = int(rng.integers(max(1, radius_base // 2), radius_base * 2 + 1))
        angle = float(rng.uniform(0.0, 180.0))
        cv2.ellipse(mask, (x, y), (rx, ry), angle, 0, 360, 0, thickness=-1)
    return gap_count


def roughen_edges(mask: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    if strength <= 0:
        return mask
    binary = (mask > 127).astype(np.uint8) * 255
    if float(rng.random()) < strength:
        kernel_size = int(rng.choice([2, 3]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if float(rng.random()) < 0.5:
            binary = cv2.dilate(binary, kernel, iterations=1)
        else:
            binary = cv2.erode(binary, kernel, iterations=1)

    noise = rng.random(binary.shape)
    boundary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
    drop = (noise < strength * 0.08) & boundary
    add = (noise > 1.0 - strength * 0.05) & cv2.dilate(binary, np.ones((3, 3), dtype=np.uint8)).astype(bool)
    binary[drop] = 0
    binary[add] = 255
    return ((binary > 127).astype(np.uint8) * 255)


def estimated_branch_count(template: MaskStats, rng: np.random.Generator, max_branches: int) -> int:
    # Branchpoint pixels appear in clusters; sqrt makes the count closer to
    # visible side-branch count than raw branchpoint-pixel totals.
    baseline = max(0.0, math.sqrt(max(0, template.branchpoint_count)))
    count = int(round(jitter_positive(baseline, rng, sigma=0.35, floor=0.0)))
    if baseline <= 0 and float(rng.random()) < 0.15:
        count = 1
    return int(np.clip(count, 0, max_branches))


def generate_skeleton_mask(
    *,
    size: int,
    template: MaskStats,
    rng: np.random.Generator,
    max_components: int,
    max_branches: int,
) -> tuple[np.ndarray, list[list[tuple[int, int]]], dict[str, Any]]:
    skeleton = np.zeros((size, size), dtype=np.uint8)
    all_paths: list[list[tuple[int, int]]] = []

    target_components = int(np.clip(round(jitter_positive(template.component_count, rng, 0.35, 1.0)), 1, max_components))
    target_length = jitter_positive(template.skeleton_length_norm * size, rng, 0.25, size * 0.25)
    length_per_component = max(size * 0.15, target_length / target_components)

    base_angle = template.orientation_deg + float(rng.normal(0.0, 12.0))
    turn_std = float(np.clip(rng.normal(0.09, 0.035), 0.025, 0.22))
    branch_count_total = 0

    for comp_idx in range(target_components):
        angle = base_angle + float(rng.normal(0.0, 35.0))
        length = jitter_positive(length_per_component, rng, 0.25, size * 0.12)
        points = random_walk_points(size=size, length=length, angle_deg=angle, turn_std=turn_std, rng=rng)
        draw_polyline(skeleton, points, thickness=1)
        all_paths.append(points)

        component_branches = estimated_branch_count(template, rng, max_branches)
        component_branches = int(np.ceil(component_branches / max(1, target_components)))
        for _ in range(component_branches):
            if len(points) < 8:
                continue
            anchor = points[int(rng.integers(3, len(points) - 3))]
            branch_angle = angle + float(rng.choice([-1.0, 1.0])) * float(rng.uniform(35.0, 85.0))
            branch_length = length * float(rng.uniform(0.15, 0.45))
            branch_points = random_walk_points(
                size=size,
                length=branch_length,
                angle_deg=branch_angle,
                turn_std=turn_std * float(rng.uniform(0.8, 1.5)),
                rng=rng,
                start=anchor,
            )
            draw_polyline(skeleton, branch_points, thickness=1)
            all_paths.append(branch_points)
            branch_count_total += 1

    params = {
        "target_components": target_components,
        "target_length": target_length,
        "base_angle_deg": base_angle,
        "turn_std": turn_std,
        "branch_count": branch_count_total,
    }
    return skeleton, all_paths, params


def rasterize_skeleton(
    skeleton_paths: list[list[tuple[int, int]]],
    template: MaskStats,
    rng: np.random.Generator,
    size: int,
    width_scale: float,
    gap_scale: float,
    roughness: float,
    target_area_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.zeros((size, size), dtype=np.uint8)
    path_length = max(1, sum(len(points) for points in skeleton_paths))
    area_width = (target_area_ratio * size * size) / path_length
    sampled_width = jitter_positive(template.width_median, rng, 0.3, 1.0) * width_scale
    base_width = np.clip(min(sampled_width, area_width * 1.35), 1.0, size * 0.04)
    width_jitter = float(np.clip(template.width_p90 / max(template.width_median, 1.0) * 0.08, 0.05, 0.35))
    gap_prob = float(np.clip((template.endpoint_count / max(template.skeleton_length, 1)) * 20.0, 0.05, 0.9)) * gap_scale
    gap_count = 0

    for points in skeleton_paths:
        draw_variable_width_polyline(mask, points, base_width, rng, width_jitter)
        gap_count += add_gaps(mask, points, rng, gap_prob=gap_prob, width=base_width)

    mask = roughen_edges(mask, rng, strength=roughness)
    mask = ((mask > 127).astype(np.uint8) * 255)
    params = {
        "base_width": float(base_width),
        "target_area_ratio": float(target_area_ratio),
        "width_jitter": width_jitter,
        "gap_prob": gap_prob,
        "gap_count": gap_count,
        "roughness": roughness,
    }
    return mask, params


def validation_ranges(real_stats: list[MaskStats]) -> dict[str, tuple[float, float]]:
    area_lo, area_hi = quantile_bounds(real_stats, "area_ratio", 1, 99)
    length_lo, length_hi = quantile_bounds(real_stats, "skeleton_length_norm", 1, 99)
    comp_lo, comp_hi = quantile_bounds(real_stats, "component_count", 1, 99)
    return {
        "area_ratio": (max(0.0002, area_lo * 0.45), max(0.002, area_hi * 1.8)),
        "skeleton_length_norm": (max(1.0, length_lo * 0.35), max(4.0, length_hi * 1.8)),
        "component_count": (max(1.0, comp_lo * 0.2), max(1.0, comp_hi * 2.0)),
    }


def is_valid_synthetic(stats: MaskStats, ranges: dict[str, tuple[float, float]]) -> bool:
    area_lo, area_hi = ranges["area_ratio"]
    length_lo, length_hi = ranges["skeleton_length_norm"]
    comp_lo, comp_hi = ranges["component_count"]
    return (
        area_lo <= stats.area_ratio <= area_hi
        and length_lo <= stats.skeleton_length_norm <= length_hi
        and comp_lo <= stats.component_count <= comp_hi
        and stats.skeleton_length > 8
    )


def is_close_to_target_area(stats: MaskStats, target_area_ratio: float) -> bool:
    lo = max(0.0002, target_area_ratio * 0.45)
    hi = max(0.0015, target_area_ratio * 3.0)
    return lo <= stats.area_ratio <= hi


def generate_one(
    *,
    image_id: str,
    real_stats: list[MaskStats],
    ranges: dict[str, tuple[float, float]],
    rng: np.random.Generator,
    size: int,
    max_components: int,
    max_branches: int,
    width_scale: float,
    gap_scale: float,
    roughness: float,
    max_attempts: int,
) -> GeneratedMask:
    last: GeneratedMask | None = None
    for attempt in range(1, max_attempts + 1):
        template = sample_template(real_stats, rng)
        area_lo, area_hi = ranges["area_ratio"]
        target_area_ratio = float(np.clip(jitter_positive(template.area_ratio, rng, 0.35, 0.0004), area_lo, area_hi))
        skeleton, paths, skeleton_params = generate_skeleton_mask(
            size=size,
            template=template,
            rng=rng,
            max_components=max_components,
            max_branches=max_branches,
        )
        mask, raster_params = rasterize_skeleton(
            paths,
            template,
            rng,
            size=size,
            width_scale=width_scale,
            gap_scale=gap_scale,
            roughness=roughness,
            target_area_ratio=target_area_ratio,
        )
        final_skeleton = skeletonize_binary((mask > 127).astype(np.uint8))
        stats = compute_mask_stats(mask, image_id)
        params = {
            "attempt": attempt,
            "template_id": template.image_id,
            "template_stats": asdict(template),
            "skeleton": skeleton_params,
            "raster": raster_params,
        }
        generated = GeneratedMask(mask=mask, skeleton=final_skeleton, params=params, stats=stats)
        last = generated
        if is_valid_synthetic(stats, ranges) and is_close_to_target_area(stats, target_area_ratio):
            return generated

    if last is None:
        raise RuntimeError("generator failed before producing a candidate")
    return last


def make_contact_sheet(paths: list[Path], out_path: Path, tile_size: int, cols: int) -> None:
    if not paths:
        return
    tiles = []
    for path in paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        tile = cv2.resize(mask, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        cv2.putText(tile, path.stem, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    cols = max(1, cols)
    rows = int(math.ceil(len(tiles) / cols))
    blank = np.zeros_like(tiles[0])
    padded = tiles + [blank] * (rows * cols - len(tiles))
    row_imgs = [cv2.hconcat(padded[i * cols : (i + 1) * cols]) for i in range(rows)]
    sheet = cv2.vconcat(row_imgs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def default_metadata_path(out_dir: Path) -> Path:
    return out_dir.parent / "metadata.jsonl"


def default_summary_path(out_dir: Path) -> Path:
    return out_dir.parent / "synthetic_mask_summary.json"


def default_contact_sheet_path(out_dir: Path) -> Path:
    return out_dir.parent / "mask_contact_sheet.png"


def write_condition_images(
    *,
    mask: np.ndarray,
    image_id: str,
    dt_out_dir: Path | None,
    topology_out_dir: Path | None,
    sigma: float,
    keypoint_radius: int,
    topology_layout: str,
) -> None:
    crack = (mask > 127).astype(np.uint8)
    if dt_out_dir is not None:
        dt_out_dir.mkdir(parents=True, exist_ok=True)
        dt = dt_heat_from_binary(crack, sigma=sigma)
        cv2.imwrite(str(dt_out_dir / f"{image_id}.png"), dt)
    if topology_out_dir is not None:
        topology_out_dir.mkdir(parents=True, exist_ok=True)
        topology = make_topology_condition(mask, sigma=sigma, keypoint_radius=keypoint_radius, layout=topology_layout)
        cv2.imwrite(str(topology_out_dir / f"{image_id}.png"), topology)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic 0/255 crack masks from learned CrackTree mask priors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--real-mask-dir", default="/CrackTree260/cond_mask")
    parser.add_argument("--out-dir", default="/work/outputs/synthetic_crack_dataset/masks")
    parser.add_argument("--num-masks", type=int, default=500)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-limit", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=60)
    parser.add_argument("--max-components", type=int, default=8)
    parser.add_argument("--max-branches", type=int, default=12)
    parser.add_argument("--width-scale", type=float, default=0.7)
    parser.add_argument("--gap-scale", type=float, default=1.0)
    parser.add_argument("--roughness", type=float, default=0.15)
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--contact-sheet", default=None)
    parser.add_argument("--contact-sheet-count", type=int, default=64)
    parser.add_argument("--contact-sheet-tile", type=int, default=128)
    parser.add_argument("--contact-sheet-cols", type=int, default=8)
    parser.add_argument("--dt-out-dir", default=None, help="Optional directory for DT ControlNet conditions.")
    parser.add_argument("--topology-out-dir", default=None, help="Optional directory for topology ControlNet conditions.")
    parser.add_argument(
        "--write-default-conditions",
        action="store_true",
        help="Write DT and topology conditions to sibling cond_dt and cond_topology directories.",
    )
    parser.add_argument("--condition-sigma", type=float, default=15.0)
    parser.add_argument("--keypoint-radius", type=int, default=3)
    parser.add_argument(
        "--topology-layout",
        choices=("skeleton_dt_width", "skeleton_dt_keypoints"),
        default="skeleton_dt_width",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_masks <= 0:
        raise ValueError("--num-masks must be positive")
    if args.size <= 16:
        raise ValueError("--size must be greater than 16")

    real_mask_dir = Path(args.real_mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_path) if args.metadata_path else default_metadata_path(out_dir)
    summary_path = Path(args.summary_path) if args.summary_path else default_summary_path(out_dir)
    contact_sheet_path = Path(args.contact_sheet) if args.contact_sheet else default_contact_sheet_path(out_dir)

    dt_out_dir = Path(args.dt_out_dir) if args.dt_out_dir else None
    topology_out_dir = Path(args.topology_out_dir) if args.topology_out_dir else None
    if args.write_default_conditions:
        dt_out_dir = dt_out_dir or (out_dir.parent / "cond_dt")
        topology_out_dir = topology_out_dir or (out_dir.parent / "cond_topology")

    real_stats = learn_mask_priors(real_mask_dir, limit=args.real_limit)
    ranges = validation_ranges(real_stats)
    rng = np.random.default_rng(args.seed)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    generated_stats: list[MaskStats] = []
    written_paths: list[Path] = []

    with metadata_path.open("w", encoding="utf-8") as mf:
        for idx in range(args.num_masks):
            image_id = f"synth_{idx:06d}"
            generated = generate_one(
                image_id=image_id,
                real_stats=real_stats,
                ranges=ranges,
                rng=rng,
                size=args.size,
                max_components=args.max_components,
                max_branches=args.max_branches,
                width_scale=args.width_scale,
                gap_scale=args.gap_scale,
                roughness=args.roughness,
                max_attempts=args.max_attempts,
            )
            out_path = out_dir / f"{image_id}.png"
            cv2.imwrite(str(out_path), generated.mask)
            write_condition_images(
                mask=generated.mask,
                image_id=image_id,
                dt_out_dir=dt_out_dir,
                topology_out_dir=topology_out_dir,
                sigma=args.condition_sigma,
                keypoint_radius=args.keypoint_radius,
                topology_layout=args.topology_layout,
            )

            row = {
                "id": image_id,
                "mask_path": str(out_path),
                "dt_path": str(dt_out_dir / f"{image_id}.png") if dt_out_dir is not None else None,
                "topology_path": str(topology_out_dir / f"{image_id}.png") if topology_out_dir is not None else None,
                "seed": args.seed,
                "params": generated.params,
                "stats": asdict(generated.stats),
            }
            mf.write(json.dumps(row, ensure_ascii=True) + "\n")
            generated_stats.append(generated.stats)
            written_paths.append(out_path)

            if (idx + 1) % 100 == 0:
                print(f"generated {idx + 1}/{args.num_masks}")

    summary = {
        "real_mask_dir": str(real_mask_dir),
        "out_dir": str(out_dir),
        "num_real_masks": len(real_stats),
        "num_synthetic_masks": len(generated_stats),
        "seed": args.seed,
        "validation_ranges": {k: [float(v[0]), float(v[1])] for k, v in ranges.items()},
        "real": summarize_stats(real_stats),
        "synthetic": summarize_stats(generated_stats),
        "metadata_path": str(metadata_path),
        "contact_sheet": str(contact_sheet_path),
        "dt_out_dir": str(dt_out_dir) if dt_out_dir is not None else None,
        "topology_out_dir": str(topology_out_dir) if topology_out_dir is not None else None,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    make_contact_sheet(
        written_paths[: args.contact_sheet_count],
        contact_sheet_path,
        tile_size=args.contact_sheet_tile,
        cols=args.contact_sheet_cols,
    )

    print(f"done. masks={len(written_paths)} out_dir={out_dir}")
    print(f"metadata={metadata_path}")
    print(f"summary={summary_path}")
    print(f"contact_sheet={contact_sheet_path}")
    if dt_out_dir is not None:
        print(f"dt_out_dir={dt_out_dir}")
    if topology_out_dir is not None:
        print(f"topology_out_dir={topology_out_dir}")


if __name__ == "__main__":
    main()
