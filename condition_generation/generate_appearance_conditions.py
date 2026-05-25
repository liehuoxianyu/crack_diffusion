#!/usr/bin/env python3
"""Generate appearance-aware ControlNet conditions from pavement images.

Default RGB layout:
  R = low-frequency shading / illumination
  G = Canny edge map
  B = high-frequency texture residual

These conditions are intended to complement binary-derived topology maps with
surface appearance cues available from an original or reference pavement image.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_jsonl_image_paths(path: Path) -> list[Path]:
    image_paths: list[Path] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            image_path = item.get("image")
            if image_path:
                image_paths.append(Path(image_path))
    return image_paths


def normalize_u8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if hi <= lo + 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - lo) / (hi - lo)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def make_appearance_condition(image_bgr: np.ndarray, blur: int, canny1: int, canny2: int) -> np.ndarray:
    if blur % 2 == 0:
        blur += 1
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    shading = cv2.GaussianBlur(gray, (blur, blur), 0)
    edges = cv2.Canny(gray, canny1, canny2)

    gray_f = gray.astype(np.float32)
    shading_f = shading.astype(np.float32)
    texture = np.abs(gray_f - shading_f)
    texture_u8 = normalize_u8(texture)

    return cv2.merge([shading, edges, texture_u8])


def output_name(path: Path) -> str:
    return f"{path.stem}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image-dir")
    group.add_argument("--image-jsonl")
    parser.add_argument("--out-dir", default="/CrackTree260/cond_appearance")
    parser.add_argument("--blur", type=int, default=51)
    parser.add_argument("--canny1", type=int, default=50)
    parser.add_argument("--canny2", type=int, default=150)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_jsonl:
        paths = read_jsonl_image_paths(Path(args.image_jsonl))
    else:
        paths = list_images(Path(args.image_dir))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError("no input images found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"skip unreadable image: {path}")
            continue
        cond = make_appearance_condition(image, blur=args.blur, canny1=args.canny1, canny2=args.canny2)
        cv2.imwrite(str(out_dir / output_name(path)), cond)
        written += 1
        if written % 100 == 0:
            print(f"processed {written}/{len(paths)}")

    print(f"done. out_dir={out_dir} num={written}")


if __name__ == "__main__":
    main()
