import os
import json
import argparse

DEFAULT_PROMPT = "a photo of pavement crack, realistic texture, high detail, natural shadowing, realistic lighting"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def stem(fn: str) -> str:
    return os.path.splitext(os.path.basename(fn))[0]

def list_images(image_dir):
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(IMG_EXTS)]
    files.sort()
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/CrackTree260")
    ap.add_argument("--image_dir", type=str, default="image")
    ap.add_argument("--out", type=str, default="/CrackTree260/train_linux.jsonl")
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)

    # 可选：提前检查条件图是否齐全（因为训练时一定会用到）
    ap.add_argument("--check_cond_dir", type=str, default="", help="例如 cond_mask 或 cond_dt；留空则不检查")
    ap.add_argument("--skip_missing_cond", action="store_true", help="缺条件图时跳过该样本（默认是报错）")
    args = ap.parse_args()

    image_dir = os.path.join(args.root, args.image_dir)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    cond_dir = os.path.join(args.root, args.check_cond_dir) if args.check_cond_dir else None
    if args.check_cond_dir and (not os.path.isdir(cond_dir)):
        raise FileNotFoundError(f"cond_dir not found: {cond_dir}")

    imgs = list_images(image_dir)
    if not imgs:
        raise RuntimeError(f"No images found in {image_dir}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n_write, n_skip = 0, 0
    last = None
    with open(args.out, "w", encoding="utf-8") as f:
        for fn in imgs:
            img_path = os.path.join(image_dir, fn)

            if cond_dir is not None:
                cond_path = os.path.join(cond_dir, stem(fn) + ".png")
                if not os.path.exists(cond_path):
                    if args.skip_missing_cond:
                        n_skip += 1
                        continue
                    raise FileNotFoundError(f"missing conditioning image: {cond_path}")

            rec = {"image": img_path, "prompt": args.prompt}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_write += 1
            last = rec

    print(f"OK: wrote {n_write} lines to {args.out}, skipped={n_skip}")
    if last:
        print("Example:", last)
    if cond_dir is not None:
        print("Checked cond_dir:", cond_dir)

if __name__ == "__main__":
    main()