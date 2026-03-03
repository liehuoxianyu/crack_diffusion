"""
FID 与 PSNR 评测：与 eval_all 输出对齐，仅针对固定 control_scale=1.0（与 metric_struct_align 一致）。
- 从 train_linux.jsonl 取 real 图路径（按 id 匹配 basename）。
- 生成图路径：EVAL_ROOT/id{id}/controlnet_{mode}/step{step}/gs{gs}_cs{cs}.png。
- PSNR：逐对 (生成图, real 图)，先统一尺寸再算。
- FID：按 (mode, step) 分别算 real 集合 vs 生成集合（需建临时目录或列表，clean-fid 支持路径列表时用列表）。
"""
import os
import json
import csv
import tempfile
import numpy as np
from PIL import Image

# ===================== 配置区 =====================
EVAL_ROOT = "/work/outputs/exp_eval_all"
EVAL_IDS_TXT = "/CrackTree260/eval_ids.txt"
REAL_IMAGE_JSONL = "/CrackTree260/train_linux.jsonl"

# 与下游 structural 分析一致：只取 control_scale=1.0
GS = 7.5
CS = 1.0
# 要评测的 step 列表（与 eval_all 的 CKPTS 一致）
STEPS = [500, 1000, 1500, 2000]
# 要评测的 mode：controlnet_binary, controlnet_dt, controlnet_binary_lora, controlnet_dt_lora
MODES = ("binary", "dt", "binary_lora", "dt_lora")

OUT_PSNR_CSV = "/work/outputs/exp_eval_all/metric_psnr_per_image.csv"
OUT_FID_PSNR_SUMMARY = "/work/outputs/exp_eval_all/metric_fid_psnr_summary.csv"
# FID 计算设备（无显卡时用 "cpu"）
FID_DEVICE = "cpu"
# ==================================================


def read_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                ids.append(t)
    return ids


def build_id_to_real_path(jsonl_path):
    """从 jsonl 的 image 字段建立 id(basename 无扩展名) -> 绝对路径。"""
    out = {}
    if not os.path.exists(jsonl_path):
        return out
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                it = json.loads(line)
                img_path = it.get("image", "")
                if not img_path:
                    continue
                # 统一为 / 并取 basename 无扩展名
                p = img_path.replace("\\", "/")
                base = os.path.splitext(os.path.basename(p))[0]
                if not base:
                    continue
                # 若路径非绝对，可在这里补前缀；当前假定 jsonl 里已是绝对路径或可访问
                if not os.path.isabs(p):
                    # 常见：/CrackTree260/... 已在 jsonl
                    pass
                out[base] = p
            except Exception:
                continue
    return out


def psnr_numpy(img1, img2, max_val=255.0):
    """两图尺寸需一致；uint8 或 float。"""
    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("shape mismatch for PSNR")
    mse = np.mean((a - b) ** 2)
    if mse <= 0:
        return float(max_val ** 2)  # 常约定为 100 或类似
    return float(10.0 * np.log10(max_val ** 2 / mse))


def get_generated_path(eval_root, id_, mode, step, gs, cs):
    return os.path.join(
        eval_root,
        f"id{id_}",
        f"controlnet_{mode}",
        f"step{step}",
        f"gs{gs}_cs{cs}.png",
    )


def main():
    os.makedirs(os.path.dirname(OUT_PSNR_CSV) or ".", exist_ok=True)
    ids = read_ids(EVAL_IDS_TXT)
    id_to_real = build_id_to_real_path(REAL_IMAGE_JSONL)
    if not id_to_real:
        print("WARNING: no real paths from", REAL_IMAGE_JSONL)

    # ---------- PSNR 逐图 ----------
    psnr_rows = []
    for id_ in ids:
        real_path = id_to_real.get(id_)
        if not real_path or not os.path.exists(real_path):
            continue
        try:
            real_img = np.array(Image.open(real_path).convert("RGB"))
        except Exception:
            continue
        h, w = real_img.shape[:2]
        for mode in MODES:
            for step in STEPS:
                gen_path = get_generated_path(EVAL_ROOT, id_, mode, step, GS, CS)
                if not os.path.exists(gen_path):
                    continue
                try:
                    gen_img = np.array(Image.open(gen_path).convert("RGB"))
                except Exception:
                    continue
                # 将生成图 resize 到 real 尺寸再算 PSNR
                from PIL import Image as PILImage
                gen_pil = PILImage.fromarray(gen_img)
                gen_resized = np.array(gen_pil.resize((w, h), PILImage.BICUBIC))
                try:
                    p = psnr_numpy(real_img, gen_resized)
                except Exception:
                    p = float("nan")
                psnr_rows.append({
                    "id": id_,
                    "mode": mode,
                    "step": step,
                    "psnr": p,
                    "real_path": real_path,
                    "gen_path": gen_path,
                })

    if psnr_rows:
        with open(OUT_PSNR_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(psnr_rows[0].keys()))
            writer.writeheader()
            writer.writerows(psnr_rows)
        print("wrote:", OUT_PSNR_CSV, "rows=", len(psnr_rows))

    # ---------- 按 (mode, step) 汇总 PSNR ----------
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in psnr_rows:
        by_key[(r["mode"], r["step"])].append(r["psnr"])

    summary_rows = []
    for (mode, step), vals in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        a = np.asarray([v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))], dtype=np.float64)
        if len(a) == 0:
            psnr_mean = psnr_median = float("nan")
        else:
            psnr_mean = float(np.mean(a))
            psnr_median = float(np.median(a))
        summary_rows.append({
            "mode": mode,
            "step": step,
            "n_pairs": len(vals),
            "psnr_mean": psnr_mean,
            "psnr_median": psnr_median,
        })

    # ---------- FID：每个 (mode, step) 一组 ----------
    for s in summary_rows:
        s["fid"] = float("nan")
    try:
        from cleanfid import fid
    except ImportError:
        print("WARNING: clean-fid not installed. pip install clean-fid. Skipping FID.")
        fid = None

    if fid is not None:
        for s in summary_rows:
            mode, step = s["mode"], s["step"]
            real_paths = []
            gen_paths = []
            for id_ in ids:
                real_path = id_to_real.get(id_)
                if not real_path or not os.path.exists(real_path):
                    continue
                gen_path = get_generated_path(EVAL_ROOT, id_, mode, step, GS, CS)
                if not os.path.exists(gen_path):
                    continue
                real_paths.append(os.path.abspath(real_path))
                gen_paths.append(os.path.abspath(gen_path))
            if len(real_paths) < 2:
                continue
            try:
                with tempfile.TemporaryDirectory(prefix="fid_real_") as real_dir, \
                     tempfile.TemporaryDirectory(prefix="fid_gen_") as gen_dir:
                    for i, (rp, gp) in enumerate(zip(real_paths, gen_paths)):
                        name = f"{i:06d}.png"
                        try:
                            os.symlink(rp, os.path.join(real_dir, name))
                            os.symlink(gp, os.path.join(gen_dir, name))
                        except OSError:
                            from shutil import copy2
                            copy2(rp, os.path.join(real_dir, name))
                            copy2(gp, os.path.join(gen_dir, name))
                    s["fid"] = fid.compute_fid(real_dir, gen_dir, device=FID_DEVICE, mode="clean")
            except Exception as e:
                print("FID error", mode, step, e)

    with open(OUT_FID_PSNR_SUMMARY, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "step", "n_pairs", "psnr_mean", "psnr_median", "fid"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print("wrote:", OUT_FID_PSNR_SUMMARY)
    for s in summary_rows:
        print(" ", s["mode"], "step%d" % s["step"], "psnr_mean=%.2f" % (s["psnr_mean"] if not np.isnan(s["psnr_mean"]) else float("nan")), "fid=%.2f" % (s["fid"] if not np.isnan(s["fid"]) else float("nan")))


if __name__ == "__main__":
    main()
