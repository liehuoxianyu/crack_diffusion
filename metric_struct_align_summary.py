import os
import re
import glob
import csv
import numpy as np
import cv2

# ===================== 配置区（只改这里） =====================
EVAL_ROOT = "/work/outputs/exp_eval_all"

# 评测哪些图片：默认只评测controlnet输出的单图（不含sd_only）
GLOB_PATTERN = "id*/controlnet_*/step*/gs*_cs*.png"
# 如果你只想评测最终一步：改成下面这行
# GLOB_PATTERN = "id*/controlnet_*/step2000/gs*_cs*.png"

COND_ROOT_BINARY = "/CrackTree260/cond_mask"
COND_ROOT_DT     = "/CrackTree260/cond_dt"

OUT_ALL_ROWS = "/work/outputs/exp_eval_all/metrics_all_rows.csv"
OUT_SUMMARY  = "/work/outputs/exp_eval_all/metrics_summary_by_step.csv"

DILATE = 5
DT_TH = 32
CANNY1 = 50
CANNY2 = 150
MAX_DIST = 15.0
# ============================================================


def load_gray(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(p)
    return im

def make_allow_region(cond_gray, mode, dilate, dt_th):
    if mode == "binary":
        crack = (cond_gray > 127).astype(np.uint8)
        if dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate+1, 2*dilate+1))
            crack = cv2.dilate(crack, k, iterations=1)
        return crack.astype(bool)
    else:
        near = (cond_gray > dt_th).astype(np.uint8)
        if dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate+1, 2*dilate+1))
            near = cv2.dilate(near, k, iterations=1)
        return near.astype(bool)

def edge_map(gen_bgr, c1, c2):
    g = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(g, c1, c2)
    return (e > 0)

def chamfer_edge_to_allow(edge_bool, allow_bool, max_dist=15.0):
    if edge_bool.sum() == 0:
        return float(max_dist), 0
    allow_u8 = allow_bool.astype(np.uint8)
    inv = 1 - allow_u8
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)
    d = dt[edge_bool]
    d = np.clip(d, 0, max_dist)
    return float(d.mean()), int(edge_bool.sum())

def recall_allow_covered_by_edge(allow_bool, edge_bool, max_dist=15.0):
    allow_u8 = allow_bool.astype(np.uint8)
    if allow_u8.sum() == 0:
        return 0.0, 0
    edge_u8 = edge_bool.astype(np.uint8)
    inv = 1 - edge_u8
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)
    covered = (dt <= max_dist) & allow_bool
    return float(covered.sum() / allow_bool.sum()), int(allow_bool.sum())

def parse_from_path(gen_path):
    """
    从路径解析：id、mode(controlnet_binary/controlnet_dt)、step、gs、cs
    例如 .../id6192/controlnet_dt/step2000/gs6.0_cs1.5.png
    """
    p = gen_path.replace("\\", "/")
    m_id = re.search(r"/id([^/]+)/", p)
    m_mode = re.search(r"/controlnet_([^/]+)/", p)
    m_step = re.search(r"/step(\d+)/", p)
    m_gs = re.search(r"/gs([0-9\.]+)_cs([0-9\.]+)\.png$", p)

    id_ = m_id.group(1) if m_id else ""
    mode = m_mode.group(1) if m_mode else ""
    step = int(m_step.group(1)) if m_step else -1
    gs = float(m_gs.group(1)) if m_gs else float("nan")
    cs = float(m_gs.group(2)) if m_gs else float("nan")
    return id_, mode, step, gs, cs

def get_cond_root(mode):
    # mode 解析出来是 "binary" 或 "dt"
    if mode == "binary":
        return COND_ROOT_BINARY
    if mode == "dt":
        return COND_ROOT_DT
    return None

def summarize_group(values):
    a = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }

def main():
    gen_paths = sorted(glob.glob(os.path.join(EVAL_ROOT, GLOB_PATTERN)))
    if not gen_paths:
        raise RuntimeError(f"No generated images matched. EVAL_ROOT={EVAL_ROOT}, pattern={GLOB_PATTERN}")

    rows = []
    for gp in gen_paths:
        id_, mode, step, gs, cs = parse_from_path(gp)
        if not id_ or mode not in ["binary", "dt"] or step < 0:
            continue

        cond_root = get_cond_root(mode)
        cond_path = os.path.join(cond_root, f"{id_}.png")
        if not os.path.exists(cond_path):
            continue

        cond = load_gray(cond_path)
        allow = make_allow_region(cond, mode, DILATE, DT_TH)

        gen = cv2.imread(gp, cv2.IMREAD_COLOR)
        if gen is None:
            continue
        edge = edge_map(gen, CANNY1, CANNY2)

        chamfer, n_edge = chamfer_edge_to_allow(edge, allow, MAX_DIST)
        recall, n_allow = recall_allow_covered_by_edge(allow, edge, MAX_DIST)
        edge_density = float(edge.mean())

        rows.append({
            "gen_path": gp,
            "id": id_,
            "mode": mode,
            "step": step,
            "guidance_scale": gs,
            "control_scale": cs,
            "dilate": DILATE,
            "dt_th": DT_TH,
            "canny1": CANNY1,
            "canny2": CANNY2,
            "max_dist": MAX_DIST,
            "chamfer_edge_to_allow_mean": chamfer,
            "allow_recall": recall,
            "edge_density": edge_density,
            "n_edge_pixels": n_edge,
            "n_allow_pixels": n_allow,
        })

    if not rows:
        raise RuntimeError("No rows collected. Check eval outputs and paths.")

    os.makedirs(os.path.dirname(OUT_ALL_ROWS) or ".", exist_ok=True)

    # 1) 写逐图明细
    with open(OUT_ALL_ROWS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 2) 按 mode+step 汇总
    groups = {}
    for r in rows:
        key = (r["mode"], r["step"])
        groups.setdefault(key, []).append(r)

    summary_rows = []
    for (mode, step), rs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        chamfers = [x["chamfer_edge_to_allow_mean"] for x in rs]
        recalls  = [x["allow_recall"] for x in rs]
        dens     = [x["edge_density"] for x in rs]

        s1 = summarize_group(chamfers)
        s2 = summarize_group(recalls)
        s3 = summarize_group(dens)

        summary_rows.append({
            "mode": mode,
            "step": step,
            "n_images": len(rs),

            "chamfer_mean": s1["mean"],
            "chamfer_median": s1["median"],
            "chamfer_p10": s1["p10"],
            "chamfer_p90": s1["p90"],

            "recall_mean": s2["mean"],
            "recall_median": s2["median"],
            "recall_p10": s2["p10"],
            "recall_p90": s2["p90"],

            "edge_density_mean": s3["mean"],
            "edge_density_median": s3["median"],
            "edge_density_p10": s3["p10"],
            "edge_density_p90": s3["p90"],
        })

    with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("wrote:", OUT_ALL_ROWS, "rows=", len(rows))
    print("wrote:", OUT_SUMMARY, "groups=", len(summary_rows))

    # 终端打印一个简短总览（每个mode最后一个step）
    last = {}
    for r in summary_rows:
        last[r["mode"]] = max(last.get(r["mode"], -1), r["step"])
    for mode, last_step in last.items():
        for r in summary_rows:
            if r["mode"] == mode and r["step"] == last_step:
                print(f"\n[{mode}] last_step={last_step} n={r['n_images']}")
                print("  chamfer_mean:", r["chamfer_mean"], "(lower better)")
                print("  recall_mean :", r["recall_mean"],  "(higher better)")
                print("  edge_density_mean:", r["edge_density_mean"])

if __name__ == "__main__":
    main()