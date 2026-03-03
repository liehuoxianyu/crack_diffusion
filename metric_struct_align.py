import os
import re
import glob
import csv
import numpy as np
import cv2

# ===================== 配置区（只改这里） =====================
EVAL_ROOT = "/work/outputs/exp_eval_all"

# 评测哪些图片：匹配 eval_all 的 controlnet_*（含 binary/dt/binary_lora/dt_lora）
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

# 边缘提取：统一流程 灰度 → 可选 CLAHE → Canny → 可选细化
USE_CLAHE = False
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
USE_THINNING = False
# IoU_r / Dice_r：对 pred_edge 与 GT 二值图分别用半径 r 的盘形膨胀后再算 IoU/Dice
R_IOU = 3
# 试跑时只处理前 N 张图（None = 不限制）
LIMIT_N = None
# 只保留该 control_scale 用于后续分析，与 FID/PSNR 一致（None = 保留全部）
CONTROL_SCALE_FILTER = 1.0
# ============================================================


def load_gray(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(p)
    return im

def make_allow_region(cond_gray, cond_type, dilate, dt_th):
    """cond_type: 'binary' 用 >127:'dt' 用 >dt_th。"""
    if cond_type == "binary":
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

def _apply_clahe(gray_uint8, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(gray_uint8)


def _apply_thinning(edge_uint8):
    """细化边缘（单像素宽）。需要 opencv-contrib 的 ximgproc，否则返回原图。"""
    try:
        return cv2.ximgproc.thinning(edge_uint8)
    except (AttributeError, cv2.error):
        return edge_uint8


def edge_map_unified(gen_bgr, c1, c2, use_clahe=False, clahe_clip=2.0, clahe_grid=(8, 8), use_thinning=False):
    """
    统一边缘提取：灰度 → 可选 CLAHE → Canny(c1,c2) → 可选细化。
    返回 bool 型二值边缘图，与原有 Chamfer/Recall 共用。
    """
    g = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        g = _apply_clahe(g, clip_limit=clahe_clip, grid_size=clahe_grid)
    e = cv2.Canny(g, c1, c2)
    if use_thinning:
        e = _apply_thinning(e)
    return (e > 0).astype(np.uint8)  # 保持 uint8 便于形态学，返回处再转 bool 也可


def disk_kernel(r):
    """半径 r 的盘形结构元（用于膨胀）。"""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def iou_r_dice_r(pred_edge_bool, gt_bin_bool, r):
    """
    带容忍半径 r 的 IoU 与 Dice：对 pred_edge 和 gt_bin 分别用半径 r 的盘形膨胀后再计算。
    pred_edge_bool, gt_bin_bool: 二值 bool 或 uint8 (0/1)。
    """
    pred = pred_edge_bool.astype(np.uint8)
    gt = gt_bin_bool.astype(np.uint8)
    if r <= 0:
        pred_dil = pred
        gt_dil = gt
    else:
        k = disk_kernel(r)
        pred_dil = cv2.dilate(pred, k)
        gt_dil = cv2.dilate(gt, k)
    pred_dil = pred_dil.astype(bool)
    gt_dil = gt_dil.astype(bool)
    inter = np.logical_and(pred_dil, gt_dil).sum()
    union = np.logical_or(pred_dil, gt_dil).sum()
    a_sum = pred_dil.sum()
    b_sum = gt_dil.sum()
    iou_r = float(inter / union) if union > 0 else 0.0
    dice_r = float(2 * inter / (a_sum + b_sum)) if (a_sum + b_sum) > 0 else 0.0
    return iou_r, dice_r


def edge_map(gen_bgr, c1, c2):
    """保留旧接口：直接 Canny，无 CLAHE/细化。内部改用统一流程以保持一致性。"""
    e = edge_map_unified(
        gen_bgr, c1, c2,
        use_clahe=USE_CLAHE,
        clahe_clip=CLAHE_CLIP_LIMIT,
        clahe_grid=CLAHE_GRID_SIZE,
        use_thinning=USE_THINNING,
    )
    return e.astype(bool)

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
    从路径解析：id、mode(controlnet_binary/controlnet_dt/controlnet_binary_lora/controlnet_dt_lora)、step、gs、cs
    例如 .../id6192/controlnet_dt/step2000/gs7.5_cs1.5.png 或 .../controlnet_dt_lora/step1000/...
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

VALID_MODES = ("binary", "dt", "binary_lora", "dt_lora")

def get_cond_root(mode):
    """binary / binary_lora 用二值条件图；dt / dt_lora 用 DT 条件图。"""
    if mode in ("binary", "binary_lora"):
        return COND_ROOT_BINARY
    if mode in ("dt", "dt_lora"):
        return COND_ROOT_DT
    return None

def cond_type_for_allow(mode):
    """用于 make_allow_region 的条件类型:binary 或 dt。"""
    return "binary" if mode in ("binary", "binary_lora") else "dt"

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
    if LIMIT_N is not None:
        gen_paths = gen_paths[: int(LIMIT_N)]
        print("LIMIT_N=%s: processing only first %d images." % (LIMIT_N, len(gen_paths)))
    if CONTROL_SCALE_FILTER is not None:
        print("CONTROL_SCALE_FILTER=%s: only images with control_scale=%s." % (CONTROL_SCALE_FILTER, CONTROL_SCALE_FILTER))

    rows = []
    for gp in gen_paths:
        id_, mode, step, gs, cs = parse_from_path(gp)
        if not id_ or mode not in VALID_MODES or step < 0:
            continue
        if CONTROL_SCALE_FILTER is not None and cs != CONTROL_SCALE_FILTER:
            continue

        cond_root = get_cond_root(mode)
        if cond_root is None:
            continue
        cond_path = os.path.join(cond_root, f"{id_}.png")
        if not os.path.exists(cond_path):
            continue

        cond = load_gray(cond_path)
        allow = make_allow_region(cond, cond_type_for_allow(mode), DILATE, DT_TH)

        gen = cv2.imread(gp, cv2.IMREAD_COLOR)
        if gen is None:
            continue
        edge = edge_map(gen, CANNY1, CANNY2)

        chamfer, n_edge = chamfer_edge_to_allow(edge, allow, MAX_DIST)
        recall, n_allow = recall_allow_covered_by_edge(allow, edge, MAX_DIST)
        edge_density = float(edge.mean())

        # IoU_r / Dice_r：GT 为裂缝二值图（cond_mask），与 pred 边缘在半径 r 膨胀后比对
        gt_bin_path = os.path.join(COND_ROOT_BINARY, f"{id_}.png")
        if os.path.exists(gt_bin_path):
            gt_gray = load_gray(gt_bin_path)
            gt_bin = (gt_gray > 127).astype(np.uint8)
            iou_r, dice_r = iou_r_dice_r(edge.astype(np.uint8), gt_bin, R_IOU)
        else:
            iou_r = dice_r = float("nan")

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
            "r_iou": R_IOU,
            "use_clahe": USE_CLAHE,
            "use_thinning": USE_THINNING,
            "chamfer_edge_to_allow_mean": chamfer,
            "allow_recall": recall,
            "edge_density": edge_density,
            "n_edge_pixels": n_edge,
            "n_allow_pixels": n_allow,
            "iou_r": iou_r,
            "dice_r": dice_r,
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
        iou_vals = [x["iou_r"] for x in rs if isinstance(x["iou_r"], (int, float)) and not (isinstance(x["iou_r"], float) and np.isnan(x["iou_r"]))]
        dice_vals = [x["dice_r"] for x in rs if isinstance(x["dice_r"], (int, float)) and not (isinstance(x["dice_r"], float) and np.isnan(x["dice_r"]))]

        s1 = summarize_group(chamfers)
        s2 = summarize_group(recalls)
        s3 = summarize_group(dens)
        s4 = summarize_group(iou_vals) if iou_vals else {"mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
        s5 = summarize_group(dice_vals) if dice_vals else {"mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}

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

            "iou_r_mean": s4["mean"],
            "iou_r_median": s4["median"],
            "iou_r_p10": s4["p10"],
            "iou_r_p90": s4["p90"],

            "dice_r_mean": s5["mean"],
            "dice_r_median": s5["median"],
            "dice_r_p10": s5["p10"],
            "dice_r_p90": s5["p90"],
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
                print("  iou_r_mean  :", r["iou_r_mean"],  "(higher better)")
                print("  dice_r_mean :", r["dice_r_mean"], "(higher better)")

if __name__ == "__main__":
    main()