import os
import cv2
import numpy as np
from tqdm import tqdm

IN_DIR = "/CrackTree260/cond_mask"
OUT_DIR = "/CrackTree260/cond_dt"
os.makedirs(OUT_DIR, exist_ok=True)

# 控制“热力图”厚度/衰减：值越大，影响范围越宽
# 细裂缝建议 10~25 之间试；先用 15
SIGMA = 15.0

def to_dt_heat(mask_u8: np.ndarray) -> np.ndarray:
    """
    mask_u8: 0/255 二值，裂缝为255
    return: 0~255 热力图，裂缝中心最亮，远离逐渐变暗
    """
    crack = (mask_u8 > 127).astype(np.uint8)
    # distanceTransform 计算到最近的 0 的距离，所以我们对 crack 做反转
    inv = 1 - crack  # 裂缝处为0，背景为1
    dt = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)

    # 将距离映射成热力值：exp(-d^2 / (2*sigma^2))
    heat = np.exp(-(dt ** 2) / (2 * (SIGMA ** 2)))
    # 裂缝中心（dt=0）heat=1，远处趋近0
    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    return heat_u8

files = [f for f in os.listdir(IN_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
files.sort()

for fn in tqdm(files, desc="DT"):
    p = os.path.join(IN_DIR, fn)
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"failed to read {p}")
    heat = to_dt_heat(m)
    out = os.path.join(OUT_DIR, os.path.splitext(fn)[0] + ".png")
    cv2.imwrite(out, heat)

print("done. out_dir =", OUT_DIR, "num =", len(files))