import os
import glob
import random
import numpy as np
from PIL import Image

DT_DIR = "/CrackTree260/cond_dt"
OUT_TXT = "/CrackTree260/eval_ids.txt"
N_TOP = 4
N_BOTTOM = 4
N_RANDOM = 2
N_RANDOM_EXTRA = 10  # 额外随机 10 张，与上面共 20 张作为 test
TH = 32         # 高亮阈值，与你训练脚本里一致即可
SEED = 123

def score_dt(path):
    im = Image.open(path).convert("L")
    arr = np.array(im, dtype=np.uint8)
    return float((arr > TH).mean())

paths = sorted(glob.glob(os.path.join(DT_DIR, "*.png")))
if not paths:
    raise RuntimeError(f"No png found in {DT_DIR}")

scores = []
for p in paths:
    s = score_dt(p)
    base = os.path.splitext(os.path.basename(p))[0]  # 6192
    scores.append((base, s))

# 过滤掉几乎全黑的（避免完全无裂缝的极端样本）
scores_nonempty = [(i, s) for (i, s) in scores if s > 0.0001]
if len(scores_nonempty) < (N_TOP + N_BOTTOM + N_RANDOM):
    scores_nonempty = scores  # 实在不够就不滤

scores_sorted = sorted(scores_nonempty, key=lambda x: x[1])
bottom = [i for i, _ in scores_sorted[:N_BOTTOM]]
top = [i for i, _ in scores_sorted[-N_TOP:]]

pool = [i for i, _ in scores_sorted]
random.seed(SEED)
rand = []
for _ in range(N_RANDOM):
    cand = random.choice(pool)
    while cand in set(top + bottom + rand):
        cand = random.choice(pool)
    rand.append(cand)

ids = top + bottom + rand

# 再随机 10 张，共 20 张作为 test（同一 SEED 可复现）
extra_rand = []
for _ in range(N_RANDOM_EXTRA):
    cand = random.choice(pool)
    while cand in set(ids + extra_rand):
        cand = random.choice(pool)
    extra_rand.append(cand)
ids = ids + extra_rand

with open(OUT_TXT, "w", encoding="utf-8") as f:
    for i in ids:
        f.write(str(i) + "\n")

print("wrote", OUT_TXT)
print("ids:", ids)