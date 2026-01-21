import os, json, shutil
from tqdm import tqdm

JSONL_IN = "/CrackTree260/train_linux.jsonl"
OUT_DIR  = "/CrackTree260/diffusers_train"

IMG_OUT  = os.path.join(OUT_DIR, "images")
COND_OUT = os.path.join(OUT_DIR, "conditioning_images")
META_OUT = os.path.join(OUT_DIR, "metadata.jsonl")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(COND_OUT, exist_ok=True)

def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    # 先硬链接（快且不占额外空间），失败就复制
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

n = 0
with open(JSONL_IN, "r", encoding="utf-8") as r, open(META_OUT, "w", encoding="utf-8") as w:
    for line in tqdm(r):
        item = json.loads(line)
        img_path  = item["image"]
        cond_path = item["conditioning_image"]
        prompt    = item.get("prompt", "")

        img_name  = os.path.basename(img_path)   # e.g. 6192.jpg
        cond_name = os.path.basename(cond_path)  # e.g. 6192.png

        link_or_copy(img_path,  os.path.join(IMG_OUT, img_name))
        link_or_copy(cond_path, os.path.join(COND_OUT, cond_name))

        # metadata.jsonl：file_name 指向 images 里的文件名；text 是prompt
        w.write(json.dumps({"file_name": img_name, "text": prompt}, ensure_ascii=False) + "\n")
        n += 1

print("DONE. out_dir:", OUT_DIR)
print("num_samples:", n)
print("metadata:", META_OUT)