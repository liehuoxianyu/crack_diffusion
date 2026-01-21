import json

IN_PATH = "/CrackTree260/train_linux.jsonl"
OUT_PATH = "/CrackTree260/train_dt.jsonl"

with open(IN_PATH, "r", encoding="utf-8") as r, open(OUT_PATH, "w", encoding="utf-8") as w:
    for line in r:
        it = json.loads(line)
        it["conditioning_image"] = it["conditioning_image"].replace("/CrackTree260/cond_mask/", "/CrackTree260/cond_dt/")
        w.write(json.dumps(it, ensure_ascii=False) + "\n")

print("wrote", OUT_PATH)