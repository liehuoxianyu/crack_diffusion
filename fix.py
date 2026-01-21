import json
import os

IN_PATH  = "/CrackTree260/train.jsonl"
OUT_PATH = "/CrackTree260/train_linux.jsonl"

LINUX_ROOT = "/CrackTree260"

def win_to_linux(p: str) -> str:
    p2 = p.replace("\\", "/")

    key = "CrackTree260/"
    idx = p2.lower().find(key.lower())
    if idx != -1:
        rel = p2[idx + len(key):]  
        return os.path.join(LINUX_ROOT, rel)

    return os.path.join(LINUX_ROOT, os.path.basename(p2))

def main():
    n = 0
    bad = 0
    with open(IN_PATH, "r", encoding="utf-8") as r, open(OUT_PATH, "w", encoding="utf-8") as w:
        for line in r:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["image"] = win_to_linux(item["image"])
            item["conditioning_image"] = win_to_linux(item["conditioning_image"])

            ok1 = os.path.exists(item["image"])
            ok2 = os.path.exists(item["conditioning_image"])
            if not (ok1 and ok2):
                bad += 1

            w.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1

    print("wrote:", OUT_PATH)
    print("num_samples:", n)
    print("not_found_after_fix:", bad)

if __name__ == "__main__":
    main()