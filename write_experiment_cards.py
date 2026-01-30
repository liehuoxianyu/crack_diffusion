import os
import json
import subprocess
from datetime import datetime

# ===================== 配置区（只改这里） =====================
RUNS = {
    "binary": "/work/outputs/exp_binary_patch512",
    "dt":     "/work/outputs/exp_dt_patch512",
}
EVAL_IDS_TXT = "/CrackTree260/eval_ids.txt"

DATA_JSONL = "/CrackTree260/train_linux.jsonl"  # 你当前 loader 用的
DATA_IMAGE_DIR = "/CrackTree260/image"
COND_DIRS = {
    "binary": "/CrackTree260/cond_mask",
    "dt":     "/CrackTree260/cond_dt",
}

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_INIT = "lllyasviel/sd-controlnet-seg"

# 你训练时使用的关键超参（写入卡片用）
TRAIN_CFG = {
    "resolution": 512,
    "learning_rate": "5e-6",
    "train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_train_steps": 2000,
    "checkpointing_steps": 500,
    "seed": 42,
    "mixed_precision": "fp16",
}

# 推理协议（写入卡片用）
INFER_CFG = {
    "prompt": "a photo of pavement crack, realistic texture, high detail, natural shadowing, realistic lighting",
    "steps": 25,
    "guidance_scales": [6.0, 7.5],
    "control_scales": [1.0, 1.5, 2.0],
    "seed_base": 42,
}
# ============================================================


def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"<failed: {e}>"

def list_checkpoints(run_dir):
    if not os.path.isdir(run_dir):
        return []
    xs = []
    for name in os.listdir(run_dir):
        if name.startswith("checkpoint-"):
            xs.append(name)
    def key(n):
        try:
            return int(n.split("-")[1])
        except:
            return 10**18
    xs.sort(key=key)
    return xs

def file_exists(p):
    return "YES" if os.path.exists(p) else "NO"

def read_eval_ids(path):
    if not os.path.exists(path):
        return []
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                ids.append(t)
    return ids

def write_card(tag, run_dir, out_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ckpts = list_checkpoints(run_dir)
    train_log = os.path.join(run_dir, "train.log")
    tb_dir = os.path.join(run_dir, "logs")

    # 尽力记录环境版本
    torch_ver = run_cmd(["python", "-c", "import torch; print(torch.__version__)"])
    diffusers_ver = run_cmd(["python", "-c", "import diffusers; print(diffusers.__version__)"])
    datasets_ver = run_cmd(["python", "-c", "import datasets; print(datasets.__version__)"])
    accelerate_ver = run_cmd(["python", "-c", "import accelerate; print(accelerate.__version__)"])

    # 关键环境变量（你当前实验设计依赖这些）
    env_keys = [
        "CRACK_JSONL", "CRACK_COND_DIR", "CRACK_USE_PATCH", "CRACK_PATCH",
        "CRACK_TRY", "CRACK_TH", "CRACK_P_RANDOM", "CRACK_CROP_SEED",
    ]
    env_dump = {k: os.environ.get(k, "<not set in this shell>") for k in env_keys}

    eval_ids = read_eval_ids(EVAL_IDS_TXT)

    md = []
    md.append(f"# Experiment Card - {tag}\n")
    md.append(f"- Generated at: `{now}`\n")
    md.append(f"- Run dir: `{run_dir}`\n\n")

    md.append("## 1. Goal\n")
    md.append("- ControlNet fine-tuning for pavement crack generation with strong structural control.\n")
    md.append("- Compare binary mask vs DT heatmap conditioning.\n\n")

    md.append("## 2. Data\n")
    md.append(f"- jsonl: `{DATA_JSONL}` (exists={file_exists(DATA_JSONL)})\n")
    md.append(f"- image dir: `{DATA_IMAGE_DIR}` (exists={file_exists(DATA_IMAGE_DIR)})\n")
    md.append(f"- cond dir ({tag}): `{COND_DIRS.get(tag,'?')}` (exists={file_exists(COND_DIRS.get(tag,'/'))})\n")
    md.append(f"- eval ids: `{EVAL_IDS_TXT}` (n={len(eval_ids)})\n")
    if eval_ids:
        md.append(f"  - ids: {', '.join(eval_ids)}\n")
    md.append("\n")

    md.append("## 3. Models\n")
    md.append(f"- base model: `{BASE_MODEL}`\n")
    md.append(f"- controlnet init: `{CONTROLNET_INIT}`\n\n")

    md.append("## 4. Training config\n")
    md.append("```json\n")
    md.append(json.dumps(TRAIN_CFG, indent=2, ensure_ascii=False))
    md.append("\n```\n\n")

    md.append("## 5. Inference protocol (evaluation)\n")
    md.append("```json\n")
    md.append(json.dumps(INFER_CFG, indent=2, ensure_ascii=False))
    md.append("\n```\n\n")

    md.append("## 6. Loader settings (env vars snapshot)\n")
    md.append("> Note: values may show `<not set in this shell>` if you run this script in a different shell.\n\n")
    md.append("```json\n")
    md.append(json.dumps(env_dump, indent=2, ensure_ascii=False))
    md.append("\n```\n\n")

    md.append("## 7. Artifacts\n")
    md.append(f"- train log: `{train_log}` (exists={file_exists(train_log)})\n")
    md.append(f"- tensorboard logdir: `{tb_dir}` (exists={file_exists(tb_dir)})\n")
    md.append(f"- checkpoints ({len(ckpts)}): {ckpts}\n\n")

    md.append("## 8. Environment\n")
    md.append(f"- torch: `{torch_ver}`\n")
    md.append(f"- diffusers: `{diffusers_ver}`\n")
    md.append(f"- datasets: `{datasets_ver}`\n")
    md.append(f"- accelerate: `{accelerate_ver}`\n\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(md)

    print("wrote", out_path)

def main():
    os.makedirs("/work/outputs", exist_ok=True)
    for tag, run_dir in RUNS.items():
        out_path = os.path.join(run_dir, "experiment_card.md")
        write_card(tag, run_dir, out_path)

if __name__ == "__main__":
    main()