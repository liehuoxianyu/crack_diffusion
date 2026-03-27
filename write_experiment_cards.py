"""
write_experiment_cards.py — 为 ControlNet/LoRA 训练 run 生成 experiment_card.md

在每个 RUNS 指定的目录下写入 experiment_card.md，记录数据路径、模型、训练/推理配置、
环境版本与产物，便于复现与对比。与 eval_all.py 的推理协议保持一致。
"""
import os
import json
import subprocess
from datetime import datetime

# ===================== 配置区（只改这里） =====================
RUNS = {
    "binary": "/work/outputs/exp_binary_patch512",
    "dt":     "/work/outputs/exp_dt_patch512",
    "lora":   "/work/outputs/exp_lora_realism",
}
EVAL_IDS_TXT = "/CrackTree260/eval_ids.txt"

DATA_JSONL = "/CrackTree260/train_linux.jsonl"
DATA_IMAGE_DIR = "/CrackTree260/image"
COND_DIRS = {
    "binary": "/CrackTree260/cond_mask",
    "dt":     "/CrackTree260/cond_dt",
}

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_INIT = "lllyasviel/sd-controlnet-seg"

# 训练超参（写入卡片用）
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

# 推理协议（与 eval_all.py 一致，写入卡片用）
INFER_CFG = {
    "prompt": "a photo of pavement crack, realistic texture, high detail, natural shadowing, realistic lighting",
    "steps": 25,
    "guidance_scales": [7.5],
    "control_scales": [0.75, 1.0, 1.25],
    "seed_base": 42,
}
# ============================================================


def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=10)
        return out.strip()
    except Exception as e:
        return f"<failed: {e}>"


def get_versions():
    """一次子进程获取 torch / diffusers / datasets / accelerate 版本。"""
    code = (
        "import json; "
        "d={}; "
        "for p in ['torch','diffusers','datasets','accelerate']: "
        "  try: m=__import__(p); d[p]=getattr(m,'__version__','?'); "
        "  except Exception as e: d[p]=f'<{e!r}>'; "
        "print(json.dumps(d))"
    )
    try:
        out = subprocess.check_output(
            ["python", "-c", code], stderr=subprocess.DEVNULL, text=True, timeout=15
        )
        return json.loads(out.strip())
    except Exception:
        return {"torch": "?", "diffusers": "?", "datasets": "?", "accelerate": "?"}


def list_checkpoints(run_dir):
    if not os.path.isdir(run_dir):
        return []
    xs = [n for n in os.listdir(run_dir) if n.startswith("checkpoint-")]
    def key(n):
        try:
            return int(n.split("-")[1])
        except (IndexError, ValueError):
            return 10**9
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


def format_eval_ids_summary(ids, max_show=10):
    """若 id 过多则只展示前几项 + ... + 后几项，避免卡片过长。"""
    n = len(ids)
    if n <= max_show:
        return ", ".join(ids)
    half = max_show // 2
    return ", ".join(ids[:half]) + " ... " + ", ".join(ids[-half:]) + f" (共 {n} 个)"

def write_card(tag, run_dir, out_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(run_dir, exist_ok=True)
    ckpts = list_checkpoints(run_dir)
    lora_weights = os.path.join(run_dir, "pytorch_lora_weights.safetensors")
    train_log = os.path.join(run_dir, "train.log")
    tb_dir = os.path.join(run_dir, "logs")

    versions = get_versions()
    env_keys = [
        "CRACK_JSONL", "CRACK_COND_DIR", "CRACK_USE_PATCH", "CRACK_PATCH",
        "CRACK_TRY", "CRACK_TH", "CRACK_P_RANDOM", "CRACK_CROP_SEED",
    ]
    env_dump = {k: os.environ.get(k, "<not set in this shell>") for k in env_keys}
    eval_ids = read_eval_ids(EVAL_IDS_TXT)
    ids_summary = format_eval_ids_summary(eval_ids) if eval_ids else "(none)"

    cond_dir = COND_DIRS.get(tag)
    is_lora = tag == "lora"
    goal_lines = (
        [
            "- LoRA fine-tuning (UNet only) for improved realism (lighting/texture).\n",
            "- No structural control; use with SD or SD+ControlNet at inference.\n",
        ]
        if is_lora
        else [
            "- ControlNet fine-tuning for pavement crack generation with strong structural control.\n",
            "- Compare binary mask vs DT heatmap conditioning.\n",
        ]
    )
    cond_line = (
        f"- cond dir: N/A (LoRA uses image+text only)\n"
        if is_lora
        else f"- cond dir ({tag}): `{cond_dir}` (exists={file_exists(cond_dir)})\n"
    )
    md = [
        f"# Experiment Card - {tag}\n",
        f"- Generated at: `{now}`\n",
        f"- Run dir: `{run_dir}`\n\n",
        "## 1. Goal\n",
        *goal_lines,
        "\n",
        "## 2. Data\n",
        f"- jsonl: `{DATA_JSONL}` (exists={file_exists(DATA_JSONL)})\n",
        f"- image dir: `{DATA_IMAGE_DIR}` (exists={file_exists(DATA_IMAGE_DIR)})\n",
        cond_line,
        f"- eval ids: `{EVAL_IDS_TXT}` (n={len(eval_ids)})\n",
        f"  - ids: {ids_summary}\n\n",
        "## 3. Models\n",
        f"- base model: `{BASE_MODEL}`\n",
        f"- controlnet init: `{CONTROLNET_INIT}`\n\n",
        "## 4. Training config\n",
        "```json\n",
        json.dumps(TRAIN_CFG, indent=2, ensure_ascii=False) + "\n",
        "```\n\n",
        "## 5. Inference protocol (evaluation)\n",
        "```json\n",
        json.dumps(INFER_CFG, indent=2, ensure_ascii=False) + "\n",
        "```\n\n",
        "## 6. Loader settings (env vars snapshot)\n",
        "> Note: values may show `<not set in this shell>` if you run this script in a different shell.\n\n",
        "```json\n",
        json.dumps(env_dump, indent=2, ensure_ascii=False) + "\n",
        "```\n\n",
        "## 7. Artifacts\n",
        f"- train log: `{train_log}` (exists={file_exists(train_log)})\n",
        f"- tensorboard logdir: `{tb_dir}` (exists={file_exists(tb_dir)})\n",
        f"- checkpoints ({len(ckpts)}): {ckpts}\n",
    ]
    if os.path.isfile(lora_weights):
        md.append(f"- LoRA weights: `{lora_weights}` (exists=YES)\n")
    md.append("\n")
    md.extend([
        "## 8. Environment\n",
        f"- torch: `{versions.get('torch', '?')}`\n",
        f"- diffusers: `{versions.get('diffusers', '?')}`\n",
        f"- datasets: `{versions.get('datasets', '?')}`\n",
        f"- accelerate: `{versions.get('accelerate', '?')}`\n\n",
    ])
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