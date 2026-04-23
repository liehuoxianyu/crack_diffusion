"""
eval_all.py — CrackTree 评测脚本

预期输出目录：OUTDIR（默认 /work/outputs/exp_eval_all），仅单张图片，无 grid、无拼接。

  OUTDIR/
  ├── settings.txt                    # 本次运行的配置（BASE_MODEL, LORA_PATH, CKPTS, PROMPT 等）
  └── id{id}/                         # 每个评测 ID 一个目录，id 来自 EVAL_IDS_TXT
      ├── cond_binary.png             # 该 ID 的二值条件图（拷贝）
      ├── cond_dt.png                 # 该 ID 的 DT 条件图（拷贝）
      │
      ├── sd_only/                    # 形式1：纯 SD
      │   └── gs{gs}.png              # 各 guidance scale 单张（如 gs7.5.png）
      │
      ├── sd_lora/                    # 形式2：SD+LoRA（仅当 LORA_PATH 有效）
      │   └── gs{gs}.png
      │
      ├── controlnet_binary/          # 形式3：SD+ControlNet（binary，无 LoRA）
      │   └── step{500,1000,1500,2000}/
      │       └── gs{gs}_cs{cs}.png   # 每个 (guidance_scale, control_scale) 一张
      │
      ├── controlnet_dt/              # 形式3：SD+ControlNet（DT，无 LoRA）
      │   └── step{500,1000,1500,2000}/
      │       └── gs{gs}_cs{cs}.png
      │
      ├── controlnet_binary_lora/     # 形式4：SD+LoRA+ControlNet binary（仅当 LORA_PATH 有效）
      │   └── step{500,1000,1500,2000}/
      │       └── gs{gs}_cs{cs}.png
      │
      └── controlnet_dt_lora/         # 形式4：SD+LoRA+ControlNet DT
          └── step{500,1000,1500,2000}/
              └── gs{gs}_cs{cs}.png
"""
import os
import torch
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from safetensors.torch import load_file

from models.cafe import CAFEEmbedding
from models.tag import inject_tag_into_controlnet

# ===================== 配置区（只改这里） =====================
# 评测输出四种形式：1=纯SD  2=SD+LoRA  3=SD+ControlNet(binary/DT)  4=SD+LoRA+ControlNet(binary/DT)
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_BASE_MODEL = "lllyasviel/sd-controlnet-seg"

# 两个实验run目录（里面有 checkpoint-500/1000/1500/2000/controlnet）
RUN_BINARY = "/work/outputs/exp_binary_patch512"
RUN_DT     = "/work/outputs/exp_dt_patch512"

# 条件图目录
COND_BINARY_DIR = "/CrackTree260/cond_mask"
COND_DT_DIR     = "/CrackTree260/cond_dt"

# 评测ID文件
EVAL_IDS_TXT = "/CrackTree260/eval_ids.txt"
# 只跑前 N 个 ID（试跑看效果）：设为 1 跑一张，None 跑全部
LIMIT_IDS = None

# 要评测哪些checkpoint
CKPTS = [500, 1000, 1500, 2000]

# 推理协议
PROMPT = "a photo of pavement crack, realistic texture, high detail, natural shadowing, realistic lighting"
NEG_PROMPT = ""   # 需要可自行加
SEED = 42
STEPS = 25
GUIDANCE_SCALES = [7.5]   # 单值减少工作量；多尺度可改为 [6.0, 7.5]
CONTROL_SCALES = [0.75, 1.0, 1.25]

# 输出目录
OUTDIR = "/work/outputs/exp_eval_all"

# 性能
USE_FP16 = True

# LoRA（可选）：仅作用于 UNet，加载顺序 base → controlnet(如有) → lora(如有)
# 设为 None 或空字符串则不加载；可为目录或 .safetensors 文件路径
LORA_PATH = "/work/outputs/exp_lora_realism"  # 设为 None 则不加载；可为目录或 .safetensors 文件
# LoRA 介入强度：1.0=全开，<1 减弱（如 0.5~0.7 可降低“介入太多”）
LORA_SCALE = 0.7
# ============================================================


def read_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                ids.append(t)
    return ids


def save_settings(path, d):
    with open(path, "w", encoding="utf-8") as f:
        for k, v in d.items():
            f.write(f"{k}: {v}\n")


def load_sd_pipe(device, dtype, use_lora=False):
    """use_lora=False: 纯 SD；use_lora=True: SD+LoRA（需 LORA_PATH 有效）"""
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)
    if use_lora and LORA_PATH and os.path.exists(LORA_PATH):
        pipe.load_lora_weights(LORA_PATH)
    return pipe


def load_cn_pipe(controlnet_dir, device, dtype, use_lora=False):
    """use_lora=False: SD+ControlNet；use_lora=True: SD+LoRA+ControlNet(需 LORA_PATH 有效）"""
    ckpt_file = os.path.join(controlnet_dir, "diffusion_pytorch_model.safetensors")
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_BASE_MODEL, torch_dtype=dtype)
    controlnet.controlnet_cond_embedding = CAFEEmbedding()
    controlnet = inject_tag_into_controlnet(controlnet)
    controlnet.load_state_dict(load_file(ckpt_file), strict=False)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)
    if use_lora and LORA_PATH and os.path.exists(LORA_PATH):
        pipe.load_lora_weights(LORA_PATH)
    return pipe


@torch.inference_mode()
def infer_sd(pipe, prompt, neg_prompt, steps, guidance, seed, device, lora_scale=None):
    g = torch.Generator(device=device).manual_seed(seed)
    kw = {}
    if lora_scale is not None:
        kw["cross_attention_kwargs"] = {"scale": float(lora_scale)}
    im = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt if neg_prompt else None,
        num_inference_steps=steps,
        guidance_scale=float(guidance),
        generator=g,
        **kw,
    ).images[0]
    return im


@torch.inference_mode()
def infer_cn(pipe, prompt, neg_prompt, cond_img, steps, guidance, control_scale, seed, device, lora_scale=None):
    g = torch.Generator(device=device).manual_seed(seed)
    kw = {}
    if lora_scale is not None:
        kw["cross_attention_kwargs"] = {"scale": float(lora_scale)}
    im = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt if neg_prompt else None,
        image=cond_img,
        num_inference_steps=steps,
        guidance_scale=float(guidance),
        controlnet_conditioning_scale=float(control_scale),
        generator=g,
        **kw,
    ).images[0]
    return im


def ensure_controlnet_dir(run_dir, step):
    p = os.path.join(run_dir, f"checkpoint-{step}", "controlnet")
    w = os.path.join(p, "diffusion_pytorch_model.safetensors")
    c = os.path.join(p, "config.json")
    if not (os.path.exists(w) and os.path.exists(c)):
        raise FileNotFoundError(f"controlnet dir invalid: {p}")
    return p


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ids = read_ids(EVAL_IDS_TXT)
    if LIMIT_IDS is not None:
        ids = ids[:LIMIT_IDS]
        print("limit to first", LIMIT_IDS, "ids:", ids)
    else:
        print("eval ids:", ids)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (USE_FP16 and device == "cuda") else torch.float32

    # 1) 纯 SD pipe（一次加载，复用）
    sd_pipe = load_sd_pipe(device, dtype, use_lora=False)

    # 保存全局设置
    save_settings(os.path.join(OUTDIR, "settings.txt"), {
        "BASE_MODEL": BASE_MODEL,
        "RUN_BINARY": RUN_BINARY,
        "RUN_DT": RUN_DT,
        "COND_BINARY_DIR": COND_BINARY_DIR,
        "COND_DT_DIR": COND_DT_DIR,
        "EVAL_IDS_TXT": EVAL_IDS_TXT,
        "CKPTS": CKPTS,
        "PROMPT": PROMPT,
        "NEG_PROMPT": NEG_PROMPT,
        "SEED": SEED,
        "STEPS": STEPS,
        "GUIDANCE_SCALES": GUIDANCE_SCALES,
        "CONTROL_SCALES": CONTROL_SCALES,
        "LORA_PATH": LORA_PATH or "",
        "LORA_SCALE": LORA_SCALE,
        "LIMIT_IDS": LIMIT_IDS,
    })

    # 2) 逐个ID评测
    for id_ in ids:
        print("\n== id", id_, "==")
        id_out = os.path.join(OUTDIR, f"id{id_}")
        os.makedirs(id_out, exist_ok=True)

        # 2.1 保存条件图
        cond_bin_path = os.path.join(COND_BINARY_DIR, f"{id_}.png")
        cond_dt_path  = os.path.join(COND_DT_DIR, f"{id_}.png")
        if os.path.exists(cond_bin_path):
            Image.open(cond_bin_path).convert("RGB").save(os.path.join(id_out, "cond_binary.png"))
        if os.path.exists(cond_dt_path):
            Image.open(cond_dt_path).convert("RGB").save(os.path.join(id_out, "cond_dt.png"))

        # 2.2 形式1：纯 SD（只跟 guidance 有关）
        sd_dir = os.path.join(id_out, "sd_only")
        os.makedirs(sd_dir, exist_ok=True)
        for r, gs in enumerate(GUIDANCE_SCALES):
            im = infer_sd(sd_pipe, PROMPT, NEG_PROMPT, STEPS, gs, SEED + r, device)
            im.save(os.path.join(sd_dir, f"gs{gs}.png"))

        # 2.3 形式2：SD+LoRA（仅当 LORA_PATH 有效时）
        if LORA_PATH and os.path.exists(LORA_PATH):
            sd_lora_pipe = load_sd_pipe(device, dtype, use_lora=True)
            sd_lora_dir = os.path.join(id_out, "sd_lora")
            os.makedirs(sd_lora_dir, exist_ok=True)
            for r, gs in enumerate(GUIDANCE_SCALES):
                im = infer_sd(sd_lora_pipe, PROMPT, NEG_PROMPT, STEPS, gs, SEED + 100 + r, device, lora_scale=LORA_SCALE)
                im.save(os.path.join(sd_lora_dir, f"gs{gs}.png"))
            del sd_lora_pipe
            torch.cuda.empty_cache()

        # 2.4 形式3：SD+ControlNet（两种：binary / DT，不加载 LoRA）
        for tag, run_dir, cond_path in [
            ("binary", RUN_BINARY, cond_bin_path),
            ("dt",     RUN_DT,     cond_dt_path),
        ]:
            if not os.path.exists(cond_path):
                print("missing cond:", cond_path, "skip", tag)
                continue

            cond_img = Image.open(cond_path).convert("RGB")

            for step in CKPTS:
                cn_dir = ensure_controlnet_dir(run_dir, step)
                pipe = load_cn_pipe(cn_dir, device, dtype, use_lora=False)

                out_dir = os.path.join(id_out, f"controlnet_{tag}", f"step{step}")
                os.makedirs(out_dir, exist_ok=True)

                idx = 0
                for gs in GUIDANCE_SCALES:
                    for cs in CONTROL_SCALES:
                        im = infer_cn(pipe, PROMPT, NEG_PROMPT, cond_img, STEPS, gs, cs, SEED + idx, device)
                        im.save(os.path.join(out_dir, f"gs{gs}_cs{cs}.png"))
                        idx += 1

                # 释放显存（重要：每个step都load一次）
                del pipe
                torch.cuda.empty_cache()

        # 2.5 形式4：SD+LoRA+ControlNet（两种：binary / DT，仅当 LORA_PATH 有效时）
        if LORA_PATH and os.path.exists(LORA_PATH):
            for tag, run_dir, cond_path in [
                ("binary_lora", RUN_BINARY, cond_bin_path),
                ("dt_lora",     RUN_DT,     cond_dt_path),
            ]:
                if not os.path.exists(cond_path):
                    print("missing cond:", cond_path, "skip", tag)
                    continue

                cond_img = Image.open(cond_path).convert("RGB")

                for step in CKPTS:
                    cn_dir = ensure_controlnet_dir(run_dir, step)
                    pipe = load_cn_pipe(cn_dir, device, dtype, use_lora=True)

                    out_dir = os.path.join(id_out, f"controlnet_{tag}", f"step{step}")
                    os.makedirs(out_dir, exist_ok=True)

                    idx = 0
                    for gs in GUIDANCE_SCALES:
                        for cs in CONTROL_SCALES:
                            im = infer_cn(pipe, PROMPT, NEG_PROMPT, cond_img, STEPS, gs, cs, SEED + 200 + idx, device, lora_scale=LORA_SCALE)
                            im.save(os.path.join(out_dir, f"gs{gs}_cs{cs}.png"))
                            idx += 1

                    del pipe
                    torch.cuda.empty_cache()

    print("\nDONE. all results saved to:", OUTDIR)


if __name__ == "__main__":
    main()