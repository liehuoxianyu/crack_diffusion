import os
import torch
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

# ===================== 配置区（只改这里） =====================
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

# 选择模式： "dt" 或 "binary"
MODE = "dt"

# ControlNet checkpoint 路径（二选一示例）
# DT（推荐）
CONTROLNET_DIR = "/work/outputs/exp_dt_patch512/checkpoint-2000/controlnet"
# Binary示例（需要时改成下面这行）
# CONTROLNET_DIR = "/work/outputs/exp_binary_patch512/checkpoint-2000/controlnet"

# 条件图路径（id6215）
COND_DT_PATH = "/CrackTree260/cond_dt/6215.png"
COND_BINARY_PATH = "/CrackTree260/cond_mask/6215.png"

# 输出目录
OUTDIR = "/work/outputs/infer_steps_id6215"

# Prompt
PROMPT = "a photo of pavement crack, realistic texture, high detail, natural shadowing, realistic lighting"
NEG_PROMPT = ""

# 推理参数
INFER_STEPS_LIST = [1, 5, 10, 15, 20, 25]
GUIDANCE_SCALE = 7.5
CONTROL_SCALE = 1.5

SEED = 42
USE_FP16 = True

# 是否也输出 SD-only 对照
DO_SD_ONLY = False
# ============================================================


def load_cond_image():
    if MODE == "dt":
        p = COND_DT_PATH
    elif MODE == "binary":
        p = COND_BINARY_PATH
    else:
        raise ValueError("MODE must be 'dt' or 'binary'")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return Image.open(p).convert("RGB"), p


def load_cn_pipe(device, dtype):
    if not os.path.exists(CONTROLNET_DIR):
        raise FileNotFoundError(CONTROLNET_DIR)
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_DIR, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
    return pipe.to(device)


def load_sd_pipe(device, dtype):
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
    return pipe.to(device)


@torch.inference_mode()
def infer_controlnet(pipe, cond_img, num_steps, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT if NEG_PROMPT else None,
        image=cond_img,
        num_inference_steps=int(num_steps),
        guidance_scale=float(GUIDANCE_SCALE),
        controlnet_conditioning_scale=float(CONTROL_SCALE),
        generator=g,
    )
    return out.images[0]


@torch.inference_mode()
def infer_sd(pipe, num_steps, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT if NEG_PROMPT else None,
        num_inference_steps=int(num_steps),
        guidance_scale=float(GUIDANCE_SCALE),
        generator=g,
    )
    return out.images[0]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (USE_FP16 and device == "cuda") else torch.float32

    cond_img, cond_path = load_cond_image()
    cond_img.save(os.path.join(OUTDIR, f"cond_{MODE}_6215.png"))

    # 保存设置
    with open(os.path.join(OUTDIR, "settings.txt"), "w", encoding="utf-8") as f:
        f.write(f"BASE_MODEL={BASE_MODEL}\n")
        f.write(f"MODE={MODE}\n")
        f.write(f"CONTROLNET_DIR={CONTROLNET_DIR}\n")
        f.write(f"COND_PATH={cond_path}\n")
        f.write(f"PROMPT={PROMPT}\n")
        f.write(f"NEG_PROMPT={NEG_PROMPT}\n")
        f.write(f"INFER_STEPS_LIST={INFER_STEPS_LIST}\n")
        f.write(f"GUIDANCE_SCALE={GUIDANCE_SCALE}\n")
        f.write(f"CONTROL_SCALE={CONTROL_SCALE}\n")
        f.write(f"SEED={SEED}\n")

    # ControlNet 推理
    cn_pipe = load_cn_pipe(device, dtype)
    for s in INFER_STEPS_LIST:
        img = infer_controlnet(cn_pipe, cond_img, s, SEED, device)
        img.save(os.path.join(OUTDIR, f"{MODE}_controlnet_steps{s:02d}.png"))
        print("saved", s)

    # 可选：SD-only 对照
    if DO_SD_ONLY:
        sd_pipe = load_sd_pipe(device, dtype)
        for s in INFER_STEPS_LIST:
            img = infer_sd(sd_pipe, s, SEED, device)
            img.save(os.path.join(OUTDIR, f"sd_only_steps{s:02d}.png"))
            print("saved sd", s)

    print("DONE. out:", OUTDIR)


if __name__ == "__main__":
    main()