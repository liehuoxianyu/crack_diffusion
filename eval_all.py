import os
import json
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

# 两个实验run目录（里面有 checkpoint-500/1000/1500/2000/controlnet）
RUN_BINARY = "/work/outputs/exp_binary_patch512"
RUN_DT     = "/work/outputs/exp_dt_patch512"

# 条件图目录
COND_BINARY_DIR = "/CrackTree260/cond_mask"
COND_DT_DIR     = "/CrackTree260/cond_dt"

# 评测ID文件
EVAL_IDS_TXT = "/CrackTree260/eval_ids.txt"

# 要评测哪些checkpoint
CKPTS = [500, 1000, 1500, 2000]

# 推理协议
PROMPT = "a photo of pavement crack, realistic texture, high detail, natural shadowing, realistic lighting"
NEG_PROMPT = ""   # 需要可自行加
SEED = 42
STEPS = 25
GUIDANCE_SCALES = [6.0, 7.5]
CONTROL_SCALES = [1.0, 1.5, 2.0]

# 输出目录
OUTDIR = "/work/outputs/exp_eval_all"

# 性能
USE_FP16 = True
# ============================================================


def read_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                ids.append(t)
    return ids


def make_grid(images, rows, cols, pad=8, bg=(0, 0, 0)):
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w + (cols - 1) * pad, rows * h + (rows - 1) * pad), bg)
    for r in range(rows):
        for c in range(cols):
            grid.paste(images[r * cols + c], (c * (w + pad), r * (h + pad)))
    return grid


def save_settings(path, d):
    with open(path, "w", encoding="utf-8") as f:
        for k, v in d.items():
            f.write(f"{k}: {v}\n")


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


def load_cn_pipe(controlnet_dir, device, dtype):
    controlnet = ControlNetModel.from_pretrained(controlnet_dir, torch_dtype=dtype)
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


@torch.inference_mode()
def infer_sd(pipe, prompt, neg_prompt, steps, guidance, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    im = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt if neg_prompt else None,
        num_inference_steps=steps,
        guidance_scale=float(guidance),
        generator=g,
    ).images[0]
    return im


@torch.inference_mode()
def infer_cn(pipe, prompt, neg_prompt, cond_img, steps, guidance, control_scale, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    im = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt if neg_prompt else None,
        image=cond_img,
        num_inference_steps=steps,
        guidance_scale=float(guidance),
        controlnet_conditioning_scale=float(control_scale),
        generator=g,
    ).images[0]
    return im


def ensure_controlnet_dir(run_dir, step):
    p = os.path.join(run_dir, f"checkpoint-{step}", "controlnet")
    w = os.path.join(p, "diffusion_pytorch_model.safetensors")
    c = os.path.join(p, "config.json")
    if not (os.path.exists(w) and os.path.exists(c)):
        raise FileNotFoundError(f"controlnet dir invalid: {p}")
    return p


def paste_side_by_side(left, right, pad=16, bg=(0, 0, 0)):
    w1, h1 = left.size
    w2, h2 = right.size
    h = max(h1, h2)
    canvas = Image.new("RGB", (w1 + pad + w2, h), bg)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (w1 + pad, 0))
    return canvas


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ids = read_ids(EVAL_IDS_TXT)
    print("eval ids:", ids)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (USE_FP16 and device == "cuda") else torch.float32

    # 1) SD-only pipe（一次加载，复用）
    sd_pipe = load_sd_pipe(device, dtype)

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

        # 2.2 SD-only：只跟 guidance 有关（不含 control_scale）
        sd_dir = os.path.join(id_out, "sd_only")
        os.makedirs(sd_dir, exist_ok=True)
        sd_imgs = []
        for r, gs in enumerate(GUIDANCE_SCALES):
            im = infer_sd(sd_pipe, PROMPT, NEG_PROMPT, STEPS, gs, SEED + r, device)
            im.save(os.path.join(sd_dir, f"gs{gs}.png"))
            sd_imgs.append(im)
        # 做一个简单横向拼接（不同 guidance）
        if len(sd_imgs) >= 2:
            strip = make_grid(sd_imgs, rows=1, cols=len(sd_imgs))
            strip.save(os.path.join(sd_dir, "grid.png"))

        # 2.3 ControlNet：binary与DT各自跑各checkpoint
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
                pipe = load_cn_pipe(cn_dir, device, dtype)

                out_dir = os.path.join(id_out, f"controlnet_{tag}", f"step{step}")
                os.makedirs(out_dir, exist_ok=True)

                grid_imgs = []
                idx = 0
                for gs in GUIDANCE_SCALES:
                    for cs in CONTROL_SCALES:
                        im = infer_cn(pipe, PROMPT, NEG_PROMPT, cond_img, STEPS, gs, cs, SEED + idx, device)
                        im.save(os.path.join(out_dir, f"gs{gs}_cs{cs}.png"))
                        grid_imgs.append(im)
                        idx += 1

                grid = make_grid(grid_imgs, rows=len(GUIDANCE_SCALES), cols=len(CONTROL_SCALES))
                grid.save(os.path.join(out_dir, "grid.png"))

                # 释放显存（重要：每个step都load一次）
                del pipe
                torch.cuda.empty_cache()

        # 2.4 自动生成对比拼图（用每条线的最终checkpoint）
        final_step = CKPTS[-1]

        # binary vs dt
        p_bin = os.path.join(id_out, "controlnet_binary", f"step{final_step}", "grid.png")
        p_dt  = os.path.join(id_out, "controlnet_dt",     f"step{final_step}", "grid.png")
        if os.path.exists(p_bin) and os.path.exists(p_dt):
            comp = paste_side_by_side(Image.open(p_bin).convert("RGB"), Image.open(p_dt).convert("RGB"))
            comp.save(os.path.join(id_out, f"compare_binary_vs_dt_step{final_step}.png"))

        # sd vs dt(final)（看ControlNet带来的可控性/结构差）
        p_sd = os.path.join(id_out, "sd_only", "grid.png")
        if os.path.exists(p_sd) and os.path.exists(p_dt):
            comp = paste_side_by_side(Image.open(p_sd).convert("RGB"), Image.open(p_dt).convert("RGB"))
            comp.save(os.path.join(id_out, f"compare_sd_vs_dt_step{final_step}.png"))

    print("\nDONE. all results saved to:", OUTDIR)


if __name__ == "__main__":
    main()