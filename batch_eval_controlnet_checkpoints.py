import os
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# =============== 配置区（你只改这里即可） ===============
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

CONTROLNET_RUN_DIR = "/work/outputs/cracktree260_controlnet_dtpatch_512"
CHECKPOINT_STEPS = [500, 1000, 1500, 2000]  # 你已有的4个

DT_DIR = "/CrackTree260/cond_dt"
EVAL_IDS_TXT = "/CrackTree260/eval_ids.txt"

OUTDIR = "/work/outputs/cracktree260_controlnet_dtpatch_512/eval_ckpts"
PROMPT = "a photo of pavement crack, realistic texture, high detail, natural shadowing, realistic lighting"

SEED = 42
STEPS = 25
GUIDANCE_SCALES = [6.0, 7.5]
CONTROL_SCALES = [1.0, 1.5, 2.0]
USE_FP16 = True
# =======================================================

def make_grid(images, rows, cols, pad=8, bg=(0,0,0)):
    w, h = images[0].size
    grid = Image.new("RGB", (cols*w + (cols-1)*pad, rows*h + (rows-1)*pad), bg)
    for r in range(rows):
        for c in range(cols):
            grid.paste(images[r*cols+c], (c*(w+pad), r*(h+pad)))
    return grid

def load_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                ids.append(t)
    return ids

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    ids = load_ids(EVAL_IDS_TXT)
    print("eval ids:", ids)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (USE_FP16 and device == "cuda") else torch.float32

    for step in CHECKPOINT_STEPS:
        cn_dir = os.path.join(CONTROLNET_RUN_DIR, f"checkpoint-{step}", "controlnet")
        ckpt_file = os.path.join(cn_dir, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f"missing controlnet weights: {ckpt_file}")

        print(f"\n=== Loading controlnet @ step {step} ===")
        controlnet = ControlNetModel.from_pretrained(cn_dir, torch_dtype=dtype)
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

        step_out = os.path.join(OUTDIR, f"step{step}")
        os.makedirs(step_out, exist_ok=True)

        for id_ in ids:
            cond_path = os.path.join(DT_DIR, f"{id_}.png")
            if not os.path.exists(cond_path):
                print("skip missing cond:", cond_path)
                continue

            cond = Image.open(cond_path).convert("RGB")
            sample_dir = os.path.join(step_out, f"id{id_}")
            os.makedirs(sample_dir, exist_ok=True)
            cond.save(os.path.join(sample_dir, "conditioning.png"))

            grid_imgs = []
            idx = 0
            for gs in GUIDANCE_SCALES:
                for cs in CONTROL_SCALES:
                    g = torch.Generator(device=device).manual_seed(SEED + idx)
                    img = pipe(
                        prompt=PROMPT,
                        image=cond,
                        num_inference_steps=STEPS,
                        guidance_scale=float(gs),
                        controlnet_conditioning_scale=float(cs),
                        generator=g,
                    ).images[0]
                    img.save(os.path.join(sample_dir, f"gs{gs}_cs{cs}.png"))
                    grid_imgs.append(img)
                    idx += 1

            grid = make_grid(grid_imgs, rows=len(GUIDANCE_SCALES), cols=len(CONTROL_SCALES))
            grid.save(os.path.join(sample_dir, "grid.png"))

            # 保存本次评测设置（科研复现用）
            with open(os.path.join(sample_dir, "settings.txt"), "w", encoding="utf-8") as f:
                f.write(f"checkpoint_step={step}\n")
                f.write(f"base_model={BASE_MODEL}\n")
                f.write(f"prompt={PROMPT}\n")
                f.write(f"seed_base={SEED}\n")
                f.write(f"steps={STEPS}\n")
                f.write(f"guidance_scales={GUIDANCE_SCALES}\n")
                f.write(f"control_scales={CONTROL_SCALES}\n")
                f.write(f"cond_image={cond_path}\n")

            print("saved", os.path.join(sample_dir, "grid.png"))

    print("\nDONE. all results in:", OUTDIR)

if __name__ == "__main__":
    main()