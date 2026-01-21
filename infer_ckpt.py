import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import os

base_model = "runwayml/stable-diffusion-v1-5"
controlnet_dir = "/work/outputs/cracktree260_controlnet_mask_512/checkpoint-1000/controlnet"

cond_path = "/CrackTree260/cond_mask/6230.png"
out_dir = "/work/outputs/cracktree260_controlnet_mask_512/infer_samples"
os.makedirs(out_dir, exist_ok=True)

prompt = "a photo of pavement crack"

controlnet = ControlNetModel.from_pretrained(controlnet_dir, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe = pipe.to("cuda")

cond = Image.open(cond_path).convert("RGB")

g = torch.Generator(device="cuda").manual_seed(42)
for i in range(4):
    img = pipe(
        prompt=prompt,
        image=cond,
        num_inference_steps=20,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        generator=g
    ).images[0]
    img.save(os.path.join(out_dir, f"step500_6197_{i}.png"))

cond.save(os.path.join(out_dir, "cond_6192.png"))
print("saved to:", out_dir)