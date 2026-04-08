import base64
import os
import time
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from api.infer import generate_image, image_to_png_bytes, save_png_to_outputs


OUTPUT_DIR = os.environ.get("WEB_DEMO_OUTPUT_DIR", "/work/outputs/web_demo")
CORS_ALLOW_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "WEB_DEMO_CORS_ALLOW_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if o.strip()
]


app = FastAPI(title="CrackTree Web Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    seed: int = Form(0),
    num_inference_steps: int = Form(25),
    guidance_scale: float = Form(7.5),
    width: int = Form(512),
    height: int = Form(512),
    enable_controlnet: bool = Form(False),
    controlnet_type: str = Form("DT"),
    controlnet_conditioning_scale: float = Form(1.0),
    enable_lora: bool = Form(False),
    lora_path: str = Form(""),
    lora_scale: float = Form(0.7),
    condition_image: Optional[UploadFile] = File(None),
):
    t0 = time.perf_counter()

    condition_image_bytes: Optional[bytes] = None
    if condition_image is not None:
        condition_image_bytes = await condition_image.read()

    # Run heavy GPU work in a thread so we don't block the event loop.
    im, meta = await run_in_threadpool(
        generate_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        enable_controlnet=enable_controlnet,
        controlnet_type=controlnet_type,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        enable_lora=enable_lora,
        lora_path=lora_path,
        lora_scale=lora_scale,
        condition_image_bytes=condition_image_bytes,
    )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    image_path = save_png_to_outputs(im, meta=meta, output_dir=OUTPUT_DIR)
    png_bytes = image_to_png_bytes(im)
    image_base64 = base64.b64encode(png_bytes).decode("utf-8")

    return JSONResponse(
        {
            "image_base64": image_base64,
            "image_path": image_path,
            "elapsed_ms": elapsed_ms,
            "config_summary": meta.config_summary,
            "mode": meta.mode,
            "seed": meta.seed_used,
        }
    )

