import base64
import io
import os
import random
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)


OUTPUT_DIR = os.environ.get("WEB_DEMO_OUTPUT_DIR", "/work/outputs/web_demo")
SD_BASE_MODEL = os.environ.get("SD_BASE_MODEL", "runwayml/stable-diffusion-v1-5")

CONTROLNET_BINARY_BASE_DIR = os.environ.get(
    "CONTROLNET_BINARY_BASE_DIR", "/work/outputs/exp_binary_patch512"
)
CONTROLNET_DT_BASE_DIR = os.environ.get("CONTROLNET_DT_BASE_DIR", "/work/outputs/exp_dt_patch512")
CONTROLNET_STEP = int(os.environ.get("CONTROLNET_STEP", "2000"))

USE_FP16 = os.environ.get("WEB_DEMO_USE_FP16", "1") not in ("0", "false", "False", "")
PIPE_CACHE_MAX = int(os.environ.get("WEB_DEMO_PIPE_CACHE_MAX", "2"))


@dataclass
class InferMeta:
    mode: str
    seed_used: int
    config_summary: str


_PIPE_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_PIPE_LOCK = threading.Lock()


def _device_and_dtype() -> Tuple[str, torch.dtype]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (USE_FP16 and device == "cuda") else torch.float32
    return device, dtype


def _ensure_controlnet_dir(controlnet_type: str) -> str:
    t = (controlnet_type or "").strip().lower()
    if t in ("dt", "struct", "structure"):
        base_dir = CONTROLNET_DT_BASE_DIR
    elif t in ("binary", "bin", "mask"):
        base_dir = CONTROLNET_BINARY_BASE_DIR
    else:
        raise ValueError(f"Unsupported controlnet_type: {controlnet_type} (need DT/Binary)")

    p = os.path.join(base_dir, f"checkpoint-{CONTROLNET_STEP}", "controlnet")
    w = os.path.join(p, "diffusion_pytorch_model.safetensors")
    c = os.path.join(p, "config.json")
    if not (os.path.exists(w) and os.path.exists(c)):
        raise FileNotFoundError(f"ControlNet checkpoint missing or invalid: {p}")
    return p


def _load_sd_pipe(lora_path: Optional[str]) -> "StableDiffusionPipeline":
    device, dtype = _device_and_dtype()
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_BASE_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        # Optional speedup; if xformers isn't available, fallback silently.
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if lora_path:
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA path not found: {lora_path}")
        # LoRA weights are loaded once into the pipe; scale is provided per request.
        pipe.load_lora_weights(lora_path)
    return pipe


def _load_cn_pipe(controlnet_type: str, lora_path: Optional[str]) -> "StableDiffusionControlNetPipeline":
    device, dtype = _device_and_dtype()
    controlnet_dir = _ensure_controlnet_dir(controlnet_type)
    controlnet = ControlNetModel.from_pretrained(controlnet_dir, torch_dtype=dtype)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if lora_path:
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA path not found: {lora_path}")
        pipe.load_lora_weights(lora_path)
    return pipe


def _get_or_create_pipe(cache_key: str, loader) -> Any:
    with _PIPE_LOCK:
        pipe = _PIPE_CACHE.get(cache_key)
        if pipe is not None:
            _PIPE_CACHE.move_to_end(cache_key)
            return pipe

    # Create outside lock: constructing pipes is expensive.
    pipe = loader()

    with _PIPE_LOCK:
        # Another request may have inserted while we were loading.
        existing = _PIPE_CACHE.get(cache_key)
        if existing is not None:
            return existing

        _PIPE_CACHE[cache_key] = pipe
        _PIPE_CACHE.move_to_end(cache_key)
        while len(_PIPE_CACHE) > PIPE_CACHE_MAX:
            old_key, old_pipe = _PIPE_CACHE.popitem(last=False)
            del old_pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return pipe


def _maybe_parse_condition_image(condition_image_bytes: Optional[bytes], width: int, height: int) -> Image.Image:
    if not condition_image_bytes:
        raise ValueError("condition_image is required when enable_controlnet=true")
    im = Image.open(io.BytesIO(condition_image_bytes)).convert("RGB")
    if im.size != (width, height):
        im = im.resize((width, height), resample=Image.Resampling.LANCZOS)
    return im


def _seed_used(seed: int) -> int:
    # StableDiffusionPipeline uses generator-based seeding.
    if seed is None or seed <= 0:
        return random.randint(0, 2**31 - 1)
    return int(seed)


def generate_image(
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    enable_controlnet: bool,
    controlnet_type: str,
    controlnet_conditioning_scale: float,
    enable_lora: bool,
    lora_path: str,
    lora_scale: float,
    condition_image_bytes: Optional[bytes],
) -> Tuple[Image.Image, InferMeta]:
    if not prompt or not prompt.strip():
        raise ValueError("prompt is required")

    # diffusers StableDiffusionPipeline requires width/height divisible by 8.
    # For a smoother web demo UX, auto-snap to nearest lower multiple of 8.
    req_width = int(width)
    req_height = int(height)
    width = max(8, req_width - (req_width % 8))
    height = max(8, req_height - (req_height % 8))

    lora_path = (lora_path or "").strip() if enable_lora else ""
    lora_path = lora_path if lora_path else None

    mode: str
    if enable_controlnet and enable_lora:
        mode = "ControlNet + LoRA"
    elif enable_controlnet and not enable_lora:
        mode = f"ControlNet ({controlnet_type})"
    elif (not enable_controlnet) and enable_lora:
        mode = "SD + LoRA"
    else:
        mode = "SD"

    device, _dtype = _device_and_dtype()
    seed_used = _seed_used(seed)
    g = torch.Generator(device=device).manual_seed(seed_used)

    kw: Dict[str, Any] = {}
    if enable_lora and lora_scale is not None:
        kw["cross_attention_kwargs"] = {"scale": float(lora_scale)}

    if enable_controlnet:
        cond_img = _maybe_parse_condition_image(condition_image_bytes, width=width, height=height)
        cache_key = f"cn::{(controlnet_type or '').strip().lower()}::{lora_path or ''}"

        def loader():
            return _load_cn_pipe(controlnet_type=controlnet_type, lora_path=lora_path)

        pipe = _get_or_create_pipe(cache_key, loader)
        im = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=cond_img,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            generator=g,
            width=int(width),
            height=int(height),
            **kw,
        ).images[0]
    else:
        cache_key = f"sd::lin{lora_path or ''}"

        def loader():
            return _load_sd_pipe(lora_path=lora_path)

        pipe = _get_or_create_pipe(cache_key, loader)
        im = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=g,
            width=int(width),
            height=int(height),
            **kw,
        ).images[0]

    config_summary = (
        f"mode={mode}, "
        f"seed={seed_used}, steps={num_inference_steps}, guidance_scale={guidance_scale}, "
        f"size={width}x{height} (requested {req_width}x{req_height}), "
        f"enable_controlnet={enable_controlnet}, controlnet_type={controlnet_type}, "
        f"controlnet_conditioning_scale={controlnet_conditioning_scale}, "
        f"enable_lora={enable_lora}, lora_path={(lora_path or '')}, lora_scale={lora_scale}"
    )
    meta = InferMeta(mode=mode, seed_used=seed_used, config_summary=config_summary)
    return im, meta


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def save_png_to_outputs(image: Image.Image, meta: InferMeta, output_dir: str = OUTPUT_DIR) -> str:
    """
    Save a generated PIL image under `outputs/web_demo/` (or overridden by env var).
    Returns absolute file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    mode_tag = (meta.mode or "unknown").strip().lower().replace(" ", "_").replace("+", "_")
    ts_ms = int(time.time() * 1000)
    fname = f"{mode_tag}_seed{meta.seed_used}_{ts_ms}.png"
    out_path = os.path.join(output_dir, fname)
    image.save(out_path, format="PNG")
    return out_path

