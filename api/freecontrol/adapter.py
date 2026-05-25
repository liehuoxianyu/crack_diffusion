import random
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image

from api.freecontrol.config import FreeControlRuntimeConfig, load_runtime_config
from api.freecontrol.types import FreeControlRequest, FreeControlResult, FreeControlT2IRequest


_FC_PIPE_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_FC_PIPE_LOCK = threading.Lock()


def _snap_size(size: int) -> int:
    n = max(8, int(size))
    return n - (n % 8)


def _seed_used(seed: int) -> int:
    if seed is None or seed <= 0:
        return random.randint(0, 2**31 - 1)
    return int(seed)


def _ensure_freecontrol_import_path(cfg: FreeControlRuntimeConfig) -> None:
    root = str(cfg.freecontrol_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def _pipeline_name_for_model(model_id: str) -> str:
    s = model_id.strip().lower()
    return "SDXLPipeline" if ("xl" in s) else "SDPipeline"


def _build_cache_key(cfg: FreeControlRuntimeConfig) -> str:
    return (
        "fc::"
        f"{cfg.model_id}::"
        f"{cfg.pca_path}::"
        f"{cfg.device}::"
        f"{cfg.cache_dtype_name()}"
    )


def _build_t2i_cache_key(cfg: FreeControlRuntimeConfig) -> str:
    return (
        "fc_t2i::"
        f"{cfg.model_id}::"
        f"{cfg.device}::"
        f"{cfg.cache_dtype_name()}"
    )


def _get_or_create_pipe(cache_key: str, loader) -> Any:
    with _FC_PIPE_LOCK:
        cached = _FC_PIPE_CACHE.get(cache_key)
        if cached is not None:
            _FC_PIPE_CACHE.move_to_end(cache_key)
            return cached

    pipe = loader()

    # Runtime config is used only for cache size; pca is not required for T2I path.
    cfg = load_runtime_config(require_pca=False)
    with _FC_PIPE_LOCK:
        exists = _FC_PIPE_CACHE.get(cache_key)
        if exists is not None:
            return exists
        _FC_PIPE_CACHE[cache_key] = pipe
        _FC_PIPE_CACHE.move_to_end(cache_key)
        while len(_FC_PIPE_CACHE) > cfg.pipe_cache_max:
            _, old_pipe = _FC_PIPE_CACHE.popitem(last=False)
            del old_pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return pipe


def _build_freecontrol_config(
    req: FreeControlRequest, cfg: FreeControlRuntimeConfig, seed_used: int
) -> Tuple[Any, Dict[str, Any]]:
    with cfg.base_config_path.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    inversion_prompt = (req.inversion_prompt or "").strip() or req.prompt
    steps = int(req.num_inference_steps)
    update = {
        "sd_config--guidance_scale": float(req.guidance_scale),
        "sd_config--steps": steps,
        "sd_config--seed": int(seed_used),
        "sd_config--prompt": req.prompt,
        "sd_config--negative_prompt": req.negative_prompt or "",
        "sd_config--obj_pairs": req.obj_pairs or "",
        "sd_config--pca_paths": [str(cfg.pca_path)],
        "data--inversion--prompt": inversion_prompt,
        "data--inversion--fixed_size": [int(req.height), int(req.width)],
        "data--inversion--num_inference_steps": int(req.inversion_num_inference_steps),
        "data--inversion--target_folder": str(cfg.inversion_cache_dir),
        "data--inversion--method": "DDIM",
        "data--inversion--policy": "share",
        "data--inversion--sd_model": cfg.model_id.replace("/", "_"),
        "guidance--pca_guidance--end_step": int(float(req.pca_guidance_end_ratio) * steps),
        "guidance--pca_guidance--weight": float(req.pca_guidance_weight),
        "guidance--pca_guidance--structure_guidance--n_components": int(req.pca_n_components),
        "guidance--pca_guidance--structure_guidance--mask_tr": float(req.pca_mask_threshold),
        "guidance--pca_guidance--structure_guidance--penalty_factor": float(req.pca_penalty_factor),
        "guidance--pca_guidance--warm_up--apply": bool(float(req.pca_warmup_ratio) > 0),
        "guidance--pca_guidance--warm_up--end_step": int(float(req.pca_warmup_ratio) * steps),
        "guidance--pca_guidance--appearance_guidance--apply": bool(float(req.appearance_threshold) > 0),
        "guidance--pca_guidance--appearance_guidance--tr": float(req.appearance_threshold),
        "guidance--pca_guidance--appearance_guidance--reg_factor": float(req.appearance_reg_factor),
        "guidance--cross_attn--end_step": int(float(req.pca_guidance_end_ratio) * steps),
        "guidance--cross_attn--weight": 0,
    }

    from libs.utils.utils import merge_sweep_config

    merged = merge_sweep_config(base_config=base, update=update)
    return OmegaConf.create(merged), update


def _load_freecontrol_pipe(cfg: FreeControlRuntimeConfig):
    _ensure_freecontrol_import_path(cfg)
    from libs.model import make_pipeline
    from libs.model.module.scheduler import CustomDDIMScheduler

    pipeline_name = _pipeline_name_for_model(cfg.model_id)
    pipe = make_pipeline(pipeline_name, cfg.model_id, torch_dtype=cfg.dtype).to(cfg.device)
    pipe.scheduler = CustomDDIMScheduler.from_pretrained(cfg.model_id, subfolder="scheduler")

    if cfg.device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe


def _load_t2i_pipe(cfg: FreeControlRuntimeConfig):
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        UniPCMultistepScheduler,
    )

    pipeline_name = _pipeline_name_for_model(cfg.model_id)
    if pipeline_name == "SDXLPipeline":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg.model_id,
            torch_dtype=cfg.dtype,
            safety_checker=None,
        ).to(cfg.device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model_id,
            torch_dtype=cfg.dtype,
            safety_checker=None,
        ).to(cfg.device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if cfg.device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe


def run_freecontrol(req: FreeControlRequest) -> FreeControlResult:
    if not req.prompt or not req.prompt.strip():
        raise ValueError("prompt is required")
    if not isinstance(req.condition_image, Image.Image):
        raise ValueError("condition_image must be a PIL.Image.Image")

    width = _snap_size(req.width)
    height = _snap_size(req.height)
    condition_image = req.condition_image.convert("RGB")
    if condition_image.size != (width, height):
        condition_image = condition_image.resize((width, height), resample=Image.Resampling.LANCZOS)

    seed_used = _seed_used(req.seed)
    cfg = load_runtime_config(require_pca=True)
    _ensure_freecontrol_import_path(cfg)
    cache_key = _build_cache_key(cfg)

    pipe = _get_or_create_pipe(cache_key, loader=lambda: _load_freecontrol_pipe(cfg))
    fc_config, _update_map = _build_freecontrol_config(req=req, cfg=cfg, seed_used=seed_used)

    generator = torch.Generator(device=cfg.device).manual_seed(seed_used)
    condition_latents = pipe.invert(img=condition_image, inversion_config=fc_config.data.inversion)
    inverted_data = {"condition_input": [condition_latents]}
    images = pipe(
        prompt=fc_config.sd_config.prompt,
        negative_prompt=fc_config.sd_config.negative_prompt,
        num_inference_steps=int(fc_config.sd_config.steps),
        guidance_scale=float(fc_config.sd_config.guidance_scale),
        generator=generator,
        config=fc_config,
        inverted_data=inverted_data,
    ).images

    out_image = images[0]
    mode = "FreeControl"
    summary = (
        f"mode={mode}, seed={seed_used}, steps={int(fc_config.sd_config.steps)}, "
        f"guidance_scale={float(fc_config.sd_config.guidance_scale)}, "
        f"size={width}x{height}, model_id={cfg.model_id}, pca_path={cfg.pca_path}"
    )
    return FreeControlResult(
        image=out_image,
        seed_used=seed_used,
        config_summary=summary,
        cache_key=cache_key,
    )


def run_freecontrol_t2i(req: FreeControlT2IRequest) -> FreeControlResult:
    if not req.prompt or not req.prompt.strip():
        raise ValueError("prompt is required")
    width = _snap_size(req.width)
    height = _snap_size(req.height)
    seed_used = _seed_used(req.seed)
    cfg = load_runtime_config(require_pca=False)
    cache_key = _build_t2i_cache_key(cfg)

    pipe = _get_or_create_pipe(cache_key, loader=lambda: _load_t2i_pipe(cfg))
    generator = torch.Generator(device=cfg.device).manual_seed(seed_used)
    out_image = pipe(
        prompt=req.prompt,
        negative_prompt=(req.negative_prompt or "") or None,
        num_inference_steps=int(req.num_inference_steps),
        guidance_scale=float(req.guidance_scale),
        generator=generator,
        width=int(width),
        height=int(height),
    ).images[0]
    summary = (
        f"mode=FreeControl-T2I, seed={seed_used}, steps={int(req.num_inference_steps)}, "
        f"guidance_scale={float(req.guidance_scale)}, size={width}x{height}, model_id={cfg.model_id}"
    )
    return FreeControlResult(
        image=out_image,
        seed_used=seed_used,
        config_summary=summary,
        cache_key=cache_key,
    )


def clear_freecontrol_cache() -> None:
    with _FC_PIPE_LOCK:
        _FC_PIPE_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
