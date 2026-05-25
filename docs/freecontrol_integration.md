# FreeControl Core Integration

This document describes the first integration phase for FreeControl in this repository.

## Scope

- Added a standalone Python adapter in `api/freecontrol`.
- No change to existing ControlNet/LoRA code paths in `api/infer.py`.
- No `web_demo` changes in this phase.

## New Module

- `api/freecontrol/types.py`: request/result dataclasses.
- `api/freecontrol/config.py`: runtime config loader and strict path validation.
- `api/freecontrol/adapter.py`: pipeline cache, config mapping, inversion + generation wrapper, and a pure T2I wrapper.
- `api/freecontrol/__init__.py`: stable imports for external callers.

## Environment Variables

- `FREECONTROL_ROOT` (default: `/work/free_control/freecontrol`)
- `FREECONTROL_BASE_CONFIG` (default: `${FREECONTROL_ROOT}/config/base.yaml`)
- `FREECONTROL_MODEL_ID` (default: `runwayml/stable-diffusion-v1-5`)
- `FREECONTROL_PCA_PATH` (required in practice; default fallback: `${FREECONTROL_ROOT}/checkpoints/pca_info.pt`)
- `FREECONTROL_INVERSION_CACHE_DIR` (default: `/work/outputs/freecontrol/latent`)
- `FREECONTROL_DEVICE` (default auto: `cuda` if available else `cpu`)
- `FREECONTROL_USE_FP16` (default: `1`)
- `FREECONTROL_PIPE_CACHE_MAX` (default: `1`)
- `FREECONTROL_STRICT_CHECKS` (default: `1`)

## Isolation Rules

- FreeControl uses a dedicated cache namespace with key prefix `fc::`.
- Existing cache keys in `api/infer.py` (`sd::`, `cn::`) are untouched.
- Integration is additive: only callers that import `api.freecontrol` use FreeControl logic.

## Minimal Python Usage

```python
from PIL import Image
from api.freecontrol import FreeControlRequest, run_freecontrol

req = FreeControlRequest(
    prompt="A photo of a lion in the desert",
    condition_image=Image.open("/path/to/condition.png").convert("RGB"),
    inversion_prompt="A photo of a dog",
    obj_pairs="(dog; lion)",
    num_inference_steps=50,
    width=512,
    height=512,
)

result = run_freecontrol(req)
result.image.save("/tmp/freecontrol_out.png")
print(result.config_summary)
```

## T2I Usage

```python
from api.freecontrol import FreeControlT2IRequest, run_freecontrol_t2i

req = FreeControlT2IRequest(
    prompt="A cinematic portrait of a lion, ultra detailed",
    num_inference_steps=30,
    guidance_scale=7.5,
    width=512,
    height=512,
)
result = run_freecontrol_t2i(req)
result.image.save("/tmp/freecontrol_t2i_out.png")
print(result.config_summary)
```

## Notes

- FreeControl depends on its own stack (`xformers`, compatible `diffusers/torch`).
- Set `FREECONTROL_PCA_PATH` to a valid `pca_info.pt` before running.
- `FREECONTROL_PCA_PATH` is only required for the FreeControl I2I path; pure T2I does not require PCA basis.
- The adapter intentionally does not expose HTTP endpoints in this phase.
