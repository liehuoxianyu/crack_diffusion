import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch


def _is_true(v: str) -> bool:
    return str(v).strip().lower() not in ("", "0", "false", "no")


@dataclass(frozen=True)
class FreeControlRuntimeConfig:
    freecontrol_root: Path
    base_config_path: Path
    model_id: str
    pca_path: Path
    inversion_cache_dir: Path
    device: str
    dtype: torch.dtype
    pipe_cache_max: int
    strict_checks: bool

    def validate(self, *, require_pca: bool = True) -> None:
        if not self.freecontrol_root.exists():
            raise FileNotFoundError(f"FREECONTROL_ROOT not found: {self.freecontrol_root}")
        if not self.base_config_path.exists():
            raise FileNotFoundError(f"FreeControl base config not found: {self.base_config_path}")
        if require_pca and (not self.pca_path.exists()):
            raise FileNotFoundError(
                "FREECONTROL_PCA_PATH not found. "
                f"Expected file at: {self.pca_path}"
            )
        self.inversion_cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_dtype_name(self) -> str:
        return "fp16" if self.dtype == torch.float16 else "fp32"


def _resolve_device_dtype() -> Tuple[str, torch.dtype]:
    user_device = os.environ.get("FREECONTROL_DEVICE", "").strip().lower()
    if user_device:
        device = user_device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = _is_true(os.environ.get("FREECONTROL_USE_FP16", "1"))
    dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32
    return device, dtype


def load_runtime_config(*, require_pca: bool = True) -> FreeControlRuntimeConfig:
    root = Path(os.environ.get("FREECONTROL_ROOT", "/work/free_control/freecontrol")).resolve()
    base_config = Path(
        os.environ.get("FREECONTROL_BASE_CONFIG", str(root / "config" / "base.yaml"))
    ).resolve()
    model_id = os.environ.get("FREECONTROL_MODEL_ID", "runwayml/stable-diffusion-v1-5").strip()
    pca_path = Path(os.environ.get("FREECONTROL_PCA_PATH", "")).expanduser()
    if not pca_path:
        pca_path = (root / "checkpoints" / "pca_info.pt").resolve()
    else:
        pca_path = pca_path.resolve()
    inversion_cache_dir = Path(
        os.environ.get("FREECONTROL_INVERSION_CACHE_DIR", "/work/outputs/freecontrol/latent")
    ).resolve()
    device, dtype = _resolve_device_dtype()
    pipe_cache_max = max(1, int(os.environ.get("FREECONTROL_PIPE_CACHE_MAX", "1")))
    strict_checks = _is_true(os.environ.get("FREECONTROL_STRICT_CHECKS", "1"))
    cfg = FreeControlRuntimeConfig(
        freecontrol_root=root,
        base_config_path=base_config,
        model_id=model_id,
        pca_path=pca_path,
        inversion_cache_dir=inversion_cache_dir,
        device=device,
        dtype=dtype,
        pipe_cache_max=pipe_cache_max,
        strict_checks=strict_checks,
    )
    if cfg.strict_checks:
        cfg.validate(require_pca=require_pca)
    return cfg
