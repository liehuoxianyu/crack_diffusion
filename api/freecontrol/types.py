from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass
class FreeControlRequest:
    prompt: str
    condition_image: Image.Image
    negative_prompt: str = ""
    seed: int = 0
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    inversion_prompt: Optional[str] = None
    inversion_num_inference_steps: int = 999
    obj_pairs: str = ""
    pca_guidance_end_ratio: float = 0.6
    pca_guidance_weight: float = 600.0
    pca_n_components: int = 64
    pca_mask_threshold: float = 0.3
    pca_penalty_factor: float = 10.0
    pca_warmup_ratio: float = 0.05
    appearance_threshold: float = 0.5
    appearance_reg_factor: float = 0.1


@dataclass
class FreeControlT2IRequest:
    prompt: str
    negative_prompt: str = ""
    seed: int = 0
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512


@dataclass
class FreeControlResult:
    image: Image.Image
    seed_used: int
    config_summary: str
    cache_key: str
