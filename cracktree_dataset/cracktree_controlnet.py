import os
import json
import datasets

_DESCRIPTION = "CrackTree260 paired dataset for ControlNet."

def _get_env(name, default=None):
    v = os.environ.get(name, default)
    return default if v is None else str(v)

class CrackTreeControlNet(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [datasets.BuilderConfig(name="default", version=VERSION)]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "conditioning_image": datasets.Image(),
                    "text": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        jsonl_path = _get_env("CRACK_JSONL", "/CrackTree260/train_linux.jsonl")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Cannot find jsonl: {jsonl_path}. Set CRACK_JSONL.")
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"jsonl_path": jsonl_path})]

    def _generate_examples(self, jsonl_path):
        # ---- Config via env ----
        COND_DIR = _get_env("CRACK_COND_DIR", "/CrackTree260/cond_mask")

        USE_PATCH = _get_env("CRACK_USE_PATCH", "1") == "1"
        PATCH = int(_get_env("CRACK_PATCH", "512"))

        TRY = int(_get_env("CRACK_TRY", "30"))
        TH = int(_get_env("CRACK_TH", "127"))

        P_RANDOM = float(_get_env("CRACK_P_RANDOM", "0.0"))
        BASE_SEED = int(_get_env("CRACK_CROP_SEED", "12345"))
        # ------------------------

        import numpy as np
        from PIL import Image
        import random

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                it = json.loads(line)
                img_path = it["image"]
                base = os.path.splitext(os.path.basename(img_path))[0]
                cond_path = os.path.join(COND_DIR, base + ".png")
                text = it.get("prompt", "")

                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"missing image: {img_path}")
                if not os.path.exists(cond_path):
                    raise FileNotFoundError(f"missing conditioning_image: {cond_path}")

                img = Image.open(img_path).convert("RGB")
                cond = Image.open(cond_path).convert("L")

                if not USE_PATCH:
                    yield i, {"image": img, "conditioning_image": cond.convert("RGB"), "text": text}
                    continue

                W, H = img.size
                if W < PATCH or H < PATCH:
                    img = img.resize((PATCH, PATCH), Image.BICUBIC)
                    cond = cond.resize((PATCH, PATCH), Image.BICUBIC)
                    yield i, {"image": img, "conditioning_image": cond.convert("RGB"), "text": text}
                    continue

                cond_np = np.array(cond, dtype=np.uint8)
                rng = random.Random(BASE_SEED + i)

                use_random = (rng.random() < P_RANDOM)

                if use_random:
                    x0 = rng.randint(0, W - PATCH)
                    y0 = rng.randint(0, H - PATCH)
                else:
                    best = (0, 0)
                    best_score = -1.0
                    for _ in range(TRY):
                        x = rng.randint(0, W - PATCH)
                        y = rng.randint(0, H - PATCH)
                        patch = cond_np[y:y+PATCH, x:x+PATCH]
                        score = float((patch > TH).mean())
                        if score > best_score:
                            best_score = score
                            best = (x, y)
                    x0, y0 = best

                img_c = img.crop((x0, y0, x0 + PATCH, y0 + PATCH))
                cond_c = cond.crop((x0, y0, x0 + PATCH, y0 + PATCH)).convert("RGB")

                yield i, {"image": img_c, "conditioning_image": cond_c, "text": text}

BUILDER_CLASS = CrackTreeControlNet