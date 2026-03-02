"""
CrackTree260 image+text only dataset for LoRA training (realism / lighting / texture).
No conditioning_image; excludes eval_ids so train/val split matches ControlNet eval.
Used by: diffusers/examples/text_to_image/train_text_to_image_lora.py
"""
import os
import json
import datasets

_DESCRIPTION = "CrackTree260 image+text for LoRA (realism). No conditioning."


def _get_env(name, default=None):
    v = os.environ.get(name, default)
    return default if v is None else str(v)


class CrackTreeLora(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [datasets.BuilderConfig(name="default", version=VERSION)]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
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
        from PIL import Image

        eval_ids_path = _get_env("CRACK_EVAL_IDS", "/CrackTree260/eval_ids.txt")
        if os.path.exists(eval_ids_path):
            with open(eval_ids_path, "r", encoding="utf-8") as ef:
                exclude_ids = set(line.strip() for line in ef if line.strip())
        else:
            exclude_ids = set()

        train_idx = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                it = json.loads(line)
                img_path = it["image"]
                base = os.path.splitext(os.path.basename(img_path))[0]
                if base in exclude_ids:
                    continue
                text = it.get("prompt", "")
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"missing image: {img_path}")
                img = Image.open(img_path).convert("RGB")
                yield train_idx, {"image": img, "text": text}
                train_idx += 1


BUILDER_CLASS = CrackTreeLora
