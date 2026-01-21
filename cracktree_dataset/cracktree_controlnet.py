import os
import json
import datasets
from PIL import Image

_DESCRIPTION = "CrackTree260 paired dataset for ControlNet: image + conditioning_image(mask) + text."

class CrackTreeControlNet(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="default", version=VERSION, description="default config"),
    ]
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
        # 优先使用环境变量指定的 jsonl
        jsonl_path = os.environ.get("CRACK_JSONL", "").strip()

        # 否则默认使用固定路径
        if not jsonl_path:
            jsonl_path = "/CrackTree260/train_linux.jsonl"

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Cannot find jsonl: {jsonl_path}. Set CRACK_JSONL or place train_linux.jsonl at /CrackTree260/")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"jsonl_path": jsonl_path},
            )
        ]

    def _generate_examples(self, jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                yield i, {
                    "image": item["image"],
                    "conditioning_image": item["conditioning_image"],
                    "text": item.get("prompt", ""),
                }