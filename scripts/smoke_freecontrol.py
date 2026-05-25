import argparse
from pathlib import Path

from PIL import Image

from api.freecontrol import FreeControlRequest, run_freecontrol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for api.freecontrol adapter")
    parser.add_argument("--condition-image", required=True, help="Path to condition image")
    parser.add_argument("--output", default="/work/outputs/freecontrol/smoke_output.png", help="Output PNG path")
    parser.add_argument("--prompt", default="A photo of a lion in the desert")
    parser.add_argument("--inversion-prompt", default="A photo of a dog")
    parser.add_argument("--obj-pairs", default="(dog; lion)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2028)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    condition_path = Path(args.condition_image)
    if not condition_path.exists():
        raise FileNotFoundError(f"Condition image not found: {condition_path}")

    condition_image = Image.open(condition_path).convert("RGB")
    req = FreeControlRequest(
        prompt=args.prompt,
        condition_image=condition_image,
        inversion_prompt=args.inversion_prompt,
        obj_pairs=args.obj_pairs,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    result = run_freecontrol(req)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(out, format="PNG")
    print(f"Saved: {out}")
    print(result.config_summary)


if __name__ == "__main__":
    main()
