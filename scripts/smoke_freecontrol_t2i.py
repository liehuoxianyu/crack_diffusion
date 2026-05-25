import argparse
from pathlib import Path

from api.freecontrol import FreeControlT2IRequest, run_freecontrol_t2i


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for api.freecontrol T2I")
    parser.add_argument("--output", default="/work/outputs/freecontrol/smoke_t2i_output.png", help="Output PNG path")
    parser.add_argument("--prompt", default="A photo of a lion in the desert, best quality, detailed")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2028)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    req = FreeControlT2IRequest(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    result = run_freecontrol_t2i(req)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(out, format="PNG")
    print(f"Saved: {out}")
    print(result.config_summary)


if __name__ == "__main__":
    main()
