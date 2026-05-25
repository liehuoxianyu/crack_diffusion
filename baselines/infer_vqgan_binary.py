#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from vqgan_binary_model import ConditionalVQGAN


def denorm(x):
    x = (x.clamp(-1, 1) + 1.0) * 0.5
    return transforms.ToPILImage()(x.cpu())


def run_checkpoint(args, checkpoint: Path, output_dir: Path):
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")

    model = ConditionalVQGAN().to(args.device)
    ckpt = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(args.input_dir).glob("*.png")):
        cond = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(args.device)
        with torch.no_grad():
            pred, _ = model(cond)
        denorm(pred[0]).save(output_dir / f"{p.stem}_fake_B.png")
    print(f"wrote VQGAN outputs checkpoint={checkpoint} to {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/work/outputs/GAN/checkpoints/vqgan_binary/latest.pt")
    ap.add_argument("--checkpoints-dir", default="/work/outputs/GAN/checkpoints/vqgan_binary")
    ap.add_argument("--steps", type=int, nargs="+", default=None, help="Infer step checkpoints into step<label>/images.")
    ap.add_argument("--input-dir", default="/work/outputs/GAN/data/paired/test/A")
    ap.add_argument("--output-dir", default="/work/outputs/GAN/raw_results/vqgan_binary/latest/images")
    ap.add_argument("--raw-root", default="/work/outputs/GAN/raw_results/vqgan_binary")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.steps:
        for step in args.steps:
            checkpoint = Path(args.checkpoints_dir) / f"step{step}.pt"
            output_dir = Path(args.raw_root) / f"step{step}" / "images"
            run_checkpoint(args, checkpoint, output_dir)
    else:
        run_checkpoint(args, Path(args.checkpoint), Path(args.output_dir))


if __name__ == "__main__":
    main()
