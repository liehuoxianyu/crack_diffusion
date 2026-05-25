#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from vqgan_binary_model import (
    ConditionalPatchDiscriminator,
    ConditionalVQGAN,
    VGGPerceptualLoss,
    hinge_d_loss,
    hinge_g_loss,
)


class PairedFolder(Dataset):
    def __init__(self, root: str, size: int = 512):
        self.a_dir = Path(root) / "A"
        self.b_dir = Path(root) / "B"
        self.ids = sorted(p.stem for p in self.a_dir.glob("*.png") if (self.b_dir / p.name).exists())
        self.tf = transforms.Compose(
            [
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        cond = Image.open(self.a_dir / f"{image_id}.png").convert("RGB")
        real = Image.open(self.b_dir / f"{image_id}.png").convert("RGB")
        return image_id, self.tf(cond), self.tf(real)


def denorm(x):
    x = (x.clamp(-1, 1) + 1.0) * 0.5
    return transforms.ToPILImage()(x.cpu())


def save_ckpt(path: Path, model, discriminator, optimizer_g, optimizer_d, step: int, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "step": step,
            "args": vars(args),
        },
        path,
    )


def write_samples(path: Path, model, samples, device: str):
    if not samples:
        return
    path.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for image_id, cond, real in samples:
            cond = cond.unsqueeze(0).to(device)
            pred, _ = model(cond)
            denorm(cond[0]).save(path / f"{image_id}_cond.png")
            denorm(real).save(path / f"{image_id}_real.png")
            denorm(pred[0]).save(path / f"{image_id}_fake.png")
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/work/outputs/GAN/data/paired/train")
    ap.add_argument("--val-root", default="/work/outputs/GAN/data/paired/test")
    ap.add_argument("--output-dir", default="/work/outputs/GAN/checkpoints/vqgan_binary")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-d", type=float, default=2e-4)
    ap.add_argument("--lambda-rec", type=float, default=1.0)
    ap.add_argument("--lambda-vq", type=float, default=1.0)
    ap.add_argument("--lambda-perceptual", type=float, default=0.1)
    ap.add_argument("--lambda-adv", type=float, default=0.1)
    ap.add_argument("--disc-start", type=int, default=1000)
    ap.add_argument("--save-steps", type=int, nargs="+", default=[5000, 10000, 15000, 20000])
    ap.add_argument("--sample-steps", type=int, default=1000)
    ap.add_argument("--log-steps", type=int, default=50)
    ap.add_argument("--disable-perceptual", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dataset = PairedFolder(args.data_root, size=args.image_size)
    if len(dataset) == 0:
        raise RuntimeError(f"No paired data found under {args.data_root}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = ConditionalVQGAN().to(args.device)
    discriminator = ConditionalPatchDiscriminator().to(args.device)
    optimizer_g = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.9))
    perceptual = None
    if not args.disable_perceptual and args.lambda_perceptual > 0:
        try:
            perceptual = VGGPerceptualLoss(args.device).eval()
        except Exception as exc:
            print(f"perceptual loss disabled: {exc}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "loss_log.txt"
    save_steps = set(args.save_steps)
    val_samples = []
    val_root = Path(args.val_root)
    if val_root.exists():
        val_dataset = PairedFolder(args.val_root, size=args.image_size)
        val_samples = [val_dataset[i] for i in range(min(4, len(val_dataset)))]

    step = 0
    model.train()
    discriminator.train()
    while step < args.max_steps:
        for _ids, cond, real in loader:
            step += 1
            cond = cond.to(args.device)
            real = real.to(args.device)

            # Generator update.
            pred, vq_loss = model(cond)
            rec_loss = F.l1_loss(pred, real)
            perc_loss = (
                perceptual(pred, real).mean()
                if perceptual is not None and args.lambda_perceptual > 0
                else pred.new_tensor(0.0)
            )
            adv_loss = (
                hinge_g_loss(discriminator(cond, pred))
                if step >= args.disc_start and args.lambda_adv > 0
                else pred.new_tensor(0.0)
            )
            g_loss = (
                args.lambda_rec * rec_loss
                + args.lambda_vq * vq_loss
                + args.lambda_perceptual * perc_loss
                + args.lambda_adv * adv_loss
            )

            optimizer_g.zero_grad(set_to_none=True)
            g_loss.backward()
            optimizer_g.step()

            # Discriminator update starts after the generator learns coarse alignment.
            if step >= args.disc_start and args.lambda_adv > 0:
                with torch.no_grad():
                    fake = model(cond)[0]
                real_logits = discriminator(cond, real)
                fake_logits = discriminator(cond, fake.detach())
                d_loss = hinge_d_loss(real_logits, fake_logits)
                optimizer_d.zero_grad(set_to_none=True)
                d_loss.backward()
                optimizer_d.step()
            else:
                d_loss = pred.new_tensor(0.0)

            if step % args.log_steps == 0:
                msg = (
                    f"step={step} g={g_loss.item():.4f} d={d_loss.item():.4f} "
                    f"rec={rec_loss.item():.4f} vq={vq_loss.item():.4f} "
                    f"perc={perc_loss.item():.4f} adv={adv_loss.item():.4f}"
                )
                print(msg)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            if step in save_steps:
                save_ckpt(out_dir / f"step{step}.pt", model, discriminator, optimizer_g, optimizer_d, step, args)
            if args.sample_steps > 0 and step % args.sample_steps == 0:
                write_samples(out_dir / "samples" / f"step{step}", model, val_samples, args.device)
            if step >= args.max_steps:
                break

    save_ckpt(out_dir / "latest.pt", model, discriminator, optimizer_g, optimizer_d, step, args)
    write_samples(out_dir / "samples" / "latest", model, val_samples, args.device)


if __name__ == "__main__":
    main()
