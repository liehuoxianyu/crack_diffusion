# Binary GAN Baselines

This document describes baseline workflows for Binary condition to crack image generation.

## Repositories

Baseline source code is kept under:

- `/work/third_party/pytorch-CycleGAN-and-pix2pix`
- `/work/third_party/VQGAN-pytorch`

Local wrappers live in `/work/baselines`.

## 1. Prepare Data

```bash
python /work/baselines/prepare_binary_paired_dataset.py \
  --jsonl /CrackTree260/train_linux.jsonl \
  --eval-ids /CrackTree260/eval_ids.txt \
  --cond-dir /CrackTree260/cond_mask \
  --out-root /work/outputs/GAN/data \
  --size 512
```

The script writes:

- `paired/train/A`, `paired/train/B`, `paired/test/A`, `paired/test/B`
- `pix2pix_aligned/train`, `pix2pix_aligned/test`
- `cyclegan_unaligned/trainA`, `trainB`, `testA`, `testB`

## 2. Train Baselines

pix2pix:

```bash
bash /work/baselines/train_pix2pix_binary.sh
```

CycleGAN:

```bash
bash /work/baselines/train_cyclegan_binary.sh
```

Conditional VQGAN:

```bash
bash /work/baselines/train_vqgan_binary.sh
```

By default these commands write checkpoints under `/work/outputs/GAN/checkpoints`.
The CycleGAN/pix2pix wrappers accept environment overrides such as `BATCH_SIZE`,
`N_EPOCHS`, `N_EPOCHS_DECAY`, and `SAVE_EPOCH_FREQ`.

VQGAN is trained by optimizer step, not epoch. The local VQGAN baseline uses a
conditional VQ generator with a PatchGAN discriminator and optional VGG
perceptual loss:

```bash
MAX_STEPS=20000 SAVE_STEPS="5000 10000 15000 20000" \
  bash /work/baselines/train_vqgan_binary.sh
```

## 3. Run Inference

pix2pix:

```bash
python /work/baselines/infer_pix2pix_binary.py --epoch latest
```

CycleGAN:

```bash
python /work/baselines/infer_cyclegan_binary.py --epochs 25 50 75 100 125 150 175 200
```

VQGAN:

```bash
python /work/baselines/infer_vqgan_binary.py --steps 5000 10000 15000 20000
```

## 4. Export to eval_all-Compatible Layout

```bash
python /work/baselines/eval_baselines_binary.py --baseline pix2pix
python /work/baselines/eval_baselines_binary.py --baseline cyclegan --labels 25 50 75 100 125 150 175 200 --strict
python /work/baselines/eval_baselines_binary.py --baseline vqgan --labels 5000 10000 15000 20000 --strict
```

Each baseline writes:

```text
/work/outputs/GAN/eval/<baseline>/id{id}/controlnet_binary/step{label}/gs7.5_cs1.0.png
```

The `controlnet_binary` name, `step<label>` directory, and `gs/cs` suffix are
compatibility shims for the existing metric scripts. Their semantics differ by
baseline:

- pix2pix/CycleGAN labels are real training epochs.
- VQGAN labels are real optimizer steps.
- Diffusion labels are diffusion checkpoint steps.

Do not export multiple labels from `latest` output. The exporter no longer does
that by default; use `--allow-latest-fallback` only for manual visual debugging,
not for metrics.

## 5. Run Metrics

```bash
bash /work/evaluation/run_eval_metrics.sh --profile baseline --baseline pix2pix
bash /work/evaluation/run_eval_metrics.sh --profile baseline --baseline cyclegan
bash /work/evaluation/run_eval_metrics.sh --profile baseline --baseline vqgan
```

The runner sets `EVAL_ROOT` and reuses:

- `/work/metric_fid_psnr.py`
- `/work/metric_struct_align.py`
