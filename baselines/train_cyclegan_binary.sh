#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/work/third_party/pytorch-CycleGAN-and-pix2pix}"
DATA_ROOT="${DATA_ROOT:-/work/outputs/GAN/data/cyclegan_unaligned}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/work/outputs/GAN/checkpoints}"
NAME="${NAME:-cyclegan_binary}"
BATCH_SIZE="${BATCH_SIZE:-1}"
N_EPOCHS="${N_EPOCHS:-100}"
N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-100}"
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-25}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"

cd "$REPO_ROOT"

python train.py \
  --dataroot "$DATA_ROOT" \
  --name "$NAME" \
  --model cycle_gan \
  --direction AtoB \
  --dataset_mode unaligned \
  --preprocess resize_and_crop \
  --load_size "$IMAGE_SIZE" \
  --crop_size "$IMAGE_SIZE" \
  --input_nc 3 \
  --output_nc 3 \
  --batch_size "$BATCH_SIZE" \
  --n_epochs "$N_EPOCHS" \
  --n_epochs_decay "$N_EPOCHS_DECAY" \
  --save_epoch_freq "$SAVE_EPOCH_FREQ" \
  --checkpoints_dir "$CHECKPOINTS_DIR"
