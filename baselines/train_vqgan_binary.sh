#!/usr/bin/env bash
set -euo pipefail

python /work/baselines/train_vqgan_binary.py \
  --data-root "${DATA_ROOT:-/work/outputs/GAN/data/paired/train}" \
  --val-root "${VAL_ROOT:-/work/outputs/GAN/data/paired/test}" \
  --output-dir "${OUTPUT_DIR:-/work/outputs/GAN/checkpoints/vqgan_binary}" \
  --image-size "${IMAGE_SIZE:-512}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --max-steps "${MAX_STEPS:-20000}" \
  --lr "${LR:-2e-4}" \
  --lr-d "${LR_D:-2e-4}" \
  --disc-start "${DISC_START:-1000}" \
  --lambda-rec "${LAMBDA_REC:-1.0}" \
  --lambda-vq "${LAMBDA_VQ:-1.0}" \
  --lambda-perceptual "${LAMBDA_PERCEPTUAL:-0.1}" \
  --lambda-adv "${LAMBDA_ADV:-0.1}" \
  --save-steps ${SAVE_STEPS:-5000 10000 15000 20000}
