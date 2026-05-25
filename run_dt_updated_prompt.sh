#!/usr/bin/env bash
set -euo pipefail

# ========== 数据加载配置（DT + updated prompts） ==========
export CRACK_JSONL="/CrackTree260/train_linux_updated.jsonl"
export CRACK_COND_DIR="/CrackTree260/cond_dt"
export CRACK_CROP_COND_DIR="/CrackTree260/cond_dt"

# patch 策略
export CRACK_USE_PATCH="1"
export CRACK_PATCH="512"
export CRACK_TRY="30"
export CRACK_TH="32"
export CRACK_P_RANDOM="0.0"
export CRACK_CROP_SEED="12345"

# ========== 训练配置 ==========
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CONTROLNET_INIT="lllyasviel/sd-controlnet-seg"
export OUTPUT_DIR="/work/outputs/exp_DT_updated_prompt_patch512"

mkdir -p "$OUTPUT_DIR"

cd /work/diffusers/examples/controlnet
rm -rf ~/.cache/huggingface/datasets ~/.cache/huggingface/modules/datasets_modules || true

accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --controlnet_model_name_or_path="$CONTROLNET_INIT" \
  --output_dir="$OUTPUT_DIR" \
  --dataset_name="/work/cracktree_dataset/cracktree_controlnet.py" \
  --image_column="image" \
  --conditioning_image_column="conditioning_image" \
  --caption_column="text" \
  --resolution=512 \
  --learning_rate=5e-6 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=2000 \
  --checkpointing_steps=500 \
  --validation_steps=500 \
  --validation_prompt "a photo of pavement crack, realistic texture, high detail, natural shadowing, realistic lighting" \
  --validation_image "/CrackTree260/cond_dt/6192.png" \
  --num_validation_images=4 \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none \
  --seed=42 \
  --report_to "tensorboard" \
  --logging_dir "$OUTPUT_DIR/logs" | tee "$OUTPUT_DIR/train.log"
