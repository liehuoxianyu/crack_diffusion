#!/usr/bin/env bash
set -euo pipefail

# ========== LoRA 训练：仅 UNet，提升真实感（光照/纹理），基座 SD v1.5 ==========
# 使用 diffusers 官方 train_text_to_image_lora.py，数据为 CrackTree image+text（排除 eval_ids）
export CRACK_JSONL="/CrackTree260/train_linux.jsonl"
export CRACK_EVAL_IDS="/CrackTree260/eval_ids.txt"

# ========== 训练配置 ==========
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/work/outputs/exp_lora_realism"

mkdir -p "$OUTPUT_DIR"

cd /work/diffusers/examples/text_to_image

# 可选：清 datasets 缓存
rm -rf ~/.cache/huggingface/datasets ~/.cache/huggingface/modules/datasets_modules || true

accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --dataset_name="/work/cracktree_dataset/cracktree_lora.py" \
  --image_column="image" \
  --caption_column="text" \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=8 \
  --max_train_steps=1500 \
  --checkpointing_steps=500 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_prompt "a photo of pavement crack, realistic texture, high detail, natural shadowing, realistic lighting" \
  --num_validation_images=4 \
  --validation_epochs=5 \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention \
  --rank=4 \
  --seed=42 \
  --report_to="tensorboard" \
  --logging_dir "$OUTPUT_DIR/logs" \
  --output_dir="$OUTPUT_DIR" | tee "$OUTPUT_DIR/train.log"

# 训练结束后 LoRA 权重在: $OUTPUT_DIR/pytorch_lora_weights.safetensors（以及 checkpoint-*/ 内）
