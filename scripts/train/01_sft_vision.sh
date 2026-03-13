#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Vision understanding SFT with ms-swift.
# Prerequisites: sft_vision_data.jsonl under OUTPUT_BASE_DIR (see config).
# Set WORKSPACE_DIR, OUTPUT_BASE_DIR, and optional CUDA env before running.
# -----------------------------------------------------------------------------
set -e

# Environment (override via export or source config.env)
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29501}"

# Paths: set OUTPUT_BASE_DIR and WORKSPACE_DIR (e.g. in config.env)
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/path/to/evpv_data}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/path/to/workspace}"
DATASET_PATH="${DATASET_PATH:-$OUTPUT_BASE_DIR/sft_vision_data.jsonl}"
VISION_SFT_OUTPUT_DIR="${VISION_SFT_OUTPUT_DIR:-$WORKSPACE_DIR/Qwen2.5VL-VisionOutput}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Dataset not found: $DATASET_PATH. Run build_vision_sft_data.py first."
  exit 1
fi

echo "Dataset: $DATASET_PATH"
echo "Output:  $VISION_SFT_OUTPUT_DIR"

swift sft \
  --model "$BASE_MODEL" \
  --dataset "$DATASET_PATH" \
  --load_from_cache_file true \
  --split_dataset_ratio 0.01 \
  --train_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --attn_impl flash_attn \
  --padding_free true \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --packing true \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing false \
  --gradient_accumulation_steps 1 \
  --eval_steps 200 \
  --save_steps 200 \
  --save_total_limit 5 \
  --logging_steps 10 \
  --max_length 8192 \
  --output_dir "$VISION_SFT_OUTPUT_DIR" \
  --warmup_ratio 0.05 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 16

echo "Vision SFT finished. Checkpoints: $VISION_SFT_OUTPUT_DIR"
