#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Vision preference (DPO) training with ms-swift.
# Prerequisites: dpo_mm_data.jsonl (preference pairs). Set paths in env.
# -----------------------------------------------------------------------------
set -e

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/path/to/evpv_data}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/path/to/workspace}"
DPO_DATASET="${DPO_DATASET:-$OUTPUT_BASE_DIR/dpo_mm_data.jsonl}"
DPO_OUTPUT_DIR="${DPO_OUTPUT_DIR:-$WORKSPACE_DIR/VisionOutput_DPO}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"

if [[ ! -f "$DPO_DATASET" ]]; then
  echo "DPO dataset not found: $DPO_DATASET. Prepare preference pairs (e.g. from swiftdata.jsonl)."
  exit 1
fi

echo "Dataset: $DPO_DATASET"
echo "Output:  $DPO_OUTPUT_DIR"

swift dpo \
  --model "$BASE_MODEL" \
  --dataset "$DPO_DATASET" \
  --train_type lora \
  --torch_dtype bfloat16 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --gradient_accumulation_steps 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --num_train_epochs 1 \
  --max_length 8192 \
  --eval_steps 200 \
  --save_steps 200 \
  --save_total_limit 5 \
  --logging_steps 10 \
  --output_dir "$DPO_OUTPUT_DIR"

echo "DPO training finished. Output: $DPO_OUTPUT_DIR"
