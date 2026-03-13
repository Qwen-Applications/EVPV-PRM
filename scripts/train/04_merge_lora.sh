#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Merge LoRA into base model with ms-swift export.
# Run after 01_sft_vision.sh (or 03_sft_step_judge.sh) to get a single deployable model.
# Set CKPT_DIR to the checkpoint to merge (e.g. checkpoint-564).
# -----------------------------------------------------------------------------
set -e

WORKSPACE_DIR="${WORKSPACE_DIR:-/path/to/workspace}"
# Base model (same as in 01_sft_vision.sh, or path to previous merged model)
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
# Training output dir that contains run subdirs (e.g. v12-20260128-080756) and checkpoints
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$WORKSPACE_DIR/Qwen2.5VL-VisionOutput}"
# Checkpoint to merge: either full path or name under latest run
CKPT_NAME="${CKPT_NAME:-checkpoint-564}"
if [[ -d "$CKPT_NAME" ]]; then
  CKPT_DIR="$CKPT_NAME"
else
  # Resolve latest run dir under TRAIN_OUTPUT_DIR
  LATEST_RUN=$(find "$TRAIN_OUTPUT_DIR" -maxdepth 1 -type d -name 'v*' 2>/dev/null | sort -r | head -1)
  CKPT_DIR="${CKPT_DIR:-$LATEST_RUN/$CKPT_NAME}"
fi
MERGED_OUTPUT_DIR="${MERGED_MODEL_DIR:-$WORKSPACE_DIR/Qwen2.5VL-VisionOutput-merged}"

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "Checkpoint not found: $CKPT_DIR. Set TRAIN_OUTPUT_DIR and CKPT_NAME, or CKPT_DIR."
  exit 1
fi

echo "Base model: $BASE_MODEL"
echo "Checkpoint: $CKPT_DIR"
echo "Output:     $MERGED_OUTPUT_DIR"

swift export \
  --model "$BASE_MODEL" \
  --train_type lora \
  --ckpt_dir "$CKPT_DIR" \
  --merge_lora true \
  --output_dir "$MERGED_OUTPUT_DIR" \
  --safe_serialization true

echo "Merge finished. Merged model: $MERGED_OUTPUT_DIR"
