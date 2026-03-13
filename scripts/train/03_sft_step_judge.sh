#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Step-discrimination SFT with ms-swift (after vision SFT).
# Uses merged vision model as base. Prerequisites: sft_processed_data_multithread.jsonl.
# -----------------------------------------------------------------------------
set -e

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"

export MASTER_PORT="${MASTER_PORT:-29501}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/path/to/evpv_data}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/path/to/workspace}"
# Path to merged vision model (after 04_merge_lora.sh from vision SFT)
MERGED_VISION_MODEL="${MERGED_VISION_MODEL:-$WORKSPACE_DIR/Qwen2.5VL-VisionOutput-merged}"
STEP_JUDGE_DATASET="${STEP_JUDGE_DATASET:-$OUTPUT_BASE_DIR/sft_processed_data_multithread.jsonl}"
STEP_JUDGE_OUTPUT_DIR="${STEP_JUDGE_SFT_OUTPUT_DIR:-$WORKSPACE_DIR/Qwen2.5VL-StepJudgeOutput}"

if [[ ! -d "$MERGED_VISION_MODEL" ]]; then
  echo "Merged vision model not found: $MERGED_VISION_MODEL. Run 01_sft_vision.sh then 04_merge_lora.sh."
  exit 1
fi
if [[ ! -f "$STEP_JUDGE_DATASET" ]]; then
  echo "Step-judge dataset not found: $STEP_JUDGE_DATASET. Run build_step_judge_sft_data.py and merge_vision_descriptions.py first."
  exit 1
fi

echo "Base model: $MERGED_VISION_MODEL"
echo "Dataset:    $STEP_JUDGE_DATASET"
echo "Output:     $STEP_JUDGE_OUTPUT_DIR"

swift sft \
  --model "$MERGED_VISION_MODEL" \
  --dataset "$STEP_JUDGE_DATASET" \
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
  --output_dir "$STEP_JUDGE_OUTPUT_DIR" \
  --warmup_ratio 0.05 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 1

echo "Step-judge SFT finished. Output: $STEP_JUDGE_OUTPUT_DIR"
