#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run_experiment.sh <config_json> <output_dir> [attn_implementation]"
  exit 1
fi

CONFIG_NAME="$1"
OUTPUT_DIR="$2"
ATTN_IMPL="${3:-eager}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

TOKENIZER_NAME="${TOKENIZER_NAME:-TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T}"
DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-wikitext-103-raw-v1}"

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision no \
  --dynamo_backend no \
  run_clm.py \
  --config_name "${CONFIG_NAME}" \
  --tokenizer_name "${TOKENIZER_NAME}" \
  --dataset_name "${DATASET_NAME}" \
  --dataset_config_name "${DATASET_CONFIG_NAME}" \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 8 \
  --auto_find_batch_size \
  --gradient_accumulation_steps 1 \
  --block_size 2048 \
  --attn_implementation "${ATTN_IMPL}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.015 \
  --learning_rate 3e-4 \
  --weight_decay 1e-1 \
  --bf16 \
  --torch_dtype bfloat16 \
  --do_train \
  --do_eval \
  --num_train_epochs 3 \
  --overwrite_output_dir \
  --save_total_limit 1 \
  --save_strategy steps \
  --save_steps 200 \
  --eval_strategy steps \
  --eval_steps 200 \
  --logging_steps 50 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_loss \
  --report_to none \
  --output_dir "${OUTPUT_DIR}"
