#!/usr/bin/env bash

set -euo pipefail

DATASET_DIR=./LLaMA-Factory/data
OUTPUT_DIR=./LLaMA-Factory/saves/smoke_test

mkdir -p "${OUTPUT_DIR}"

llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --dataset c4 \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --finetuning_type cola \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --bf16 True \
  --fp16 False \
  --cola_debug 2>&1 | tee train_moe_debug.log