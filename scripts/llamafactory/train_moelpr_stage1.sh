#!/usr/bin/env bash

set -euo pipefail

DATASET_DIR=./LLaMA-Factory/data
OUTPUT_DIR=./LLaMA-Factory/saves/moelpr_stage1
LANG_MAP='{"en":"germanic","fr":"romance","de":"germanic"}'

mkdir -p "${OUTPUT_DIR}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${OUTPUT_DIR}/hf_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --dataset moelpr_stage1_demo \
  --dataset_dir "${DATASET_DIR}" \
  --template alpaca \
  --finetuning_type moelpr \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --logging_steps 10 \
  --save_steps 50 \
  --bf16 False \
  --fp16 False \
  --language_column language \
  --language_map "${LANG_MAP}" \
  --moelpr_stage 1 \
  --moelpr_num_experts 4 \
  --moelpr_top_k 1 \
  --moelpr_layers_to_transform "0,1,2" \
  --moelpr_aux_loss_coef 0.01 \
  --moelpr_debug_mode True \
  "$@" \
  2>&1 | tee "${OUTPUT_DIR}/train_moelpr_stage1.log"
