#!/usr/bin/env bash

set -euo pipefail

DATASET_DIR=./LLaMA-Factory/data
OUTPUT_DIR=./LLaMA-Factory/saves/mola_sft

mkdir -p "${OUTPUT_DIR}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${OUTPUT_DIR}/hf_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

llamafactory-cli train \
  --stage pt \
  --do_train \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --dataset c4_demo \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --finetuning_type mola \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 2e-4 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target "q_proj,k_proj,v_proj,o_proj" \
  --bf16 True \
  --fp16 False \
  --mola_num_experts 4 \
  --mola_top_k 2 \
  --mola_use_null_expert \
  --mola_router_aux_loss_coef 0.01 \
  --mola_null_expert_penalty 0.1 \
  2>&1 | tee "${OUTPUT_DIR}/train_mola.log"
