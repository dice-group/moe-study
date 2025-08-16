#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=1

cd LLaMA-Factory
echo "$PWD"


llamafactory-cli train \
  --stage sft --do_train true \
  --model_name_or_path mistralai/Mixtral-8x7B-v0.1 \
  --finetuning_type lora \
  --lora_target gate,gate2,w1,w2,w3 \
  --lora_rank 8 \
  --learning_rate 5e-5 \
  --quantization_bit 8 \
  --dataset wikiqa \
  --max_samples 1000 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --bf16 True \
  --save_steps 100 \
  --logging_steps 10 \
  --output_dir ./outputs/mixtral_ada/wino \
  --overwrite_output_dir True \
  --moe_aux_loss_coef 0.02 \
  --template default

