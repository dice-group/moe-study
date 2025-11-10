#!/bin/bash

export WANDB_PROJECT="multilingual_tokenizer_perplexity_eval_belebele"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
mkdir -p logs

nohup python train_base_hf.py \
  --model_name meta-llama/Llama-3.2-1B \
  --dataset_path ./loss_analysis_dataset_low_resource/combined_low_resource.jsonl \
  --tokenizer_path meta-llama/Llama-3.2-1B \
  --output_dir ./loss_analysis_output/base \
  --run_name low_resource_only_llama-original \
  --per_device_batch_size 2 \
  --grad_accum_steps 1 \
  --max_steps 10000 \
  --save_steps 2000 \
  --enable_lm_eval \
  --lm_eval_interval 2000 \
  --lm_eval_tasks belebele_lao_Laoo belebele_ory_Orya belebele_kat_Geor belebele_tel_Telu \
  --lm_eval_batch_size 2 \
  --lm_eval_limit 500 \
  --lm_eval_cuda_devices "0" \
  --lm_eval_wandb_project "lm_eval_project" \
  --lm_eval_model_path meta-llama/Llama-3.2-1B \
  > logs/train_base_orig.log 2>&1 &

echo $! > train_base_orig.pid
echo "Training started with PID $(cat train_base_orig.pid)"