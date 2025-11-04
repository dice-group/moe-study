#!/bin/bash

export WANDB_PROJECT="multilingual_tokenizer_perplexity_eval"
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs

nohup python train_base_hf.py \
  --model_name meta-llama/Llama-3.2-1B \
  --dataset_path ./loss_analysis_dataset/combined.jsonl \
  --tokenizer_path /data/project_data/moe_study/trained_multilingual_tokenizers/256k_vocab/llama-3.2-1B \
  --output_dir ./loss_analysis_output/multilingual_tokenizer \
  --per_device_batch_size 2 \
  --grad_accum_steps 1 \
  --run_name llama32-1b-multilingual-tokenizer > logs/train_multilingual.log 2>&1 &

echo $! > train_multilingual.pid
echo "Training started with PID $(cat train_multilingual.pid)"
