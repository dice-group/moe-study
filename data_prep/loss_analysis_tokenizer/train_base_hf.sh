#!/bin/bash
#SBATCH --job-name=llama_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

export WANDB_PROJECT="multilingual_tokenizer_perplexity_eval_belebele"
export TOKENIZERS_PARALLELISM=false

mkdir -p logs

srun python train_base_hf.py \
  --model_name meta-llama/Llama-3.2-1B \
  --dataset_path ./loss_analysis_dataset_low_resource/combined_low_resource.jsonl \
  --tokenizer_path meta-llama/Llama-3.2-1B \
  --output_dir ./loss_analysis_output/base \
  --run_name low_resource_only_llama-original \
  --per_device_batch_size 2 \
  --grad_accum_steps 1 \
  --max_steps 10000 \
  --save_steps 2000
