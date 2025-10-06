#!/bin/bash
#SBATCH --job-name=tok_metrics
#SBATCH --output=logs/compute_tok_metrics_llama3.2-3b_%j.out
#SBATCH --error=logs/compute_tok_metrics_llama3.2-3b_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --partition=normal

python -u compute_tokenizer_coverage.py \
  --data_dir /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset \
  --tokenizer meta-llama/Llama-3.2-3B \
  --max_lines 5000000000000000 \
  --num_workers 64 \
  --output_csv /scratch/hpc-prf-merlin/moe-study/data_prep/metrics/tokenizer_metrics_llama3.2-3b_all_.csv
