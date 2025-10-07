#!/bin/bash
#SBATCH --job-name=tok_metrics
#SBATCH --output=logs/compute_tok_metrics_llama3.2-1b_%j.out
#SBATCH --error=logs/compute_tok_metrics_llama3.2-1b_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --partition=normal

python -u compute_tokenizer_coverage.py \
  --data_dir /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset \
  --tokenizer /scratch/hpc-prf-merlin/project_data/moe_study/trained_multilingual_tokenizer_llama-3.2-1B \
  --max_lines 5000 \
  --num_workers 64 \
  --output_csv /scratch/hpc-prf-merlin/moe-study/data_prep/metrics/multilingual_retrained_tokenizer_metrics_llama3.2-1b_all_.csv
