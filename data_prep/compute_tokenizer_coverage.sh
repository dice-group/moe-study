#!/bin/bash
#SBATCH --job-name=tok_metrics
#SBATCH --output=tok_metrics_%j.out
#SBATCH --error=tok_metrics_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=normal

python -u compute_tokenizer_coverage.py \
  --data_dir /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset \
  --tokenizer meta-llama/Llama-3.2-1B \
  --max_lines 5000 \
  --num_workers 32 \
  --output_csv /scratch/hpc-prf-merlin/moe-study/data_prep/metrics/tokenizer_metrics.csv
