#!/bin/bash
#SBATCH --job-name=tokenize_arrow
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=logs/tokenize_arrow_%x_%j.out
#SBATCH --error=logs/tokenize_arrow_%x_%j.err
#SBATCH --partition=normal


srun python -u tokenize_training_data.py \
  --parent-dir /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/samples \
  --out-dir /scratch/hpc-prf-merlin/project_data/moe_study/test_tokenized_arrow \
  --tokenizer llama3.2-1b \
  --workers 16
