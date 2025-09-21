#!/bin/bash
#SBATCH --job-name=prep_tokenizer_data
#SBATCH --output=logs/prep_tokenizer_data_%j.out
#SBATCH --error=logs/prep_tokenizer_data_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -p normal

srun python subsample_tokenizer.py
