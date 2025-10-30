#!/bin/bash
#SBATCH --job-name=extend_tok
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/extend_with_merges_%x_%j.out
#SBATCH -p normal

export TOKENIZERS_PARALLELISM=false  # prevent HF from internal threading issues

srun python -u extend_tok_most_important.py
