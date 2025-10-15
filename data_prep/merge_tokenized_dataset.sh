#!/bin/bash
#SBATCH --job-name=merge_tok
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/merge_%A.out
#SBATCH -p normal

ROOT=/scratch/hpc-prf-merlin/project_data/moe_study/tokenized

echo "Starting merge on $ROOT ..."
python merge_tokenized_datasets.py \
    --root $ROOT \
    --batch-size 20
