#!/bin/bash
#SBATCH --job-name=train_toke
#SBATCH --output=logs/train_tokenizer_%j.out
#SBATCH --error=logs/train_tokenizer_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH -p normal 

srun python train_tokenizer.py \
    --data_dir /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/samples \
    --sample_sizes_tsv /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/tokenizer_lang_sample_sizes.tsv \
    --output_dir /scratch/hpc-prf-merlin/project_data/moe_study/trained_tokenizer \
    --vocab_size 250000