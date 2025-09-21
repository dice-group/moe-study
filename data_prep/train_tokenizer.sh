#!/bin/bash
#SBATCH --job-name=train_toke
#SBATCH --output=logs/train_tokenizer_%j.out
#SBATCH --error=logs/train_tokenizer_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -p normal 

srun python -u train_tokenizer.py \
    --data_dir /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset \
    --output_dir /scratch/hpc-prf-merlin/project_data/moe_study/trained_multilingual_tokenizer_llama-3.2-3B \
    --vocab_size 300000 \
    --model_name meta-llama/Llama-3.2-3B