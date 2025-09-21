#!/bin/bash
#SBATCH --job-name=train_toke_llama_8b
#SBATCH --output=logs/train_tokenizer_llama_8b%j.out
#SBATCH --error=logs/train_tokenizer_llama_8b%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH -p normal 

srun python -u train_tokenizer.py \
    --data_dir /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset \
    --output_dir /scratch/hpc-prf-merlin/project_data/moe_study/trained_multilingual_tokenizer_llama-3.1-8B \
    --vocab_size 300000 \
    --model_name meta-llama/Llama-3.1-8B