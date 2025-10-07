#!/bin/bash
#SBATCH --job-name=train_toke_llama_3.2-18b
#SBATCH --output=logs/train_tokenizer_llama_3.2-18b%j.out
#SBATCH --error=logs/train_tokenizer_llama_3.2-18b%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH -p normal 

srun python -u train_tokenizer.py \
    --data_dir /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset \
    --output_dir /scratch/hpc-prf-merlin/project_data/moe_study/250K_vocab_size/trained_multilingual_tokenizer_llama-3.2-1B \
    --vocab_size 250000 \
    --model_name meta-llama/Llama-3.2-1B