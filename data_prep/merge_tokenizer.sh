#!/bin/bash
#SBATCH --job-name=merge_tokenizer_llama3.1-8b
#SBATCH --output=logs/merge_tokenizers_vocab_llama3.1-8b_%j.out
#SBATCH --error=logs/merge_tokenizers_vocab_llama3.1-8b_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00

srun python -u merge_tokenizer.py \
  --base_tokenizer meta-llama/Llama-3.1-8B \
  --target_tokenizer /scratch/hpc-prf-merlin/project_data/moe_study/trained_multilingual_tokenizer_llama-3.1-8B \
  --data_dir /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset \
  --min_freq 5 \
  --output_dir /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/extended_tokenizer/llama-3.1-8B_extended \
  --freq_output /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/extended_tokenizer/llama-3.1-8B_extended/token_frequencies.tsv