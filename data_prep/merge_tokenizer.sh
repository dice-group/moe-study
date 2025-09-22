#!/bin/bash
#SBATCH --job-name=merge_tokenizer
#SBATCH --output=logs/merge_tokenizers_vocab_%j.out
#SBATCH --error=logs/merge_tokenizers_vocab_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

srun python merge_tokenizer.py \
  --base_tokenizer meta-llama/Llama-3.2-1B \
  --target_tokenizer /scratch/hpc-prf-merlin/project_data/moe_study/trained_multilingual_tokenizer_llama-3.2-1B \
  --data_dir /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset \
  --min_freq 5 \
  --output_dir /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/extended_tokenizer/llama-3.2-1b_extended \
  --freq_output /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/extended_tokenizer/llama-3.2-1b_extended/token_frequencies.tsv