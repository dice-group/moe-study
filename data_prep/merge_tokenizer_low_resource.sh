#!/bin/bash
#SBATCH --job-name=merge_selected_llama3.2-1b
#SBATCH --output=logs/merge_tok/merge_selected_llama3.2-1b_%A_%a.out
#SBATCH --error=logs/merge_tok/merge_selected_llama3.2-1b_%A_%a.err
#SBATCH --array=0-57
#SBATCH --mem=64G
#SBATCH -p normal
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00

LANG_LIST=/scratch/hpc-prf-merlin/moe-study/data_prep/langs_selected.txt
LANG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$LANG_LIST")

DATA_DIR=/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset
OUTPUT_DIR=/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/extended_tokenizer/test
BASE_TOKENIZER=meta-llama/Llama-3.2-1B
TARGET_TOKENIZER=/scratch/hpc-prf-merlin/project_data/moe_study/trained_multilingual_tokenizer_llama-3.2-1B

echo "Processing language: $LANG"

srun python -u merge_tokenizer.py \
  --base_tokenizer "$BASE_TOKENIZER" \
  --target_tokenizer "$TARGET_TOKENIZER" \
  --data_dir "$DATA_DIR" \
  --min_freq 1 \
  --limit 500 \
  --output_dir "$OUTPUT_DIR" \
  --lang "$LANG"
