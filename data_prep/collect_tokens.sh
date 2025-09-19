#!/bin/bash
#SBATCH --job-name=tok-collect
#SBATCH --array=0-15
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/tok_collect_%A_%a.out
#SBATCH --error=logs/tok_collect_%A_%a.err
#SBATCH -p normal 

DATA_ROOT=/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/samples
OUTPUT_DIR=/scratch/hpc-prf-merlin/project_data/moe_study/tokenizer_ext
SCRIPT=extend_llama_tokenizer.py
BASE_MODEL=meta-llama/Llama-3.2-1b
MIN_FREQ=20
TARGET_VOCAB=5000

echo "[SLURM $SLURM_ARRAY_TASK_ID] Collecting tokens + frequencies for llama tokenizer extension"
srun python $SCRIPT \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --base_model $BASE_MODEL \
    --min_freq $MIN_FREQ \
    --target_vocab $TARGET_VOCAB \
    --debug
