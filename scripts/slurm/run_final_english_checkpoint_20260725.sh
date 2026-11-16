#!/usr/bin/env bash
# RUN_ID, CHECKPOINT, TASKS_FILE, and OUTPUT_DIR are supplied by sbatch.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -eo pipefail
: "${RUN_ID:?RUN_ID is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${TASKS_FILE:?TASKS_FILE is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

module purge
module load toolchain/foss/2024a system/CUDA/12.6.0
source /opt/software/pc2/EB-SW/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate merlin
set -u

ROOT=/scratch/hpc-prf-merlin/joel/moe-study
cd "$ROOT"
CHECKPOINT="$(realpath "$CHECKPOINT")"
TASKS_FILE="$(realpath "$TASKS_FILE")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

export HF_HOME=/scratch/hpc-prf-merlin/shared_cache/huggingface
export HF_DATASETS_CACHE=/scratch/hpc-prf-merlin/shared_cache/huggingface/datasets
export PYTHONPATH="$ROOT:$ROOT/LLaMA-Factory/src:${PYTHONPATH:-}"
export WANDB_MODE=disabled

echo "[GPU_PREFLIGHT] run=${RUN_ID}"
timeout --signal=TERM --kill-after=30 330s python -u -m pipeline.gpu_preflight \
  --required-gpu-count 1 --required-gpu-name A100 --timeout-seconds 300 --check-ecc
echo "[GPU_PREFLIGHT_OK]"

TASKS_CSV="$(paste -sd, "$TASKS_FILE")"
exec bash scripts/eval/lm_eval_checkpoint_adapter_only.sh \
  --checkpoint "$CHECKPOINT" \
  --tokenizer meta-llama/Llama-3.1-8B \
  --tasks "$TASKS_CSV" \
  --batch-size auto \
  --output-dir "$OUTPUT_DIR" \
  --wandb-project moe_study_eval \
  --wandb-prefix "eval/final_english/${RUN_ID}" \
  --wandb-mode disabled \
  --lang-mode no_ids
