#!/usr/bin/env bash
# RUN_ROOT, RUN_ID, TASKS_FILE, SHARD, and OUTPUT_DIR are supplied by sbatch.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${TASKS_FILE:?TASKS_FILE is required}"
: "${SHARD:?SHARD is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
SHARD_PARTS="${SHARD_PARTS:-1}"
SHARD_PART_INDEX="${SHARD_PART_INDEX:-0}"

ROOT=/scratch/hpc-prf-merlin/joel/moe-study
PYTHON=/scratch/hpc-prf-merlin/joel/conda/envs/merlin/bin/python

module purge
module load toolchain/foss/2024a system/CUDA/12.6.0
set +u
source /opt/software/pc2/EB-SW/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate merlin
set -u
cd "$ROOT"

RUN_ROOT="$(realpath "$RUN_ROOT")"
TASKS_FILE="$(realpath "$TASKS_FILE")"
if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing non-empty benchmark output: $OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

export HF_HOME=/scratch/hpc-prf-merlin/shared_cache/huggingface
export HF_DATASETS_CACHE=/scratch/hpc-prf-merlin/shared_cache/huggingface/datasets
export PYTHONPATH="$ROOT:$ROOT/LLaMA-Factory/src:${PYTHONPATH:-}"
export WANDB_MODE=disabled

echo "[GPU_PREFLIGHT] run=${RUN_ID} shard=${SHARD} part=$((SHARD_PART_INDEX + 1))/${SHARD_PARTS}"
timeout --signal=TERM --kill-after=30 330s "$PYTHON" -u -m pipeline.gpu_preflight \
  --required-gpu-count 1 --required-gpu-name A100 --timeout-seconds 300 --check-ecc
echo "[GPU_PREFLIGHT_OK]"

CHECKPOINT="$("$PYTHON" scripts/analysis/resolve_direct_adapter_checkpoint.py \
  --run-root "$RUN_ROOT" --run-id "$RUN_ID")"
SHARD_TASKS="$OUTPUT_DIR/selected_tasks.txt"
"$PYTHON" -u scripts/analysis/select_final_benchmark_shard.py \
  --tasks-file "$TASKS_FILE" \
  --shard "$SHARD" \
  --parts "$SHARD_PARTS" \
  --part-index "$SHARD_PART_INDEX" \
  --output-file "$SHARD_TASKS"
TASKS_CSV="$(paste -sd, "$SHARD_TASKS")"

exec bash scripts/eval/lm_eval_checkpoint_adapter_only.sh \
  --checkpoint "$CHECKPOINT" \
  --tokenizer meta-llama/Llama-3.1-8B \
  --tasks "$TASKS_CSV" \
  --batch-size auto \
  --output-dir "$OUTPUT_DIR/results" \
  --wandb-project moe_study_eval \
  --wandb-prefix "eval/final_lora/${RUN_ID}/${SHARD}_part${SHARD_PART_INDEX}of${SHARD_PARTS}" \
  --wandb-mode disabled \
  --lang-mode no_ids
