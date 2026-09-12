#!/usr/bin/env bash
# TASKS_FILE, SHARD, and OUTPUT_DIR are required.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -eo pipefail
: "${TASKS_FILE:?TASKS_FILE is required}"
: "${SHARD:?SHARD is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

module purge
module load toolchain/foss/2024a system/CUDA/12.6.0
source /opt/software/pc2/EB-SW/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate merlin
set -u

ROOT=/scratch/hpc-prf-merlin/joel/moe-study
cd "$ROOT"
TASKS_FILE="$(realpath "$TASKS_FILE")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

export HF_HOME=/scratch/hpc-prf-merlin/shared_cache/huggingface
export HF_DATASETS_CACHE=/scratch/hpc-prf-merlin/shared_cache/huggingface/datasets
export PYTHONPATH="$ROOT/scripts/eval/latest_checkpoint/lm-evaluation-harness:$ROOT/LLaMA-Factory/src:$ROOT:${PYTHONPATH:-}"
export WANDB_MODE=disabled

echo "[GPU_PREFLIGHT] baseline shard=${SHARD}"
timeout --signal=TERM --kill-after=30 330s python -u -m pipeline.gpu_preflight \
  --required-gpu-count 1 --required-gpu-name A100 --timeout-seconds 300 --check-ecc
echo "[GPU_PREFLIGHT_OK]"

HARNESS="$ROOT/scripts/eval/latest_checkpoint/lm-evaluation-harness"
python "$HARNESS/lm_eval/tasks/generate_local_task_configs.py"
INCLUDE_PATH="$HARNESS/lm_eval/tasks/flores_en_perplexity/configs:$HARNESS/lm_eval/tasks/sib200/configs:$HARNESS/lm_eval/tasks/taxi1500/configs"
TASKS="$(paste -sd, "$TASKS_FILE")"
cd "$HARNESS"

exec python -u -m lm_eval \
  --model hf \
  --model_args "pretrained=meta-llama/Llama-3.1-8B,tokenizer=meta-llama/Llama-3.1-8B,dtype=bfloat16" \
  --tasks "$TASKS" \
  --batch_size auto \
  --output_path "$OUTPUT_DIR" \
  --include_path "$INCLUDE_PATH" \
  --seed 42
