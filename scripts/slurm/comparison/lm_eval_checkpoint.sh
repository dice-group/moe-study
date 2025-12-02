#!/bin/bash
#SBATCH --job-name=cmp-lmeval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${MODULE_INIT:-}" ]]; then
  eval "${MODULE_INIT}"
fi

if [[ -n "${CONDA_BASE:-}" && -n "${CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH:?CHECKPOINT_PATH not set}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${CHECKPOINT_PATH}}"
TASKS="${LM_EVAL_TASKS:-belebele}"
BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
RESULTS_DIR="${LM_EVAL_RESULTS_DIR:-${CHECKPOINT_PATH}/lm_eval_results}"
LIMIT="${LM_EVAL_LIMIT:-0}"
mkdir -p "${RESULTS_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"
output_json="${RESULTS_DIR}/lm_eval_${timestamp}.json"

wandb_group="${WANDB_GROUP:-lm-eval}"
wandb_project="${WANDB_PROJECT:-lm-eval}"
wandb_name="${WANDB_NAME:-${wandb_group}-${timestamp}}"
wandb_args="project=${wandb_project},group=${wandb_group},job_type=lm-eval,name=${wandb_name}"
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  wandb_args+=",entity=${WANDB_ENTITY}"
fi

cmd=(
  lm_eval
  --model hf
  --model_args pretrained="${CHECKPOINT_PATH}",tokenizer="${TOKENIZER_PATH}",trust_remote_code=True
  --tasks "${TASKS}"
  --batch_size "${BATCH_SIZE}"
  --output_path "${output_json}"
  --wandb_args "${wandb_args}"
)

if [[ "${LIMIT}" != "0" && -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi

"${cmd[@]}"
