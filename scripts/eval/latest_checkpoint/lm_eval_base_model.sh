#!/bin/bash
#SBATCH --job-name=lm-eval-base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/lm_eval_base_%j.log

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/cluster/user/moe-study}"
HARNESS_ROOT="${LM_EVAL_HARNESS_PATH:-/cluster/user/moe-study/scripts/eval/latest_checkpoint/lm-evaluation-harness}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.2-1B}"
TASKS_FILE="${TASKS_FILE:-${REPO_ROOT}/configs/lm_eval_tasks_10_langs.txt}"
BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LIMIT="${LM_EVAL_LIMIT:-}"
WANDB_PROJECT="${WANDB_PROJECT:-moe-study-comparison_labeled_eval}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-llama32-1b-base-10langs}"
WANDB_NAME="${WANDB_NAME:-llama32-1b-base-10langs_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/scratch/anonymous-project/project_data/moe_study/moe_study_comparison_outputs_labeled/llama-3.2-1B_tokenizer-10_langs/base_model_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUT_ROOT}/$(date +%Y%m%d_%H%M%S)}"

join_include_paths() {
  local parts=()
  local item
  for item in "$@"; do
    [[ -d "$item" ]] || continue
    parts+=("$item")
  done
  local IFS=:
  printf '%s' "${parts[*]}"
}

ensure_external_task_configs() {
  python "${HARNESS_ROOT}/lm_eval/tasks/generate_local_task_configs.py"
}

FLORES_CONFIGS="${HARNESS_ROOT}/lm_eval/tasks/flores_en_perplexity/configs"
SIB200_TASKS="${HARNESS_ROOT}/lm_eval/tasks/sib200/configs"
TAXI1500_TASKS="${HARNESS_ROOT}/lm_eval/tasks/taxi1500/configs"
ensure_external_task_configs
INCLUDE_PATH="${LM_EVAL_INCLUDE_PATH:-$(join_include_paths "${FLORES_CONFIGS}" "${SIB200_TASKS}" "${TAXI1500_TASKS}")}"

if [[ ! -d "${HARNESS_ROOT}" ]]; then
  echo "[ERROR] Harness not found: ${HARNESS_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${TASKS_FILE}" ]]; then
  echo "[ERROR] Tasks file not found: ${TASKS_FILE}" >&2
  exit 1
fi

TASKS=$(grep -v '^[[:space:]]*#' "${TASKS_FILE}" | awk 'NF' | paste -sd, -)
if [[ -z "${TASKS}" ]]; then
  echo "[ERROR] No tasks found in ${TASKS_FILE}" >&2
  exit 1
fi

if [[ -n "${MODULE_INIT:-}" ]]; then
  eval "${MODULE_INIT}"
fi

set +u
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-${CONDA_ENV:-}}"
if [[ -n "${CONDA_BASE:-}" && -n "${EVAL_CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${EVAL_CONDA_ENV}"
fi
set -u

mkdir -p "${OUTPUT_DIR}"
cd "${HARNESS_ROOT}"

WANDB_ARGS="project=${WANDB_PROJECT},group=${WANDB_GROUP},name=${WANDB_NAME},job_type=lm-eval-base"
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ARGS="${WANDB_ARGS},entity=${WANDB_ENTITY}"
fi

CMD=(
  python -m lm_eval
  --model hf
  --model_args "pretrained=${MODEL_NAME_OR_PATH},tokenizer=${MODEL_NAME_OR_PATH},dtype=bfloat16,device_map=auto"
  --tasks "${TASKS}"
  --batch_size "${BATCH_SIZE}"
  --output_path "${OUTPUT_DIR}/base_lm_eval.json"
  --wandb_args "${WANDB_ARGS}"
)

if [[ -n "${INCLUDE_PATH}" ]]; then
  CMD+=(--include_path "${INCLUDE_PATH}")
fi
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

echo "[INFO] Base model eval"
echo "  model:      ${MODEL_NAME_OR_PATH}"
echo "  tasks file: ${TASKS_FILE}"
echo "  output:     ${OUTPUT_DIR}"
echo "  wandb:      ${WANDB_PROJECT} / ${WANDB_GROUP} / ${WANDB_NAME}"

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[INFO] Done"
