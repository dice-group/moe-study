#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_BIN_DEFAULT="/anonymous-institution/users/j/anonymous-user/profiles/unix/cs/.conda/envs/merlin/bin"
if [[ -d "${CONDA_BIN_DEFAULT}" ]]; then
  export PATH="${CONDA_BIN_DEFAULT}:${PATH}"
fi

MODE="${MODE:-both}" # both | healthy | malformed
TRAIN_MODE="${TRAIN_MODE:-parallel}" # parallel | sequential
PARALLEL_LAUNCH_DELAY_SEC="${PARALLEL_LAUNCH_DELAY_SEC:-8}"
DRY_RUN="${DRY_RUN:-0}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.2-1B}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

HEALTHY_TOKENIZED_PATH="${HEALTHY_TOKENIZED_PATH:-/tmp/tokenized_10langs_masked}"
MALFORMED_TOKENIZED_PATH="${MALFORMED_TOKENIZED_PATH:-${REPO_ROOT}/tokenized_data/10_langs_ranks/rank_00047}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/cola_rank47_loss_repro_${TIMESTAMP}}"
HEALTHY_OUTPUT_DIR="${HEALTHY_OUTPUT_DIR:-${OUTPUT_ROOT}/healthy_masked}"
MALFORMED_OUTPUT_DIR="${MALFORMED_OUTPUT_DIR:-${OUTPUT_ROOT}/malformed_rank47}"

WANDB_PROJECT="${WANDB_PROJECT:-moe-study-cola-rank47-loss-repro}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-cola-rank47-loss-repro-${TIMESTAMP}}"
WANDB_MODE="${WANDB_MODE:-online}"

DATASET_NAME="${DATASET_NAME:-rank47_loss_repro}"
USE_ACCELERATE="${USE_ACCELERATE:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-300}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-12}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-12}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
AUTO_FIND_BATCH_SIZE="${AUTO_FIND_BATCH_SIZE:-true}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-25}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-False}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-8}"
PREPROCESSING_BATCH_SIZE="${PREPROCESSING_BATCH_SIZE:-100000}"
DISABLE_GRADIENT_CHECKPOINTING="${DISABLE_GRADIENT_CHECKPOINTING:-True}"
BF16="${BF16:-True}"
FP16="${FP16:-False}"
FLASH_ATTN="${FLASH_ATTN:-sdpa}"
COLA_NUM_EXPERTS="${COLA_NUM_EXPERTS:-4}"
COLA_TOP_K="${COLA_TOP_K:-2}"
COLA_STRATEGY="${COLA_STRATEGY:-fully}"
USE_COLA_EXPERTS="${USE_COLA_EXPERTS:-False}"
SEED="${SEED:-42}"

HEALTHY_GPU_ID="${HEALTHY_GPU_ID:-0}"
MALFORMED_GPU_ID="${MALFORMED_GPU_ID:-1}"
HEALTHY_MASTER_PORT="${HEALTHY_MASTER_PORT:-29701}"
MALFORMED_MASTER_PORT="${MALFORMED_MASTER_PORT:-29702}"

mkdir -p "${OUTPUT_ROOT}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Required file does not exist: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Required directory does not exist: ${path}" >&2
    exit 1
  fi
}

validate_train_config() {
  if [[ "${GPUS_PER_NODE}" != "1" ]]; then
    echo "[ERROR] This launcher expects one GPU per run (GPUS_PER_NODE=1). Current: ${GPUS_PER_NODE}" >&2
    exit 1
  fi
  if ! [[ "${HEALTHY_GPU_ID}" =~ ^[0-9]+$ && "${MALFORMED_GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] HEALTHY_GPU_ID and MALFORMED_GPU_ID must be integer GPU indices." >&2
    exit 1
  fi
  if [[ "${TRAIN_MODE}" == "parallel" && "${HEALTHY_GPU_ID}" == "${MALFORMED_GPU_ID}" ]]; then
    echo "[ERROR] Parallel mode requires distinct GPUs. Got HEALTHY_GPU_ID=${HEALTHY_GPU_ID}, MALFORMED_GPU_ID=${MALFORMED_GPU_ID}." >&2
    exit 1
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_count
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
    if (( HEALTHY_GPU_ID >= gpu_count || MALFORMED_GPU_ID >= gpu_count )); then
      echo "[ERROR] GPU index out of range. gpu_count=${gpu_count}, HEALTHY_GPU_ID=${HEALTHY_GPU_ID}, MALFORMED_GPU_ID=${MALFORMED_GPU_ID}" >&2
      exit 1
    fi
  fi
}

run_variant() {
  local label="$1"       # healthy | malformed
  local gpu_id="$2"
  local tokenized_path="$3"
  local output_dir="$4"
  local master_port="$5"
  local run_async="$6"   # 0 | 1
  local run_name="cola-rank47-repro-${label}-${TIMESTAMP}"
  local log_path="${output_dir}.log"

  require_dir "${tokenized_path}"
  mkdir -p "${output_dir}"

  local run_block
  run_block() {
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export MASTER_PORT="${master_port}"
    export USE_ACCELERATE
    export GPUS_PER_NODE
    export ACCELERATE_CONFIG

    export WANDB_PROJECT
    export WANDB_RUN_GROUP
    export WANDB_MODE
    export WANDB_NAME="${run_name}"

    export MODEL_NAME_OR_PATH
    export OUTPUT_DIR="${output_dir}"
    export GENERIC_TOKENIZED_PATH="${tokenized_path}"
    export GENERIC_DATASET_NAME="${DATASET_NAME}"
    export GENERIC_EVAL_DATASET_NAME="${DATASET_NAME}"

    export NUM_TRAIN_EPOCHS
    export MAX_STEPS
    export LEARNING_RATE
    export PER_DEVICE_TRAIN_BATCH_SIZE
    export PER_DEVICE_EVAL_BATCH_SIZE
    export GRADIENT_ACCUMULATION_STEPS
    export AUTO_FIND_BATCH_SIZE
    export LOGGING_STEPS
    export EVAL_STEPS
    export SAVE_STEPS
    export SAVE_TOTAL_LIMIT
    export SAVE_ONLY_MODEL
    export PREPROCESSING_NUM_WORKERS
    export PREPROCESSING_BATCH_SIZE
    export DISABLE_GRADIENT_CHECKPOINTING
    export BF16
    export FP16
    export FLASH_ATTN
    export COLA_NUM_EXPERTS
    export COLA_TOP_K
    export COLA_STRATEGY
    export USE_COLA_EXPERTS
    export SEED

    echo "[INFO] [${label}] GPU=${gpu_id} tokenized_path=${tokenized_path}"
    echo "[INFO] [${label}] max_steps=${MAX_STEPS} train_bs=${PER_DEVICE_TRAIN_BATCH_SIZE} eval_bs=${PER_DEVICE_EVAL_BATCH_SIZE} grad_acc=${GRADIENT_ACCUMULATION_STEPS} master_port=${master_port}"
    echo "[INFO] [${label}] Logging to ${log_path}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[DRY_RUN] bash ${REPO_ROOT}/scripts/slurm/comparison/cola_job.sh"
    else
      bash "${REPO_ROOT}/scripts/slurm/comparison/cola_job.sh" >"${log_path}" 2>&1
    fi
  }

  if [[ "${run_async}" == "1" ]]; then
    run_block &
    LAST_LAUNCHED_PID=$!
  else
    run_block
  fi
}

train_single() {
  local label="$1"
  require_file "${REPO_ROOT}/scripts/slurm/comparison/cola_job.sh"
  validate_train_config
  if [[ "${label}" == "healthy" ]]; then
    run_variant "healthy" "${HEALTHY_GPU_ID}" "${HEALTHY_TOKENIZED_PATH}" "${HEALTHY_OUTPUT_DIR}" "${HEALTHY_MASTER_PORT}" "0"
  elif [[ "${label}" == "malformed" ]]; then
    run_variant "malformed" "${MALFORMED_GPU_ID}" "${MALFORMED_TOKENIZED_PATH}" "${MALFORMED_OUTPUT_DIR}" "${MALFORMED_MASTER_PORT}" "0"
  else
    echo "[ERROR] Invalid label for train_single: ${label}" >&2
    exit 1
  fi
}

train_both() {
  require_file "${REPO_ROOT}/scripts/slurm/comparison/cola_job.sh"
  require_dir "${HEALTHY_TOKENIZED_PATH}"
  require_dir "${MALFORMED_TOKENIZED_PATH}"
  validate_train_config

  if [[ "${TRAIN_MODE}" == "parallel" ]]; then
    echo "[INFO] Launching in parallel (healthy on GPU${HEALTHY_GPU_ID}, malformed on GPU${MALFORMED_GPU_ID})"
    run_variant "healthy" "${HEALTHY_GPU_ID}" "${HEALTHY_TOKENIZED_PATH}" "${HEALTHY_OUTPUT_DIR}" "${HEALTHY_MASTER_PORT}" "1"
    local pid_healthy="${LAST_LAUNCHED_PID}"
    if [[ "${PARALLEL_LAUNCH_DELAY_SEC}" -gt 0 ]]; then
      sleep "${PARALLEL_LAUNCH_DELAY_SEC}"
    fi
    run_variant "malformed" "${MALFORMED_GPU_ID}" "${MALFORMED_TOKENIZED_PATH}" "${MALFORMED_OUTPUT_DIR}" "${MALFORMED_MASTER_PORT}" "1"
    local pid_malformed="${LAST_LAUNCHED_PID}"

    local rc=0
    wait "${pid_healthy}" || rc=$?
    wait "${pid_malformed}" || rc=$?
    if [[ "${rc}" -ne 0 ]]; then
      echo "[ERROR] At least one run failed. Check logs:" >&2
      echo "  - ${HEALTHY_OUTPUT_DIR}.log" >&2
      echo "  - ${MALFORMED_OUTPUT_DIR}.log" >&2
      exit "${rc}"
    fi
    return
  fi

  if [[ "${TRAIN_MODE}" == "sequential" ]]; then
    train_single "healthy"
    train_single "malformed"
    return
  fi

  echo "[ERROR] Invalid TRAIN_MODE=${TRAIN_MODE}. Use sequential|parallel." >&2
  exit 1
}

print_summary() {
  cat <<EOF
[INFO] CoLA loss reproduction setup
  healthy tokenized path:   ${HEALTHY_TOKENIZED_PATH}
  malformed tokenized path: ${MALFORMED_TOKENIZED_PATH}
  output root:              ${OUTPUT_ROOT}
  W&B project:              ${WANDB_PROJECT}
  W&B group:                ${WANDB_RUN_GROUP}
  mode:                     ${MODE}
  train mode:               ${TRAIN_MODE}
  max steps:                ${MAX_STEPS}
  healthy gpu:              ${HEALTHY_GPU_ID}
  malformed gpu:            ${MALFORMED_GPU_ID}
EOF
}

main() {
  print_summary
  case "${MODE}" in
    both)
      train_both
      ;;
    healthy)
      train_single "healthy"
      ;;
    malformed)
      train_single "malformed"
      ;;
    *)
      echo "[ERROR] Invalid MODE=${MODE}. Use both|healthy|malformed." >&2
      exit 1
      ;;
  esac
  echo "[INFO] Done."
}

main "$@"
