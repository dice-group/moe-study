#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_BIN_DEFAULT="/anonymous-institution/users/j/anonymous-user/profiles/unix/cs/.conda/envs/merlin/bin"
if [[ -d "${CONDA_BIN_DEFAULT}" ]]; then
  export PATH="${CONDA_BIN_DEFAULT}:${PATH}"
fi

MODE="${MODE:-all}" # tokenize | train | train_masked | train_unmasked | all
TRAIN_MODE="${TRAIN_MODE:-sequential}" # sequential | parallel (used when MODE=train/all)
PARALLEL_LAUNCH_DELAY_SEC="${PARALLEL_LAUNCH_DELAY_SEC:-10}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_RETOKENIZE="${FORCE_RETOKENIZE:-0}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.2-1B}"
SAMPLED_ROOT="${SAMPLED_ROOT:-${REPO_ROOT}/sampled_data/llama-3.2-1B/200_langs}"
SHARD_DIR="${SHARD_DIR:-${SAMPLED_ROOT}/samples}"
TOKENIZER_PY="${TOKENIZER_PY:-${REPO_ROOT}/data_prep/merlin_data_prep/distributed_data_processor/src/tokenize_slurm.py}"
PYTHON_BIN="${PYTHON_BIN:-/anonymous-institution/users/j/anonymous-user/profiles/unix/cs/.conda/envs/merlin/bin/python}"

TOKENIZED_ROOT="${TOKENIZED_ROOT:-/tmp/tokenized_200langs_padmask_ablation_1b}"
TOKENIZED_MASKED_DIR="${TOKENIZED_MASKED_DIR:-${TOKENIZED_ROOT}/masked}"
TOKENIZED_UNMASKED_DIR="${TOKENIZED_UNMASKED_DIR:-${TOKENIZED_ROOT}/unmasked}"
LANGUAGE_MAP_PATH="${LANGUAGE_MAP_PATH:-${TOKENIZED_ROOT}/language_map_200langs_identity.json}"

MAX_LENGTH="${MAX_LENGTH:-1024}"
EVAL_FRACTION="${EVAL_FRACTION:-0.02}"
EVAL_SEED="${EVAL_SEED:-42}"
NUM_PROC="${NUM_PROC:-8}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-1.0}"
SAMPLE_SEED="${SAMPLE_SEED:-123}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
WANDB_PROJECT="${WANDB_PROJECT:-moe-study-cola-padmask-ablation-200langs}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-cola-padmask-200langs-${TIMESTAMP}}"
WANDB_MODE="${WANDB_MODE:-online}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/cola_padmask_200langs_${TIMESTAMP}}"
MASKED_OUTPUT_DIR="${MASKED_OUTPUT_DIR:-${OUTPUT_ROOT}/cola_masked}"
UNMASKED_OUTPUT_DIR="${UNMASKED_OUTPUT_DIR:-${OUTPUT_ROOT}/cola_unmasked}"

DATASET_NAME="${DATASET_NAME:-ablation200}"
USE_ACCELERATE="${USE_ACCELERATE:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-12}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-12}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MASKED_PER_DEVICE_TRAIN_BATCH_SIZE="${MASKED_PER_DEVICE_TRAIN_BATCH_SIZE:-${PER_DEVICE_TRAIN_BATCH_SIZE}}"
MASKED_PER_DEVICE_EVAL_BATCH_SIZE="${MASKED_PER_DEVICE_EVAL_BATCH_SIZE:-${PER_DEVICE_EVAL_BATCH_SIZE}}"
MASKED_GRADIENT_ACCUMULATION_STEPS="${MASKED_GRADIENT_ACCUMULATION_STEPS:-${GRADIENT_ACCUMULATION_STEPS}}"
UNMASKED_PER_DEVICE_TRAIN_BATCH_SIZE="${UNMASKED_PER_DEVICE_TRAIN_BATCH_SIZE:-${PER_DEVICE_TRAIN_BATCH_SIZE}}"
UNMASKED_PER_DEVICE_EVAL_BATCH_SIZE="${UNMASKED_PER_DEVICE_EVAL_BATCH_SIZE:-${PER_DEVICE_EVAL_BATCH_SIZE}}"
UNMASKED_GRADIENT_ACCUMULATION_STEPS="${UNMASKED_GRADIENT_ACCUMULATION_STEPS:-${GRADIENT_ACCUMULATION_STEPS}}"
AUTO_FIND_BATCH_SIZE="${AUTO_FIND_BATCH_SIZE:-true}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-400}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-False}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-8}"
PREPROCESSING_BATCH_SIZE="${PREPROCESSING_BATCH_SIZE:-100000}"
DISABLE_GRADIENT_CHECKPOINTING="${DISABLE_GRADIENT_CHECKPOINTING:-True}"
BF16="${BF16:-True}"
FP16="${FP16:-False}"
FLASH_ATTN="${FLASH_ATTN:-sdpa}"
MASKED_GPU_ID="${MASKED_GPU_ID:-0}"
UNMASKED_GPU_ID="${UNMASKED_GPU_ID:-1}"
MASKED_MASTER_PORT="${MASKED_MASTER_PORT:-29601}"
UNMASKED_MASTER_PORT="${UNMASKED_MASTER_PORT:-29602}"

mkdir -p "${TOKENIZED_ROOT}" "${OUTPUT_ROOT}"

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN] %q ' "$@"
    echo
  else
    "$@"
  fi
}

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
  if ! [[ "${MASKED_GPU_ID}" =~ ^[0-9]+$ && "${UNMASKED_GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] MASKED_GPU_ID and UNMASKED_GPU_ID must be integer GPU indices." >&2
    exit 1
  fi
  if [[ "${TRAIN_MODE}" == "parallel" && "${MASKED_GPU_ID}" == "${UNMASKED_GPU_ID}" ]]; then
    echo "[ERROR] Parallel mode requires distinct GPUs. Got MASKED_GPU_ID=${MASKED_GPU_ID}, UNMASKED_GPU_ID=${UNMASKED_GPU_ID}." >&2
    exit 1
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_count
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
    if (( MASKED_GPU_ID >= gpu_count || UNMASKED_GPU_ID >= gpu_count )); then
      echo "[ERROR] GPU index out of range. gpu_count=${gpu_count}, MASKED_GPU_ID=${MASKED_GPU_ID}, UNMASKED_GPU_ID=${UNMASKED_GPU_ID}" >&2
      exit 1
    fi
  fi
}

prepare_language_map() {
  require_dir "${SHARD_DIR}"
  run_cmd "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

shard_dir = Path(${SHARD_DIR@Q})
map_path = Path(${LANGUAGE_MAP_PATH@Q})
langs = sorted([p.name for p in shard_dir.iterdir() if p.is_dir()])
if not langs:
    raise SystemExit(f"No language directories found in {shard_dir}")
mapping = {lang: lang for lang in langs}
map_path.parent.mkdir(parents=True, exist_ok=True)
map_path.write_text(json.dumps(mapping, ensure_ascii=True, indent=2), encoding="utf-8")
print(f"[INFO] Wrote language map to {map_path} with {len(mapping)} entries")
PY
}

tokenize_variant() {
  local variant="$1"         # masked | unmasked
  local mask_flag="$2"       # true | false
  local out_dir="$3"

  if [[ -f "${out_dir}/dataset_info.json" && "${FORCE_RETOKENIZE}" != "1" ]]; then
    echo "[INFO] Reusing existing tokenized dataset (${variant}): ${out_dir}"
    return
  fi

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  local cmd=(
    "${PYTHON_BIN}" "${TOKENIZER_PY}"
    --shard_dir "${SHARD_DIR}"
    --save_tokenized_data_dir "${out_dir}"
    --model_name "${MODEL_NAME_OR_PATH}"
    --max_length "${MAX_LENGTH}"
    --num_proc "${NUM_PROC}"
    --eval_fraction "${EVAL_FRACTION}"
    --eval_seed "${EVAL_SEED}"
    --sample_fraction "${SAMPLE_FRACTION}"
    --sample_seed "${SAMPLE_SEED}"
    --language-map "${LANGUAGE_MAP_PATH}"
    --mask_pad_labels "${mask_flag}"
  )

  if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max_samples "${MAX_SAMPLES}")
  fi

  echo "[INFO] Tokenizing ${variant} dataset..."
  run_cmd "${cmd[@]}"
}

tokenize_all() {
  require_file "${TOKENIZER_PY}"
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  fi

  prepare_language_map
  tokenize_variant "masked" "true" "${TOKENIZED_MASKED_DIR}"
  tokenize_variant "unmasked" "false" "${TOKENIZED_UNMASKED_DIR}"
}

run_cola_variant() {
  local label="$1"         # masked | unmasked
  local gpu_id="$2"        # 0 | 1
  local tokenized_path="$3"
  local output_dir="$4"
  local master_port="$5"
  local run_async="$6"     # 0 | 1
  local run_name="cola-padmask-${label}-200langs-${TIMESTAMP}"
  local log_path="${output_dir}.log"
  local train_bs
  local eval_bs
  local grad_acc

  if [[ "${label}" == "masked" ]]; then
    train_bs="${MASKED_PER_DEVICE_TRAIN_BATCH_SIZE}"
    eval_bs="${MASKED_PER_DEVICE_EVAL_BATCH_SIZE}"
    grad_acc="${MASKED_GRADIENT_ACCUMULATION_STEPS}"
  else
    train_bs="${UNMASKED_PER_DEVICE_TRAIN_BATCH_SIZE}"
    eval_bs="${UNMASKED_PER_DEVICE_EVAL_BATCH_SIZE}"
    grad_acc="${UNMASKED_GRADIENT_ACCUMULATION_STEPS}"
  fi

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
    export LEARNING_RATE
    export PER_DEVICE_TRAIN_BATCH_SIZE="${train_bs}"
    export PER_DEVICE_EVAL_BATCH_SIZE="${eval_bs}"
    export GRADIENT_ACCUMULATION_STEPS="${grad_acc}"
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

    echo "[INFO] [${label}] GPU=${gpu_id} tokenized_path=${tokenized_path}"
    echo "[INFO] [${label}] train_bs=${train_bs} eval_bs=${eval_bs} grad_acc=${grad_acc} auto_find_batch_size=${AUTO_FIND_BATCH_SIZE} master_port=${master_port}"
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
  if [[ "${label}" == "masked" ]]; then
    require_dir "${TOKENIZED_MASKED_DIR}"
    run_cola_variant "masked" "${MASKED_GPU_ID}" "${TOKENIZED_MASKED_DIR}" "${MASKED_OUTPUT_DIR}" "${MASKED_MASTER_PORT}" "0"
  elif [[ "${label}" == "unmasked" ]]; then
    require_dir "${TOKENIZED_UNMASKED_DIR}"
    run_cola_variant "unmasked" "${UNMASKED_GPU_ID}" "${TOKENIZED_UNMASKED_DIR}" "${UNMASKED_OUTPUT_DIR}" "${UNMASKED_MASTER_PORT}" "0"
  else
    echo "[ERROR] Invalid label for train_single: ${label}" >&2
    exit 1
  fi
}

train_ab() {
  require_file "${REPO_ROOT}/scripts/slurm/comparison/cola_job.sh"
  require_dir "${TOKENIZED_MASKED_DIR}"
  require_dir "${TOKENIZED_UNMASKED_DIR}"
  validate_train_config

  if [[ "${TRAIN_MODE}" == "parallel" ]]; then
    echo "[INFO] Launching CoLA A/B runs in parallel (masked on GPU${MASKED_GPU_ID}, unmasked on GPU${UNMASKED_GPU_ID})"
    run_cola_variant "masked" "${MASKED_GPU_ID}" "${TOKENIZED_MASKED_DIR}" "${MASKED_OUTPUT_DIR}" "${MASKED_MASTER_PORT}" "1"
    local pid_masked="${LAST_LAUNCHED_PID}"
    if [[ "${PARALLEL_LAUNCH_DELAY_SEC}" -gt 0 ]]; then
      echo "[INFO] Waiting ${PARALLEL_LAUNCH_DELAY_SEC}s before launching unmasked run to reduce startup collisions."
      sleep "${PARALLEL_LAUNCH_DELAY_SEC}"
    fi
    run_cola_variant "unmasked" "${UNMASKED_GPU_ID}" "${TOKENIZED_UNMASKED_DIR}" "${UNMASKED_OUTPUT_DIR}" "${UNMASKED_MASTER_PORT}" "1"
    local pid_unmasked="${LAST_LAUNCHED_PID}"

    local rc=0
    wait "${pid_masked}" || rc=$?
    wait "${pid_unmasked}" || rc=$?
    if [[ "${rc}" -ne 0 ]]; then
      echo "[ERROR] At least one training run failed. Check logs:" >&2
      echo "  - ${MASKED_OUTPUT_DIR}.log" >&2
      echo "  - ${UNMASKED_OUTPUT_DIR}.log" >&2
      exit "${rc}"
    fi
    return
  fi

  if [[ "${TRAIN_MODE}" == "sequential" ]]; then
    echo "[INFO] Launching CoLA runs sequentially (masked, then unmasked)."
    train_single "masked"
    train_single "unmasked"
    return
  fi

  echo "[ERROR] Invalid TRAIN_MODE=${TRAIN_MODE}. Use sequential|parallel." >&2
  exit 1
}

print_summary() {
  cat <<EOF
[INFO] A/B setup prepared
  sampled shard dir:    ${SHARD_DIR}
  masked tokenized dir: ${TOKENIZED_MASKED_DIR}
  unmasked tokenized:   ${TOKENIZED_UNMASKED_DIR}
  output root:          ${OUTPUT_ROOT}
  W&B project:          ${WANDB_PROJECT}
  W&B group:            ${WANDB_RUN_GROUP}
  mode:                 ${MODE}
  train mode:           ${TRAIN_MODE}
  flash attn backend:   ${FLASH_ATTN}
  masked gpu:           ${MASKED_GPU_ID}
  unmasked gpu:         ${UNMASKED_GPU_ID}
  auto find batch:      ${AUTO_FIND_BATCH_SIZE}
EOF
}

main() {
  require_dir "${SHARD_DIR}"
  print_summary

  case "${MODE}" in
    tokenize)
      tokenize_all
      ;;
    train)
      train_ab
      ;;
    train_masked)
      train_single "masked"
      ;;
    train_unmasked)
      train_single "unmasked"
      ;;
    all)
      tokenize_all
      train_ab
      ;;
    *)
      echo "[ERROR] Invalid MODE=${MODE}. Use tokenize|train|train_masked|train_unmasked|all." >&2
      exit 1
      ;;
  esac

  echo "[INFO] Done."
}

main "$@"
