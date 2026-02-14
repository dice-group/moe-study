#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODE="${MODE:-all}" # all | prepare | submit
DRY_RUN="${DRY_RUN:-0}"
FORCE_REBUILD_MASKED="${FORCE_REBUILD_MASKED:-0}"
DEPEND_ON_PREP_FOR_BOTH="${DEPEND_ON_PREP_FOR_BOTH:-1}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-padmask_ablation_${TIMESTAMP}}"

TOKENIZED_BASE="${TOKENIZED_BASE:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/fw_l2v_samples_moe_study_comparison}"
TOKENIZER_NAME="${TOKENIZER_NAME:-llama-3.2-1B_tokenizer}"
DATASET_NAME="${DATASET_NAME:-10_langs}"
SOURCE_TOKENIZED_PATH="${SOURCE_TOKENIZED_PATH:-${TOKENIZED_BASE}/${TOKENIZER_NAME}/${DATASET_NAME}}"
MASKED_TOKENIZED_PATH="${MASKED_TOKENIZED_PATH:-${TOKENIZED_BASE}/${TOKENIZER_NAME}/${DATASET_NAME}_padmasked_labels}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.2-1B}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/LLaMA-Factory/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

WANDB_PROJECT="${WANDB_PROJECT:-moe-study-cola-padmask-ablation-10langs}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-cola-padmask-ablation-${DATASET_NAME}-${TIMESTAMP}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-l3-1b-base-10langs-cola-padmask-ab}"

OUTPUT_BASE="${OUTPUT_BASE:-/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/${TOKENIZER_NAME}-${DATASET_NAME}/cola}"
OUTPUT_DIR_UNMASKED="${OUTPUT_DIR_UNMASKED:-${OUTPUT_BASE}/${RUN_TAG}/unmasked}"
OUTPUT_DIR_MASKED="${OUTPUT_DIR_MASKED:-${OUTPUT_BASE}/${RUN_TAG}/masked}"

LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/scripts/slurm/comparison/logs/padmask_ablation}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"

# Match paper-best CoLA settings used in comparison_config_llama3_2_1b.json
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-21}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-21}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
EVAL_STEPS="${EVAL_STEPS:-3000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
AUTO_FIND_BATCH_SIZE="${AUTO_FIND_BATCH_SIZE:-false}"
FLASH_ATTN="${FLASH_ATTN:-fa2}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
COLA_NUM_A="${COLA_NUM_A:-1}"
COLA_NUM_B="${COLA_NUM_B:-3}"
COLA_STRATEGY="${COLA_STRATEGY:-fully}"
LLAMAFACTORY_ADAPTER_MILESTONE_STEPS="${LLAMAFACTORY_ADAPTER_MILESTONE_STEPS:-10000}"

# H100 request for training (and prep by default to keep scheduling simple).
TRAIN_PARTITION="${TRAIN_PARTITION:-gpu}"
TRAIN_GPU_TYPE="${TRAIN_GPU_TYPE:-h100}"
TRAIN_GPUS="${TRAIN_GPUS:-1}"
TRAIN_CPUS_PER_TASK="${TRAIN_CPUS_PER_TASK:-32}"
TRAIN_MEM="${TRAIN_MEM:-128G}"
TRAIN_TIME="${TRAIN_TIME:-24:00:00}"

PREP_PARTITION="${PREP_PARTITION:-gpu}"
PREP_GPU_TYPE="${PREP_GPU_TYPE:-h100}"
PREP_GPUS="${PREP_GPUS:-1}"
PREP_CPUS_PER_TASK="${PREP_CPUS_PER_TASK:-32}"
PREP_MEM="${PREP_MEM:-128G}"
PREP_TIME="${PREP_TIME:-04:00:00}"
PREP_NUM_PROC="${PREP_NUM_PROC:-32}"

MASK_SCRIPT="${REPO_ROOT}/scripts/slurm/comparison/mask_pad_labels_dataset.py"
TRAIN_SCRIPT="${REPO_ROOT}/scripts/slurm/comparison/cola_job.sh"

mkdir -p "${LOG_ROOT}" "${OUTPUT_BASE}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Missing directory: ${path}" >&2
    exit 1
  fi
}

run_or_echo() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN] %q ' "$@" >&2
    echo >&2
    return 0
  fi
  "$@"
}

append_extra_sbatch_args() {
  local -n _arr_ref="$1"
  if [[ -z "${SBATCH_EXTRA_ARGS}" ]]; then
    return
  fi
  local -a extra=()
  # shellcheck disable=SC2206
  extra=(${SBATCH_EXTRA_ARGS})
  _arr_ref+=("${extra[@]}")
}

submit_prepare_job() {
  local prep_log="${LOG_ROOT}/prepare_masked_${RUN_TAG}.log"
  local wrap_cmd
  wrap_cmd="cd ${REPO_ROOT} && ${PYTHON_BIN} ${MASK_SCRIPT} --input-path ${SOURCE_TOKENIZED_PATH} --output-path ${MASKED_TOKENIZED_PATH} --num-proc ${PREP_NUM_PROC}"
  if [[ "${FORCE_REBUILD_MASKED}" == "1" ]]; then
    wrap_cmd="${wrap_cmd} --overwrite"
  fi

  local -a sbatch_cmd=(
    sbatch
    --parsable
    --job-name "cola-padmask-prep"
    --output "${prep_log}"
    --partition "${PREP_PARTITION}"
    --gres "gpu:${PREP_GPU_TYPE}:${PREP_GPUS}"
    --cpus-per-task "${PREP_CPUS_PER_TASK}"
    --mem "${PREP_MEM}"
    --time "${PREP_TIME}"
  )
  append_extra_sbatch_args sbatch_cmd
  sbatch_cmd+=(--wrap "${wrap_cmd}")

  local prep_id
  if [[ "${DRY_RUN}" == "1" ]]; then
    run_or_echo "${sbatch_cmd[@]}"
    prep_id="DRYRUN_PREP"
  else
    prep_id="$("${sbatch_cmd[@]}")"
  fi
  echo "${prep_id}"
}

submit_train_job() {
  local label="$1"           # masked | unmasked
  local tokenized_path="$2"
  local output_dir="$3"
  local dependency_id="${4:-}"

  local run_name="${WANDB_NAME_PREFIX}-${label}-${TIMESTAMP}"
  local log_path="${LOG_ROOT}/train_${label}_${RUN_TAG}.log"
  mkdir -p "${output_dir}"

  local -a sbatch_cmd=(
    sbatch
    --parsable
    --job-name "cola-padmask-${label}"
    --output "${log_path}"
    --partition "${TRAIN_PARTITION}"
    --gres "gpu:${TRAIN_GPU_TYPE}:${TRAIN_GPUS}"
    --cpus-per-task "${TRAIN_CPUS_PER_TASK}"
    --mem "${TRAIN_MEM}"
    --time "${TRAIN_TIME}"
  )
  if [[ -n "${dependency_id}" ]]; then
    sbatch_cmd+=(--dependency "afterok:${dependency_id}")
  fi
  append_extra_sbatch_args sbatch_cmd
  sbatch_cmd+=("${TRAIN_SCRIPT}")

  local -a env_cmd=(
    env
    "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
    "GENERIC_TOKENIZED_PATH=${tokenized_path}"
    "GENERIC_DATASET_NAME=${DATASET_NAME}"
    "GENERIC_EVAL_DATASET_NAME=${DATASET_NAME}"
    "OUTPUT_DIR=${output_dir}"
    "DATASET_DIR=${DATASET_DIR}"
    "WANDB_PROJECT=${WANDB_PROJECT}"
    "WANDB_RUN_GROUP=${WANDB_RUN_GROUP}"
    "WANDB_MODE=${WANDB_MODE}"
    "WANDB_NAME=${run_name}"
    "GPUS_PER_NODE=1"
    "USE_ACCELERATE=1"
    "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
    "LEARNING_RATE=${LEARNING_RATE}"
    "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
    "PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}"
    "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
    "EVAL_STEPS=${EVAL_STEPS}"
    "SAVE_STEPS=${SAVE_STEPS}"
    "SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
    "AUTO_FIND_BATCH_SIZE=${AUTO_FIND_BATCH_SIZE}"
    "FLASH_ATTN=${FLASH_ATTN}"
    "LORA_R=${LORA_R}"
    "LORA_ALPHA=${LORA_ALPHA}"
    "COLA_NUM_A=${COLA_NUM_A}"
    "COLA_NUM_B=${COLA_NUM_B}"
    "COLA_STRATEGY=${COLA_STRATEGY}"
    "LLAMAFACTORY_ADAPTER_MILESTONE_STEPS=${LLAMAFACTORY_ADAPTER_MILESTONE_STEPS}"
  )
  if [[ -n "${MAX_STEPS}" ]]; then
    env_cmd+=("MAX_STEPS=${MAX_STEPS}")
  fi

  local job_id
  if [[ "${DRY_RUN}" == "1" ]]; then
    run_or_echo "${env_cmd[@]}" "${sbatch_cmd[@]}"
    job_id="DRYRUN_${label^^}"
  else
    job_id="$("${env_cmd[@]}" "${sbatch_cmd[@]}")"
  fi
  echo "${job_id}"
}

print_summary() {
  cat <<EOF
[INFO] Cluster CoLA pad-mask A/B launcher
  mode:                     ${MODE}
  source tokenized path:    ${SOURCE_TOKENIZED_PATH}
  masked tokenized path:    ${MASKED_TOKENIZED_PATH}
  output unmasked:          ${OUTPUT_DIR_UNMASKED}
  output masked:            ${OUTPUT_DIR_MASKED}
  W&B project:              ${WANDB_PROJECT}
  W&B group:                ${WANDB_RUN_GROUP}
  train resources:          partition=${TRAIN_PARTITION}, gpu=${TRAIN_GPU_TYPE}:${TRAIN_GPUS}, cpu=${TRAIN_CPUS_PER_TASK}, mem=${TRAIN_MEM}, time=${TRAIN_TIME}
  prep resources:           partition=${PREP_PARTITION}, gpu=${PREP_GPU_TYPE}:${PREP_GPUS}, cpu=${PREP_CPUS_PER_TASK}, mem=${PREP_MEM}, time=${PREP_TIME}
EOF
}

main() {
  require_file "${MASK_SCRIPT}"
  require_file "${TRAIN_SCRIPT}"
  require_dir "${SOURCE_TOKENIZED_PATH}"
  require_dir "${DATASET_DIR}"

  print_summary

  local prep_job_id=""
  local masked_ready=0
  if [[ -d "${MASKED_TOKENIZED_PATH}" && -f "${MASKED_TOKENIZED_PATH}/dataset_info.json" && "${FORCE_REBUILD_MASKED}" != "1" ]]; then
    masked_ready=1
    echo "[INFO] Reusing existing masked dataset: ${MASKED_TOKENIZED_PATH}"
  fi

  if [[ "${MODE}" == "prepare" || "${MODE}" == "all" ]]; then
    if [[ "${masked_ready}" == "1" && "${MODE}" == "prepare" ]]; then
      echo "[INFO] Nothing to do (masked dataset already exists)."
      return 0
    fi
    if [[ "${masked_ready}" == "0" || "${FORCE_REBUILD_MASKED}" == "1" ]]; then
      prep_job_id="$(submit_prepare_job)"
      echo "[INFO] Submitted prepare job: ${prep_job_id}"
    fi
  fi

  if [[ "${MODE}" == "prepare" ]]; then
    return 0
  fi

  if [[ "${MODE}" != "submit" && "${MODE}" != "all" ]]; then
    echo "[ERROR] Invalid MODE=${MODE}. Use all|prepare|submit." >&2
    exit 1
  fi

  local dep_masked=""
  local dep_unmasked=""
  if [[ -n "${prep_job_id}" && "${prep_job_id}" != DRYRUN_PREP ]]; then
    dep_masked="${prep_job_id}"
    if [[ "${DEPEND_ON_PREP_FOR_BOTH}" == "1" ]]; then
      dep_unmasked="${prep_job_id}"
    fi
  fi

  local unmasked_job_id
  local masked_job_id
  unmasked_job_id="$(submit_train_job "unmasked" "${SOURCE_TOKENIZED_PATH}" "${OUTPUT_DIR_UNMASKED}" "${dep_unmasked}")"
  masked_job_id="$(submit_train_job "masked" "${MASKED_TOKENIZED_PATH}" "${OUTPUT_DIR_MASKED}" "${dep_masked}")"

  echo "[INFO] Submitted unmasked job: ${unmasked_job_id}"
  echo "[INFO] Submitted masked job:   ${masked_job_id}"
  if [[ -n "${prep_job_id}" ]]; then
    echo "[INFO] Prepare job:           ${prep_job_id}"
  fi
}

main "$@"
