#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
COMPARISON_DIR="${SCRIPT_DIR}"

# Execution mode:
# - slurm (default): submit via sbatch
# - local: run job scripts directly on this machine
RUN_MODE="${RUN_MODE:-slurm}"
DRY_RUN="${DRY_RUN:-0}"
LOCAL_PARALLEL="${LOCAL_PARALLEL:-0}"

CONDA_BASE="${CONDA_BASE:-}"
CONDA_ENV="${CONDA_ENV:-}"
MODULE_INIT="${MODULE_INIT:-}"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/LLaMA-Factory/data}"
TOKENIZED_BASE_DIR="${TOKENIZED_BASE_DIR:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/fw_l2v_samples_moe_study_comparison_labeled}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/scripts/slurm/comparison/logs/matrix_1b_10langs}"

WANDB_PROJECT="${WANDB_PROJECT:-moe-study-comparison}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-llama3.2-1B-10langs-$(date +%Y%m%d_%H%M%S)}"

FLASH_ATTN="${FLASH_ATTN:-fa2}"
AUTO_FIND_BATCH_SIZE="${AUTO_FIND_BATCH_SIZE:-false}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
EVAL_STEPS="${EVAL_STEPS:-3000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-32}"
PREPROCESSING_BATCH_SIZE="${PREPROCESSING_BATCH_SIZE:-100000}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
USE_ACCELERATE="${USE_ACCELERATE:-1}"

DEFAULT_GPU_TYPE="${DEFAULT_GPU_TYPE:-h100}"
DEFAULT_GPU_COUNT="${DEFAULT_GPU_COUNT:-1}"
DEFAULT_PARTITION="${DEFAULT_PARTITION:-gpu}"
DEFAULT_WALLTIME="${DEFAULT_WALLTIME:-24:00:00}"
DEFAULT_CPUS_PER_TASK="${DEFAULT_CPUS_PER_TASK:-32}"
DEFAULT_MEM="${DEFAULT_MEM:-128G}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"

# -----------------------------------------------------------------------------
# Edit these arrays for quick experiments
# -----------------------------------------------------------------------------

# <tier_id>|<dataset_name>|<tokenizer_name>|<model_name_or_path>
LANGUAGE_TIERS=(
  "tier10|10_langs|llama-3.2-1B_tokenizer|meta-llama/Llama-3.2-1B"
  # "tier10_200langs_sample|200_langs|llama-3.2-1B_tokenizer|meta-llama/Llama-3.2-1B"
)

# <variant_label>|<train_bs>|<extra_env_pairs>
# NOTE: extra env pairs are space-separated KEY=VALUE assignments.
COLA_VARIANTS=(
  "paper-best|21|COLA_NUM_A=1 COLA_NUM_B=3 COLA_STRATEGY=fully LORA_R=8 LORA_ALPHA=16"
)

HYDRALORA_VARIANTS=(
  "paper-best|17|USE_HYDRALORA_EXPERTS=True HYDRALORA_NUM_EXPERTS=3 HYDRALORA_TOP_K=1 HYDRALORA_NUM_B=3 LORA_R=8 LORA_ALPHA=16"
)

ADAMOLE_VARIANTS=(
  "paper-best|15|ADAMOLE_NUM_EXPERTS=4 ADAMOLE_AUX_LOSS_COEF=0.001 ADAMOLE_MAX_THRESHOLD=0.25 LORA_R=8 LORA_ALPHA=16"
)

LORA_VARIANTS=(
  "paper-best|21|LORA_R=8 LORA_ALPHA=16 LORA_DROPOUT=0.05"
)

# Resource overrides per tier id.
declare -A TIER_WALLTIME_MAP=(
  ["tier10"]="${DEFAULT_WALLTIME}"
)
declare -A TIER_GPU_COUNT_MAP=(
  ["tier10"]="${DEFAULT_GPU_COUNT}"
)
declare -A TIER_GPU_TYPE_MAP=(
  ["tier10"]="${DEFAULT_GPU_TYPE}"
)
declare -A TIER_PARTITION_MAP=(
  ["tier10"]="${DEFAULT_PARTITION}"
)
declare -A TIER_CPUS_PER_TASK_MAP=(
  ["tier10"]="${DEFAULT_CPUS_PER_TASK}"
)
declare -A TIER_MEM_MAP=(
  ["tier10"]="${DEFAULT_MEM}"
)

declare -A APPROACH_SCRIPT=(
  ["cola"]="cola_job.sh"
  ["hydralora"]="hydralora_job.sh"
  ["adamole"]="adamole_job.sh"
  ["lora"]="lora_job.sh"
)

declare -A APPROACH_VARIANTS_REF=(
  ["cola"]="COLA_VARIANTS"
  ["hydralora"]="HYDRALORA_VARIANTS"
  ["adamole"]="ADAMOLE_VARIANTS"
  ["lora"]="LORA_VARIANTS"
)

sanitize() {
  echo "$1" | tr '/ :|' '_'
}

print_cmd() {
  printf '[CMD] '
  printf '%q ' "$@"
  echo
}

require_path() {
  local p="$1"
  local label="$2"
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] ${label} not found: ${p}" >&2
    exit 1
  fi
}

ensure_tools() {
  if [[ "${RUN_MODE}" == "slurm" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[WARN] sbatch not found; continuing because DRY_RUN=1."
      else
        echo "[ERROR] RUN_MODE=slurm but sbatch is not available in PATH." >&2
        exit 1
      fi
    fi
  fi
}

submit_or_run_job() {
  local tier_id="$1"
  local dataset_name="$2"
  local tokenizer_name="$3"
  local model_name="$4"
  local approach="$5"
  local variant_label="$6"
  local train_bs="$7"
  local extra_env_pairs="$8"
  local timestamp="$9"

  local script_name="${APPROACH_SCRIPT[${approach}]}"
  local script_path="${COMPARISON_DIR}/${script_name}"
  require_path "${script_path}" "job script"

  local tokenized_path="${TOKENIZED_BASE_DIR}/${tokenizer_name}/${dataset_name}"
  require_path "${tokenized_path}" "tokenized dataset"

  local tier_slug
  local variant_slug
  tier_slug="$(sanitize "${tier_id}")"
  variant_slug="$(sanitize "${variant_label}")"
  local run_name="l3-1b-${dataset_name}-${approach}-${variant_slug}-${timestamp}"
  local output_dir="${OUTPUT_ROOT}/${tokenizer_name}-${dataset_name}/${approach}/${variant_slug}_${timestamp}"
  local log_path="${LOG_ROOT}/${tier_slug}_${approach}_${variant_slug}_${timestamp}.log"
  mkdir -p "${output_dir}" "$(dirname "${log_path}")"

  local -a env_assignments=(
    "REPO_ROOT=${REPO_ROOT}"
    "DATASET_DIR=${DATASET_DIR}"
    "MODEL_NAME_OR_PATH=${model_name}"
    "GENERIC_TOKENIZED_PATH=${tokenized_path}"
    "GENERIC_DATASET_NAME=${dataset_name}"
    "GENERIC_EVAL_DATASET_NAME=${dataset_name}"
    "OUTPUT_DIR=${output_dir}"
    "WANDB_PROJECT=${WANDB_PROJECT}"
    "WANDB_ENTITY=${WANDB_ENTITY}"
    "WANDB_RUN_GROUP=${WANDB_RUN_GROUP}"
    "WANDB_NAME=${run_name}"
    "FLASH_ATTN=${FLASH_ATTN}"
    "AUTO_FIND_BATCH_SIZE=${AUTO_FIND_BATCH_SIZE}"
    "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
    "LEARNING_RATE=${LEARNING_RATE}"
    "PER_DEVICE_TRAIN_BATCH_SIZE=${train_bs}"
    "PER_DEVICE_EVAL_BATCH_SIZE=${train_bs}"
    "EVAL_STEPS=${EVAL_STEPS}"
    "SAVE_STEPS=${SAVE_STEPS}"
    "SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
    "PREPROCESSING_NUM_WORKERS=${PREPROCESSING_NUM_WORKERS}"
    "PREPROCESSING_BATCH_SIZE=${PREPROCESSING_BATCH_SIZE}"
    "GPUS_PER_NODE=${GPUS_PER_NODE}"
    "USE_ACCELERATE=${USE_ACCELERATE}"
  )

  if [[ -n "${CONDA_BASE}" ]]; then
    env_assignments+=("CONDA_BASE=${CONDA_BASE}")
  fi
  if [[ -n "${CONDA_ENV}" ]]; then
    env_assignments+=("CONDA_ENV=${CONDA_ENV}")
  fi
  if [[ -n "${MODULE_INIT}" ]]; then
    env_assignments+=("MODULE_INIT=${MODULE_INIT}")
  fi

  if [[ -n "${extra_env_pairs}" ]]; then
    # shellcheck disable=SC2206
    local extra_arr=( ${extra_env_pairs} )
    env_assignments+=("${extra_arr[@]}")
  fi

  if [[ "${RUN_MODE}" == "slurm" ]]; then
    local gpu_type="${TIER_GPU_TYPE_MAP[${tier_id}]:-${DEFAULT_GPU_TYPE}}"
    local gpu_count="${TIER_GPU_COUNT_MAP[${tier_id}]:-${DEFAULT_GPU_COUNT}}"
    local partition="${TIER_PARTITION_MAP[${tier_id}]:-${DEFAULT_PARTITION}}"
    local walltime="${TIER_WALLTIME_MAP[${tier_id}]:-${DEFAULT_WALLTIME}}"
    local cpus="${TIER_CPUS_PER_TASK_MAP[${tier_id}]:-${DEFAULT_CPUS_PER_TASK}}"
    local mem="${TIER_MEM_MAP[${tier_id}]:-${DEFAULT_MEM}}"

    local -a cmd=(env)
    cmd+=("${env_assignments[@]}")
    cmd+=(
      sbatch
      --output "${log_path}"
      --partition "${partition}"
      --gres "gpu:${gpu_type}:${gpu_count}"
      --cpus-per-task "${cpus}"
      --mem "${mem}"
      --time "${walltime}"
    )
    if [[ -n "${SBATCH_EXTRA_ARGS}" ]]; then
      # shellcheck disable=SC2206
      local sbatch_extra_arr=( ${SBATCH_EXTRA_ARGS} )
      cmd+=("${sbatch_extra_arr[@]}")
    fi
    cmd+=("${script_path}")

    if [[ "${DRY_RUN}" == "1" ]]; then
      print_cmd "${cmd[@]}"
    else
      local output
      output="$("${cmd[@]}")"
      echo "[INFO] ${approach}/${variant_label}/${tier_id}: ${output}"
    fi
    return
  fi

  local -a local_cmd=(env)
  local_cmd+=("${env_assignments[@]}")
  local_cmd+=(bash "${script_path}")

  if [[ "${DRY_RUN}" == "1" ]]; then
    print_cmd "${local_cmd[@]}"
    return
  fi

  echo "[INFO] Running local ${approach}/${variant_label}/${tier_id} -> ${log_path}"
  if [[ "${LOCAL_PARALLEL}" == "1" ]]; then
    "${local_cmd[@]}" >"${log_path}" 2>&1 &
    LOCAL_JOB_PIDS+=("$!")
  else
    "${local_cmd[@]}" >"${log_path}" 2>&1
  fi
}

main() {
  ensure_tools
  require_path "${DATASET_DIR}" "DATASET_DIR"
  require_path "${TOKENIZED_BASE_DIR}" "TOKENIZED_BASE_DIR"
  mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"

  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"

  echo "[INFO] Matrix launcher"
  echo "  mode:           ${RUN_MODE}"
  echo "  dry run:        ${DRY_RUN}"
  echo "  tokenized base: ${TOKENIZED_BASE_DIR}"
  echo "  output root:    ${OUTPUT_ROOT}"
  echo "  wandb project:  ${WANDB_PROJECT}"
  echo "  wandb group:    ${WANDB_RUN_GROUP}"

  for tier_spec in "${LANGUAGE_TIERS[@]}"; do
    IFS='|' read -r tier_id dataset_name tokenizer_name model_name <<<"${tier_spec}"
    echo "[INFO] Tier ${tier_id}: dataset=${dataset_name} tokenizer=${tokenizer_name} model=${model_name}"

    for approach in cola hydralora adamole lora; do
      local variants_ref_name="${APPROACH_VARIANTS_REF[${approach}]}"
      declare -n variants_ref="${variants_ref_name}"
      for variant_spec in "${variants_ref[@]}"; do
        IFS='|' read -r variant_label train_bs extra_env_pairs <<<"${variant_spec}"
        submit_or_run_job \
          "${tier_id}" "${dataset_name}" "${tokenizer_name}" "${model_name}" \
          "${approach}" "${variant_label}" "${train_bs}" "${extra_env_pairs}" "${timestamp}"
      done
    done
  done

  if [[ "${RUN_MODE}" == "local" && "${LOCAL_PARALLEL}" == "1" && "${DRY_RUN}" != "1" ]]; then
    local rc=0
    for pid in "${LOCAL_JOB_PIDS[@]:-}"; do
      wait "${pid}" || rc=$?
    done
    exit "${rc}"
  fi
}

main "$@"
