#!/usr/bin/env bash

# Logic module for matrix launchers. Requires config arrays to be sourced first.

if [[ -z "${COMPARISON_DIR:-}" ]]; then
  COMPARISON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "${COMPARISON_DIR}/../../.." && pwd)"
fi

MATRIX_RUN_MODE="${RUN_MODE:-slurm}"
MATRIX_DRY_RUN="${DRY_RUN:-0}"
MATRIX_LOCAL_PARALLEL="${LOCAL_PARALLEL:-0}"

MATRIX_CONDA_BASE="${CONDA_BASE:-}"
MATRIX_CONDA_ENV="${CONDA_ENV:-}"
MATRIX_MODULE_INIT="${MODULE_INIT:-}"

MATRIX_DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/LLaMA-Factory/data}"
MATRIX_TOKENIZED_BASE_DIR="${TOKENIZED_BASE_DIR:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/fw_l2v_samples_moe_study_comparison_labeled}"
MATRIX_OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs}"
MATRIX_LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/scripts/slurm/comparison/logs/matrix}"

MATRIX_WANDB_PROJECT="${WANDB_PROJECT:-moe-study-comparison}"
MATRIX_WANDB_ENTITY="${WANDB_ENTITY:-}"
MATRIX_WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-matrix-$(date +%Y%m%d_%H%M%S)}"

MATRIX_FLASH_ATTN="${FLASH_ATTN:-fa2}"
MATRIX_AUTO_FIND_BATCH_SIZE="${AUTO_FIND_BATCH_SIZE:-false}"
MATRIX_NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MATRIX_LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MATRIX_SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
MATRIX_PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-32}"
MATRIX_PREPROCESSING_BATCH_SIZE="${PREPROCESSING_BATCH_SIZE:-100000}"
MATRIX_USE_ACCELERATE="${USE_ACCELERATE:-1}"
MATRIX_LORA_R="${LORA_R:-8}"
MATRIX_LORA_ALPHA="${LORA_ALPHA:-16}"
MATRIX_LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MATRIX_LORA_TARGETS="${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj}"

MATRIX_DEFAULT_GPU_TYPE="${DEFAULT_GPU_TYPE:-h100}"
MATRIX_DEFAULT_GPU_COUNT="${DEFAULT_GPU_COUNT:-1}"
MATRIX_DEFAULT_PARTITION="${DEFAULT_PARTITION:-gpu}"
MATRIX_DEFAULT_WALLTIME="${DEFAULT_WALLTIME:-27:00:00}"
MATRIX_DEFAULT_CPUS_PER_TASK="${DEFAULT_CPUS_PER_TASK:-32}"
MATRIX_DEFAULT_MEM="${DEFAULT_MEM:-128G}"
MATRIX_SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"

MATRIX_RUN_APPROACHES="${RUN_APPROACHES:-cola hydralora adamole lora}"

declare -a MATRIX_LOCAL_JOB_PIDS=()

matrix_sanitize() {
  echo "$1" | tr '/ :|' '_'
}

matrix_print_cmd() {
  printf '[CMD] '
  printf '%q ' "$@"
  echo
}

matrix_require_path() {
  local p="$1"
  local label="$2"
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] ${label} not found: ${p}" >&2
    exit 1
  fi
}

matrix_ensure_tools() {
  if [[ "${MATRIX_RUN_MODE}" == "slurm" ]] && ! command -v sbatch >/dev/null 2>&1; then
    if [[ "${MATRIX_DRY_RUN}" == "1" ]]; then
      echo "[WARN] sbatch not found; continuing because DRY_RUN=1."
    else
      echo "[ERROR] RUN_MODE=slurm but sbatch is not available in PATH." >&2
      exit 1
    fi
  fi
}

matrix_tier_gpu_hours_from_dataset() {
  local dataset_name="$1"
  local head="${dataset_name%%_*}"
  if [[ ! "${head}" =~ ^[0-9]+$ ]]; then
    echo ""
    return
  fi
  case "${head}" in
    5) echo "24.00" ;;
    10) echo "27.69" ;;
    22) echo "36.55" ;;
    46) echo "54.28" ;;
    96) echo "91.20" ;;
    200) echo "168.00" ;;
    *) echo "" ;;
  esac
}

matrix_format_sbatch_time_from_hours() {
  local hours="$1"
  awk -v h="${hours}" 'BEGIN {
    m = int((h * 60.0) + 0.999999);
    d = int(m / (24 * 60));
    r = m % (24 * 60);
    hh = int(r / 60);
    mm = r % 60;
    if (d > 0) {
      printf "%d-%02d:%02d:00", d, hh, mm;
    } else {
      printf "%02d:%02d:00", hh, mm;
    }
  }'
}

matrix_resolve_walltime() {
  local tier_id="$1"
  local dataset_name="$2"
  local gpu_count="$3"

  local override="${TIER_WALLTIME_MAP[${tier_id}]:-}"
  if [[ -n "${override}" ]]; then
    echo "${override}"
    return
  fi

  local gpu_hours
  gpu_hours="$(matrix_tier_gpu_hours_from_dataset "${dataset_name}")"
  if [[ -z "${gpu_hours}" ]]; then
    echo "${MATRIX_DEFAULT_WALLTIME}"
    return
  fi

  local wall_hours
  wall_hours="$(awk -v gh="${gpu_hours}" -v g="${gpu_count}" 'BEGIN { if (g <= 0) g = 1; printf "%.6f", gh / g }')"
  matrix_format_sbatch_time_from_hours "${wall_hours}"
}

matrix_is_enabled_approach() {
  local target="$1"
  local a
  for a in ${MATRIX_RUN_APPROACHES}; do
    if [[ "${a}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

matrix_resolve_tokenized_path() {
  local tokenizer_name="$1"
  local dataset_name="$2"
  local tokenized_override="$3"

  if [[ -n "${tokenized_override}" ]]; then
    if [[ "${tokenized_override}" = /* ]]; then
      echo "${tokenized_override}"
    else
      echo "${MATRIX_TOKENIZED_BASE_DIR}/${tokenized_override}"
    fi
    return
  fi

  echo "${MATRIX_TOKENIZED_BASE_DIR}/${tokenizer_name}/${dataset_name}"
}

matrix_submit_or_run_job() {
  local tier_id="$1"
  local dataset_name="$2"
  local tokenizer_name="$3"
  local model_name="$4"
  local tokenized_override="$5"
  local approach="$6"
  local variant_label="$7"
  local train_bs="$8"
  local grad_acc="$9"
  local eval_steps="${10}"
  local save_steps="${11}"
  local extra_env_pairs="${12}"
  local timestamp="${13}"

  local script_name="${APPROACH_SCRIPT[${approach}]}"
  local script_path="${COMPARISON_DIR}/${script_name}"
  matrix_require_path "${script_path}" "job script"

  local tokenized_path
  tokenized_path="$(matrix_resolve_tokenized_path "${tokenizer_name}" "${dataset_name}" "${tokenized_override}")"
  matrix_require_path "${tokenized_path}" "tokenized dataset"

  local tier_slug
  local variant_slug
  local model_size_tag
  tier_slug="$(matrix_sanitize "${tier_id}")"
  variant_slug="$(matrix_sanitize "${variant_label}")"
  model_size_tag="$(
    echo "${model_name}" | sed -nE 's/.*([^0-9]|^)(8|3|1)[Bb]([^0-9]|$).*/\2b/p' | head -n1
  )"
  if [[ -z "${model_size_tag}" ]]; then
    model_size_tag="$(
      echo "${tokenizer_name}" | sed -nE 's/.*([^0-9]|^)(8|3|1)[Bb]([^0-9]|$).*/\2b/p' | head -n1
    )"
  fi
  if [[ -z "${model_size_tag}" ]]; then
    model_size_tag="unk"
    echo "[WARN] Could not infer model size for run naming from model='${model_name}' tokenizer='${tokenizer_name}'." >&2
  fi

  local run_name="l3-${model_size_tag}-${dataset_name}-${approach}-${variant_slug}-${timestamp}"
  local output_dir="${MATRIX_OUTPUT_ROOT}/${tokenizer_name}-${dataset_name}/${approach}/${variant_slug}_${timestamp}"
  local log_path="${MATRIX_LOG_ROOT}/${tier_slug}_${approach}_${variant_slug}_${timestamp}.log"
  mkdir -p "${output_dir}" "$(dirname "${log_path}")"

  local gpu_type="${TIER_GPU_TYPE_MAP[${tier_id}]:-${MATRIX_DEFAULT_GPU_TYPE}}"
  local gpu_count="${TIER_GPU_COUNT_MAP[${tier_id}]:-${MATRIX_DEFAULT_GPU_COUNT}}"
  local partition="${TIER_PARTITION_MAP[${tier_id}]:-${MATRIX_DEFAULT_PARTITION}}"
  local walltime
  walltime="$(matrix_resolve_walltime "${tier_id}" "${dataset_name}" "${gpu_count}")"
  local cpus="${TIER_CPUS_PER_TASK_MAP[${tier_id}]:-${MATRIX_DEFAULT_CPUS_PER_TASK}}"
  local mem="${TIER_MEM_MAP[${tier_id}]:-${MATRIX_DEFAULT_MEM}}"

  local -a env_assignments=(
    "REPO_ROOT=${REPO_ROOT}"
    "DATASET_DIR=${MATRIX_DATASET_DIR}"
    "MODEL_NAME_OR_PATH=${model_name}"
    "GENERIC_TOKENIZED_PATH=${tokenized_path}"
    "GENERIC_DATASET_NAME=${dataset_name}"
    "GENERIC_EVAL_DATASET_NAME=${dataset_name}"
    "OUTPUT_DIR=${output_dir}"
    "WANDB_PROJECT=${MATRIX_WANDB_PROJECT}"
    "WANDB_ENTITY=${MATRIX_WANDB_ENTITY}"
    "WANDB_RUN_GROUP=${MATRIX_WANDB_RUN_GROUP}"
    "WANDB_NAME=${run_name}"
    "FLASH_ATTN=${MATRIX_FLASH_ATTN}"
    "AUTO_FIND_BATCH_SIZE=${MATRIX_AUTO_FIND_BATCH_SIZE}"
    "NUM_TRAIN_EPOCHS=${MATRIX_NUM_TRAIN_EPOCHS}"
    "LEARNING_RATE=${MATRIX_LEARNING_RATE}"
    "LORA_R=${MATRIX_LORA_R}"
    "LORA_ALPHA=${MATRIX_LORA_ALPHA}"
    "LORA_DROPOUT=${MATRIX_LORA_DROPOUT}"
    "LORA_TARGETS=${MATRIX_LORA_TARGETS}"
    "PER_DEVICE_TRAIN_BATCH_SIZE=${train_bs}"
    "PER_DEVICE_EVAL_BATCH_SIZE=${train_bs}"
    "GRADIENT_ACCUMULATION_STEPS=${grad_acc}"
    "EVAL_STEPS=${eval_steps}"
    "SAVE_STEPS=${save_steps}"
    "SAVE_TOTAL_LIMIT=${MATRIX_SAVE_TOTAL_LIMIT}"
    "PREPROCESSING_NUM_WORKERS=${MATRIX_PREPROCESSING_NUM_WORKERS}"
    "PREPROCESSING_BATCH_SIZE=${MATRIX_PREPROCESSING_BATCH_SIZE}"
    "GPUS_PER_NODE=${gpu_count}"
    "USE_ACCELERATE=${MATRIX_USE_ACCELERATE}"
  )

  if [[ -n "${MATRIX_CONDA_BASE}" ]]; then
    env_assignments+=("CONDA_BASE=${MATRIX_CONDA_BASE}")
  fi
  if [[ -n "${MATRIX_CONDA_ENV}" ]]; then
    env_assignments+=("CONDA_ENV=${MATRIX_CONDA_ENV}")
  fi
  if [[ -n "${MATRIX_MODULE_INIT}" ]]; then
    env_assignments+=("MODULE_INIT=${MATRIX_MODULE_INIT}")
  fi

  if [[ -n "${extra_env_pairs}" ]]; then
    # shellcheck disable=SC2206
    local extra_arr=( ${extra_env_pairs} )
    env_assignments+=("${extra_arr[@]}")
  fi

  echo "[INFO] ${approach}/${variant_label}/${tier_id}: bs=${train_bs} grad_acc=${grad_acc} eval_steps=${eval_steps} save_steps=${save_steps} gpus=${gpu_count} walltime=${walltime}"

  if [[ "${MATRIX_RUN_MODE}" == "slurm" ]]; then
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
    if [[ -n "${MATRIX_SBATCH_EXTRA_ARGS}" ]]; then
      # shellcheck disable=SC2206
      local sbatch_extra_arr=( ${MATRIX_SBATCH_EXTRA_ARGS} )
      cmd+=("${sbatch_extra_arr[@]}")
    fi
    cmd+=("${script_path}")

    if [[ "${MATRIX_DRY_RUN}" == "1" ]]; then
      matrix_print_cmd "${cmd[@]}"
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

  if [[ "${MATRIX_DRY_RUN}" == "1" ]]; then
    matrix_print_cmd "${local_cmd[@]}"
    return
  fi

  echo "[INFO] Running local ${approach}/${variant_label}/${tier_id} -> ${log_path}"
  if [[ "${MATRIX_LOCAL_PARALLEL}" == "1" ]]; then
    "${local_cmd[@]}" >"${log_path}" 2>&1 &
    MATRIX_LOCAL_JOB_PIDS+=("$!")
  else
    "${local_cmd[@]}" >"${log_path}" 2>&1
  fi
}

run_matrix() {
  matrix_ensure_tools

  if [[ ! -d "${MATRIX_DATASET_DIR}" ]]; then
    echo "[WARN] DATASET_DIR not found, creating empty dir: ${MATRIX_DATASET_DIR}"
    mkdir -p "${MATRIX_DATASET_DIR}"
  fi
  matrix_require_path "${MATRIX_TOKENIZED_BASE_DIR}" "TOKENIZED_BASE_DIR"
  mkdir -p "${MATRIX_LOG_ROOT}" "${MATRIX_OUTPUT_ROOT}"

  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"

  echo "[INFO] Matrix launcher"
  echo "  mode:           ${MATRIX_RUN_MODE}"
  echo "  dry run:        ${MATRIX_DRY_RUN}"
  echo "  tokenized base: ${MATRIX_TOKENIZED_BASE_DIR}"
  echo "  output root:    ${MATRIX_OUTPUT_ROOT}"
  echo "  wandb project:  ${MATRIX_WANDB_PROJECT}"
  echo "  wandb group:    ${MATRIX_WANDB_RUN_GROUP}"
  echo "  approaches:     ${MATRIX_RUN_APPROACHES}"

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

  local tier_spec
  for tier_spec in "${LANGUAGE_TIERS[@]}"; do
    local tier_id
    local dataset_name
    local tokenizer_name
    local model_name
    local tokenized_override
    IFS='|' read -r tier_id dataset_name tokenizer_name model_name tokenized_override <<<"${tier_spec}"

    echo "[INFO] Tier ${tier_id}: dataset=${dataset_name} tokenizer=${tokenizer_name} model=${model_name}"

    local approach
    for approach in cola hydralora adamole lora; do
      if ! matrix_is_enabled_approach "${approach}"; then
        continue
      fi

      local variants_ref_name="${APPROACH_VARIANTS_REF[${approach}]}"
      declare -n variants_ref="${variants_ref_name}"

      local variant_spec
      for variant_spec in "${variants_ref[@]}"; do
        local variant_label
        local train_bs
        local grad_acc
        local eval_steps
        local save_steps
        local extra_env_pairs

        IFS='|' read -r variant_label train_bs grad_acc eval_steps save_steps extra_env_pairs <<<"${variant_spec}"

        if [[ -z "${variant_label}" || -z "${train_bs}" || -z "${grad_acc}" || -z "${eval_steps}" || -z "${save_steps}" ]]; then
          echo "[ERROR] Invalid variant spec for ${approach}: ${variant_spec}" >&2
          exit 1
        fi

        matrix_submit_or_run_job \
          "${tier_id}" "${dataset_name}" "${tokenizer_name}" "${model_name}" "${tokenized_override:-}" \
          "${approach}" "${variant_label}" "${train_bs}" "${grad_acc}" \
          "${eval_steps}" "${save_steps}" "${extra_env_pairs:-}" "${timestamp}"
      done
    done
  done

  if [[ "${MATRIX_RUN_MODE}" == "local" && "${MATRIX_LOCAL_PARALLEL}" == "1" && "${MATRIX_DRY_RUN}" != "1" ]]; then
    local rc=0
    local pid
    for pid in "${MATRIX_LOCAL_JOB_PIDS[@]:-}"; do
      wait "${pid}" || rc=$?
    done
    exit "${rc}"
  fi
}
