#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/comparison.env}"
source "${CONFIG_PATH}"

required_vars=(
  CONDA_BASE
  CONDA_ENV
  MODEL_NAME_OR_PATH
  GENERIC_TOKENIZED_PATH
  MOELPR_TOKENIZED_PATH
  MOELPR_LANGUAGE_COLUMN
  MOELPR_LANGUAGE_MAP
  MOELPR_ORIGINAL_LANGUAGE
  OUTPUT_ROOT
)

for var in "${required_vars[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "[ERROR] ${var} must be defined in ${CONFIG_PATH}" >&2
    exit 1
  fi
done

for path_var in GENERIC_TOKENIZED_PATH MOELPR_TOKENIZED_PATH MOELPR_LANGUAGE_MAP; do
  if [[ ! -e "${!path_var}" ]]; then
    echo "[ERROR] ${path_var} points to ${!path_var}, which does not exist." >&2
    exit 1
  fi
done

mkdir -p logs/comparison "${OUTPUT_ROOT}"

export CONDA_BASE CONDA_ENV MODULE_INIT
export MODEL_NAME_OR_PATH GENERIC_TOKENIZED_PATH MOELPR_TOKENIZED_PATH
export MOELPR_LANGUAGE_COLUMN MOELPR_LANGUAGE_MAP MOELPR_ORIGINAL_LANGUAGE
export DATASET_DIR WANDB_PROJECT WANDB_ENTITY WANDB_RUN_GROUP
export GENERIC_DATASET_NAME

timestamp="$(date +%Y%m%d_%H%M%S)"

ADAMOLE_OUT="${OUTPUT_ROOT}/adamole_${timestamp}"
MOLA_OUT="${OUTPUT_ROOT}/mola_${timestamp}"
MOELPR_STAGE1_OUT="${OUTPUT_ROOT}/moelpr_stage1_${timestamp}"
MOELPR_STAGE2_OUT="${OUTPUT_ROOT}/moelpr_stage2_${timestamp}"

echo "[INFO] Outputs:"
echo "  AdaMoLE -> ${ADAMOLE_OUT}"
echo "  MoLA    -> ${MOLA_OUT}"
echo "  MoE-LPR Stage1 -> ${MOELPR_STAGE1_OUT}"
echo "  MoE-LPR Stage2 -> ${MOELPR_STAGE2_OUT}"

submit_job() {
  local name=$1
  local script=$2
  shift 2
  local env_assignments=("$@")
  local cmd=(env)
  for assignment in "${env_assignments[@]}"; do
    cmd+=("${assignment}")
  done
  cmd+=("sbatch")
  if [[ -n "${SBATCH_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    cmd+=(${SBATCH_ARGS})
  fi
  cmd+=("${script}")
  local output
  if ! output=$("${cmd[@]}"); then
    echo "[ERROR] Failed to submit ${name} job." >&2
    exit 1
  fi
  if [[ "${output}" != *"Submitted batch job"* ]]; then
    echo "${output}" >&2
    echo "[ERROR] sbatch did not acknowledge submission for ${name}." >&2
    exit 1
  fi
  local job_id
  job_id=$(awk '{print $4}' <<<"${output}")
  echo "[INFO] Submitted ${name} job ${job_id}"
  printf "%s" "${job_id}"
}

ADAMOLE_JOB_ID=$(submit_job "AdaMoLE" scripts/slurm/comparison/adamole_job.sh \
  "OUTPUT_DIR=${ADAMOLE_OUT}" \
  "WANDB_NAME=adamole-${timestamp}")

MOLA_JOB_ID=$(submit_job "MoLA" scripts/slurm/comparison/mola_job.sh \
  "OUTPUT_DIR=${MOLA_OUT}" \
  "WANDB_NAME=mola-${timestamp}")

MOELPR_JOB_ID=$(submit_job "MoE-LPR" scripts/slurm/comparison/moelpr_job.sh \
  "MOELPR_STAGE1_OUTPUT=${MOELPR_STAGE1_OUT}" \
  "MOELPR_STAGE2_OUTPUT=${MOELPR_STAGE2_OUT}" \
  "WANDB_STAGE1_NAME=moelpr-stage1-${timestamp}" \
  "WANDB_STAGE2_NAME=moelpr-stage2-${timestamp}")

cat <<SUMMARY
========================================
Submitted comparison jobs at ${timestamp}
  AdaMoLE : ${ADAMOLE_JOB_ID}
  MoLA    : ${MOLA_JOB_ID}
  MoE-LPR : ${MOELPR_JOB_ID}

Outputs stored under ${OUTPUT_ROOT}
Monitor training metrics in W&B project ${WANDB_PROJECT:-moe-study-comparison}
========================================
SUMMARY
