#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPARISON_DIR="${SCRIPT_DIR}/comparison"
CALL_DIR="$(pwd)"
LOG_DIR="${LOG_DIR:-${CALL_DIR}/logs/comparison}"

cd "${REPO_ROOT}"

CONDA_BASE=${CONDA_BASE:-/opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3}
CONDA_ENV=${CONDA_ENV:-moe_study_factory}
MODULE_INIT=${MODULE_INIT:-module purge && module load toolchain/foss/2024a system/CUDA/12.6.0 lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0}

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.2-1B}
GENERIC_TOKENIZED_PATH=${GENERIC_TOKENIZED_PATH:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter/llama-3.1-8B_tokenizer/5_langs}
GENERIC_DATASET_NAME=${GENERIC_DATASET_NAME:-c4}
MOELPR_TOKENIZED_PATH=${MOELPR_TOKENIZED_PATH:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/moelpr/llama-3.1-8B_tokenizer/moelpr_stage1_5newlangs}
MOELPR_LANGUAGE_COLUMN=${MOELPR_LANGUAGE_COLUMN:-language}
MOELPR_LANGUAGE_MAP=${MOELPR_LANGUAGE_MAP:-/scratch/hpc-prf-merlin/joel/moe-study/configs/moelpr_lang_map.json}
MOELPR_ORIGINAL_LANGUAGE=${MOELPR_ORIGINAL_LANGUAGE:-english}
DATASET_DIR=${DATASET_DIR:-./LLaMA-Factory/data}

OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/hpc-prf-merlin/project_data/moe_study/comparison_runs}
WANDB_PROJECT=${WANDB_PROJECT:-moe-study-comparison}
WANDB_ENTITY=${WANDB_ENTITY:-}
default_wandb_group="refactor-benchmark"
if [[ -z "${WANDB_RUN_GROUP+x}" ]]; then
  WANDB_RUN_GROUP="${default_wandb_group}"
fi
SBATCH_ARGS=${SBATCH_ARGS:-}

required_paths=(
  GENERIC_TOKENIZED_PATH
  MOELPR_TOKENIZED_PATH
  MOELPR_LANGUAGE_MAP
  DATASET_DIR
)

for path_var in "${required_paths[@]}"; do
  if [[ ! -e "${!path_var}" ]]; then
    echo "[ERROR] ${path_var} points to ${!path_var}, which does not exist." >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

export CONDA_BASE CONDA_ENV MODULE_INIT
export MODEL_NAME_OR_PATH GENERIC_TOKENIZED_PATH MOELPR_TOKENIZED_PATH
export MOELPR_LANGUAGE_COLUMN MOELPR_LANGUAGE_MAP MOELPR_ORIGINAL_LANGUAGE
export DATASET_DIR WANDB_PROJECT WANDB_ENTITY WANDB_RUN_GROUP
export GENERIC_DATASET_NAME

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ "${WANDB_RUN_GROUP}" == "${default_wandb_group}" ]]; then
  WANDB_RUN_GROUP="${WANDB_RUN_GROUP}-${timestamp}"
fi

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
  local log_prefix=$3
  shift 3
  local env_assignments=("$@")
  local cmd=(env)
  for assignment in "${env_assignments[@]}"; do
    cmd+=("${assignment}")
  done
  cmd+=("sbatch" "--output=${LOG_DIR}/${log_prefix}_%j.log")
  if [[ -n "${SBATCH_ARGS:-}" ]]; then
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

ADAMOLE_JOB_ID=$(submit_job "AdaMoLE" "${COMPARISON_DIR}/adamole_job.sh" "adamole" \
  "OUTPUT_DIR=${ADAMOLE_OUT}" \
  "WANDB_NAME=adamole-${timestamp}")

MOLA_JOB_ID=$(submit_job "MoLA" "${COMPARISON_DIR}/adamoe_1_mola_job.sh" "mola" \
  "OUTPUT_DIR=${MOLA_OUT}" \
  "WANDB_NAME=mola-${timestamp}")

MOELPR_JOB_ID=$(submit_job "MoE-LPR" "${COMPARISON_DIR}/moelpr_job.sh" "moelpr" \
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
