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
GENERIC_TOKENIZED_PATH=${GENERIC_TOKENIZED_PATH:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/moe_study_comparison/llama-3.2-1B_tokenizer/22_langs}
GENERIC_DATASET_NAME=${GENERIC_DATASET_NAME:-c4}
MOELPR_TOKENIZED_PATH=${MOELPR_TOKENIZED_PATH:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/moe_study_comparison/llama-3.2-1B_tokenizer/22_langs} # TODO: Stage 2 should eventually point to its replay subset.
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

declare -A JOB_IDS=()

ADA_VARIANTS=(
  "default|ADAMOLE_NUM_EXPERTS=4|ADAMOLE_AUX_LOSS_COEF=0.001|ADAMOLE_MAX_THRESHOLD=0.25"
  "low-aux|ADAMOLE_NUM_EXPERTS=4|ADAMOLE_AUX_LOSS_COEF=0.0005|ADAMOLE_MAX_THRESHOLD=0.25"
  "high-aux|ADAMOLE_NUM_EXPERTS=4|ADAMOLE_AUX_LOSS_COEF=0.003|ADAMOLE_MAX_THRESHOLD=0.25"
)

MOLA_VARIANTS=(
  "default|MOLA_TOP_K=1|MOLA_NULL_EXPERT_PENALTY=0|MOLA_ROUTER_AUX_LOSS_COEF=0.02"
  "low-aux|MOLA_TOP_K=1|MOLA_NULL_EXPERT_PENALTY=0|MOLA_ROUTER_AUX_LOSS_COEF=0.01"
  "high-aux|MOLA_TOP_K=1|MOLA_NULL_EXPERT_PENALTY=0|MOLA_ROUTER_AUX_LOSS_COEF=0.03"
)

HYDRA_VARIANTS=(
  "shared-lora|USE_HYDRALORA_EXPERTS=False|HYDRALORA_NUM_EXPERTS=1|HYDRALORA_TOP_K=1"
  "hydra-default|USE_HYDRALORA_EXPERTS=True|HYDRALORA_NUM_EXPERTS=4|HYDRALORA_TOP_K=1"
  "hydra-high|USE_HYDRALORA_EXPERTS=True|HYDRALORA_NUM_EXPERTS=8|HYDRALORA_TOP_K=1"
)

COLA_VARIANTS=(
  "flat-2x3|USE_COLA_EXPERTS=False|COLA_NUM_A=2|COLA_NUM_B=3|COLA_STRATEGY=fully|COLA_TOP_K=1"
  "collab-rand|USE_COLA_EXPERTS=False|COLA_NUM_A=1|COLA_NUM_B=4|COLA_STRATEGY=random|COLA_TOP_K=1"
  "dense-1x4|USE_COLA_EXPERTS=False|COLA_NUM_A=1|COLA_NUM_B=4|COLA_STRATEGY=fully|COLA_TOP_K=1"
)

MOELPR_VARIANTS=(
  "default|MOELPR_STAGE1_AUX_LOSS_COEF=0.05|MOELPR_STAGE2_LPR_LOSS_COEF=0.1|MOELPR_TOP_K=1"
  "strong-prior|MOELPR_STAGE1_AUX_LOSS_COEF=0.08|MOELPR_STAGE2_LPR_LOSS_COEF=0.2|MOELPR_TOP_K=1"
  "low-aux|MOELPR_STAGE1_AUX_LOSS_COEF=0.03|MOELPR_STAGE2_LPR_LOSS_COEF=0.05|MOELPR_TOP_K=1"
)

run_sweep() {
  local approach=$1
  local script=$2
  local log_prefix=$3
  local -n variants_ref=$4
  local base_dir="${OUTPUT_ROOT}/${approach}"
  mkdir -p "${base_dir}"
  for spec in "${variants_ref[@]}"; do
    IFS='|' read -r label env1 env2 env3 env4 <<<"${spec}"
    local out_dir="${base_dir}/${label}_${timestamp}"
    mkdir -p "${out_dir}"
    local envs=(
      "OUTPUT_DIR=${out_dir}"
      "WANDB_NAME=${approach}-${label}-${timestamp}"
    )
    for extra in "${env1}" "${env2}" "${env3}" "${env4}"; do
      if [[ -n "${extra}" ]]; then
        envs+=("${extra}")
      fi
    done
    job_id=$(submit_job "${approach}-${label}" "${script}" "${log_prefix}_${label}" "${envs[@]}")
    JOB_IDS["${approach}-${label}"]="${job_id}"
  done
}

run_sweep "adamole" "${COMPARISON_DIR}/adamole_job.sh" "adamole" ADA_VARIANTS
run_sweep "mola" "${COMPARISON_DIR}/adamoe_1_mola_job.sh" "mola" MOLA_VARIANTS
run_sweep "hydralora" "${COMPARISON_DIR}/hydralora_job.sh" "hydralora" HYDRA_VARIANTS
run_sweep "cola" "${COMPARISON_DIR}/cola_job.sh" "cola" COLA_VARIANTS

for spec in "${MOELPR_VARIANTS[@]}"; do
  IFS='|' read -r label env1 env2 env3 <<<"${spec}"
  stage1_dir="${OUTPUT_ROOT}/moelpr/${label}_stage1_${timestamp}"
  stage2_dir="${OUTPUT_ROOT}/moelpr/${label}_stage2_${timestamp}"
  mkdir -p "${stage1_dir}" "${stage2_dir}"
  envs=(
    "MOELPR_STAGE1_OUTPUT=${stage1_dir}"
    "MOELPR_STAGE2_OUTPUT=${stage2_dir}"
    "WANDB_STAGE1_NAME=moelpr-stage1-${label}-${timestamp}"
    "WANDB_STAGE2_NAME=moelpr-stage2-${label}-${timestamp}"
  )
  for extra in "${env1}" "${env2}" "${env3}"; do
    if [[ -n "${extra}" ]]; then
      envs+=("${extra}")
    fi
  done
  job_id=$(submit_job "moelpr-${label}" "${COMPARISON_DIR}/moelpr_job.sh" "moelpr_${label}" "${envs[@]}")
  JOB_IDS["moelpr-${label}"]="${job_id}"
done

cat <<SUMMARY
========================================
Submitted comparison sweeps at ${timestamp}
$(for key in "${!JOB_IDS[@]}"; do printf "  %-25s : %s\n" "${key}" "${JOB_IDS[$key]}"; done | sort)

Outputs stored under ${OUTPUT_ROOT}
Monitor training metrics in W&B project ${WANDB_PROJECT:-moe-study-comparison}
========================================
SUMMARY
