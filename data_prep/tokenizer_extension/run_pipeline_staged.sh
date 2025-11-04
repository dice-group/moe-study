#!/bin/bash
# Submit each tokenizer-extension pipeline stage as a separate Slurm job.
#
# Usage:
#   ./run_pipeline_stages_slurm.sh /path/to/config.yaml
#
# Optional environment overrides:
        #   LOG_ROOT=/scratch/user/logs         # where per-stage Slurm logs are written
#   PARTITION=normal                    # Slurm partition

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG_PATH" >&2
  exit 1
fi

CONFIG_PATH="$1"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

PARTITION=${PARTITION:-normal}
OUTPUT_ROOT="/scratch/hpc-prf-merlin/joel/moe-study/data_prep/tokenizer_extension/outputs"
RUN_ID=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${OUTPUT_ROOT}/run_${RUN_ID}"
LOG_ROOT="${RUN_DIR}/logs"

mkdir -p "$LOG_ROOT"
MASTER_LOG="${LOG_ROOT}/master_${RUN_ID}.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "Run directory: $RUN_DIR"
echo "Master log: $MASTER_LOG"



# we declare the cpu / mem etc definitions for each stage first
declare -A CPUS=(
  [base_coverage]=64
  [allocation]=4
  [training]=32
  [extension]=8
  [extended_coverage]=64
  [comparison]=2
)

declare -A MEM=(
  [base_coverage]="64G"
  [allocation]="8G"
  [training]="128G"
  [extension]="96G"
  [extended_coverage]="64G"
  [comparison]="8G"
)

declare -A TIME=(
  [base_coverage]="02:00:00"
  [allocation]="00:30:00"
  [training]="08:00:00"
  [extension]="04:00:00"
  [extended_coverage]="02:00:00"
  [comparison]="00:30:00"
)

STAGES=(base_coverage allocation training extension extended_coverage comparison)

previous_job=""

for stage in "${STAGES[@]}"; do
  job_name="tokext_${stage}"
  job_log_out="${LOG_ROOT}/${stage}_%j.out"
  job_log_err="${LOG_ROOT}/${stage}_%j.err"

  echo "------------------------------------------------------------"
  echo "Preparing stage: ${stage}"
  echo "  job_name   : ${job_name}"
  echo "  config     : ${CONFIG_PATH}"
  echo "  cpus       : ${CPUS[$stage]}"
  echo "  memory     : ${MEM[$stage]}"
  echo "  time limit : ${TIME[$stage]}"
  echo "  stdout log : ${job_log_out}"
  echo "  stderr log : ${job_log_err}"
  if [[ -n "$previous_job" ]]; then
    echo "  dependency : afterok:${previous_job}"
  else
    echo "  dependency : none (first stage)"
  fi

  sbatch_args=(
    --parsable
    --job-name="$job_name"
    --partition="$PARTITION"
    --cpus-per-task="${CPUS[$stage]}"
    --mem="${MEM[$stage]}"
    --time="${TIME[$stage]}"
    --output="$job_log_out"
    --error="$job_log_err"
  )

  if [[ -n "$previous_job" ]]; then
    sbatch_args+=(--dependency="afterok:${previous_job}")
  fi

  wrap_cmd=(
    "srun"
    "python"
    "-m"
    "tokenizer_extension.pipeline_slurm"
    "--config"
    "$CONFIG_PATH"
    "--stage"
    "$stage"
  )

  echo "  command    : ${wrap_cmd[*]}"
  echo "Submitting stage '${stage}'..."
  job_id=$(sbatch "${sbatch_args[@]}" --wrap "${wrap_cmd[*]}")
  echo "Stage '${stage}' submitted as job ${job_id}"
  previous_job="$job_id"
done

echo "------------------------------------------------------------"
echo "All stages submitted. Tail logs under: ${LOG_ROOT}"
echo "Final stage job ID: ${previous_job}"
