#!/usr/bin/env bash
set -euo pipefail

# Submit one Slurm job per pipeline stage with afterok dependencies.
# Each stage job runs:
#   python -m pipeline.run --start-at <stage> --stop-after <stage>
# The stage itself still uses profile executor settings (including stage-level overrides).

PROFILE="${PROFILE:-hala_otus}"
CONFIG_PATH="${CONFIG_PATH:-configs/pipeline}"
START_STAGE="${START_STAGE:-sample}"
STOP_STAGE="${STOP_STAGE:-report}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Orchestrator jobs (the jobs that run pipeline.run for each single stage).
ORCH_PARTITION="${ORCH_PARTITION:-normal}"
ORCH_CPUS="${ORCH_CPUS:-1}"
ORCH_MEM="${ORCH_MEM:-4G}"
ORCH_TIME="${ORCH_TIME:-04:00:00}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/pipeline_runs/${PROFILE}/logs/dependency_chain}"
mkdir -p "${LOG_DIR}"

STAGES=(
  sample
  tokenizer_extend
  embedding_init
  tokenize
  audit
  train
  embedding_verify
  benchmark
  baseline_eval
  summary
  report
)

stage_index() {
  local target="$1"
  local i
  for i in "${!STAGES[@]}"; do
    if [[ "${STAGES[$i]}" == "${target}" ]]; then
      echo "${i}"
      return 0
    fi
  done
  return 1
}

START_IDX="$(stage_index "${START_STAGE}")" || {
  echo "[ERROR] Unknown START_STAGE=${START_STAGE}" >&2
  exit 1
}
STOP_IDX="$(stage_index "${STOP_STAGE}")" || {
  echo "[ERROR] Unknown STOP_STAGE=${STOP_STAGE}" >&2
  exit 1
}
if (( STOP_IDX < START_IDX )); then
  echo "[ERROR] STOP_STAGE must come after START_STAGE" >&2
  exit 1
fi

DEP=""
for ((i=START_IDX; i<=STOP_IDX; i++)); do
  STAGE="${STAGES[$i]}"
  TS="$(date +%Y%m%d_%H%M%S)"
  OUT_LOG="${LOG_DIR}/${STAGE}.orch.${TS}.out"

  CMD="cd ${REPO_ROOT} && ${PYTHON_BIN} -m pipeline.run --config ${CONFIG_PATH} --profile ${PROFILE} --start-at ${STAGE} --stop-after ${STAGE}"
  SBATCH_ARGS=(
    --parsable
    --partition="${ORCH_PARTITION}"
    --cpus-per-task="${ORCH_CPUS}"
    --mem="${ORCH_MEM}"
    --time="${ORCH_TIME}"
    --job-name="pipe-${PROFILE}-${STAGE}"
    --output="${OUT_LOG}"
  )
  if [[ -n "${DEP}" ]]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEP}")
  fi

  JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" --wrap "${CMD}")"
  echo "[SUBMIT] stage=${STAGE} job_id=${JOB_ID} depends_on=${DEP:-none} log=${OUT_LOG}"
  DEP="${JOB_ID}"
done

echo "[DONE] Submitted dependency chain for profile=${PROFILE} stages=${START_STAGE}..${STOP_STAGE}"
