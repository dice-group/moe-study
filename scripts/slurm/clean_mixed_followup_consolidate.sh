#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${PROOF_PIPELINE_REPO_ROOT:-/scratch/hpc-prf-merlin/joel/moe-study}"
PYTHON_BIN="${PROOF_PIPELINE_PYTHON:-/scratch/hpc-prf-merlin/joel/conda/envs/merlin/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python executable is unavailable: ${PYTHON_BIN}" >&2
  exit 1
fi
cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/analysis/consolidate_clean_mixed_followup.py
"${PYTHON_BIN}" -m pipeline.run \
  --config configs/pipeline \
  --profile moe_study_mixed_rank_expert_followup_clean22_consolidated_v13 \
  --start-at summary \
  --stop-after report
