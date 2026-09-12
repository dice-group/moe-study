#!/usr/bin/env bash
# Generic Noctua2 worker for the final top-5 routing analysis.
set -eo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <config> <run-id> <canary|full>" >&2
  exit 2
fi

CONFIG=$1
RUN_ID=$2
MODE=$3
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"

module purge
module load toolchain/foss/2024a system/CUDA/12.6.0
set +u
source /opt/software/pc2/EB-SW/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate merlin
set -u

cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ARGS=(
  python -m pipeline_tools.routing_analysis
  --config "${CONFIG}"
  --run-id "${RUN_ID}"
  --validate-static
)
if [[ "${MODE}" == "canary" ]]; then
  ARGS+=(--canary --skip-figures)
elif [[ "${MODE}" != "full" ]]; then
  echo "Unsupported mode: ${MODE}" >&2
  exit 2
fi

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-local} mode=${MODE} run_id=${RUN_ID}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${ARGS[@]}"
