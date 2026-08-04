#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_JSON="${SCRIPT_DIR}/comparison/comparison_config_llama3_2_3b.json"

export CONDA_BASE="${CONDA_BASE:-/opt/software/cluster/EB-SW/software/Miniforge3/25.3.0-3}"
export CONDA_ENV="${CONDA_ENV:-merlin}"
export MODULE_INIT="${MODULE_INIT:-module purge && module load toolchain/foss/2024a system/CUDA/12.6.0 lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0}"

python "${SCRIPT_DIR}/comparison/run_comparison.py" --config "${CONFIG_JSON}"
