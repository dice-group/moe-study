#!/usr/bin/env bash
set -euo pipefail

module purge
module load toolchain/foss/2024a system/CUDA/12.6.0

PYTHON_BIN="${PYTHON_BIN:-/scratch/hpc-prf-merlin/joel/conda/envs/merlin/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[PREFLIGHT_ERROR] Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

cd /scratch/hpc-prf-merlin/joel/moe-study
export PYTHONPATH="${PWD}:${PWD}/LLaMA-Factory/src:${PYTHONPATH:-}"
PREFLIGHT_GPU_COUNT="${PREFLIGHT_GPU_COUNT:-4}"

echo "[PREFLIGHT_START] $(date -u +%Y-%m-%dT%H:%M:%SZ) job=${SLURM_JOB_ID:-none} node=$(hostname) gpus=${PREFLIGHT_GPU_COUNT}"
timeout --signal=TERM --kill-after=30 330s \
  "${PYTHON_BIN}" -m pipeline.gpu_preflight \
    --required-gpu-count "${PREFLIGHT_GPU_COUNT}" \
    --required-gpu-name A100 \
    --timeout-seconds 300 \
    --check-ecc \
    --check-nvlink \
    --check-torch-distributed
echo "[PREFLIGHT_END] $(date -u +%Y-%m-%dT%H:%M:%SZ) node=$(hostname)"
