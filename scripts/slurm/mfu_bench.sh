#!/bin/bash

set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda activate "${CONDA_ENV}"
  fi
fi

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.2-1B}
BATCH_SIZE=${BATCH_SIZE:-1}
SEQ_LENGTH=${SEQ_LENGTH:-1024}
NUM_STEPS=${NUM_STEPS:-50}
FLASH_ATTN=${FLASH_ATTN:-auto}
DEEPSPEED_STAGE=${DEEPSPEED_STAGE:-0}

METHODS=${METHODS:-"cola hydralora adamole mola moelpr"}
TORCHRUN=${TORCHRUN:-torchrun}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-${REPO_ROOT}/LLaMA-Factory}"

mkdir -p "${REPO_ROOT}/logs"

for method in ${METHODS}; do
  echo "[mfu] Running ${method} for ${MODEL_NAME_OR_PATH}"
  (
    cd "${LLAMA_FACTORY_DIR}"
  "${TORCHRUN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" scripts/stat_utils/cal_mfu.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --seq_length "${SEQ_LENGTH}" \
    --num_steps "${NUM_STEPS}" \
    --finetuning_type "${method}" \
    --flash_attn "${FLASH_ATTN}" \
    --deepspeed_stage "${DEEPSPEED_STAGE}"
  )
done
