#!/bin/bash

set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda activate "${CONDA_ENV}"
  fi
fi

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.2-1B}
TOKENIZED_PATH=${TOKENIZED_PATH:-/data/project_data/moe_study/tokenized/test_llama_3.2-1b/}
OUTPUT_ROOT=${OUTPUT_ROOT:-./outputs/throughput}
TEMPLATE=${TEMPLATE:-llama3}

MAX_STEPS=${MAX_STEPS:-50}
LOGGING_STEPS=${LOGGING_STEPS:-10}
CUTOFF_LEN=${CUTOFF_LEN:-1024}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
SEED=${SEED:-42}
BF16=${BF16:-true}
FP16=${FP16:-false}
PURE_BF16=${PURE_BF16:-true}

METHODS=${METHODS:-"cola hydralora adamole mola moelpr"}
TORCHRUN=${TORCHRUN:-torchrun}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-${REPO_ROOT}/LLaMA-Factory}"
TRAIN_SCRIPT="${LLAMA_FACTORY_DIR}/src/train.py"

mkdir -p "${OUTPUT_ROOT}" "${REPO_ROOT}/logs"

for method in ${METHODS}; do
  out_dir="${OUTPUT_ROOT}/${method}"
  extra_args=()
  if [[ "${method}" == "moelpr" ]]; then
    extra_args+=(--moelpr_stage 1)
  fi

  echo "[throughput] Running ${method} -> ${out_dir}"
  (
    cd "${LLAMA_FACTORY_DIR}"
    "${TORCHRUN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" \
      --stage sft \
      --do_train \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" \
      --dataset c4_demo \
      --template "${TEMPLATE}" \
      --tokenized_path "${TOKENIZED_PATH}" \
      --cutoff_len "${CUTOFF_LEN}" \
      --finetuning_type "${method}" \
      --output_dir "${out_dir}" \
      --overwrite_output_dir \
      --max_steps "${MAX_STEPS}" \
      --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --learning_rate "${LEARNING_RATE}" \
      --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
      --warmup_ratio "${WARMUP_RATIO}" \
      --logging_steps "${LOGGING_STEPS}" \
      --report_to none \
      --save_strategy no \
      --save_safetensors false \
      --include_effective_tokens_per_second true \
      --include_num_input_tokens_seen true \
      --seed "${SEED}" \
      --bf16 "${BF16}" \
      --fp16 "${FP16}" \
      --pure_bf16 "${PURE_BF16}" \
      --lora_target all \
      "${extra_args[@]}"
  )
done
