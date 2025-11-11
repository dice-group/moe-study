#!/bin/bash
set -euo pipefail

MODEL_DIR=${1:-}
TASKS=${2:-}
TOKENIZER_PATH=${3:-}
RUN_LABEL=${4:-}
RUN_ID=${5:-}

if [[ -z "$MODEL_DIR" || -z "$TASKS" ]]; then
  echo "Usage: lm_eval.sh MODEL_DIR TASKS [TOKENIZER_PATH]" >&2
  exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory '$MODEL_DIR' does not exist." >&2
  exit 1
fi

CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
WANDB_PROJECT=${WANDB_PROJECT:-lm_eval_project}
BATCH_SIZE=${BATCH_SIZE:-2}
LIMIT=${LIMIT:-500}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-lm_eval}
LM_EVAL_BIN=${LM_EVAL_BIN:-lm_eval}

shopt -s nullglob
CHECKPOINTS=("$MODEL_DIR"/checkpoint-*)
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  CHECKPOINTS=("$MODEL_DIR")
fi

for CKPT in "${CHECKPOINTS[@]}"; do
  [[ -d "$CKPT" ]] || continue
  out_dir="$CKPT/$OUTPUT_SUFFIX"
  mkdir -p "$out_dir"
  ckpt_name=$(basename "$CKPT")
  echo "Evaluating ${ckpt_name} on $TASKS"

  run_name="${RUN_LABEL:-$(basename "$MODEL_DIR")}-eval-${ckpt_name}"
  run_id="${RUN_ID:-${ckpt_name}-$(date +%s)}"
  wandb_args="project=${WANDB_PROJECT},group=lm_eval,job_type=${ckpt_name},name=${run_name},id=${run_id}"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" WANDB_PROJECT="$WANDB_PROJECT" \
    "$LM_EVAL_BIN" \
      --model hf \
      --model_args "pretrained=$CKPT,tokenizer=${TOKENIZER_PATH:-$CKPT}" \
      --tasks "$TASKS" \
      --device cuda \
      --batch_size "$BATCH_SIZE" \
      --limit "$LIMIT" \
      --output_path "$out_dir" \
      --wandb_args "$wandb_args"
done
