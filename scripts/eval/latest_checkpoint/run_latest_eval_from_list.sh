#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Edit this list as needed.
DIRS=(
  # active run set: llama-3.2-1B 10 langs (labeled)
  "/scratch/anonymous-project/project_data/moe_study/moe_study_comparison_outputs_labeled/llama-3.2-1B_tokenizer-10_langs/adamole/paper-best_20260210_021455"
  "/scratch/anonymous-project/project_data/moe_study/moe_study_comparison_outputs_labeled/llama-3.2-1B_tokenizer-10_langs/cola/paper-best_20260210_021455"
  "/scratch/anonymous-project/project_data/moe_study/moe_study_comparison_outputs_labeled/llama-3.2-1B_tokenizer-10_langs/hydralora/paper-best_20260210_021455"
  "/scratch/anonymous-project/project_data/moe_study/moe_study_comparison_outputs_labeled/llama-3.2-1B_tokenizer-10_langs/lora/paper-best_20260210_021455"
)

# Defaults (override via env if needed)
EVAL_TIME="${EVAL_TIME:-12:00:00}"
EVAL_GPUS="${EVAL_GPUS:-1}"
EVAL_GPU_TYPE="${EVAL_GPU_TYPE:-h100}"
EVAL_PARTITION="${EVAL_PARTITION:-gpu}"
WANDB_PROJECT="${WANDB_PROJECT:-moe-study-comparison_eval}"
LOG_ROOT="${LOG_ROOT:-${PWD}/logs}"

exec python3 "${REPO_ROOT}/scripts/eval/latest_checkpoint/submit_latest_eval.py" \
  --paths "${DIRS[@]}" \
  --wandb-project "${WANDB_PROJECT}" \
  --eval-partition "${EVAL_PARTITION}" \
  --eval-time "${EVAL_TIME}" \
  --eval-gpus "${EVAL_GPUS}" \
  --eval-gpu-type "${EVAL_GPU_TYPE}" \
  --log-root "${LOG_ROOT}" \
  --auto-tasks-by-lang-tier
