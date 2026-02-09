#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Edit this list as needed.
DIRS=(
  # llama 3.1 8B models 10 langs
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.1-8B_tokenizer-10_langs/adamole/paper-best_20260120_222802"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.1-8B_tokenizer-10_langs/mola/paper-best_20260120_222802"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.1-8B_tokenizer-10_langs/cola/paper-best_20260114_012804"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.1-8B_tokenizer-10_langs/hydralora/paper-best_20260114_012804"

  # # llama 3.1 8B models 96 langs
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.1-8B_tokenizer-96_langs/adamole/paper-best_20260120_222802"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.1-8B_tokenizer-96_langs/mola/paper-best_20260120_222802"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.1-8B_tokenizer-96_langs/cola/paper-best_20260114_012128"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.1-8B_tokenizer-96_langs/hydralora/paper-best_20260114_012128"

  # # llama 3.1 8B models 200 langs
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.1-8B_tokenizer-200_langs/adamole/paper-best_20260120_222802"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.1-8B_tokenizer-200_langs/mola/paper-best_20260120_222802"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.1-8B_tokenizer-200_langs/cola/paper-best_20260114_003639"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.1-8B_tokenizer-200_langs/hydralora/paper-best_20260114_003639"

  # # llama 3.2 1B models 10 langs
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-1B_tokenizer-10_langs/adamole/paper-best_20260125_183236"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-1B_tokenizer-10_langs/cola/paper-best_20260125_183236"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-1B_tokenizer-10_langs/hydralora/paper-best_20260125_183236"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-1B_tokenizer-10_langs/mola/paper-best_20260125_183236"
  
  # # llama 3.2 1B models 96 langs
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-1B_tokenizer-96_langs/adamole/paper-best_20260123_173907"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-1B_tokenizer-96_langs/cola/paper-best_20260123_173907"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-1B_tokenizer-96_langs/hydralora/paper-best_20260123_173907"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-1B_tokenizer-96_langs/mola/paper-best_20260123_173907"
  
  # # llama 3.2 1B models 200 langs
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-1B_tokenizer-200_langs/adamole/paper-best_20260123_162950"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-1B_tokenizer-200_langs/cola/paper-best_20260123_162950"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-1B_tokenizer-200_langs/hydralora/paper-best_20260123_162950"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-1B_tokenizer-200_langs/mola/paper-best_20260123_162950"

  # # llama 3.2 3B models 10 langs
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-3B_tokenizer-10_langs/adamole/paper-best_20260126_133154"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-3B_tokenizer-10_langs/cola/paper-best_20260126_133154"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-3B_tokenizer-10_langs/hydralora/paper-best_20260126_133154"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-3B_tokenizer-10_langs/mola/paper-best_20260126_133154"

  # # llama 3.2 3B models 96 langs
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-3B_tokenizer-96_langs/adamole/paper-best_20260124_225927"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-3B_tokenizer-96_langs/cola/paper-best_20260124_225927"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-3B_tokenizer-96_langs/hydralora/paper-best_20260124_225927"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-3B_tokenizer-96_langs/mola/paper-best_20260124_225927"

  # llama 3.2 3B models 200 langs
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-3B_tokenizer-200_langs/adamole/paper-best_20260124_223511"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-3B_tokenizer-200_langs/cola/paper-best_20260124_223511"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-3B_tokenizer-200_langs/hydralora/paper-best_20260124_223511"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-3B_tokenizer-200_langs/mola/paper-best_20260124_223511"
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
