#!/usr/bin/env bash

set -euo pipefail

# By default this script only prints commands.
# Add --execute to launch jobs.

EXECUTE=0
MODEL="meta-llama/Llama-3.2-1B"
OUTPUT_ROOT="./outputs/domain_verification"
MULTI_TOKENIZED_PATH="${MULTI_TOKENIZED_PATH:-}"
NPROC_PER_NODE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=1; shift 1 ;;
    --model) MODEL="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --multi-tokenized-path) MULTI_TOKENIZED_PATH="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: plan_overnight.sh [--execute] [--model MODEL] [--output-root DIR] [--multi-tokenized-path PATH] [--nproc-per-node N]"
      exit 0
      ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

run_cmd() {
  if [[ "$EXECUTE" == "1" ]]; then
    echo "[RUN] $*"
    eval "$*"
  else
    echo "[PLAN] $*"
  fi
}

echo "=== Domain Verification Training Plan ==="
echo "Model: ${MODEL}"
echo "Output root: ${OUTPUT_ROOT}"
echo "nproc_per_node: ${NPROC_PER_NODE}"

declare -a methods=(full lora adamole mola)
declare -a tracks=(gsm8k dolly)

for track in "${tracks[@]}"; do
  for method in "${methods[@]}"; do
    run_cmd "scripts/domain_verification/train_domain_track.sh --method ${method} --track ${track} --model ${MODEL} --output-root ${OUTPUT_ROOT} --seed 42 --nproc-per-node ${NPROC_PER_NODE}"
  done
done

if [[ -n "$MULTI_TOKENIZED_PATH" ]]; then
  for method in "${methods[@]}"; do
    run_cmd "scripts/domain_verification/train_domain_track.sh --method ${method} --track multilingual10 --model ${MODEL} --output-root ${OUTPUT_ROOT} --tokenized-path ${MULTI_TOKENIZED_PATH} --seed 42 --nproc-per-node ${NPROC_PER_NODE}"
  done
else
  echo "[PLAN] Skipping multilingual10 track because --multi-tokenized-path is not provided."
fi

echo ""
echo "=== Evaluation Plan ==="
run_cmd "scripts/domain_verification/eval_domain_suite.sh --suite gsm8k --model ${MODEL} --output-dir ${OUTPUT_ROOT}/eval/base"
run_cmd "scripts/domain_verification/eval_domain_suite.sh --suite mmlu --model ${MODEL} --output-dir ${OUTPUT_ROOT}/eval/base"
run_cmd "scripts/domain_verification/eval_domain_suite.sh --suite humaneval --model ${MODEL} --output-dir ${OUTPUT_ROOT}/eval/base"

for track in "${tracks[@]}"; do
  for method in "${methods[@]}"; do
    ckpt="${OUTPUT_ROOT}/${track}/${method}/seed_42"
    case "$track" in
      gsm8k)
        if [[ "$method" == "full" ]]; then
          run_cmd "scripts/domain_verification/eval_domain_suite.sh --suite gsm8k --model ${ckpt} --output-dir ${OUTPUT_ROOT}/eval/${track}/${method}"
        else
          run_cmd "scripts/domain_verification/eval_domain_suite.sh --suite gsm8k --checkpoint ${ckpt} --output-dir ${OUTPUT_ROOT}/eval/${track}/${method}"
        fi
        ;;
      dolly)
        if [[ "$method" == "full" ]]; then
          run_cmd "scripts/domain_verification/eval_domain_suite.sh --suite mmlu --model ${ckpt} --output-dir ${OUTPUT_ROOT}/eval/${track}/${method}"
        else
          run_cmd "scripts/domain_verification/eval_domain_suite.sh --suite mmlu --checkpoint ${ckpt} --output-dir ${OUTPUT_ROOT}/eval/${track}/${method}"
        fi
        ;;
    esac
  done
done

if [[ -n "$MULTI_TOKENIZED_PATH" ]]; then
  for method in "${methods[@]}"; do
    ckpt="${OUTPUT_ROOT}/multilingual10/${method}/seed_42"
    if [[ "$method" == "full" ]]; then
      run_cmd "scripts/domain_verification/eval_domain_suite.sh --suite multilingual10 --model ${ckpt} --output-dir ${OUTPUT_ROOT}/eval/multilingual10/${method}"
    else
      run_cmd "scripts/domain_verification/eval_domain_suite.sh --suite multilingual10 --checkpoint ${ckpt} --output-dir ${OUTPUT_ROOT}/eval/multilingual10/${method}"
    fi
  done
fi
