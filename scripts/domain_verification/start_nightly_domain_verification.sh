#!/usr/bin/env bash

set -euo pipefail

# Quick launcher for overnight domain-verification training on 2 GPUs.
# Defaults are chosen from smoke/calibration runs for stable execution.

RUN_DATE="$(date +%Y%m%d)"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/domain_verification_night_${RUN_DATE}}"
GPU_IDS="${GPU_IDS:-0,1}"
TRACKS="${TRACKS:-gsm8k,dolly,code}"
METHODS="${METHODS:-full,lora,adamole,mola}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
REPORT_TO="${REPORT_TO:-wandb}"

# AdaMoLE auxiliary loss currently triggers backward-graph errors at >0.
# Keep it at 0.0 for stable overnight runs.
ADAMOLE_AUX_LOSS_COEF="${ADAMOLE_AUX_LOSS_COEF:-0.0}"

RUN_EVAL=0
if [[ "${1:-}" == "--with-eval" ]]; then
  RUN_EVAL=1
fi

mkdir -p "${OUTPUT_ROOT}"

CMD=(
  bash scripts/domain_verification/run_nightly_queue_2gpu.sh
  --output-root "${OUTPUT_ROOT}"
  --tracks "${TRACKS}"
  --methods "${METHODS}"
  --gpu-ids "${GPU_IDS}"
  --report-to "${REPORT_TO}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM}"
  --adamole-aux-loss-coef "${ADAMOLE_AUX_LOSS_COEF}"
)

if [[ "${RUN_EVAL}" -eq 0 ]]; then
  CMD+=(--skip-eval)
fi

nohup "${CMD[@]}" > "${OUTPUT_ROOT}/launcher.log" 2>&1 &
LAUNCHER_PID=$!
echo "${LAUNCHER_PID}" > "${OUTPUT_ROOT}/launcher.pid"

echo "Started nightly queue."
echo "pid=${LAUNCHER_PID}"
echo "output_root=${OUTPUT_ROOT}"
echo "log=${OUTPUT_ROOT}/launcher.log"
echo "status=${OUTPUT_ROOT}/nightly_queue_logs/train_status.tsv"
if [[ "${RUN_EVAL}" -eq 1 ]]; then
  echo "eval_status=${OUTPUT_ROOT}/nightly_queue_logs/eval_status.tsv"
fi

