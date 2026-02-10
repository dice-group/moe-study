#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_nightly_queue_2gpu.sh [options]

Description:
  Queue-based launcher for domain verification on multiple GPUs.
  It pins one job per GPU and automatically starts the next queued job
  when any running job finishes.

Options:
  --model <hf_model_or_path>           Default: meta-llama/Llama-3.2-1B
  --output-root <path>                 Default: ./outputs/domain_verification
  --tracks <csv>                       Default: gsm8k,dolly,code
  --methods <csv>                      Default: full,lora,adamole,mola
  --gpu-ids <csv>                      Default: 0,1
  --seed <int>                         Default: 42
  --tokenized-path <path>              Required only if tracks include multilingual10
  --report-to <none|wandb>             Default: none
  --nproc-per-node <int>               Default: 1
  --train-batch-size <int>             Default: 8
  --eval-batch-size <int>              Default: 8
  --grad-accum <int>                   Default: 2
  --adamole-aux-loss-coef <float>      Default: 0.001
  --max-steps <int>                    Optional smoke override
  --max-samples <int>                  Optional smoke override
  --skip-eval                          Train only
  --eval-limit <float>                 Optional lm-eval limit
  --include-base-eval                  Also run base-model eval suite after training

Examples:
  scripts/domain_verification/run_nightly_queue_2gpu.sh \
    --output-root ./outputs/domain_verification_night1 \
    --gpu-ids 0,1 \
    --report-to wandb \
    --include-base-eval

  scripts/domain_verification/run_nightly_queue_2gpu.sh \
    --tracks gsm8k,dolly \
    --methods full,lora,mola \
    --max-steps 10 --max-samples 512
USAGE
}

MODEL="meta-llama/Llama-3.2-1B"
OUTPUT_ROOT="./outputs/domain_verification"
TRACKS_CSV="gsm8k,dolly,code"
METHODS_CSV="full,lora,adamole,mola"
GPU_IDS_CSV="0,1"
SEED="42"
TOKENIZED_PATH=""
REPORT_TO="none"
NPROC_PER_NODE="1"
TRAIN_BATCH_SIZE="8"
EVAL_BATCH_SIZE="8"
GRAD_ACCUM="2"
ADAMOLE_AUX_LOSS_COEF="0.001"
MAX_STEPS=""
MAX_SAMPLES=""
RUN_EVAL=1
EVAL_LIMIT=""
INCLUDE_BASE_EVAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --tracks) TRACKS_CSV="$2"; shift 2 ;;
    --methods) METHODS_CSV="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --tokenized-path) TOKENIZED_PATH="$2"; shift 2 ;;
    --report-to) REPORT_TO="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
    --adamole-aux-loss-coef) ADAMOLE_AUX_LOSS_COEF="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --skip-eval) RUN_EVAL=0; shift 1 ;;
    --eval-limit) EVAL_LIMIT="$2"; shift 2 ;;
    --include-base-eval) INCLUDE_BASE_EVAL=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

IFS=',' read -r -a TRACKS <<<"$TRACKS_CSV"
IFS=',' read -r -a METHODS <<<"$METHODS_CSV"
IFS=',' read -r -a GPUS <<<"$GPU_IDS_CSV"

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPUs resolved from --gpu-ids"
  exit 1
fi

if printf '%s\n' "${TRACKS[@]}" | rg -qx "multilingual10"; then
  if [[ -z "$TOKENIZED_PATH" ]]; then
    echo "Track multilingual10 requires --tokenized-path"
    exit 1
  fi
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${OUTPUT_ROOT}/nightly_queue_logs"
mkdir -p "$LOG_DIR"

TRAIN_STATUS_TSV="${LOG_DIR}/train_status.tsv"
EVAL_STATUS_TSV="${LOG_DIR}/eval_status.tsv"
echo -e "track\tmethod\tgpu\trc\tseconds\tlog_file" >"$TRAIN_STATUS_TSV"
echo -e "kind\ttrack\tmethod\tsuite\tgpu\trc\tseconds\tlog_file" >"$EVAL_STATUS_TSV"

declare -a RUNNING_PIDS=()
declare -a AVAILABLE_GPUS=("${GPUS[@]}")
declare -A PID_GPU PID_TRACK PID_METHOD PID_KIND PID_SUITE PID_LOG PID_START
declare -a TRAIN_SUCCESS=()

contains_running_pid() {
  local pid="$1"
  for p in "${RUNNING_PIDS[@]}"; do
    [[ "$p" == "$pid" ]] && return 0
  done
  return 1
}

collect_finished() {
  local phase="$1"
  local new_running=()
  for pid in "${RUNNING_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      new_running+=("$pid")
      continue
    fi

    set +e
    wait "$pid"
    local rc=$?
    set -e

    local gpu="${PID_GPU[$pid]}"
    local track="${PID_TRACK[$pid]}"
    local method="${PID_METHOD[$pid]}"
    local kind="${PID_KIND[$pid]}"
    local suite="${PID_SUITE[$pid]}"
    local logfile="${PID_LOG[$pid]}"
    local start_ts="${PID_START[$pid]}"
    local end_ts
    end_ts=$(date +%s)
    local seconds=$((end_ts - start_ts))

    AVAILABLE_GPUS+=("$gpu")

    if [[ "$phase" == "train" ]]; then
      echo -e "${track}\t${method}\t${gpu}\t${rc}\t${seconds}\t${logfile}" >>"$TRAIN_STATUS_TSV"
      if [[ "$rc" -eq 0 ]]; then
        TRAIN_SUCCESS+=("${track}|${method}")
      fi
    else
      echo -e "${kind}\t${track}\t${method}\t${suite}\t${gpu}\t${rc}\t${seconds}\t${logfile}" >>"$EVAL_STATUS_TSV"
    fi

    unset PID_GPU["$pid"] PID_TRACK["$pid"] PID_METHOD["$pid"] PID_KIND["$pid"] PID_SUITE["$pid"] PID_LOG["$pid"] PID_START["$pid"]
  done
  RUNNING_PIDS=("${new_running[@]}")
}

launch_train_job() {
  local gpu="$1"
  local track="$2"
  local method="$3"
  local logfile="${LOG_DIR}/train_${track}_${method}_seed${SEED}.log"

  local cmd=(
    bash scripts/domain_verification/train_domain_track.sh
    --method "$method"
    --track "$track"
    --model "$MODEL"
    --output-root "$OUTPUT_ROOT"
    --seed "$SEED"
    --nproc-per-node "$NPROC_PER_NODE"
    --train-batch-size "$TRAIN_BATCH_SIZE"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --grad-accum "$GRAD_ACCUM"
    --report-to "$REPORT_TO"
    --adamole-aux-loss-coef "$ADAMOLE_AUX_LOSS_COEF"
  )
  if [[ "$track" == "multilingual10" ]]; then
    cmd+=(--tokenized-path "$TOKENIZED_PATH")
  fi
  if [[ -n "$MAX_STEPS" ]]; then
    cmd+=(--max-steps "$MAX_STEPS")
  fi
  if [[ -n "$MAX_SAMPLES" ]]; then
    cmd+=(--max-samples "$MAX_SAMPLES")
  fi

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "${cmd[@]}"
  ) >"$logfile" 2>&1 &
  local pid=$!

  RUNNING_PIDS+=("$pid")
  PID_GPU["$pid"]="$gpu"
  PID_TRACK["$pid"]="$track"
  PID_METHOD["$pid"]="$method"
  PID_KIND["$pid"]="train"
  PID_SUITE["$pid"]="-"
  PID_LOG["$pid"]="$logfile"
  PID_START["$pid"]="$(date +%s)"
  echo "[TRAIN-LAUNCH] gpu=${gpu} ${track}/${method} pid=${pid}"
}

launch_eval_job() {
  local gpu="$1"
  local kind="$2"
  local track="$3"
  local method="$4"
  local suite="$5"
  local arg_type="$6"
  local arg_value="$7"

  local logfile="${LOG_DIR}/eval_${kind}_${track}_${method}_${suite}.log"
  local cmd=(
    bash scripts/domain_verification/eval_domain_suite.sh
    --suite "$suite"
    --output-dir "${OUTPUT_ROOT}/eval/${track}/${method}"
  )
  if [[ "$arg_type" == "model" ]]; then
    cmd+=(--model "$arg_value")
  else
    cmd+=(--checkpoint "$arg_value")
  fi
  if [[ -n "$EVAL_LIMIT" ]]; then
    cmd+=(--limit "$EVAL_LIMIT")
  fi

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "${cmd[@]}"
  ) >"$logfile" 2>&1 &
  local pid=$!

  RUNNING_PIDS+=("$pid")
  PID_GPU["$pid"]="$gpu"
  PID_TRACK["$pid"]="$track"
  PID_METHOD["$pid"]="$method"
  PID_KIND["$pid"]="$kind"
  PID_SUITE["$pid"]="$suite"
  PID_LOG["$pid"]="$logfile"
  PID_START["$pid"]="$(date +%s)"
  echo "[EVAL-LAUNCH] gpu=${gpu} ${kind}/${track}/${method} suite=${suite} pid=${pid}"
}

run_train_queue() {
  local -a queue=()
  for track in "${TRACKS[@]}"; do
    for method in "${METHODS[@]}"; do
      queue+=("${track}|${method}")
    done
  done

  echo "[TRAIN] total_jobs=${#queue[@]} gpus=${GPU_IDS_CSV}"
  while [[ "${#queue[@]}" -gt 0 || "${#RUNNING_PIDS[@]}" -gt 0 ]]; do
    while [[ "${#queue[@]}" -gt 0 && "${#AVAILABLE_GPUS[@]}" -gt 0 ]]; do
      local gpu="${AVAILABLE_GPUS[0]}"
      AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:1}")
      local job="${queue[0]}"
      queue=("${queue[@]:1}")
      IFS='|' read -r track method <<<"$job"
      launch_train_job "$gpu" "$track" "$method"
    done

    if [[ "${#RUNNING_PIDS[@]}" -gt 0 ]]; then
      set +e
      wait -n
      set -e
      collect_finished "train"
    fi
  done
}

suite_for_track() {
  case "$1" in
    gsm8k) echo "gsm8k" ;;
    dolly) echo "mmlu" ;;
    code) echo "humaneval" ;;
    multilingual10) echo "multilingual10" ;;
    *) return 1 ;;
  esac
}

run_eval_queue() {
  local -a queue=()
  if [[ "$INCLUDE_BASE_EVAL" -eq 1 ]]; then
    queue+=("base|-|-|gsm8k|model|${MODEL}")
    queue+=("base|-|-|mmlu|model|${MODEL}")
    queue+=("base|-|-|mmlu_stem|model|${MODEL}")
    queue+=("base|-|-|humaneval|model|${MODEL}")
    if printf '%s\n' "${TRACKS[@]}" | rg -qx "multilingual10"; then
      queue+=("base|-|-|multilingual10|model|${MODEL}")
    fi
  fi

  local item
  for item in "${TRAIN_SUCCESS[@]}"; do
    IFS='|' read -r track method <<<"$item"
    local suite
    suite="$(suite_for_track "$track")" || continue
    local path="${OUTPUT_ROOT}/${track}/${method}/seed_${SEED}"
    if [[ "$method" == "full" ]]; then
      queue+=("train|${track}|${method}|${suite}|model|${path}")
    else
      queue+=("train|${track}|${method}|${suite}|checkpoint|${path}")
    fi
  done

  echo "[EVAL] total_jobs=${#queue[@]} gpus=${GPU_IDS_CSV}"
  while [[ "${#queue[@]}" -gt 0 || "${#RUNNING_PIDS[@]}" -gt 0 ]]; do
    while [[ "${#queue[@]}" -gt 0 && "${#AVAILABLE_GPUS[@]}" -gt 0 ]]; do
      local gpu="${AVAILABLE_GPUS[0]}"
      AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:1}")
      local job="${queue[0]}"
      queue=("${queue[@]:1}")
      IFS='|' read -r kind track method suite arg_type arg_value <<<"$job"
      launch_eval_job "$gpu" "$kind" "$track" "$method" "$suite" "$arg_type" "$arg_value"
    done

    if [[ "${#RUNNING_PIDS[@]}" -gt 0 ]]; then
      set +e
      wait -n
      set -e
      collect_finished "eval"
    fi
  done
}

echo "=== Night Queue Launcher ==="
echo "model=${MODEL}"
echo "output_root=${OUTPUT_ROOT}"
echo "tracks=${TRACKS_CSV}"
echo "methods=${METHODS_CSV}"
echo "gpu_ids=${GPU_IDS_CSV}"
echo "report_to=${REPORT_TO}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "train_batch_size=${TRAIN_BATCH_SIZE}"
echo "eval_batch_size=${EVAL_BATCH_SIZE}"
echo "grad_accum=${GRAD_ACCUM}"
echo "adamole_aux_loss_coef=${ADAMOLE_AUX_LOSS_COEF}"
echo "max_steps=${MAX_STEPS:-<none>}"
echo "max_samples=${MAX_SAMPLES:-<none>}"
echo "eval=${RUN_EVAL}"
echo "eval_limit=${EVAL_LIMIT:-<none>}"

run_train_queue

echo "[TRAIN] done. status: ${TRAIN_STATUS_TSV}"
if [[ "$RUN_EVAL" -eq 1 ]]; then
  run_eval_queue
  echo "[EVAL] done. status: ${EVAL_STATUS_TSV}"
else
  echo "[EVAL] skipped by --skip-eval"
fi

echo "Done."
