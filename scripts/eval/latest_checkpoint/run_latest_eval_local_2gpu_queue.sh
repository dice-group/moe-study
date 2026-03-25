#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${SCRIPT_DIR}/lm_eval_harness_latest.sh}"

RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/checkpoints/slurm_sync/moe_study_comparison_outputs_labeled/llama-3.1-8B_tokenizer-10_langs}"
DIRS=(
  "${RUNS_ROOT}/adamole/paper-best_20260212_134639"
  "${RUNS_ROOT}/cola/paper-best_20260212_134639"
  "${RUNS_ROOT}/hydralora/paper-best_20260212_134639"
  "${RUNS_ROOT}/lora/paper-best_20260213_060802"
)

PATHS_FILE="${PATHS_FILE:-}"
PATHS_FILE_MODE="${PATHS_FILE_MODE:-replace}"

WANDB_PROJECT="${WANDB_PROJECT:-moe-study-comparison_eval}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/scripts/eval/latest_checkpoint/logs}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-lm_eval_latest}" # output under each run dir
GPU_LIST="${GPU_LIST:-0,1}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LM_EVAL_LANG_MODE="${LM_EVAL_LANG_MODE:-both}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"
LM_EVAL_DEVICE="${LM_EVAL_DEVICE:-cuda}"
LM_EVAL_LOG_ROUTER_METRICS="${LM_EVAL_LOG_ROUTER_METRICS:-true}"
LM_EVAL_FORCE_DEVICE="${LM_EVAL_FORCE_DEVICE:-true}"
LM_EVAL_PYTHON_BIN="${LM_EVAL_PYTHON_BIN:-python3}"
LM_EVAL_ENABLE_WANDB="${LM_EVAL_ENABLE_WANDB:-1}"
AUTO_TASKS_BY_LANG_TIER="${AUTO_TASKS_BY_LANG_TIER:-1}"
TASKS_FILE_DEFAULT="${TASKS_FILE_DEFAULT:-${REPO_ROOT}/configs/lm_eval_tasks.txt}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"

HARNESS_ROOT="${LM_EVAL_HARNESS_PATH:-${REPO_ROOT}/scripts/eval/latest_checkpoint/lm-evaluation-harness}"
FLORES_CONFIGS="${HARNESS_ROOT}/lm_eval/tasks/flores_en_perplexity/configs"

join_include_paths() {
  local parts=()
  local item
  for item in "$@"; do
    [[ -d "$item" ]] || continue
    parts+=("$item")
  done
  local IFS=:
  printf '%s' "${parts[*]}"
}

ensure_external_task_configs() {
  "${LM_EVAL_PYTHON_BIN}" "${HARNESS_ROOT}/lm_eval/tasks/generate_local_task_configs.py"
}

SIB200_TASKS="${HARNESS_ROOT}/lm_eval/tasks/sib200/configs"
TAXI1500_TASKS="${HARNESS_ROOT}/lm_eval/tasks/taxi1500/configs"
ensure_external_task_configs
if [[ -z "${LM_EVAL_INCLUDE_PATH:-}" ]]; then
  LM_EVAL_INCLUDE_PATH="$(join_include_paths "${FLORES_CONFIGS}" "${SIB200_TASKS}" "${TAXI1500_TASKS}")"
fi

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

label_from_run_dir() {
  local run_dir="$1"
  local run_label approach_label dataset_label
  run_label="$(basename "$run_dir")"
  approach_label="$(basename "$(dirname "$run_dir")")"
  dataset_label="$(basename "$(dirname "$(dirname "$run_dir")")")"
  printf '%s\n' "${dataset_label}-${approach_label}-${run_label}"
}

to_abs_path() {
  local p="$1"
  if [[ "$p" == /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s\n' "${REPO_ROOT}/${p}"
  fi
}

latest_adapter_for_run() {
  local run_dir="$1"
  local best_step=-1
  local best_ckpt=""
  local best_adapter=""
  local ckpt base step adapter

  shopt -s nullglob
  for ckpt in "$run_dir"/checkpoint-*; do
    [[ -d "$ckpt" ]] || continue
    base="$(basename "$ckpt")"
    [[ "$base" =~ ^checkpoint-([0-9]+)$ ]] || continue

    step="${BASH_REMATCH[1]}"
    adapter="$ckpt"
    [[ -f "${adapter}/adapter_config.json" ]] || adapter="${ckpt}_adapter"
    [[ -f "${adapter}/adapter_config.json" ]] || continue

    if (( 10#$step > best_step )); then
      best_step=$((10#$step))
      best_ckpt="checkpoint-${step}"
      best_adapter="$adapter"
    fi
  done
  shopt -u nullglob

  [[ -n "$best_adapter" ]] || return 1
  printf '%s\t%s\n' "$best_adapter" "$best_ckpt"
}

tasks_file_for_run() {
  local run_dir="$1"
  local selected="$TASKS_FILE_DEFAULT"
  local candidate=""

  if [[ "$AUTO_TASKS_BY_LANG_TIER" == "1" ]]; then
    case "$run_dir" in
      *10_langs*) candidate="${REPO_ROOT}/configs/lm_eval_tasks_10_langs.txt" ;;
      *96_langs*) candidate="${REPO_ROOT}/configs/lm_eval_tasks_96_langs.txt" ;;
      *200_langs*) candidate="${REPO_ROOT}/configs/lm_eval_tasks_200_langs.txt" ;;
    esac
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      selected="$candidate"
    fi
  fi
  printf '%s\n' "$selected"
}

load_run_paths() {
  declare -a paths=("${DIRS[@]}")

  if [[ -n "$PATHS_FILE" ]]; then
    [[ -f "$PATHS_FILE" ]] || die "PATHS_FILE not found: $PATHS_FILE"
    case "$PATHS_FILE_MODE" in
      replace) paths=() ;;
      append) ;;
      *) die "PATHS_FILE_MODE must be replace or append, got: $PATHS_FILE_MODE" ;;
    esac
    while IFS= read -r line || [[ -n "$line" ]]; do
      line="${line%%#*}"
      line="${line#"${line%%[![:space:]]*}"}"
      line="${line%"${line##*[![:space:]]}"}"
      [[ -n "$line" ]] || continue
      paths+=("${line/#\~/$HOME}")
    done < "$PATHS_FILE"
  fi

  [[ "${#paths[@]}" -gt 0 ]] || die "No run directories configured."
  printf '%s\n' "${paths[@]}"
}

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  die "Eval script not found: $EVAL_SCRIPT"
fi
if [[ "$LM_EVAL_PYTHON_BIN" == */* ]]; then
  [[ -x "$LM_EVAL_PYTHON_BIN" ]] || die "LM_EVAL_PYTHON_BIN not executable: $LM_EVAL_PYTHON_BIN"
else
  command -v "$LM_EVAL_PYTHON_BIN" >/dev/null 2>&1 || die "LM_EVAL_PYTHON_BIN not found in PATH: $LM_EVAL_PYTHON_BIN"
fi

mkdir -p "$LOG_ROOT"
IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
[[ "${#GPU_IDS[@]}" -gt 0 ]] || die "No GPUs configured. Set GPU_LIST, e.g. GPU_LIST=0,1"

mapfile -t RUN_PATHS < <(load_run_paths)

declare -a JOBS=()
for raw_run in "${RUN_PATHS[@]}"; do
  run_dir="$(to_abs_path "$raw_run")"
  if [[ ! -d "$run_dir" ]]; then
    echo "[WARN] run dir not found, skipping: $run_dir" >&2
    continue
  fi

  latest="$(latest_adapter_for_run "$run_dir" || true)"
  if [[ -z "$latest" ]]; then
    echo "[WARN] no adapter checkpoints found under: $run_dir" >&2
    continue
  fi
  IFS=$'\t' read -r adapter_dir base_ckpt <<< "$latest"

  tasks_file="$(tasks_file_for_run "$run_dir")"
  if [[ ! -f "$tasks_file" ]]; then
    echo "[WARN] tasks file not found, skipping $run_dir: $tasks_file" >&2
    continue
  fi

  label="$(label_from_run_dir "$run_dir")"
  JOBS+=("${run_dir}"$'\t'"${adapter_dir}"$'\t'"${base_ckpt}"$'\t'"${tasks_file}"$'\t'"${label}")
done

[[ "${#JOBS[@]}" -gt 0 ]] || die "No runnable checkpoints discovered."

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
echo "[INFO] Prepared ${#JOBS[@]} eval jobs across ${#GPU_IDS[@]} GPUs (${GPU_LIST})"

declare -a SLOT_PID=()
declare -a SLOT_DESC=()
for _ in "${GPU_IDS[@]}"; do
  SLOT_PID+=("")
  SLOT_DESC+=("")
done

FAILED=0

wait_slot() {
  local slot="$1"
  local pid="${SLOT_PID[$slot]}"
  local desc="${SLOT_DESC[$slot]}"
  [[ -n "$pid" ]] || return 0

  if wait "$pid"; then
    echo "[INFO] Completed: $desc" >&2
  else
    echo "[ERROR] Failed: $desc" >&2
    FAILED=1
  fi
  SLOT_PID[$slot]=""
  SLOT_DESC[$slot]=""
}

for i in "${!JOBS[@]}"; do
  slot=$((i % ${#GPU_IDS[@]}))
  gpu_id="${GPU_IDS[$slot]}"
  IFS=$'\t' read -r run_dir adapter_dir base_ckpt tasks_file label <<< "${JOBS[$i]}"

  wait_slot "$slot"

  output_dir="${run_dir}/${OUTPUT_SUBDIR}/${base_ckpt}"
  if [[ "$SKIP_EXISTING" == "1" && -d "$output_dir" ]]; then
    if compgen -G "${output_dir}/*.json" >/dev/null || compgen -G "${output_dir}/*.jsonl" >/dev/null; then
      echo "[INFO] Skipping existing results: $output_dir" >&2
      continue
    fi
  fi

  log_file="${LOG_ROOT}/kiss_eval_${label}_${base_ckpt}_gpu${gpu_id}_${RUN_STAMP}.log"

  cmd=(
    env
    CUDA_VISIBLE_DEVICES="${gpu_id}"
    REPO_ROOT="${REPO_ROOT}"
    CHECKPOINT_PATH="${adapter_dir}"
    OUTPUT_DIR="${output_dir}"
    TASKS_FILE="${tasks_file}"
    WANDB_PROJECT="${WANDB_PROJECT}"
    WANDB_ENTITY="${WANDB_ENTITY}"
    WANDB_GROUP="${label}"
    WANDB_NAME="${label}-${base_ckpt}"
    LM_EVAL_HARNESS_PATH="${HARNESS_ROOT}"
    LM_EVAL_INCLUDE_PATH="${LM_EVAL_INCLUDE_PATH:-}"
    LM_EVAL_PYTHON_BIN="${LM_EVAL_PYTHON_BIN}"
    LM_EVAL_ENABLE_WANDB="${LM_EVAL_ENABLE_WANDB}"
    LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE}"
    LM_EVAL_LANG_MODE="${LM_EVAL_LANG_MODE}"
    LM_EVAL_LIMIT="${LM_EVAL_LIMIT}"
    LM_EVAL_DEVICE="${LM_EVAL_DEVICE}"
    LM_EVAL_LOG_ROUTER_METRICS="${LM_EVAL_LOG_ROUTER_METRICS}"
    LM_EVAL_FORCE_DEVICE="${LM_EVAL_FORCE_DEVICE}"
    bash "${EVAL_SCRIPT}"
  )

  echo "[INFO] Launching on GPU ${gpu_id}: ${label} @ ${base_ckpt}" >&2
  echo "       adapter=${adapter_dir}" >&2
  echo "       tasks=${tasks_file}" >&2
  echo "       output=${output_dir}" >&2
  echo "       log=${log_file}" >&2

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY-RUN] ' >&2
    printf '%q ' "${cmd[@]}" >&2
    echo >&2
    continue
  fi

  "${cmd[@]}" > "${log_file}" 2>&1 &
  pid="$!"
  SLOT_PID[$slot]="$pid"
  SLOT_DESC[$slot]="${label} @ ${base_ckpt} (GPU ${gpu_id}, PID ${pid})"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[INFO] DRY_RUN=1, no jobs started." >&2
  exit 0
fi

for slot in "${!SLOT_PID[@]}"; do
  wait_slot "$slot"
done

if [[ "$FAILED" == "1" ]]; then
  echo "[ERROR] One or more eval jobs failed." >&2
  exit 1
fi

echo "[INFO] All eval jobs completed successfully." >&2
