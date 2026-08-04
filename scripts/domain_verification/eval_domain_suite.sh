#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  eval_domain_suite.sh --suite <gsm8k|mmlu|mmlu_stem|humaneval|multilingual10> [options]

Options:
  --model <base_model_or_path>         Evaluate a base/full model path directly
  --checkpoint <adapter_checkpoint>    Evaluate a PEFT adapter checkpoint
  --tokenizer <tokenizer_path>         Optional tokenizer override
  --output-dir <path>                  Default: ./outputs/domain_verification/eval
  --batch-size <value>                 Default: auto
  --limit <float>                      Optional lm-eval sample limit
  --device <device>                    Default: cuda
  --tasks-file <path>                  Optional override
  --lang-mode <both|single|none>       Default: both (used by language wrapper)
  --dry-run                            Print command only

Examples:
  scripts/domain_verification/eval_domain_suite.sh --suite gsm8k --model meta-llama/Llama-3.2-1B
  scripts/domain_verification/eval_domain_suite.sh --suite mmlu --checkpoint outputs/.../checkpoint-200
USAGE
}

SUITE=""
MODEL=""
CHECKPOINT=""
TOKENIZER=""
OUTPUT_DIR="./outputs/domain_verification/eval"
BATCH_SIZE="auto"
LIMIT=""
DEVICE="cuda"
TASKS_FILE=""
LANG_MODE="both"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) SUITE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --tokenizer) TOKENIZER="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --tasks-file) TASKS_FILE="$2"; shift 2 ;;
    --lang-mode) LANG_MODE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$SUITE" ]]; then
  usage
  exit 1
fi
if [[ -z "$MODEL" && -z "$CHECKPOINT" ]]; then
  echo "Provide either --model or --checkpoint"
  exit 1
fi
if [[ -n "$MODEL" && -n "$CHECKPOINT" ]]; then
  echo "Use only one of --model or --checkpoint"
  exit 1
fi

if [[ -n "$TASKS_FILE" ]]; then
  if [[ ! -f "$TASKS_FILE" ]]; then
    echo "Tasks file not found: $TASKS_FILE"
    exit 1
  fi
  TASKS=$(grep -v '^[[:space:]]*#' "$TASKS_FILE" | awk 'NF' | paste -sd, -)
  if [[ -z "$TASKS" ]]; then
    echo "No tasks resolved from $TASKS_FILE"
    exit 1
  fi
else
  case "$SUITE" in
    gsm8k)
      TASKS="gsm8k"
      ;;
    mmlu)
      TASKS="mmlu"
      ;;
    mmlu_stem)
      TASKS="mmlu_stem"
      ;;
    humaneval)
      TASKS="humaneval"
      ;;
    multilingual10)
      TASKS="belebele_arb_Arab,belebele_arb_Latn,belebele_azj_Latn,belebele_eng_Latn,belebele_hun_Latn,belebele_ind_Latn,belebele_jpn_Jpan,belebele_kor_Hang,belebele_rus_Cyrl,belebele_vie_Latn"
      ;;
    *)
      echo "Unsupported suite: $SUITE"
      exit 1
      ;;
  esac
fi

MERLIN_BIN="/anonymous-institution/users/j/anonymous-user/profiles/unix/cs/.conda/envs/merlin/bin"
LM_EVAL_BIN="${LM_EVAL_BIN:-${MERLIN_BIN}/lm_eval}"
PYTHON_BIN="${PYTHON_BIN:-${MERLIN_BIN}/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMAFACTORY_SRC="${REPO_ROOT}/LLaMA-Factory/src"

if [[ -d "$LLAMAFACTORY_SRC" ]]; then
  export PYTHONPATH="${LLAMAFACTORY_SRC}:${PYTHONPATH:-}"
fi
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
if [[ "$SUITE" == "humaneval" ]]; then
  export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
fi

mkdir -p "$OUTPUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="${OUTPUT_DIR}/${SUITE}_${STAMP}.json"

if [[ -n "$CHECKPOINT" ]]; then
  ADAPTER_DIR="$CHECKPOINT"
  if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" && -f "${ADAPTER_DIR}_adapter/adapter_config.json" ]]; then
    ADAPTER_DIR="${ADAPTER_DIR}_adapter"
  fi
  if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
    echo "adapter_config.json not found in ${CHECKPOINT}"
    exit 1
  fi

  BASE_MODEL=$("$PYTHON_BIN" - <<'PY' "$ADAPTER_DIR/adapter_config.json"
import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg.get("base_model_name_or_path", ""))
PY
)
  if [[ -z "$BASE_MODEL" ]]; then
    echo "Could not resolve base model from adapter config"
    exit 1
  fi
  TOKENIZER_USE="${TOKENIZER:-$BASE_MODEL}"

  USE_LANG_WRAPPER=0
  if [[ "$SUITE" == "multilingual10" ]]; then
    HAS_LANGUAGE_LIST=$("$PYTHON_BIN" - <<'PY' "$ADAPTER_DIR/adapter_config.json"
import json, sys
cfg = json.load(open(sys.argv[1]))
print("1" if cfg.get("language_list") else "0")
PY
)
    if [[ "$HAS_LANGUAGE_LIST" == "1" ]]; then
      USE_LANG_WRAPPER=1
    fi
  fi

  if [[ "$USE_LANG_WRAPPER" == "1" ]]; then
    cmd=(
      "$PYTHON_BIN" scripts/eval/lm_eval_language_ids.py
      --checkpoint "$ADAPTER_DIR"
      --tokenizer "$TOKENIZER_USE"
      --tasks "$TASKS"
      --output-dir "$OUTPUT_DIR"
      --batch-size "$BATCH_SIZE"
      --mode "$LANG_MODE"
      --device "$DEVICE"
    )
    if [[ -n "$LIMIT" ]]; then
      cmd+=(--limit "$LIMIT")
    fi
  else
    MODEL_ARGS="pretrained=${BASE_MODEL},peft=${ADAPTER_DIR},tokenizer=${TOKENIZER_USE},trust_remote_code=True"
    cmd=(
      "$LM_EVAL_BIN"
      --model hf
      --model_args "$MODEL_ARGS"
      --tasks "$TASKS"
      --batch_size "$BATCH_SIZE"
      --output_path "$OUTFILE"
    )
    if [[ "$SUITE" == "humaneval" ]]; then
      cmd+=(--confirm_run_unsafe_code)
    fi
    if [[ -n "$LIMIT" ]]; then
      cmd+=(--limit "$LIMIT")
    fi
  fi
else
  TOKENIZER_USE="${TOKENIZER:-$MODEL}"
  MODEL_ARGS="pretrained=${MODEL},tokenizer=${TOKENIZER_USE},trust_remote_code=True"
  cmd=(
    "$LM_EVAL_BIN"
    --model hf
    --model_args "$MODEL_ARGS"
    --tasks "$TASKS"
    --batch_size "$BATCH_SIZE"
    --output_path "$OUTFILE"
  )
  if [[ "$SUITE" == "humaneval" ]]; then
    cmd+=(--confirm_run_unsafe_code)
  fi
  if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit "$LIMIT")
  fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY RUN COMMAND:\n%s\n' "${cmd[*]}"
  exit 0
fi

"${cmd[@]}"
