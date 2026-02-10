#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  train_domain_track.sh --method <full|lora|adamole|mola> --track <gsm8k|dolly|code|multilingual10> [options]

Options:
  --model <hf_model_or_path>           Default: meta-llama/Llama-3.2-1B
  --dataset-dir <path>                 Default: ./LLaMA-Factory/data
  --tokenized-path <path>              Required for track multilingual10 when loading pre-tokenized data
  --output-root <path>                 Default: ./outputs/domain_verification
  --seed <int>                         Default: 42
  --max-steps <int>                    Optional smoke-test override
  --max-samples <int>                  Optional smoke-test override
  --nproc-per-node <int>               Torchrun workers per node (default: 1)
  --train-batch-size <int>             Default: 2
  --eval-batch-size <int>              Default: 2
  --grad-accum <int>                   Default: 8
  --epochs <float>                     Default: 1.0
  --learning-rate <float>              Default: 2e-4
  --val-size <float>                   Default: 0.05 (ignored for multilingual10)
  --dataset-name <name>                Optional override for dataset name
  --eval-dataset-name <name>           Optional override for eval dataset name
  --report-to <none|wandb>             Default: none
  --adamole-aux-loss-coef <float>      Default: 0.001
  --flash-attn <auto|fa2|sdpa|...>     Default: auto
  --dry-run                            Print command without executing

Examples:
  scripts/domain_verification/train_domain_track.sh --method adamole --track gsm8k --dry-run
  scripts/domain_verification/train_domain_track.sh --method full --track dolly --max-steps 5
  scripts/domain_verification/train_domain_track.sh --method mola --track multilingual10 \
    --tokenized-path /scratch/.../llama-3.2-1B_tokenizer/10_langs
USAGE
}

METHOD=""
TRACK=""
MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-1B"
DATASET_DIR="./LLaMA-Factory/data"
TOKENIZED_PATH=""
OUTPUT_ROOT="./outputs/domain_verification"
SEED="42"
MAX_STEPS=""
MAX_SAMPLES=""
TRAIN_BATCH_SIZE="2"
EVAL_BATCH_SIZE="2"
GRAD_ACCUM="8"
EPOCHS="1.0"
LEARNING_RATE="2e-4"
VAL_SIZE="0.05"
DATASET_NAME_OVERRIDE=""
EVAL_DATASET_NAME_OVERRIDE=""
REPORT_TO="none"
ADAMOLE_AUX_LOSS_COEF="0.001"
DRY_RUN=0
FLASH_ATTN="${FLASH_ATTN:-auto}"
NPROC_PER_NODE="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --track) TRACK="$2"; shift 2 ;;
    --model) MODEL_NAME_OR_PATH="$2"; shift 2 ;;
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --tokenized-path) TOKENIZED_PATH="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --val-size) VAL_SIZE="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME_OVERRIDE="$2"; shift 2 ;;
    --eval-dataset-name) EVAL_DATASET_NAME_OVERRIDE="$2"; shift 2 ;;
    --report-to) REPORT_TO="$2"; shift 2 ;;
    --adamole-aux-loss-coef) ADAMOLE_AUX_LOSS_COEF="$2"; shift 2 ;;
    --flash-attn) FLASH_ATTN="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$METHOD" || -z "$TRACK" ]]; then
  usage
  exit 1
fi

case "$METHOD" in
  full|lora|adamole|mola) ;;
  *) echo "Unsupported method: $METHOD"; exit 1 ;;
esac

case "$TRACK" in
  gsm8k)
    DATASET_NAME="${DATASET_NAME_OVERRIDE:-gsm8k_train}"
    EVAL_DATASET_NAME="${EVAL_DATASET_NAME_OVERRIDE:-}"
    USE_VAL_SPLIT=1
    ;;
  dolly)
    DATASET_NAME="${DATASET_NAME_OVERRIDE:-dolly_15k_en}"
    EVAL_DATASET_NAME="${EVAL_DATASET_NAME_OVERRIDE:-}"
    USE_VAL_SPLIT=1
    ;;
  code)
    DATASET_NAME="${DATASET_NAME_OVERRIDE:-codealpaca}"
    EVAL_DATASET_NAME="${EVAL_DATASET_NAME_OVERRIDE:-}"
    USE_VAL_SPLIT=1
    ;;
  multilingual10)
    DATASET_NAME="${DATASET_NAME_OVERRIDE:-10_langs}"
    EVAL_DATASET_NAME="${EVAL_DATASET_NAME_OVERRIDE:-10_langs}"
    USE_VAL_SPLIT=0
    if [[ -z "$TOKENIZED_PATH" ]]; then
      echo "Track multilingual10 requires --tokenized-path"
      exit 1
    fi
    ;;
  *) echo "Unsupported track: $TRACK"; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MERLIN_BIN="/upb/users/j/joeldag/profiles/unix/cs/.conda/envs/merlin/bin"
PYTHON_BIN="${PYTHON_BIN:-${MERLIN_BIN}/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${MERLIN_BIN}/torchrun}"
TRAIN_ENTRY="${REPO_ROOT}/LLaMA-Factory/src/train.py"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python not found: $PYTHON_BIN"
  exit 1
fi
if [[ ! -x "$TORCHRUN_BIN" ]]; then
  echo "torchrun not found: $TORCHRUN_BIN"
  exit 1
fi
if [[ ! -f "$TRAIN_ENTRY" ]]; then
  echo "train entrypoint not found: $TRAIN_ENTRY"
  exit 1
fi

RUN_DIR="${OUTPUT_ROOT}/${TRACK}/${METHOD}/seed_${SEED}"
mkdir -p "$RUN_DIR"

cmd=(
  "$TORCHRUN_BIN"
  --standalone
  --nnodes=1
  --nproc_per_node "$NPROC_PER_NODE"
  "$TRAIN_ENTRY"
  --stage sft
  --do_train
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --flash_attn "$FLASH_ATTN"
  --dataset "$DATASET_NAME"
  --dataset_dir "$DATASET_DIR"
  --template llama3
  --finetuning_type "$METHOD"
  --output_dir "$RUN_DIR"
  --overwrite_output_dir
  --run_name "domain-${TRACK}-${METHOD}-s${SEED}"
  --num_train_epochs "$EPOCHS"
  --learning_rate "$LEARNING_RATE"
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --per_device_train_batch_size "$TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size "$EVAL_BATCH_SIZE"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --logging_steps 20
  --logging_first_step True
  --eval_strategy steps
  --eval_steps 200
  --save_steps 200
  --save_total_limit 2
  --save_safetensors True
  --save_only_model True
  --seed "$SEED"
  --disable_gradient_checkpointing False
  --bf16 True
  --fp16 False
  --preprocessing_num_workers 8
  --preprocessing_batch_size 10000
  --report_to "$REPORT_TO"
  --include_effective_tokens_per_second true
  --include_num_input_tokens_seen true
)

if [[ "$USE_VAL_SPLIT" == "1" ]]; then
  cmd+=(--val_size "$VAL_SIZE")
elif [[ -n "$EVAL_DATASET_NAME" ]]; then
  cmd+=(--eval_dataset "$EVAL_DATASET_NAME")
fi

if [[ -n "$TOKENIZED_PATH" ]]; then
  cmd+=(--tokenized_path "$TOKENIZED_PATH")
fi

if [[ "$METHOD" == "lora" ]]; then
  cmd+=(
    --lora_rank 8
    --lora_alpha 16
    --lora_dropout 0.05
    --lora_target "q_proj,k_proj,v_proj,o_proj"
  )
elif [[ "$METHOD" == "adamole" ]]; then
  cmd+=(
    --lora_rank 8
    --lora_alpha 16
    --lora_dropout 0.05
    --lora_target "q_proj,k_proj,v_proj,o_proj"
    --adamole_num_experts 4
    --adamole_max_threshold 0.25
    --adamole_aux_loss_coef "$ADAMOLE_AUX_LOSS_COEF"
    --adamole_debug_mode False
  )
elif [[ "$METHOD" == "mola" ]]; then
  cmd+=(
    --lora_rank 8
    --lora_alpha 16
    --lora_dropout 0.05
    --lora_target "q_proj,k_proj,v_proj,o_proj"
    --mola_num_experts 4
    --mola_top_k 2
    --mola_use_null_expert True
    --mola_router_aux_loss_coef 0.001
    --mola_null_expert_penalty 0.0
    --mola_aux_loss_annealing False
    --mola_debug_mode False
  )
fi

if [[ -n "$MAX_STEPS" ]]; then
  cmd+=(--max_steps "$MAX_STEPS")
fi
if [[ -n "$MAX_SAMPLES" ]]; then
  cmd+=(--max_samples "$MAX_SAMPLES")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY RUN COMMAND:\n%s\n' "${cmd[*]}"
  exit 0
fi

export PYTHONPATH="${REPO_ROOT}/LLaMA-Factory/src:${PYTHONPATH:-}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
"${cmd[@]}"
