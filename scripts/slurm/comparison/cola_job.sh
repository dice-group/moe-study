#!/bin/bash
#SBATCH --job-name=cmp-cola
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=64G
#SBATCH --time=38:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/comparison/cola_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ -n "${MODULE_INIT:-}" ]]; then
  eval "${MODULE_INIT}"
fi

if [[ -n "${CONDA_BASE:-}" && -n "${CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-moe-study-comparison}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-moe-study}"
export WANDB_NAME="${WANDB_NAME:-cola}"
export WANDB_TAGS="comparison,cola"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR not set}"
mkdir -p "${OUTPUT_DIR}"

DATASET_DIR="${DATASET_DIR:-./LLaMA-Factory/data}"
TOKENIZED_PATH="${GENERIC_TOKENIZED_PATH:?GENERIC_TOKENIZED_PATH not set}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH not set}"
DATASET_NAME="${GENERIC_DATASET_NAME:-c4}"
EVAL_DATASET_NAME="${GENERIC_EVAL_DATASET_NAME:-${DATASET_NAME}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG_FILE:-${REPO_ROOT}/configs/deepspeed/comparison_ds_config.json}"
if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "[ERROR] DeepSpeed config ${DEEPSPEED_CONFIG} not found." >&2
  exit 1
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  PORT_SEED="${SLURM_JOB_ID}"
else
  PORT_SEED=$((RANDOM + 10000))
fi
MASTER_PORT="${MASTER_PORT:-$((15000 + PORT_SEED % 20000))}"
NNODES="${SLURM_JOB_NUM_NODES:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] ${PYTHON_BIN} not found in PATH." >&2
  exit 1
fi
LLAMAFACTORY_MODULE="${LLAMAFACTORY_MODULE:-llamafactory.launcher}"
TRAIN_SCRIPT="${REPO_ROOT}/LLaMA-Factory/src/train.py"
USE_ACCELERATE="${USE_ACCELERATE:-0}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"
if [[ "${USE_ACCELERATE}" == "1" ]]; then
  if [[ -z "${ACCELERATE_CONFIG}" ]]; then
    echo "[ERROR] ACCELERATE_CONFIG not set for accelerate launch." >&2
    exit 1
  fi
  if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
    echo "[ERROR] Accelerate config ${ACCELERATE_CONFIG} not found." >&2
    exit 1
  fi
  if ! command -v accelerate >/dev/null 2>&1; then
    echo "[ERROR] accelerate launcher not found in PATH." >&2
    exit 1
  fi
  export ACCELERATE_USE_FSDP="${ACCELERATE_USE_FSDP:-1}"
  LAUNCH_CMD=(accelerate launch --config_file "${ACCELERATE_CONFIG}" "${TRAIN_SCRIPT}")
  echo "[INFO] Using Accelerate config ${ACCELERATE_CONFIG}"
else
  DEEPSPEED_BIN="${DEEPSPEED_BIN:-$(command -v deepspeed || true)}"
  if [[ -z "${DEEPSPEED_BIN}" ]]; then
    echo "[ERROR] deepspeed launcher not found in PATH." >&2
    exit 1
  fi
  LAUNCH_CMD=(
    "${DEEPSPEED_BIN}"
    "--num_nodes=${NNODES}"
    "--num_gpus=${GPUS_PER_NODE}"
    "--master_port=${MASTER_PORT}"
    "${PYTHON_BIN}"
    -m
    "${LLAMAFACTORY_MODULE}"
  )
  echo "[INFO] Using DeepSpeed config ${DEEPSPEED_CONFIG} on ${GPUS_PER_NODE} GPU(s) per node (master_port=${MASTER_PORT})"
fi
DEEPSPEED_ARGS=()
if [[ "${USE_ACCELERATE}" != "1" ]]; then
  DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
TRAIN_LR="${LEARNING_RATE:-2e-4}"
TRAIN_BS="${PER_DEVICE_TRAIN_BATCH_SIZE:-12}"
EVAL_BS="${PER_DEVICE_EVAL_BATCH_SIZE:-${TRAIN_BS}}"
GRAD_ACC="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_EXPERTS="${COLA_NUM_EXPERTS:-4}"
TOP_K="${COLA_TOP_K:-2}"
STRATEGY="${COLA_STRATEGY:-fully}"
RANDOM_ID=$(printf "%04d" $((RANDOM % 10000)))
RUN_NAME="${WANDB_NAME:-cola}_ep${TRAIN_EPOCHS}_bs${TRAIN_BS}x${GRAD_ACC}_lr${TRAIN_LR}_nexp${NUM_EXPERTS}_k${TOP_K}_${STRATEGY}_${RANDOM_ID}"
export WANDB_NAME="${RUN_NAME}"

echo "[INFO] Running CoLA training into ${OUTPUT_DIR}"

"${LAUNCH_CMD[@]}" \
  --stage sft \
  --do_train \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset "${DATASET_NAME}" \
  --eval_dataset "${EVAL_DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --finetuning_type cola \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --run_name "${RUN_NAME}" \
  --num_train_epochs "${TRAIN_EPOCHS}" \
  --learning_rate "${TRAIN_LR}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --per_device_train_batch_size "${TRAIN_BS}" \
  --per_device_eval_batch_size "${EVAL_BS}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --eval_strategy "${EVAL_STRATEGY:-steps}" \
  --eval_steps "${EVAL_STEPS:-200}" \
  --save_steps "${SAVE_STEPS:-200}" \
  --max_steps "${MAX_STEPS:-0}" \
  --disable_gradient_checkpointing "${DISABLE_GRADIENT_CHECKPOINTING:-True}" \
  --bf16 "${BF16:-False}" \
  --fp16 "${FP16:-True}" \
  --tokenized_path "${TOKENIZED_PATH}" \
  --lora_rank "${LORA_R:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --lora_target "${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj}" \
  --num_A "${COLA_NUM_A:-2}" \
  --num_B "${COLA_NUM_B:-3}" \
  --cola_strategy "${COLA_STRATEGY:-fully}" \
  --use_cola_experts "${USE_COLA_EXPERTS:-False}" \
  --cola_num_experts "${COLA_NUM_EXPERTS:-4}" \
  --cola_top_k "${COLA_TOP_K:-2}" \
  --cola_debug "${COLA_DEBUG:-False}" \
  "${DEEPSPEED_ARGS[@]}" \
  --report_to wandb \
  --include_effective_tokens_per_second true \
  --include_num_input_tokens_seen true

echo "[INFO] CoLA job completed"
