#!/bin/bash
#SBATCH --job-name=cmp-moelpr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=64G
#SBATCH --time=38:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/comparison/moelpr_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="${REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

if [[ -n "${MODULE_INIT:-}" ]]; then
  eval "${MODULE_INIT}"
fi

set +u
if [[ -n "${CONDA_BASE:-}" && -n "${CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi
set -u

export PYTHONUNBUFFERED=1
export LLAMAFACTORY_SAVE_ADAPTER_CHECKPOINTS="${LLAMAFACTORY_SAVE_ADAPTER_CHECKPOINTS:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-moe-study-comparison}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-moe-study}"

DATASET_DIR="${DATASET_DIR:-./LLaMA-Factory/data}"
TOKENIZED_PATH="${MOELPR_TOKENIZED_PATH:?MOELPR_TOKENIZED_PATH not set}"
TOKENIZED_PATH2="${MOELPR_TOKENIZED_PATH_STAGE2:-${MOELPR_TOKENIZED_PATH2:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/moelpr/llama-3.2-1B_tokenizer/moelpr_stage2_eng}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH not set}"
LANGUAGE_COLUMN="${MOELPR_LANGUAGE_COLUMN:?MOELPR_LANGUAGE_COLUMN not set}"
LANGUAGE_MAP="${MOELPR_LANGUAGE_MAP:?MOELPR_LANGUAGE_MAP not set}"
ORIGINAL_LANGUAGE="${MOELPR_ORIGINAL_LANGUAGE:?MOELPR_ORIGINAL_LANGUAGE not set}"
DATASET_NAME="${MOELPR_DATASET_NAME:-moelpr_5_langs}"
EVAL_DATASET_NAME="${MOELPR_EVAL_DATASET_NAME:-${DATASET_NAME}}"
FLASH_ATTN="${FLASH_ATTN:-fa2}"
AUTO_FIND_BATCH_SIZE="${AUTO_FIND_BATCH_SIZE:-false}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
PREPROCESSING_BATCH_SIZE="${PREPROCESSING_BATCH_SIZE:-100000}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
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
USE_ACCELERATE="${USE_ACCELERATE:-1}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"
if [[ "${USE_ACCELERATE}" != "1" ]]; then
  echo "[ERROR] USE_ACCELERATE=0 is no longer supported in comparison jobs." >&2
  exit 1
fi
if ! command -v accelerate >/dev/null 2>&1; then
  echo "[ERROR] accelerate launcher not found in PATH." >&2
  exit 1
fi
source "${REPO_ROOT}/scripts/slurm/comparison/accelerate_launch.sh"
build_accelerate_launch "${TRAIN_SCRIPT}" "${ACCELERATE_CONFIG}" "${REPO_ROOT}"

STAGE1_OUT="${MOELPR_STAGE1_OUTPUT:?MOELPR_STAGE1_OUTPUT not set}"
STAGE2_OUT="${MOELPR_STAGE2_OUTPUT:?MOELPR_STAGE2_OUTPUT not set}"
mkdir -p "${STAGE1_OUT}" "${STAGE2_OUT}"

STAGE1_EPOCHS="${MOELPR_STAGE1_EPOCHS:-1}"
STAGE1_LR="${MOELPR_STAGE1_LR:-2e-4}"
STAGE1_BS="${MOELPR_STAGE1_BATCH:-${PER_DEVICE_TRAIN_BATCH_SIZE:-18}}"
STAGE2_EPOCHS="${MOELPR_STAGE2_EPOCHS:-1}"
STAGE2_LR="${MOELPR_STAGE2_LR:-1e-4}"
STAGE2_BS="${MOELPR_STAGE2_BATCH:-${PER_DEVICE_TRAIN_BATCH_SIZE:-18}}"
GRAD_ACC="${GRADIENT_ACCUMULATION_STEPS:-2}"
TOP_K="${MOELPR_TOP_K:-1}"
RANDOM_ID=$(printf "%04d" $((RANDOM % 10000)))
RUN_NAME_STAGE1="${WANDB_STAGE1_NAME:-moelpr-stage1}_ep${STAGE1_EPOCHS}_bs${STAGE1_BS}x${GRAD_ACC}_lr${STAGE1_LR}_k${TOP_K}_${RANDOM_ID}"
RUN_NAME_STAGE2="${WANDB_STAGE2_NAME:-moelpr-stage2}_ep${STAGE2_EPOCHS}_bs${STAGE2_BS}x${GRAD_ACC}_lr${STAGE2_LR}_k${TOP_K}_${RANDOM_ID}"

echo "[INFO] Starting MoE-LPR Stage 1 into ${STAGE1_OUT}"
export WANDB_NAME="${RUN_NAME_STAGE1}"
export WANDB_TAGS="comparison,moelpr,stage1"

AUTO_FIND_BATCH_SIZE_FLAG=()
if [[ "${AUTO_FIND_BATCH_SIZE}" == "true" || "${AUTO_FIND_BATCH_SIZE}" == "True" || "${AUTO_FIND_BATCH_SIZE}" == "1" ]]; then
  AUTO_FIND_BATCH_SIZE_FLAG=(--auto_find_batch_size)
fi

"${LAUNCH_PREFIX[@]}" "${LAUNCH_CMD[@]}" "${ENTRYPOINT[@]}" \
  --stage sft \
  --do_train \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --flash_attn "${FLASH_ATTN}" \
  --resize_vocab "${RESIZE_VOCAB:-False}" \
  --dataset "${DATASET_NAME}" \
  --eval_dataset "${EVAL_DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --finetuning_type moelpr \
  --output_dir "${STAGE1_OUT}" \
  --overwrite_output_dir \
  --run_name "${RUN_NAME_STAGE1}" \
  --num_train_epochs "${STAGE1_EPOCHS}" \
  --learning_rate "${STAGE1_LR}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --per_device_train_batch_size "${STAGE1_BS}" \
  --per_device_eval_batch_size "${STAGE1_BS}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --logging_first_step "${LOGGING_FIRST_STEP:-True}" \
  --eval_strategy "${EVAL_STRATEGY:-steps}" \
  --eval_steps "${EVAL_STEPS:-2000}" \
  --save_steps "${SAVE_STEPS:-2000}" \
  --save_safetensors "${SAVE_SAFETENSORS:-False}" \
  --save_only_model "${SAVE_ONLY_MODEL:-True}" \
  --bf16 "${BF16:-True}" \
  --fp16 "${FP16:-False}" \
  --seed "${SEED:-42}" \
  --tokenized_path "${TOKENIZED_PATH}" \
  --language_column "${LANGUAGE_COLUMN}" \
  --language_map "${LANGUAGE_MAP}" \
  --moelpr_stage 1 \
  --moelpr_num_experts "${MOELPR_NUM_EXPERTS:-4}" \
  --moelpr_top_k "${MOELPR_TOP_K:-1}" \
  --moelpr_aux_loss_coef "${MOELPR_STAGE1_AUX_LOSS_COEF:-0.05}" \
  --moelpr_debug_mode "${MOELPR_DEBUG_MODE:-False}" \
  "${AUTO_FIND_BATCH_SIZE_FLAG[@]}" \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
  --preprocessing_batch_size "${PREPROCESSING_BATCH_SIZE}" \
  --report_to wandb \
  --include_effective_tokens_per_second true \
  --include_num_input_tokens_seen true

echo "[INFO] Stage 1 finished, starting Stage 2 using ${STAGE1_OUT}"
export WANDB_NAME="${RUN_NAME_STAGE2}"
export WANDB_TAGS="comparison,moelpr,stage2"

"${LAUNCH_PREFIX[@]}" "${LAUNCH_CMD[@]}" "${ENTRYPOINT[@]}" \
  --stage sft \
  --do_train \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --flash_attn "${FLASH_ATTN}" \
  --resize_vocab "${RESIZE_VOCAB:-False}" \
  --dataset "${DATASET_NAME}" \
  --eval_dataset "${EVAL_DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --finetuning_type moelpr \
  --output_dir "${STAGE2_OUT}" \
  --overwrite_output_dir \
  --run_name "${RUN_NAME_STAGE2}" \
  --num_train_epochs "${STAGE2_EPOCHS}" \
  --learning_rate "${STAGE2_LR}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --per_device_train_batch_size "${STAGE2_BS}" \
  --per_device_eval_batch_size "${STAGE2_BS}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --logging_first_step "${LOGGING_FIRST_STEP:-True}" \
  --eval_strategy "${EVAL_STRATEGY:-steps}" \
  --eval_steps "${EVAL_STEPS:-200}" \
  --save_steps "${SAVE_STEPS:-200}" \
  --save_safetensors "${SAVE_SAFETENSORS:-False}" \
  --save_only_model "${SAVE_ONLY_MODEL:-True}" \
  --bf16 "${BF16:-True}" \
  --fp16 "${FP16:-False}" \
  --seed "${SEED:-42}" \
  --tokenized_path "${TOKENIZED_PATH2}" \
  --adapter_name_or_path "${STAGE1_OUT}" \
  --language_column "${LANGUAGE_COLUMN}" \
  --language_map "${LANGUAGE_MAP}" \
  --moelpr_stage 2 \
  --moelpr_num_experts "${MOELPR_NUM_EXPERTS:-4}" \
  --moelpr_top_k "${MOELPR_TOP_K:-1}" \
  --moelpr_aux_loss_coef "${MOELPR_STAGE2_AUX_LOSS_COEF:-0.01}" \
  --moelpr_lpr_loss_coef "${MOELPR_STAGE2_LPR_LOSS_COEF:-0.1}" \
  --moelpr_original_language "${ORIGINAL_LANGUAGE}" \
  --moelpr_debug_mode "${MOELPR_DEBUG_MODE:-False}" \
  "${AUTO_FIND_BATCH_SIZE_FLAG[@]}" \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
  --preprocessing_batch_size "${PREPROCESSING_BATCH_SIZE}" \
  --report_to wandb \
  --include_effective_tokens_per_second true \
  --include_num_input_tokens_seen true

echo "[INFO] MoE-LPR Stage 2 completed"
