#!/bin/bash
#SBATCH --job-name=cmp-moelpr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/comparison/moelpr_%j.log

set -euo pipefail

if [[ -n "${MODULE_INIT:-}" ]]; then
  eval "${MODULE_INIT}"
fi

if [[ -n "${CONDA_BASE:-}" && -n "${CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-moe-study-comparison}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-moe-study}"

DATASET_DIR="${DATASET_DIR:-./LLaMA-Factory/data}"
TOKENIZED_PATH="${MOELPR_TOKENIZED_PATH:?MOELPR_TOKENIZED_PATH not set}" # TODO: for Stage 2, point to replay data (<1%) if available.
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH not set}"
LANGUAGE_COLUMN="${MOELPR_LANGUAGE_COLUMN:?MOELPR_LANGUAGE_COLUMN not set}"
LANGUAGE_MAP="${MOELPR_LANGUAGE_MAP:?MOELPR_LANGUAGE_MAP not set}"
ORIGINAL_LANGUAGE="${MOELPR_ORIGINAL_LANGUAGE:?MOELPR_ORIGINAL_LANGUAGE not set}"
DATASET_NAME="${MOELPR_DATASET_NAME:-moelpr_5_langs}"
EVAL_DATASET_NAME="${MOELPR_EVAL_DATASET_NAME:-${DATASET_NAME}}"

STAGE1_OUT="${MOELPR_STAGE1_OUTPUT:?MOELPR_STAGE1_OUTPUT not set}"
STAGE2_OUT="${MOELPR_STAGE2_OUTPUT:?MOELPR_STAGE2_OUTPUT not set}"
mkdir -p "${STAGE1_OUT}" "${STAGE2_OUT}"

STAGE1_EPOCHS="${MOELPR_STAGE1_EPOCHS:-1}"
STAGE1_LR="${MOELPR_STAGE1_LR:-2e-4}"
STAGE1_BS="${MOELPR_STAGE1_BATCH:-2}"
STAGE2_EPOCHS="${MOELPR_STAGE2_EPOCHS:-1}"
STAGE2_LR="${MOELPR_STAGE2_LR:-1e-4}"
STAGE2_BS="${MOELPR_STAGE2_BATCH:-2}"
GRAD_ACC="${GRADIENT_ACCUMULATION_STEPS:-2}"
TOP_K="${MOELPR_TOP_K:-1}"
RANDOM_ID=$(printf "%04d" $((RANDOM % 10000)))
RUN_NAME_STAGE1="${WANDB_STAGE1_NAME:-moelpr-stage1}_ep${STAGE1_EPOCHS}_bs${STAGE1_BS}x${GRAD_ACC}_lr${STAGE1_LR}_k${TOP_K}_${RANDOM_ID}"
RUN_NAME_STAGE2="${WANDB_STAGE2_NAME:-moelpr-stage2}_ep${STAGE2_EPOCHS}_bs${STAGE2_BS}x${GRAD_ACC}_lr${STAGE2_LR}_k${TOP_K}_${RANDOM_ID}"

echo "[INFO] Starting MoE-LPR Stage 1 into ${STAGE1_OUT}"
export WANDB_NAME="${RUN_NAME_STAGE1}"
export WANDB_TAGS="comparison,moelpr,stage1"

llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
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
  --evaluation_strategy "${EVAL_STRATEGY:-steps}" \
  --eval_steps "${EVAL_STEPS:-200}" \
  --save_steps "${SAVE_STEPS:-200}" \
  --max_steps "${MOELPR_STAGE1_MAX_STEPS:-0}" \
  --bf16 "${BF16:-False}" \
  --fp16 "${FP16:-True}" \
  --seed "${SEED:-42}" \
  --tokenized_path "${TOKENIZED_PATH}" \
  --language_column "${LANGUAGE_COLUMN}" \
  --language_map "${LANGUAGE_MAP}" \
  --moelpr_stage 1 \
  --moelpr_num_experts "${MOELPR_NUM_EXPERTS:-4}" \
  --moelpr_top_k "${MOELPR_TOP_K:-1}" \
  --moelpr_aux_loss_coef "${MOELPR_STAGE1_AUX_LOSS_COEF:-0.05}" \
  --moelpr_debug_mode "${MOELPR_DEBUG_MODE:-False}" \
  --report_to wandb \
  --include_effective_tokens_per_second true \
  --include_num_input_tokens_seen true

echo "[INFO] Stage 1 finished, starting Stage 2 using ${STAGE1_OUT}"
export WANDB_NAME="${RUN_NAME_STAGE2}"
export WANDB_TAGS="comparison,moelpr,stage2"

llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
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
  --evaluation_strategy "${EVAL_STRATEGY:-steps}" \
  --eval_steps "${EVAL_STEPS:-200}" \
  --save_steps "${SAVE_STEPS:-200}" \
  --max_steps "${MOELPR_STAGE2_MAX_STEPS:-0}" \
  --bf16 "${BF16:-False}" \
  --fp16 "${FP16:-True}" \
  --seed "${SEED:-42}" \
  --tokenized_path "${TOKENIZED_PATH}" \
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
  --report_to wandb \
  --include_effective_tokens_per_second true \
  --include_num_input_tokens_seen true

echo "[INFO] MoE-LPR Stage 2 completed"
