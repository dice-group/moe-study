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
TOKENIZED_PATH="${MOELPR_TOKENIZED_PATH:?MOELPR_TOKENIZED_PATH not set}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH not set}"
LANGUAGE_COLUMN="${MOELPR_LANGUAGE_COLUMN:?MOELPR_LANGUAGE_COLUMN not set}"
LANGUAGE_MAP="${MOELPR_LANGUAGE_MAP:?MOELPR_LANGUAGE_MAP not set}"
ORIGINAL_LANGUAGE="${MOELPR_ORIGINAL_LANGUAGE:?MOELPR_ORIGINAL_LANGUAGE not set}"
DATASET_NAME="${MOELPR_DATASET_NAME:-moelpr_5_langs}"

STAGE1_OUT="${MOELPR_STAGE1_OUTPUT:?MOELPR_STAGE1_OUTPUT not set}"
STAGE2_OUT="${MOELPR_STAGE2_OUTPUT:?MOELPR_STAGE2_OUTPUT not set}"
mkdir -p "${STAGE1_OUT}" "${STAGE2_OUT}"

echo "[INFO] Starting MoE-LPR Stage 1 into ${STAGE1_OUT}"
export WANDB_NAME="${WANDB_STAGE1_NAME:-moelpr-stage1}"
export WANDB_TAGS="comparison,moelpr,stage1"

llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset "${DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --finetuning_type moelpr \
  --output_dir "${STAGE1_OUT}" \
  --overwrite_output_dir \
  --num_train_epochs "${MOELPR_STAGE1_EPOCHS:-1}" \
  --learning_rate "${MOELPR_STAGE1_LR:-2e-4}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --per_device_train_batch_size "${MOELPR_STAGE1_BATCH:-2}" \
  --per_device_eval_batch_size "${MOELPR_STAGE1_BATCH:-2}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-2}" \
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
  --moelpr_aux_loss_coef "${MOELPR_STAGE1_AUX_LOSS_COEF:-0.01}" \
  --moelpr_debug_mode "${MOELPR_DEBUG_MODE:-False}" \
  --report_to wandb

echo "[INFO] Stage 1 finished, starting Stage 2 using ${STAGE1_OUT}"
export WANDB_NAME="${WANDB_STAGE2_NAME:-moelpr-stage2}"
export WANDB_TAGS="comparison,moelpr,stage2"

llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path "${STAGE1_OUT}" \
  --dataset "${DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --finetuning_type moelpr \
  --output_dir "${STAGE2_OUT}" \
  --overwrite_output_dir \
  --num_train_epochs "${MOELPR_STAGE2_EPOCHS:-1}" \
  --learning_rate "${MOELPR_STAGE2_LR:-1e-4}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --per_device_train_batch_size "${MOELPR_STAGE2_BATCH:-2}" \
  --per_device_eval_batch_size "${MOELPR_STAGE2_BATCH:-2}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-2}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --evaluation_strategy "${EVAL_STRATEGY:-steps}" \
  --eval_steps "${EVAL_STEPS:-200}" \
  --save_steps "${SAVE_STEPS:-200}" \
  --max_steps "${MOELPR_STAGE2_MAX_STEPS:-0}" \
  --bf16 "${BF16:-False}" \
  --fp16 "${FP16:-True}" \
  --seed "${SEED:-42}" \
  --tokenized_path "${TOKENIZED_PATH}" \
  --language_column "${LANGUAGE_COLUMN}" \
  --language_map "${LANGUAGE_MAP}" \
  --moelpr_stage 2 \
  --moelpr_num_experts "${MOELPR_NUM_EXPERTS:-4}" \
  --moelpr_top_k "${MOELPR_TOP_K:-1}" \
  --moelpr_aux_loss_coef "${MOELPR_STAGE2_AUX_LOSS_COEF:-0.0}" \
  --moelpr_lpr_loss_coef "${MOELPR_STAGE2_LPR_LOSS_COEF:-0.1}" \
  --moelpr_original_language "${ORIGINAL_LANGUAGE}" \
  --moelpr_debug_mode "${MOELPR_DEBUG_MODE:-False}" \
  --report_to wandb

echo "[INFO] MoE-LPR Stage 2 completed"
