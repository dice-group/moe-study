#!/bin/bash
#SBATCH --job-name=cmp-mola
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/comparison/mola_%j.log

set -euo pipefail

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
export WANDB_NAME="${WANDB_NAME:-mola}"
export WANDB_TAGS="comparison,mola"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR not set}"
mkdir -p "${OUTPUT_DIR}"

DATASET_DIR="${DATASET_DIR:-./LLaMA-Factory/data}"
TOKENIZED_PATH="${GENERIC_TOKENIZED_PATH:?GENERIC_TOKENIZED_PATH not set}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH not set}"
DATASET_NAME="${GENERIC_DATASET_NAME:-c4}"
EVAL_DATASET_NAME="${GENERIC_EVAL_DATASET_NAME:-${DATASET_NAME}}"

echo "[INFO] Running MoLA training into ${OUTPUT_DIR}"

llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset "${DATASET_NAME}" \
  --eval_dataset "${EVAL_DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --finetuning_type mola \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --learning_rate "${LEARNING_RATE:-2e-4}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-2}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --evaluation_strategy "${EVAL_STRATEGY:-steps}" \
  --eval_steps "${EVAL_STEPS:-200}" \
  --save_steps "${SAVE_STEPS:-200}" \
  --max_steps "${MAX_STEPS:-0}" \
  --bf16 "${BF16:-False}" \
  --fp16 "${FP16:-True}" \
  --seed "${SEED:-42}" \
  --tokenized_path "${TOKENIZED_PATH}" \
  --lora_rank "${LORA_R:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --lora_target "${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj}" \
  --mola_num_experts "${MOLA_NUM_EXPERTS:-4}" \
  --mola_top_k "${MOLA_TOP_K:-3}" \
  --mola_use_null_expert "${MOLA_USE_NULL_EXPERT:-True}" \
  --mola_router_aux_loss_coef "${MOLA_ROUTER_AUX_LOSS_COEF:-0.01}" \
  --mola_null_expert_penalty "${MOLA_NULL_EXPERT_PENALTY:-0.1}" \
  --mola_aux_loss_annealing "${MOLA_AUX_LOSS_ANNEALING:-False}" \
  --mola_debug_mode "${MOLA_DEBUG_MODE:-False}" \
  --report_to wandb \
  --include_effective_tokens_per_second true \
  --include_num_input_tokens_seen true

echo "[INFO] MoLA job completed"
