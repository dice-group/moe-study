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
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$((10000 + RANDOM % 20000))}"
NNODES="${NNODES:-1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:?DEEPSPEED_CONFIG not set}"
LAUNCH_CMD=(torchrun --nnodes="${NNODES}" --nproc_per_node="${GPUS_PER_NODE}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")
echo "[INFO] Using ${GPUS_PER_NODE} GPU(s) per node (DeepSpeed config: ${DEEPSPEED_CONFIG})"

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

"${LAUNCH_CMD[@]}" llamafactory-cli train \
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
  --evaluation_strategy "${EVAL_STRATEGY:-steps}" \
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
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --report_to wandb \
  --include_effective_tokens_per_second true \
  --include_num_input_tokens_seen true

echo "[INFO] CoLA job completed"
