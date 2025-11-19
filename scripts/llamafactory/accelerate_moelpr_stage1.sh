#!/bin/bash
#SBATCH --job-name=moelpr-stage1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/train_acc_moelpr_stage1_%j.log
#SBATCH --partition=gpu

set -euo pipefail
module purge
module load toolchain/foss/2024a
module load system/CUDA/12.6.0
module load lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0

source /opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
conda activate cola_llama_factory
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count());"

export WANDB_PROJECT="${WANDB_PROJECT:-llama31_moelpr_stage1}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-moelpr_stage1}"

DATASET_DIR=./LLaMA-Factory/data
DATASET_NAME=${DATASET_NAME:-moelpr_5_langs}
FINETUNING_TYPE=moelpr
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/hpc-prf-merlin/project_data/moe_study/saves/moelpr_stage1_acc}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.1-8B}
ACCEL_CONFIG=${ACCEL_CONFIG:-./LLaMA-Factory/examples/accelerate/fsdp_4gpu_config.yaml}
export ACCELERATE_USE_FSDP=1
TOKENIZED_PATH=${TOKENIZED_PATH:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter/llama-3.1-8B_tokenizer/5_langs}
LANGUAGE_COLUMN=${LANGUAGE_COLUMN:-language}
LANGUAGE_MAP=${LANGUAGE_MAP:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter/moelpr_5_langs_lang_map.json}

############ Training hyperparameters ############
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
EVAL_STEPS=${EVAL_STEPS:-200}
SEED=${SEED:-42}
LOGGING_STEPS=${LOGGING_STEPS:-10}
LOGGING_FIRST_STEP=True
BF16=${BF16:-True}
FP16=${FP16:-False}
PURE_BF16=${PURE_BF16:-True}
DISABLE_TQDM=${DISABLE_TQDM:-False}
USE_REENTRANT_GC=${USE_REENTRANT_GC:-False}
MAX_STEPS=${MAX_STEPS:-500}

MOELPR_STAGE=1
MOELPR_NUM_EXPERTS=${MOELPR_NUM_EXPERTS:-4}
MOELPR_TOP_K=${MOELPR_TOP_K:-1}
MOELPR_AUX_LOSS_COEF=${MOELPR_AUX_LOSS_COEF:-0.01}

NUM_LANGS=$(echo "${TOKENIZED_PATH}" | sed -n 's#.*/\([0-9]\+\)_langs.*#\1#p')
NUM_LANGS=${NUM_LANGS:-all}

RUN_NAME="moelpr_stage1_${NUM_LANGS}langs_$(date +%Y%m%d_%H%M%S)"
WANDB_TAGS="moelpr,stage1,accelerate"
WANDB_CONFIG_JSON=$(cat <<JSON
{
  "model_name_or_path": "${MODEL_NAME_OR_PATH}",
  "tokenized_path": "${TOKENIZED_PATH}",
  "finetuning_type": "${FINETUNING_TYPE}",
  "dataset": "${DATASET_NAME}",
  "moelpr_stage": ${MOELPR_STAGE},
  "moelpr_num_experts": ${MOELPR_NUM_EXPERTS},
  "moelpr_top_k": ${MOELPR_TOP_K},
  "moelpr_aux_loss_coef": ${MOELPR_AUX_LOSS_COEF},
  "per_device_train_batch_size": ${PER_DEVICE_TRAIN_BATCH_SIZE},
  "gradient_accumulation_steps": ${GRADIENT_ACCUMULATION_STEPS},
  "learning_rate": ${LEARNING_RATE},
  "lr_scheduler_type": "${LR_SCHEDULER_TYPE}",
  "warmup_ratio": ${WARMUP_RATIO},
  "num_train_epochs": ${NUM_TRAIN_EPOCHS},
  "bf16": ${BF16},
  "fp16": ${FP16},
  "pure_bf16": ${PURE_BF16},
  "seed": ${SEED},
  "max_steps": ${MAX_STEPS},
  "num_langs": "${NUM_LANGS}"
}
JSON
)
export WANDB_TAGS
export WANDB_CONFIG_JSON

if [[ ! -f "$ACCEL_CONFIG" ]]; then
  echo "[ERROR] Accelerate config $ACCEL_CONFIG not found." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

TOKENIZED_ARGS=()
if [[ -n "$TOKENIZED_PATH" ]]; then
  if [[ ! -d "$TOKENIZED_PATH" ]]; then
    echo "[ERROR] TOKENIZED_PATH ${TOKENIZED_PATH} does not exist." >&2
    exit 1
  fi
  TOKENIZED_ARGS+=(--tokenized_path "${TOKENIZED_PATH}")
fi

echo "[INFO] Starting Accelerate-backed MoE-LPR Stage 1 training at $(date)"
accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  ./LLaMA-Factory/src/train.py \
  --stage sft \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps ${EVAL_STEPS} \
  --run_name "${RUN_NAME}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset "${DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --finetuning_type "${FINETUNING_TYPE}" \
  --output_dir "${OUTPUT_DIR}" \
  "${TOKENIZED_ARGS[@]}" \
  --overwrite_output_dir \
  --num_train_epochs ${NUM_TRAIN_EPOCHS} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --warmup_ratio ${WARMUP_RATIO} \
  --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
  --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --logging_steps ${LOGGING_STEPS} \
  --logging_first_step ${LOGGING_FIRST_STEP} \
  --seed ${SEED} \
  --bf16 ${BF16} \
  --fp16 ${FP16} \
  --pure_bf16 ${PURE_BF16} \
  --disable_tqdm ${DISABLE_TQDM} \
  --use_reentrant_gc ${USE_REENTRANT_GC} \
  --max_steps ${MAX_STEPS} \
  --moelpr_stage ${MOELPR_STAGE} \
  --moelpr_num_experts ${MOELPR_NUM_EXPERTS} \
  --moelpr_top_k ${MOELPR_TOP_K} \
  --moelpr_aux_loss_coef ${MOELPR_AUX_LOSS_COEF} \
  --language_column "${LANGUAGE_COLUMN}" \
  --language_map "${LANGUAGE_MAP}"

echo "[INFO] Stage 1 MoE-LPR training completed at $(date)"
