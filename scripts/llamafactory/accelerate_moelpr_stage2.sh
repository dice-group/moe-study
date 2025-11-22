#!/bin/bash
#SBATCH --job-name=moelpr-stage2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/train_acc_moelpr_stage2_%j.log
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

export WANDB_PROJECT="${WANDB_PROJECT:-llama31_moelpr_stage2}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-moelpr_stage2}"

DATASET_DIR=./LLaMA-Factory/data
DATASET_NAME=${DATASET_NAME:-moelpr_5_langs}
FINETUNING_TYPE=moelpr
BASE_MODEL=${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.2-1B}
STAGE1_ADAPTER=${ADAPTER_NAME_OR_PATH:-/scratch/hpc-prf-merlin/project_data/moe_study/saves/moelpr_stage1_acc}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/hpc-prf-merlin/project_data/moe_study/saves/moelpr_stage2_acc}
ACCEL_CONFIG=${ACCEL_CONFIG:-./LLaMA-Factory/examples/accelerate/fsdp_4gpu_config.yaml}
export ACCELERATE_USE_FSDP=1
TOKENIZED_PATH=${TOKENIZED_PATH:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter/llama-3.1-8B_tokenizer/5_langs} # TODO: Stage 2 should point to replay data (<1%) when available.
LANGUAGE_COLUMN=${LANGUAGE_COLUMN:-language}
LANGUAGE_MAP=${LANGUAGE_MAP:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter/moelpr_5_langs_lang_map.json}
MOELPR_ORIGINAL_LANGUAGE=${MOELPR_ORIGINAL_LANGUAGE:-eng_Latn}

############ Training hyperparameters ############
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
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
MAX_STEPS=${MAX_STEPS:-200}

MOELPR_STAGE=2
MOELPR_NUM_EXPERTS=${MOELPR_NUM_EXPERTS:-4}
MOELPR_TOP_K=${MOELPR_TOP_K:-1}
MOELPR_LPR_LOSS_COEF=${MOELPR_LPR_LOSS_COEF:-0.1}

NUM_LANGS=$(echo "${TOKENIZED_PATH}" | sed -n 's#.*/\([0-9]\+\)_langs.*#\1#p')
NUM_LANGS=${NUM_LANGS:-all}

RUN_NAME="moelpr_stage2_${NUM_LANGS}langs_${MOELPR_ORIGINAL_LANGUAGE}_$(date +%Y%m%d_%H%M%S)"
WANDB_TAGS="moelpr,stage2,accelerate"
WANDB_CONFIG_JSON=$(cat <<JSON
{
  "model_name_or_path": "${BASE_MODEL}",
  "tokenized_path": "${TOKENIZED_PATH}",
  "finetuning_type": "${FINETUNING_TYPE}",
  "dataset": "${DATASET_NAME}",
  "moelpr_stage": ${MOELPR_STAGE},
  "moelpr_num_experts": ${MOELPR_NUM_EXPERTS},
  "moelpr_top_k": ${MOELPR_TOP_K},
  "moelpr_lpr_loss_coef": ${MOELPR_LPR_LOSS_COEF},
  "moelpr_original_language": "${MOELPR_ORIGINAL_LANGUAGE}",
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

echo "[INFO] Starting Accelerate-backed MoE-LPR Stage 2 (router review) at $(date)"
accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  ./LLaMA-Factory/src/train.py \
  --stage sft \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps ${EVAL_STEPS} \
  --run_name "${RUN_NAME}" \
  --model_name_or_path "${BASE_MODEL}" \
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
  --moelpr_lpr_loss_coef ${MOELPR_LPR_LOSS_COEF} \
  --moelpr_original_language "${MOELPR_ORIGINAL_LANGUAGE}" \
  --adapter_name_or_path "${STAGE1_ADAPTER}" \
  --language_column "${LANGUAGE_COLUMN}" \
  --language_map "${LANGUAGE_MAP}"

echo "[INFO] Stage 2 MoE-LPR training completed at $(date)"
