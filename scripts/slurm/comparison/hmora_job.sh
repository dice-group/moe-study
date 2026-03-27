#!/bin/bash
#SBATCH --job-name=cmp-hmora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=64G
#SBATCH --time=38:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/comparison/hmora_%j.log

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

export WANDB_PROJECT="${WANDB_PROJECT:-moe-study-comparison}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-moe-study}"
export WANDB_NAME="${WANDB_NAME:-hmora}"
export WANDB_TAGS="comparison,hmora"
export PYTHONUNBUFFERED=1
export LLAMAFACTORY_SAVE_ADAPTER_CHECKPOINTS="${LLAMAFACTORY_SAVE_ADAPTER_CHECKPOINTS:-1}"

OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR not set}"
mkdir -p "${OUTPUT_DIR}"

# Avoid NFS-backed caches (may cause slowdowns/hangs on cluster)
CACHE_ROOT="${CACHE_ROOT:-${TMPDIR:-/tmp}/${USER}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/.triton/autotune}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${XDG_CACHE_HOME}/torch_extensions}"
mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

DATASET_DIR="${DATASET_DIR:-./LLaMA-Factory/data}"
TOKENIZED_PATH="${GENERIC_TOKENIZED_PATH:?GENERIC_TOKENIZED_PATH not set}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH not set}"
DATASET_NAME="${GENERIC_DATASET_NAME:-c4}"
EVAL_DATASET_NAME="${GENERIC_EVAL_DATASET_NAME:-${DATASET_NAME}}"
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

TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
TRAIN_LR="${LEARNING_RATE:-2e-4}"
TRAIN_BS="${PER_DEVICE_TRAIN_BATCH_SIZE:-12}"
EVAL_BS="${PER_DEVICE_EVAL_BATCH_SIZE:-${TRAIN_BS}}"
GRAD_ACC="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_EXPERTS="${HMORA_NUM_EXPERTS:-8}"
TOP_K="${HMORA_TOP_K:-2}"
RUN_NAME="${WANDB_NAME:-hmora}_nexp${NUM_EXPERTS}_k${TOP_K}"
export WANDB_NAME="${RUN_NAME}"

AUTO_FIND_BATCH_SIZE_FLAG=()
if [[ "${AUTO_FIND_BATCH_SIZE}" == "true" || "${AUTO_FIND_BATCH_SIZE}" == "True" || "${AUTO_FIND_BATCH_SIZE}" == "1" ]]; then
  AUTO_FIND_BATCH_SIZE_FLAG=(--auto_find_batch_size)
fi

HMORA_TARGET_MODULES_LORA_ARG=()
if [[ -n "${HMORA_TARGET_MODULES_LORA:-}" ]]; then
  HMORA_TARGET_MODULES_LORA_ARG=(--hmora_target_modules_lora "${HMORA_TARGET_MODULES_LORA}")
fi

HMORA_TASK_TOKEN_ID_ARG=()
if [[ -n "${HMORA_TASK_TOKEN_ID:-}" ]]; then
  HMORA_TASK_TOKEN_ID_ARG=(--hmora_task_token_id "${HMORA_TASK_TOKEN_ID}")
fi

echo "[INFO] Running HMoRA training into ${OUTPUT_DIR}"

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
  --finetuning_type hmora \
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
  --logging_steps "${LOGGING_STEPS:-50}" \
  --logging_first_step "${LOGGING_FIRST_STEP:-True}" \
  --eval_strategy "${EVAL_STRATEGY:-steps}" \
  --eval_steps "${EVAL_STEPS:-2000}" \
  --save_steps "${SAVE_STEPS:-2000}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --save_safetensors "${SAVE_SAFETENSORS:-False}" \
  --save_only_model "${SAVE_ONLY_MODEL:-True}" \
  --seed "${SEED:-42}" \
  --disable_gradient_checkpointing "${DISABLE_GRADIENT_CHECKPOINTING:-True}" \
  --bf16 "${BF16:-True}" \
  --fp16 "${FP16:-False}" \
  --tokenized_path "${TOKENIZED_PATH}" \
  --lora_rank "${LORA_R:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --lora_target "${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj}" \
  --hmora_num_experts "${HMORA_NUM_EXPERTS:-8}" \
  --hmora_dropout "${HMORA_DROPOUT:-0.1}" \
  --hmora_top_k_routing_strategy "${HMORA_TOP_K_ROUTING_STRATEGY:-True}" \
  --hmora_top_k "${HMORA_TOP_K:-2}" \
  --hmora_use_task_router "${HMORA_USE_TASK_ROUTER:-True}" \
  --hmora_task_router_only "${HMORA_TASK_ROUTER_ONLY:-False}" \
  --hmora_share_router_for_qkv "${HMORA_SHARE_ROUTER_FOR_QKV:-True}" \
  --hmora_share_router_for_w_i "${HMORA_SHARE_ROUTER_FOR_W_I:-True}" \
  --hmora_num_router_mlp_layers "${HMORA_NUM_ROUTER_MLP_LAYERS:-1}" \
  --hmora_router_hidden_dim "${HMORA_ROUTER_HIDDEN_DIM:-32}" \
  --hmora_epsilon_alpha "${HMORA_EPSILON_ALPHA:-4.0}" \
  --hmora_alpha_shift "${HMORA_ALPHA_SHIFT:--2.0}" \
  --hmora_alpha_low_bound "${HMORA_ALPHA_LOW_BOUND:-0.2}" \
  --hmora_alpha_up_bound "${HMORA_ALPHA_UP_BOUND:-0.8}" \
  --hmora_use_load_balancing_loss "${HMORA_USE_LOAD_BALANCING_LOSS:-False}" \
  --hmora_use_div_loss "${HMORA_USE_DIV_LOSS:-True}" \
  --hmora_gamma_div_certain_t "${HMORA_GAMMA_DIV_CERTAIN_T:-0.4}" \
  --hmora_gamma_div_balance_t "${HMORA_GAMMA_DIV_BALANCE_T:-1.0}" \
  --hmora_gamma_div_certain_s "${HMORA_GAMMA_DIV_CERTAIN_S:-0.4}" \
  --hmora_gamma_div_balance_s "${HMORA_GAMMA_DIV_BALANCE_S:-1.0}" \
  --hmora_lambda_auxiliary "${HMORA_LAMBDA_AUXILIARY:-0.003}" \
  --hmora_lambda_lm "${HMORA_LAMBDA_LM:-1.0}" \
  --hmora_use_hydra_lora "${HMORA_USE_HYDRA_LORA:-False}" \
  --hmora_eta_b "${HMORA_ETA_B:-1.0}" \
  --hmora_use_language_ids_as_task_ids "${HMORA_USE_LANGUAGE_IDS_AS_TASK_IDS:-True}" \
  --hmora_task_token "${HMORA_TASK_TOKEN:-?}" \
  --hmora_num_encoder_layer "${HMORA_NUM_ENCODER_LAYER:-1}" \
  "${HMORA_TARGET_MODULES_LORA_ARG[@]}" \
  "${HMORA_TASK_TOKEN_ID_ARG[@]}" \
  "${AUTO_FIND_BATCH_SIZE_FLAG[@]}" \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
  --preprocessing_batch_size "${PREPROCESSING_BATCH_SIZE}" \
  --report_to wandb \
  --include_effective_tokens_per_second true \
  --include_num_input_tokens_seen true

echo "[INFO] HMoRA job completed"
