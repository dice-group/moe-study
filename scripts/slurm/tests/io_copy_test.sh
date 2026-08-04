#!/bin/bash
#SBATCH --job-name=io-copy-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/io_copy_test_%j.log

set -euo pipefail

REPO_ROOT="/cluster/user/moe-study"
export SLURM_CPU_BIND="${SLURM_CPU_BIND:-none}"
export SLURM_CPU_BIND_VERBOSE="${SLURM_CPU_BIND_VERBOSE:-0}"

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

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/scripts/slurm/tests/io_copy_test_runs}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.2-1B}"
TOKENIZED_BASE="${TOKENIZED_BASE:-/scratch/anonymous-project/project_data/moe_study/tokenized/fw_l2v_samples_moe_study_comparison}"
TOKENIZER_NAME="${TOKENIZER_NAME:-llama-3.2-1B_tokenizer}"
DATASET_NAME="${DATASET_NAME:-200_langs}"
TOKENIZED_SRC="${TOKENIZED_SRC:-${TOKENIZED_BASE}/${TOKENIZER_NAME}/${DATASET_NAME}}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/LLaMA-Factory/data}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/LLaMA-Factory/examples/accelerate/fsdp_4gpu_config.yaml}"

MAX_STEPS="${MAX_STEPS:-200}"
TRAIN_BS="${PER_DEVICE_TRAIN_BATCH_SIZE:-21}"
GRAD_ACC="${GRADIENT_ACCUMULATION_STEPS:-2}"
LR="${LEARNING_RATE:-2e-4}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
RESIZE_VOCAB="${RESIZE_VOCAB:-False}"
SAVE_STEPS="${SAVE_STEPS:-20000}"
COLA_NUM_A="${COLA_NUM_A:-1}"
COLA_NUM_B="${COLA_NUM_B:-3}"
COLA_STRATEGY="${COLA_STRATEGY:-fully}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${REPO_ROOT}/LLaMA-Factory/src/train.py"

mkdir -p "${OUTPUT_DIR}"
if [[ ! -d "${TOKENIZED_SRC}" ]]; then
  echo "[ERROR] TOKENIZED_SRC not found: ${TOKENIZED_SRC}" >&2
  exit 1
fi

run_train() {
  local label="$1"
  local tokenized_path="$2"
  local run_dir="${OUTPUT_DIR}/${label}"
  mkdir -p "${run_dir}"
  echo "[INFO] Starting ${label} with tokenized_path=${tokenized_path}"
  accelerate launch --config_file "${ACCELERATE_CONFIG}" "${TRAIN_SCRIPT}" \
    --stage sft \
    --do_train \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --flash_attn fa2 \
    --dataset "${DATASET_NAME}" \
    --dataset_dir "${DATASET_DIR}" \
    --template llama3 \
    --finetuning_type cola \
    --resize_vocab "${RESIZE_VOCAB}" \
    --output_dir "${run_dir}" \
    --overwrite_output_dir \
    --max_steps "${MAX_STEPS}" \
    --learning_rate "${LR}" \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --per_device_train_batch_size "${TRAIN_BS}" \
    --per_device_eval_batch_size "${TRAIN_BS}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --logging_steps "${LOGGING_STEPS}" \
    --eval_strategy no \
    --save_steps "${SAVE_STEPS}" \
    --disable_gradient_checkpointing True \
    --report_to none \
    --bf16 True \
    --fp16 False \
    --tokenized_path "${tokenized_path}" \
    --lora_rank "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout 0.05 \
    --lora_target q_proj,k_proj,v_proj,o_proj \
    --num_A "${COLA_NUM_A}" \
    --num_B "${COLA_NUM_B}" \
    --cola_strategy "${COLA_STRATEGY}"
}

start_base=$(date +%s)
run_train "baseline" "${TOKENIZED_SRC}"
end_base=$(date +%s)
echo "[INFO] baseline runtime: $((end_base - start_base))s"

LOCAL_ROOT="${LOCAL_TOKENIZED_ROOT:-${SLURM_TMPDIR:-/tmp}/${USER}/tokenized_copy}"
LOCAL_PATH="${LOCAL_ROOT}/$(basename "${TOKENIZED_SRC}")"
mkdir -p "${LOCAL_ROOT}"
echo "[INFO] Copying tokenized data to ${LOCAL_PATH}"
copy_start=$(date +%s)
rm -rf "${LOCAL_PATH}"
rsync -aL --info=stats2 "${TOKENIZED_SRC}/" "${LOCAL_PATH}/"
copy_end=$(date +%s)
echo "[INFO] copy runtime: $((copy_end - copy_start))s"

if find "${LOCAL_PATH}" -name "*.arrow" -print -quit | grep -q .; then
  start_local=$(date +%s)
  run_train "local_copy" "${LOCAL_PATH}"
  end_local=$(date +%s)
  echo "[INFO] local_copy runtime: $((end_local - start_local))s"
else
  echo "[WARN] Local copy missing arrow shards; skipping local_copy run."
fi

echo "[INFO] Done. Compare effective_tokens_per_second in logs for baseline vs local_copy."
