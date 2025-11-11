#!/bin/bash
set -euo pipefail

#LANGS=(
#  lao_Laoo ory_Orya tel_Telu asm_Beng
#  english rus_Cyrl fin_Latn spa_Latn
#  hin_Deva kat_Geor slv_Latn srp_Cyrl
#)
LANGS=(lao_Laoo)

MODEL_NAME=${MODEL_NAME:-meta-llama/Llama-3.1-8B}
EXTENDED_TOKENIZER=${EXTENDED_TOKENIZER:-/scratch/hpc-prf-merlin/project_data/moe_study/trained_multilingual_tokenizers/256k_vocab/llama-3.1-8B/}
DATASET_DIR=${DATASET_DIR:-./loss_analysis_dataset}
FALLBACK_DATASET=${FALLBACK_DATASET:-$DATASET_DIR/english.jsonl.gz}
OUTPUT_ROOT=${OUTPUT_ROOT:-./loss_analysis_output}
WANDB_PROJECT=${WANDB_PROJECT:-multilingual_tokenizer_perplexity_eval}
LM_EVAL_PROJECT=${LM_EVAL_PROJECT:-multilingual_tokenizer_lm_eval}
LOG_DIR=${LOG_DIR:-./logs}
mkdir -p "$LOG_DIR"

dataset_for_lang() {
  local lang="$1"
  local gz_path="$DATASET_DIR/${lang}.jsonl.gz"
  if [[ -f "$gz_path" ]]; then
    echo "$gz_path"
    return
  fi
  if [[ -f "$FALLBACK_DATASET" ]]; then
    echo "$FALLBACK_DATASET"
    return
  fi
  echo "Missing gzip dataset for ${lang} (expected ${gz_path}) and fallback ${FALLBACK_DATASET}." >&2
  exit 1
}

eval_task_for_lang() {
  local lang="$1"
  case "$lang" in
    english) echo "belebele_eng_Latn" ;;
    *) echo "belebele_${lang}" ;;
  esac
}

submit_job() {
  local lang="$1"
  local tokenizer_label="$2" # base or extended
  local tokenizer_path="$3"
  local dataset_path_gz
  dataset_path_gz=$(dataset_for_lang "$lang")
  local eval_task
  eval_task=$(eval_task_for_lang "$lang")
  local job_name="loss-${lang//[^a-zA-Z0-9]/}-${tokenizer_label}"
  local output_dir="${OUTPUT_ROOT}/${lang}/${tokenizer_label}"
  local log_file="${LOG_DIR}/${job_name}-%j.out"
  local submission
  local job_id

  submission=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name:0:63}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=5:00:00
#SBATCH --output=$log_file
#SBATCH --partition=gpu
set -euo pipefail

export WANDB_PROJECT="$WANDB_PROJECT"
export TOKENIZERS_PARALLELISM=false

LANG="$lang"
TOKENIZER_LABEL="$tokenizer_label"
TOKENIZER_PATH="$tokenizer_path"
DATASET_PATH_GZ="$dataset_path_gz"
EVAL_TASK="$eval_task"
OUTPUT_DIR="$output_dir"
RUN_NAME="loss-\${LANG}-\${TOKENIZER_LABEL}-\${SLURM_JOB_ID}"
mkdir -p "\$OUTPUT_DIR"

TMP_NAME="\$(basename "\${DATASET_PATH_GZ%.gz}")"
DATASET_PATH="\$OUTPUT_DIR/\$TMP_NAME"
echo "Decompressing \$DATASET_PATH_GZ → \$DATASET_PATH"
gzip -cd "\$DATASET_PATH_GZ" > "\$DATASET_PATH"

EXTRA_TOKENIZER_ARGS=()
LM_EVAL_TOKENIZER_ARGS=()
if [[ -n "\$TOKENIZER_PATH" ]]; then
  EXTRA_TOKENIZER_ARGS+=(--tokenizer_path "\$TOKENIZER_PATH")
  LM_EVAL_TOKENIZER_ARGS+=(--tokenizer-path "\$TOKENIZER_PATH")
fi

python train_base_hf.py \
  --model_name "$MODEL_NAME" \
  --dataset_path "\$DATASET_PATH" \
  --output_dir "\$OUTPUT_DIR" \
  --run_name "\$RUN_NAME" \
  --per_device_batch_size 2 \
  --grad_accum_steps 1 \
  --max_steps 10000 \
  --save_steps 2000 \
  --enable_lm_eval \
  --lm_eval_interval 2000 \
  --lm_eval_tasks "\$EVAL_TASK" \
  --lm_eval_batch_size 2 \
  --lm_eval_limit 500 \
  --lm_eval_cuda_devices "0" \
  --lm_eval_wandb_project "$LM_EVAL_PROJECT" \
  "\${EXTRA_TOKENIZER_ARGS[@]}"

./lm_eval.sh \
  --model-dir "\$OUTPUT_DIR" \
  --tasks "\$EVAL_TASK" \
  --cuda-devices "0" \
  --wandb-project "$LM_EVAL_PROJECT" \
  --batch-size 2 \
  --limit 500 \
  "\${LM_EVAL_TOKENIZER_ARGS[@]}"
EOF
)

  job_id=$(echo "$submission" | awk '{print $4}')
  echo "Submitted $lang/$tokenizer_label (dataset: $dataset_path_gz, eval: $eval_task) → job $job_id"
}

for lang in "${LANGS[@]}"; do
  submit_job "$lang" base ""
  submit_job "$lang" extended "$EXTENDED_TOKENIZER"
done
