#!/bin/bash
#SBATCH --job-name=cmp-watch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1-00:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${MODULE_INIT:-}" ]]; then
  eval "${MODULE_INIT}"
fi

if [[ -n "${CONDA_BASE:-}" && -n "${CONDA_ENV:-}" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

python3 "${SCRIPT_DIR}/watch_checkpoints.py" \
  --watch-dir "${WATCH_DIR:?WATCH_DIR not set}" \
  --label "${WATCH_LABEL:?WATCH_LABEL not set}" \
  --eval-script "${EVAL_SCRIPT:?EVAL_SCRIPT not set}" \
  --eval-log-dir "${EVAL_LOG_DIR:?EVAL_LOG_DIR not set}" \
  --tokenizer-path "${TOKENIZER_PATH:?TOKENIZER_PATH not set}" \
  --wandb-project "${WANDB_PROJECT:-}" \
  --wandb-entity "${WANDB_ENTITY:-}" \
  --wandb-group "${WANDB_GROUP:-}" \
  --wandb-name-prefix "${WATCH_LABEL:?WATCH_LABEL not set}" \
  --tasks "${LM_EVAL_TASKS:-belebele}" \
  --batch-size "${LM_EVAL_BATCH_SIZE:-auto}" \
  --poll-interval "${LM_EVAL_POLL_INTERVAL:-120}" \
  --idle-timeout "${LM_EVAL_IDLE_TIMEOUT:-60}" \
  --max-duration "${LM_EVAL_MAX_DURATION:-1440}" \
  --results-dir "${LM_EVAL_RESULTS_DIR:-${WATCH_DIR}/lm_eval_results}" \
  --eval-partition "${EVAL_PARTITION:-}" \
  --eval-time "${EVAL_TIME:-}" \
  --eval-gpus "${EVAL_GPUS:-1}" \
  --eval-extra-sbatch "${EVAL_EXTRA_SBATCH:-}"
