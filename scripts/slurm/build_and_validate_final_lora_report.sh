#!/usr/bin/env bash
# PROFILE is supplied by sbatch.
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
: "${PROFILE:?PROFILE is required}"

ROOT=/scratch/hpc-prf-merlin/joel/moe-study
PYTHON=/scratch/hpc-prf-merlin/joel/conda/envs/merlin/bin/python

module purge
module load toolchain/foss/2024a system/CUDA/12.6.0
set +u
source /opt/software/pc2/EB-SW/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate merlin
set -u
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/LLaMA-Factory/src:${PYTHONPATH:-}"
export WANDB_MODE=disabled

"$PYTHON" -u -m pipeline.run \
  --config configs/pipeline \
  --profile "$PROFILE" \
  --start-at summary \
  --stop-after report \
  --force-stage summary \
  --force-stage report \
  --executor-type local

exec "$PYTHON" -u scripts/analysis/validate_final_lora_submission.py \
  --profile "$PROFILE" --validate-results
