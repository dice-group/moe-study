#!/usr/bin/env bash
# RECOVERY_ROOT may be overridden through sbatch --export.
#SBATCH --job-name=merge_v2_baselines
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
ROOT=/scratch/hpc-prf-merlin/joel/moe-study
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
RECOVERY_ROOT="${RECOVERY_ROOT:-$ROOT/outputs/pipeline_runs/moe_study_v2_missing_baseline_recovery_20260725}"

exec /scratch/hpc-prf-merlin/joel/conda/envs/merlin/bin/python \
  scripts/analysis/consolidate_missing_v2_baselines.py \
  --recovery-root "$RECOVERY_ROOT"
