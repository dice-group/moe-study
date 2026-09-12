#!/usr/bin/env bash
#SBATCH --job-name=eng_consolidate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
ROOT=/scratch/hpc-prf-merlin/joel/moe-study
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec /scratch/hpc-prf-merlin/joel/conda/envs/merlin/bin/python \
  scripts/analysis/consolidate_final_english_evaluations.py
