#!/usr/bin/env bash
# Submit with: sbatch --export=ALL,PANEL=v1_96 scripts/slurm/build_final_combined_top5_report.sh
# Valid panels: v1_96, v1_200, v2_96, v2_200
#SBATCH --job-name=final_top5_report
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
: "${PANEL:?PANEL is required (v1_96, v1_200, v2_96, or v2_200)}"

ROOT=/scratch/hpc-prf-merlin/joel/moe-study
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

args=(/scratch/hpc-prf-merlin/joel/conda/envs/merlin/bin/python \
  scripts/analysis/build_final_combined_top5_report.py \
  --panel "$PANEL" \
  --overwrite)
if [[ "${INTERIM:-false}" == "true" ]]; then
  args+=(--interim)
fi
if [[ -n "${GROUPS:-}" ]]; then
  args+=(--groups "$GROUPS")
fi
exec "${args[@]}"
