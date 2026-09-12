#!/usr/bin/env bash
# RUN_ID, SHARD_ROOT, BENCHMARK_ROOT, SELECTED_TASKS_FILE, and CHECKPOINT are required.
#SBATCH --job-name=consolidate_final_bench
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
: "${RUN_ID:?RUN_ID is required}"
: "${SHARD_ROOT:?SHARD_ROOT is required}"
: "${BENCHMARK_ROOT:?BENCHMARK_ROOT is required}"
: "${SELECTED_TASKS_FILE:?SELECTED_TASKS_FILE is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
PANEL="${PANEL:-v1_200}"

ROOT=/scratch/hpc-prf-merlin/joel/moe-study
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

args=(/scratch/hpc-prf-merlin/joel/conda/envs/merlin/bin/python \
  scripts/analysis/consolidate_final_benchmark_shards.py \
  --run-id "$RUN_ID" \
  --shard-root "$SHARD_ROOT" \
  --benchmark-root "$BENCHMARK_ROOT" \
  --selected-tasks-file "$SELECTED_TASKS_FILE" \
  --checkpoint "$CHECKPOINT" \
  --panel "$PANEL")
if [[ -n "${BELEBELE_SUBSHARDS:-}" ]]; then
  args+=(--belebele-subshards "${BELEBELE_SUBSHARDS//+/,}")
fi
exec "${args[@]}"
