#!/usr/bin/env bash
# RUN_ROOT, RUN_ID, SHARD_ROOT, BENCHMARK_ROOT, TASKS_FILE, and PANEL are required.
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${SHARD_ROOT:?SHARD_ROOT is required}"
: "${BENCHMARK_ROOT:?BENCHMARK_ROOT is required}"
: "${TASKS_FILE:?TASKS_FILE is required}"
: "${PANEL:?PANEL is required}"

ROOT=/scratch/hpc-prf-merlin/joel/moe-study
PYTHON=/scratch/hpc-prf-merlin/joel/conda/envs/merlin/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT="$("$PYTHON" scripts/analysis/resolve_direct_adapter_checkpoint.py \
  --run-root "$RUN_ROOT" --run-id "$RUN_ID")"

args=("$PYTHON" -u scripts/analysis/consolidate_final_benchmark_shards.py \
  --run-id "$RUN_ID" \
  --shard-root "$SHARD_ROOT" \
  --benchmark-root "$BENCHMARK_ROOT" \
  --selected-tasks-file "$TASKS_FILE" \
  --checkpoint "$CHECKPOINT" \
  --panel "$PANEL")
if [[ -n "${BELEBELE_SUBSHARDS:-}" ]]; then
  args+=(--belebele-subshards "${BELEBELE_SUBSHARDS//+/,}")
fi
exec "${args[@]}"
