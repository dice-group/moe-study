#!/bin/bash
# Run with srun inside an existing one-node, four-A100 salloc allocation.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
runtime="$PWD/configs/cross_arch_study/noctua2_runtime.json"
module purge
mapfile -t runtime_modules < <(jq -er '.modules[]' "$runtime")
module load "${runtime_modules[@]}"
source "$(jq -er .conda_base "$runtime")/etc/profile.d/conda.sh"
export PROOF_PIPELINE_PYTHON=$(jq -er .python "$runtime")
conda activate "$(dirname "$(dirname "$PROOF_PIPELINE_PYTHON")")"
export PYTHONPATH="$PWD/configs/cross_arch_study/python_startup:$PWD:$PWD/LLaMA-Factory/src:$PWD/scripts/eval/latest_checkpoint/lm-evaluation-harness"
if [[ -f "/dev/shm/moe-study-runtime-${SLURM_JOB_ID}/libtorch_python.so" ]]; then
    export MOE_STUDY_LOCAL_TORCH_LIBS="/dev/shm/moe-study-runtime-${SLURM_JOB_ID}"
    export LD_LIBRARY_PATH="$MOE_STUDY_LOCAL_TORCH_LIBS:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
if [[ -f "/dev/shm/moe-study-python-${SLURM_JOB_ID}/transformers/__init__.py" ]]; then
    export MOE_STUDY_LOCAL_PYTHON="/dev/shm/moe-study-python-${SLURM_JOB_ID}"
fi
export MOE_STUDY_DEBUG_STACKS=1
export HF_HOME=$(jq -er .hf_home "$runtime")
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PROOF_PIPELINE_PRESERVE_ENV=1
export PROOF_PIPELINE_REPO_ROOT="$PWD"
export PYTHONUNBUFFERED=1
if [[ "${1:-}" == --verify-only ]]; then
    shift
    exec "$PROOF_PIPELINE_PYTHON" -m pipeline_tools.cross_arch_debug_verify "$@"
fi
exec "$PROOF_PIPELINE_PYTHON" -m pipeline_tools.cross_arch_debug "$@"
