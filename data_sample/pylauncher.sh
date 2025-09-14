#!/bin/bash
set -eu

PY_MODULE=$2
MODE=$1

# Determine the directory of the current script
CUR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # This gets overriden on the next 'source' call

SETUP_DIR=$CUR_SCRIPT_DIR

source $CUR_SCRIPT_DIR/env.sh

CUR_VENV_NAME=data_sample_venv
CUR_VENV_DIR=$PROJECT_ROOT/src/virt-envs/$CUR_VENV_NAME

source $(conda info --base)/etc/profile.d/conda.sh

# activating custom enviroment
conda activate $CUR_VENV_DIR

# Capture any additional CLI arguments
EXTRA_ARGS="${@:3}"

if [ "$MODE" == "debug" ]; then
    echo "Debug mode enabled, waiting for remote debugger to attach.."
    python -u -m debugpy --wait-for-client --listen 0.0.0.0:12121 -m $PY_MODULE $EXTRA_ARGS
else
    python -m $PY_MODULE $EXTRA_ARGS
fi