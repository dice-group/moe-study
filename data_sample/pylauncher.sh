#!/bin/bash
set -eu

echo Current arguments "${@}"

PY_MODULE=$2
MODE=$1

# Remove the first two arguments so "$@" now contains only the
# user‑provided options for the Python module.
shift 2

# Skip conda activation for "nested" mode
if [ "$MODE" != "nested" ]; then
    # Reference: https://stackoverflow.com/questions/56962129/how-to-get-original-location-of-script-used-for-slurm-job
    # Determine the directory of the current script
    if [ -n "${SLURM_JOB_ID:-}" ];  then
        # check the original location through scontrol and $SLURM_JOB_ID
        CUR_SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
    else
        # otherwise: started with bash. Get the real location.
        CUR_SCRIPT_PATH=$(realpath $0)
    fi

    # Get only the directory component
    CUR_SCRIPT_DIR=$(dirname "$CUR_SCRIPT_PATH")

    SETUP_DIR=$CUR_SCRIPT_DIR/setup

    source $SETUP_DIR/env.sh

    CUR_VENV_NAME=data_sample_venv
    CUR_VENV_DIR=$PROJECT_ROOT/src/virt-envs/$CUR_VENV_NAME

    source $(conda info --base)/etc/profile.d/conda.sh

    # activating custom enviroment
    conda activate $CUR_VENV_DIR
fi


echo Current venv: $CONDA_PREFIX
which python

if [ "$MODE" == "debug" ]; then
    echo "Debug mode enabled, waiting for remote debugger to attach.."
    python -u -m debugpy --wait-for-client --listen 0.0.0.0:12121 -m "$PY_MODULE" "$@"
else
    python -m "$PY_MODULE" "$@"
fi