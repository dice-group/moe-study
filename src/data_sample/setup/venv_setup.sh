#!/bin/bash
set -eu
# Determine the directory of the current script
CUR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # This gets overriden on the next 'source' call

SETUP_DIR=$CUR_SCRIPT_DIR

source $CUR_SCRIPT_DIR/env.sh

CUR_VENV_NAME=data_sample_venv
CUR_VENV_DIR=$PROJECT_ROOT/src/virt-envs/$CUR_VENV_NAME

source $(conda info --base)/etc/profile.d/conda.sh

if [[ ! -d "$CUR_VENV_DIR" ]]; then
  conda create -y --prefix "$CUR_VENV_DIR" python=3.12
else
  echo "Conda environment already exists at $CUR_VENV_DIR – skipping creation."
fi
# deactivating default environment
conda deactivate
# activating custom enviroment
conda activate $CUR_VENV_DIR

echo $(python --version)

echo $(pip --version)

pip install -r $SETUP_DIR"/requirements.txt"

ipython kernel install --user --name=$CUR_VENV_NAME