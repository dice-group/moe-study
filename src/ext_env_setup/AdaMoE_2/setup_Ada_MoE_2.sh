#!/usr/bin/env bash
set -e

REPO="https://github.com/hiyouga/LLaMA-Factory.git"
[ -d LLaMA-Factory ] || { \
  git clone "$REPO" LLaMA-Factory && \
  cd LLaMA-Factory && \
  git checkout 4ba7de0; \
}

# create environment if not already done
ENV="adamole"
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV"; then
  conda env create -n "$ENV" -f llamaf.yml
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
echo " \n ##### Environment '$ENV' ready ##### \n"

# Insert files into LLaMA-Factory
cp adaptive_moe_utils.py LLaMA-Factory/src/llamafactory/model
cp loader.py LLaMA-Factory/src/llamafactory/model

##### Data Preparation, simplified for now ##### 
cp ../MoE-LPR/wiki_demo.txt LLaMA-Factory/scripts/data

##### Optional, directly train automatically ##### 
bash train_Ada_MoE_2.sh