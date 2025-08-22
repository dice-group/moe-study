#!/usr/bin/env bash
set -e

# repo + env name
REPO="https://github.com/NJUNLP/MoE-LPR.git"
ENV="moelpr"

# clone repo if not already done, create environment
[ -d MoE-LPR ] || git clone "$REPO"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV"; then
  conda env create -n "$ENV" -f moe-lpr.yml
  conda run -n "$ENV" pip install --editable MoE-LPR/peft/
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
echo " \n ##### Environment '$ENV' ready ##### \n"

# Insert files into Moe-LPR
cp -rf scripts MoE-LPR/LLaMA-Factory/scripts/


##### Data Preparation, simplified for now ##### 
cp wiki_demo.txt MoE-LPR/LLaMA-Factory/scripts/data

##### Optional, directly train automatically ##### 
bash train_MoE_LPR_stage1.sh