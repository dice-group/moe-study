#!/usr/bin/env bash
set -e  # stop if any command fails

# repo + env name
REPO="https://github.com/NJUNLP/MoE-LPR.git"
ENV="moelpr"

# clone repo if not already done, create environment
[ -d MoE-LPR ] || git clone "$REPO"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV"; then
  conda env create -n "$ENV" -f MoE-LPR/environment.yml
  conda run -n "$ENV" pip install --editable MoE-LPR/peft/
  conda run -n "$ENV" pip install --editable MoE-LPR/transformers/
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
echo "Environment '$ENV' ready. Add your data prep commands here."

# Insert files into Moe-LPR
cp -rf scripts MoE-LPR/LLaMA-Factory/scripts/


##### Data Preparation, simplified for now ##### 
cp wiki_demo.txt MoE-LPR/LLaMA-Factory/scripts/data

##### Training ##### 
cd MoE-LPR/LLaMA-Factory
bash scripts/stage1.sh
