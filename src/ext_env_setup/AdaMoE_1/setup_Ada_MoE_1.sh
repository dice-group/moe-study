#!/usr/bin/env bash
set -e

REPO="https://github.com/GCYZSL/MoLA.git"
[ -d MoLA ] || { \
  git clone "$REPO" MoLA && \
  cd MoLA && \
  git checkout 01785c84c70846999638b11b88409fdf86361d9a; \
}

# create environment if not already done
ENV="mola"
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV"; then
  conda env create -n "$ENV" -f llamaf.yml
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
echo " \n ##### Environment '$ENV' ready ##### \n"

# copy modified scripts of authors into MOLA
cp mola_lora_hacked.py MoLA/src/
cp mola_modeling_llama_hacked.py MoLA/src/

##### Data Preparation, simplified for now ##### 
cp ../MoE-LPR/wiki_demo.txt MoLA

##### Optional, directly train automatically ##### 
bash train_Ada_MoE_1.sh