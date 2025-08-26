#!/usr/bin/env bash
set -e

REPO="https://github.com/zefang-liu/AdaMoLE.git"
[ -d MoLA ] || { \
  git clone "$REPO" AdaMoLE && \
  cd AdaMoLE && \
  git checkout 5e8599a7ad6fafde03aaad8466f583f5f28c22ac; \
}

# create environment if not already done
ENV="adamole"
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV"; then
  conda create -n "$ENV" python=3.10 -y
  conda activate "$ENV"
  pip install -r requirements.txt
fi


source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
echo " \n ##### Environment '$ENV' ready ##### \n"


##### Data Preparation, simplified for now ##### 
cp ../MoE-LPR/wiki_demo.txt AdaMoLE
# Tbd

##### Optional, directly train automatically ##### 
bash train_Ada_MoLE.sh