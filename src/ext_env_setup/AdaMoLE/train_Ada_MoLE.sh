#!/usr/bin/env bash
set -e

cd AdaMoLE
export CUDA_VISIBLE_DEVICES=1
echo "$PWD"

python train.py @configs/llama_adamole_csr_train.config