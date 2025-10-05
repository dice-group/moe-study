#!/bin/bash

# refines the router using a small sample of the original language data.
# teaches the router to send original language tokens to Expert 0.
#Ensure the model_name_or_path in the config points to the output of your Stage 1 run.

export CUDA_VISIBLE_DEVICES=0

python -m src.train_hf \
    --config configs/train_moelpr_stage2_example.yaml \
    --use_method moelpr \
    --moelpr_stage 2 \
    --moelpr_num_experts 4 \
    --moelpr_top_k 1 \
    --moelpr_lpr_loss_coef 0.1 \
    --moelpr_layers_to_transform "0,1,2,3"
