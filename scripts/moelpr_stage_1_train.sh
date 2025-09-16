#!/bin/bash

# MoE-LPR Stage 1: Post-pretraining
# This stage trains the new experts and the router on the (new) target languages.
# The original model parameters (Expert 0) are frozen. Router is biased after this stage, since it hasnt seen old langs

export CUDA_VISIBLE_DEVICES=0

python -m src.train_hf \
    --config configs/train_moelpr_stage1_example.yaml \
    --use_method moelpr \
    --moelpr_stage 1 \
    --moelpr_num_experts 6 \
    --moelpr_top_k 1 \
    --moelpr_aux_loss_coef 0.01 \
    --moelpr_debug_mode \
    --moelpr_layers_to_transform "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
