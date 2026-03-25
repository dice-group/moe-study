#!/bin/bash
# MixLoRA training script
# Paper: "MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts"
#
# Attention projections use standard LoRA.
# FFN projections use top-k routed LoRA experts with an auxiliary router loss.
set -e

export CUDA_VISIBLE_DEVICES=0
export LOG_LEVEL=DEBUG

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python "${REPO_ROOT}/LLaMA-Factory/src/train.py" \
  "${REPO_ROOT}/configs/train_mixlora.yaml" \
  --finetuning_type mixlora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target "q_proj,k_proj,v_proj,o_proj" \
  --mixlora_num_experts 8 \
  --mixlora_top_k 2 \
  --mixlora_router_init_range 0.02 \
  --mixlora_jitter_noise 0.0 \
  --mixlora_router_loss true \
  --mixlora_router_aux_loss_coef 0.001 \
  --mixlora_moe_target_modules "gate_proj,down_proj,up_proj" \
  --gradient_checkpointing true \
  --report_to wandb \
  --logging_steps 1 \
  --save_steps 500 \
  2>&1 | tee "${REPO_ROOT}/outputs/mixlora_train.log"
