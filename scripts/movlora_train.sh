#!/bin/bash
# MoV-LoRA Training Script
# Paper: "Pushing Mixture of Experts to the Limit: Extremely Parameter Efficient MoE
#         for Instruction Tuning" (Zadouri et al., ICLR 2024)
#
# MoV-LoRA uses per-layer learned routers for token-level expert selection.
# Unlike MoE-LoRA, no task/language IDs are needed — routing is learned end-to-end.
# Each expert has full rank r (NOT r/N as in MoE-LoRA).
#
# Paper defaults: 8 experts, soft routing (all experts contribute), lr=3e-4.
# For sparse routing, set --movlora_top_k 2 (paper finds soft > sparse).
set -e

export CUDA_VISIBLE_DEVICES=0
export LOG_LEVEL=DEBUG

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python "${REPO_ROOT}/LLaMA-Factory/src/train.py" \
  "${REPO_ROOT}/configs/train_movlora.yaml" \
  --finetuning_type movlora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj" \
  --movlora_num_experts 8 \
  --movlora_top_k 0 \
  --movlora_router_temperature 1.0 \
  --movlora_router_jitter_noise 0.0 \
  --movlora_router_bias false \
  --movlora_router_init_std 0.02 \
  --movlora_router_ignore_padding_tokens false \
  --gradient_checkpointing true \
  --report_to wandb \
  --logging_steps 1 \
  --save_steps 500 \
  2>&1 | tee "${REPO_ROOT}/outputs/movlora_train.log"
