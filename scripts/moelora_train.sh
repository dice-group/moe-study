#!/bin/bash
# MoE-LoRA Training Script
# Paper: "When MOE Meets LLMs: Parameter Efficient Fine-tuning for Multi-task Medical Applications"
#        (Liu et al., SIGIR 2024)
#
# Adapted for massively multilingual SFT with language-ID routing.
# Language IDs replace task IDs; the gate embedding maps language -> expert weights.
#
# Paper defaults: 8 experts, rank 16 (2 per expert), dense softmax gate.
# For sparse gate (paper: top-K=2 is optimal), set --moelora_sparse_top_k 2.
set -e

export CUDA_VISIBLE_DEVICES=0
export LOG_LEVEL=DEBUG

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python "${REPO_ROOT}/LLaMA-Factory/src/train.py" \
  "${REPO_ROOT}/configs/train_moelora.yaml" \
  --finetuning_type moelora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj" \
  --moelora_num_experts 8 \
  --moelora_task_num 1 \
  --moelora_task_embedding_dim 64 \
  --moelora_sparse_top_k 0 \
  --moelora_gate_temperature 1.0 \
  --language_column "language" \
  --language_map "${REPO_ROOT}/configs/moelpr_lang_map.json" \
  --gradient_checkpointing true \
  --report_to wandb \
  --logging_steps 1 \
  --save_steps 500 \
  2>&1 | tee "${REPO_ROOT}/outputs/moelora_train.log"
