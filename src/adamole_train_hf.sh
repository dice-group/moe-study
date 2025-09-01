#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

python -m llamafactory_moe_study.train_hf \
  --config /upb/users/j/joeldag/profiles/unix/cs/MERLIN/moe-study/src/llamafactory_moe_study/config/train_adamole_hf.yaml \
  --use_method adamole \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_targets "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj" \
  --adamole_num_experts 4 \
  --adamole_max_threshold 0.25 \
  --adamole_aux_coef 0.01 \
  --gradient_checkpointing \