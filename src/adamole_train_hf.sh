#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export LOG_LEVEL=DEBUG

python -m llamafactory_moe_study.train_hf \
  --config /upb/users/j/joeldag/profiles/unix/cs/MERLIN/moe-study/src/llamafactory_moe_study/config/train_adamole_hf.yaml \
  --use_method adamole \
  --lora_r 4 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_targets "q_proj,k_proj,v_proj,o_proj" \
  --adamole_num_experts 8 \
  --adamole_max_threshold 0.25 \
  --adamole_aux_coef 0.001 \
  --gradient_checkpointing \
  |& tee output.log
