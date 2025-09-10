#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export LOG_LEVEL=DEBUG

python -m llamafactory_moe_study.train_hf \
  --config /upb/users/j/joeldag/profiles/unix/cs/MERLIN/moe-study/src/llamafactory_moe_study/config/train_mola.yaml \
  --use_method mola \
  --lora_r 4 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_targets "q_proj,k_proj,v_proj,o_proj" \
  --mola_num_experts 4 \
  --mola_top_k 2 \
  --mola_debug_mode \
  --mola_use_null_exper |& tee output.log
