#!/bin/bash
set -e

python -m llamafactory_moe_study.train_hf \
  --config /upb/users/j/joeldag/profiles/unix/cs/MERLIN/moe-study/src/llamafactory_moe_study/config/train_mola.yaml \
  --use_method adamoe_mola \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_targets "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
  --mola_num_experts 2 \
  --mola_top_k 2
