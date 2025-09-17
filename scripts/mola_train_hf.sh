#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export LOG_LEVEL=DEBUG

python src/train_hf.py \
  --config /upb/users/j/joeldag/profiles/unix/cs/MERLIN/moe-study/configs/train_mola.yaml \
  --use_method mola \
  --lora_r 4 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_targets "q_proj,k_proj,v_proj,o_proj" \
  --mola_num_experts 4 \
  --mola_top_k 2 \
  --mola_use_null_exper \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --do_harness_eval \
  --harness_tasks winogrande |& tee output.log
