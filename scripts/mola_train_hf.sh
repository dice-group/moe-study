#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export LOG_LEVEL=DEBUG

#Paper setup:
# num_experts (true experts) = 4
# lora rank is 8
# they trained on 3 seeds and averaged results TODO?

python src/train_hf.py \
  --config /anonymous-institution/users/j/anonymous-user/profiles/unix/cs/MERLIN/moe-study/configs/train_mola.yaml \
  --use_method mola \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_targets "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj" \
  --mola_num_experts 4 \
  --mola_top_k 2 \
  --mola_use_null_expert \
  --eval_strategy steps \
  --eval_steps 500 \
  --do_harness_eval \
  --logging_steps 1 \
  --save_steps 10 \
  --harness_tasks winogrande |& tee output.log
