#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=1

cd MoLA
echo "$PWD"

python mola_training.py \
          --base_model "unsloth/llama-2-7b" \
          --data_path "./scienceqa/science_qa.hf" \
          --output_dir "./scienceqa_mola" \
          --batch_size 8 \
          --micro_batch_size 2 \
          --num_epochs 5 \
          --learning_rate 3e-4 \
          --cutoff_len 256 \
          --val_set_size 1 \
          --lora_r "8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8" \
          --lora_alpha 16 \
          --lora_dropout 0.05 \
          --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
          --number_experts "4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4" \
          --top_k "4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4" \
          --train_on_inputs \
          --group_by_length \
          --add_eos_token