#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export LOG_LEVEL=DEBUG

echo "Starting training job"
python src/train_hf.py \
  --config /upb/users/j/joeldag/profiles/unix/cs/MERLIN/moe-study/configs/train_adamole_hf.yaml \
  --tokenizer /data/MERLIN/multilingual_tokenizer/250K_vocab_size/trained_multilingual_tokenizer_llama-3.2-1B \
  --tokenized_dataset_dir /data/MERLIN/tokenized_test_data \
  --use_method adamole \
  --lora_r 4 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_targets "q_proj,k_proj,v_proj,o_proj" \
  --adamole_num_experts 8 \
  --adamole_max_threshold 0.25 \
  --adamole_aux_coef 0.001 \
  --gradient_checkpointing \
  --do_harness_eval
  --harness_tasks belebele hellaswag arc xnli \
  --wandb_project moe-study |& tee output.log
echo "Starting training job"