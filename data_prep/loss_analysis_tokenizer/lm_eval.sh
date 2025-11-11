#!/bin/bash
export WANDB_PROJECT="multilingual_tokenizer_perplexity_eval_belebele_lao_Laoo"
export CUDA_VISIBLE_DEVICES=0

MODEL_DIR="loss_analysis_output/base_lao_Laoo"
EVAL_NAME="belebele_lao_Laoo"

for CKPT in "$MODEL_DIR"/checkpoint-*; do
  echo "Evaluating $(basename $CKPT) ..."
  lm_eval \
    --model hf \
    --model_args "pretrained=$CKPT,tokenizer=$CKPT" \
    --tasks $EVAL_NAME \
    --device cuda \
    --batch_size 1 \
    --output_path "$CKPT/${EVAL_NAME}_results.json"
done
