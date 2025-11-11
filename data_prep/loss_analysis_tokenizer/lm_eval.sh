#!/bin/bash
export WANDB_PROJECT="multilingual_tokenizer_perplexity_eval_belebele_lao_Laoo"
export CUDA_VISIBLE_DEVICES=0

MODEL_DIR="loss_analysis_output/base_lao_Laoo"
EVAL_NAME="belebele_lao_Laoo"

shopt -s nullglob
CHECKPOINTS=("$MODEL_DIR"/checkpoint-*)
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "No checkpoints found under $MODEL_DIR" >&2
  exit 1
fi

for CKPT in "${CHECKPOINTS[@]}"; do
  echo "Evaluating $(basename $CKPT) ..."
  lm_eval \
    --model hf \
    --model_args "pretrained=$CKPT,tokenizer=$CKPT" \
    --tasks $EVAL_NAME \
    --device cuda \
    --batch_size 1 \
    --output_path "$CKPT/${EVAL_NAME}_results.json"
done
