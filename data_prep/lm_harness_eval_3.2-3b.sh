#!/bin/bash
# Run multilingual evaluation for Llama-3.2-3B
# Tasks: Knowledge (MMLU), Reasoning (Hellaswag), NLI (XNLI), QA (Belebele), Translation (FLORES)

export CUDA_VISIBLE_DEVICES=1
MODEL="meta-llama/Llama-3.2-3B"
OUT_DIR="./metrics/llama32_3b"
OUT_FILE="$OUT_DIR/llama32_3b_eval.json"
DETAIL_DIR="$OUT_DIR/per_language"

mkdir -p "$DETAIL_DIR"

lm_eval \
  --model hf \
  --model_args pretrained=$MODEL,tokenizer=$MODEL \
  --tasks hellaswag,xnli,belebele,flores,include_base_44_* \
  --device cuda \
  --batch_size auto \
  --num_fewshot 0 \
  --output_path "$OUT_FILE" \
  --include_path "$DETAIL_DIR" \
  --wandb_args "project=llama3.2-3b_multilingual_eval,name=llama3.2_3b_global_suite"

echo "Evaluation complete. Results saved to $OUT_FILE and $DETAIL_DIR/"
