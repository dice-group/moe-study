#!/bin/bash
# Run multilingual evaluation for Llama-3.2-1B
# Tasks: Knowledge (MMLU), Reasoning (Hellaswag), NLI (XNLI), QA (Belebele), Translation (FLORES)

export CUDA_VISIBLE_DEVICES=0
MODEL="meta-llama/Llama-3.2-1B"
OUT_DIR="./metrics/llama32_1b_belebele"
OUT_FILE="$OUT_DIR/llama32_1b_eval_belebele.json"
DETAIL_DIR="$OUT_DIR/per_language"

mkdir -p "$DETAIL_DIR"

lm_eval \
  --model hf \
  --model_args pretrained=$MODEL,tokenizer=$MODEL \
  --tasks belebele \
  --batch_size auto \
  --limit 10 \
  --log_samples \
  --output_path "$OUT_FILE" \
  --include_path "$DETAIL_DIR" \
  --wandb_args "project=llama32_multilingual_eval_belebele,name=llama32_1b_global_suite_belebele"

echo "Evaluation complete. Results saved to $OUT_FILE and $DETAIL_DIR/"
