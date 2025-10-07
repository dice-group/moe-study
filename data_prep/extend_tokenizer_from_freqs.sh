#!/bin/bash

LANG_DIR=/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/extended_tokenizer/test
BASE_TOKENIZER=meta-llama/Llama-3.2-1B
ALLOC_CSV=/scratch/hpc-prf-merlin/moe-study/data_prep/metrics/vocab_extension_plan_ranking_based_all_llama_3.2-1b.csv
OUTPUT_DIR=/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/extended_tokenizer/test/extended_test_final_llama3.2-1b

python extend_tokenizer_from_freqs.py \
  --base_tokenizer "$BASE_TOKENIZER" \
  --lang_dir "$LANG_DIR" \
  --alloc_csv "$ALLOC_CSV" \
  --output_dir "$OUTPUT_DIR"
