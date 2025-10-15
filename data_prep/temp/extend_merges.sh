#!/bin/bash
BASE="/scratch/hpc-prf-merlin/shared_cache/huggingface/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/"
TARGET="/scratch/hpc-prf-merlin/project_data/moe_study/250K_vocab_size/trained_multilingual_tokenizer_llama-3.2-1B/"
OUT_DIR="extended_tokenizer"

python extend_merges.py \
  --base_tokenizer "$BASE" \
  --target_tokenizer "$TARGET" \
  --output_dir "$OUT_DIR"
