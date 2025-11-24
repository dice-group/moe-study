#!/usr/bin/env bash
set -euo pipefail


bash "tokenize_all_tokenizers_lang_combinations.sh" \
  --shard-dir /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/sharded_samples \
  --output-base /scratch/hpc-prf-merlin/project_data/moe_study/tokenized/moe_study_comparison \
  --job-prefix tok_moe_study_comparison
