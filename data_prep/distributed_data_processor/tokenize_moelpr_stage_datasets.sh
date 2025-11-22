#!/bin/bash

# Tokenize the MoE-LPR Stage 1 (new languages) and Stage 2 (router replay) datasets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stage 1: languages that also exist in Belebele (no English)
STAGE1_SHARD_DIR=/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/sharded_samples
STAGE1_LANGUAGES="wbm_Latn tel_Telu fas_Arab crl_Cans abk_Cyrl"
STAGE1_TOKENIZER=meta-llama/Llama-3.1-8B
STAGE1_OUTPUT=/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/moelpr/llama-3.1-8B_tokenizer/moelpr_stage1_5newlangs

# Stage 2: original-language replay (English only)
STAGE2_SHARD_DIR=/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/sharded_samples
STAGE2_LANGUAGES="english"
STAGE2_TOKENIZER=meta-llama/Llama-3.1-8B
STAGE2_OUTPUT=/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/moelpr/llama-3.1-8B_tokenizer/moelpr_stage2_eng
STAGE2_LANGUAGE_RATES_FILE="${SCRIPT_DIR}/../../configs/moelpr_stage2_language_rates.json"

NUM_PROC=4
MERGE_WORKERS=4
NUM_RANKS=5

echo "[INFO] Tokenizing MoE-LPR Stage 1 languages: ${STAGE1_LANGUAGES}"
bash "${SCRIPT_DIR}/tokenize_and_merge_pipeline.sh" \
  --shard-dir "${STAGE1_SHARD_DIR}" \
  --tokenizer "${STAGE1_TOKENIZER}" \
  --output-root "${STAGE1_OUTPUT}" \
  --languages "${STAGE1_LANGUAGES}" \
  --job-prefix moelpr_stage1_tok \
  --num-proc "${NUM_PROC}" \
  --num-ranks "${NUM_RANKS}" \
  --merge-workers "${MERGE_WORKERS}"

echo "[INFO] Tokenizing MoE-LPR Stage 2 languages: ${STAGE2_LANGUAGES}"
LANG_RATE_FILE="${STAGE2_LANGUAGE_RATES_FILE}"

bash "${SCRIPT_DIR}/tokenize_and_merge_pipeline.sh" \
  --shard-dir "${STAGE2_SHARD_DIR}" \
  --tokenizer "${STAGE2_TOKENIZER}" \
  --output-root "${STAGE2_OUTPUT}" \
  --languages "${STAGE2_LANGUAGES}" \
  --job-prefix moelpr_stage2_tok \
  --num-proc "${NUM_PROC}" \
  --num-ranks 150 \
  --merge-workers "${MERGE_WORKERS}" \
  --language-rates "${LANG_RATE_FILE}"

echo "[INFO] Stage datasets ready:"
echo "  Stage 1 dataset -> ${STAGE1_OUTPUT}"
echo "  Stage 2 dataset -> ${STAGE2_OUTPUT}"
