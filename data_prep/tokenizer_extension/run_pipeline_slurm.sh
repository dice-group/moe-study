#!/bin/bash
#SBATCH --job-name=train_extend_tokenizer
#SBATCH --output=logs/tokenize_extension/train_extend_%j.out
#SBATCH --error=logs/tokenize_extension/train_extend_%j.err
#SBATCH --partition=normal
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH -p normal

set -euo pipefail

DATA_DIR=/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset
BASE_TOKENIZER_ID=meta-llama/Llama-3.2-1B
BASE_TOKENIZER_PATH=/scratch/hpc-prf-merlin/shared_cache/huggingface/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08
TRAIN_OUTPUT_DIR=/scratch/hpc-prf-merlin/project_data/moe_study/250K_vocab_size/trained_multilingual_tokenizer_llama-3.2-1B
ALLOCATION_CSV=/scratch/hpc-prf-merlin/moe-study/data_prep/metrics/vocab_extension_plan_ranking_based_all_llama_3.2-1b.csv
BASE_METRICS_CSV=/scratch/hpc-prf-merlin/moe-study/data_prep/metrics/tokenizer_metrics_llama3.2-1b_all_.csv
EXTENDED_TOKENIZER_DIR=/scratch/hpc-prf-merlin/joel/moe-study/data_prep/temp/merged_extended_tokenizer
EXTENDED_METRICS_CSV=/scratch/hpc-prf-merlin/joel/moe-study/data_prep/temp/merged_extended_tokenizer/test_tokenizer_coverage.csv
COMPARISON_CSV=/scratch/hpc-prf-merlin/joel/moe-study/data_prep/metrics/tokenizer_comparison_llama3.2-1b.csv

mkdir -p logs/tokenize_extension

srun python -u -m tokenize_extension \
  --train-multilingual \
  --train-data-dir "${DATA_DIR}" \
  --train-output-dir "${TRAIN_OUTPUT_DIR}" \
  --train-vocab-size 250000 \
  --train-base-model "${BASE_TOKENIZER_ID}" \
  --base-metrics-csv "${BASE_METRICS_CSV}" \
  --allocation-csv "${ALLOCATION_CSV}" \
  --extend \
  --extension-base-path "${BASE_TOKENIZER_PATH}" \
  --extension-multilingual-path "${TRAIN_OUTPUT_DIR}" \
  --extension-data-dir "${DATA_DIR}" \
  --extension-output-dir "${EXTENDED_TOKENIZER_DIR}" \
  --extension-sample-docs 50 \
  --extension-vocab-cap 256000 \
  --compute-extended-coverage \
  --extended-data-dir "${DATA_DIR}" \
  --extended-tokenizer "${EXTENDED_TOKENIZER_DIR}" \
  --extended-max-lines 0 \
  --extended-num-workers 64 \
  --extended-coverage-output "${EXTENDED_METRICS_CSV}" \
  --comparison-output-csv "${COMPARISON_CSV}"

