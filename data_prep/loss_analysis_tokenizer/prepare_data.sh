#!/bin/bash

DATA_ROOT="/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/samples"
OUT_DATA_DIR="/scratch/hpc-prf-merlin/joel/moe-study/data_prep/loss_analysis_tokenizer/loss_analysis_dataset"

echo "Starting data prep"
echo "Logs will be written to: $LOG_FILE"

python prepare_loss_analysis_dataset.py \
  --root "$DATA_ROOT" \
  --out-dir "$OUT_DATA_DIR" \
  --allow-repeat

echo "Data prep completed"
echo "Output written to: $OUT_DATA_DIR"
