#!/bin/bash
# MoV-LoRA Training Script
# Paper: "Pushing Mixture of Experts to the Limit: Extremely Parameter Efficient MoE
#         for Instruction Tuning" (Zadouri et al., ICLR 2024)
#
# MoV-LoRA uses per-layer learned routers for token-level expert selection.
# Unlike MoE-LoRA, no task/language IDs are needed — routing is learned end-to-end.
# Pass extra CLI overrides after the script path if needed.
set -e

export CUDA_VISIBLE_DEVICES=0
export LOG_LEVEL=DEBUG

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python "${REPO_ROOT}/LLaMA-Factory/src/train.py" \
  "${REPO_ROOT}/configs/train_movlora.yaml" \
  "$@" \
  2>&1 | tee "${REPO_ROOT}/outputs/movlora_train.log"
