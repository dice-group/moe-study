#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_JSON="${CONFIG_JSON:-${SCRIPT_DIR}/comparison/comparison_config.json}"

if [[ ! -f "${CONFIG_JSON}" ]]; then
  echo "[ERROR] CONFIG_JSON ${CONFIG_JSON} not found." >&2
  exit 1
fi

python "${SCRIPT_DIR}/comparison/run_comparison.py" --config "${CONFIG_JSON}"
