#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
COMPARISON_DIR="${SCRIPT_DIR}"

CONFIG_PATH="${CONFIG:-${SCRIPT_DIR}/configs/matrix_config.sh}"
if [[ $# -ge 1 ]]; then
  CONFIG_PATH="$1"
  shift
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  if [[ -f "${SCRIPT_DIR}/${CONFIG_PATH}" ]]; then
    CONFIG_PATH="${SCRIPT_DIR}/${CONFIG_PATH}"
  else
    echo "[ERROR] Config file not found: ${CONFIG_PATH}" >&2
    exit 1
  fi
fi

# shellcheck disable=SC1090
source "${CONFIG_PATH}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/matrix_runner.sh"

run_matrix "$@"
