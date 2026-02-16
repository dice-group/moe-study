#!/usr/bin/env bash
# Launch the moe-study pipeline for one or more profiles.
#
# Usage:
#   ./scripts/launch_pipeline.sh                              # run local_mini only
#   ./scripts/launch_pipeline.sh --profile gpu_1b_tier10_all_approaches
#   ./scripts/launch_pipeline.sh --profile-list configs/pipeline/profile_lists/smoke_test.txt
#   ./scripts/launch_pipeline.sh --profile local_mini --dry-run
#   ./scripts/launch_pipeline.sh --profile local_mini --start-at train --stop-after train
#
# Profile lists are text files with one profile name per line.
# Lines starting with '#' are ignored — comment out profiles to skip them.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_BUNDLE="${CONFIG_BUNDLE:-configs/pipeline}"

PROFILES=()
EXTRA_ARGS=()
PROFILE_LIST=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILES+=("$2")
      shift 2
      ;;
    --profile-list)
      PROFILE_LIST="$2"
      shift 2
      ;;
    --config)
      CONFIG_BUNDLE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# Read profiles from file if --profile-list was given.
if [[ -n "${PROFILE_LIST}" ]]; then
  if [[ ! -f "${REPO_ROOT}/${PROFILE_LIST}" ]]; then
    echo "[ERROR] Profile list not found: ${PROFILE_LIST}" >&2
    exit 1
  fi
  while IFS= read -r line; do
    line="${line%%#*}"      # strip comments
    line="${line// /}"      # strip whitespace
    [[ -z "${line}" ]] && continue
    PROFILES+=("${line}")
  done < "${REPO_ROOT}/${PROFILE_LIST}"
fi

# Default to local_mini if no profiles specified.
if [[ ${#PROFILES[@]} -eq 0 ]]; then
  PROFILES=("local_mini")
fi

echo "============================================"
echo " moe-study pipeline launcher"
echo " Config:   ${CONFIG_BUNDLE}"
echo " Profiles: ${PROFILES[*]}"
echo " Extra:    ${EXTRA_ARGS[*]:-<none>}"
echo "============================================"

FAILED=()
for profile in "${PROFILES[@]}"; do
  echo ""
  echo ">>> Running profile: ${profile}"
  echo "--------------------------------------------"
  if python3 -m pipeline.run \
       --config "${CONFIG_BUNDLE}" \
       --profile "${profile}" \
       "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; then
    echo ">>> Profile ${profile}: DONE"
  else
    echo ">>> Profile ${profile}: FAILED (exit code $?)" >&2
    FAILED+=("${profile}")
  fi
done

echo ""
echo "============================================"
echo " Summary: ${#PROFILES[@]} profile(s) attempted"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo " FAILED: ${FAILED[*]}"
  exit 1
else
  echo " All profiles completed successfully."
fi
echo "============================================"
