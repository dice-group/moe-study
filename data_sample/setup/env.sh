#!/bin/bash
## NOTE: If you are working on a slurm-based environment, have: export SLURM_ACTIVE=true
CUR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # This gets overriden on the next 'source' call

# Assign project root directory variable to three levels above
export PROJECT_ROOT="$(cd "${CUR_SCRIPT_DIR}/../.." && pwd)"

# Check if SLURM is active and source the cluster‑specific config
if [[ "$SLURM_ACTIVE" == "true" ]]; then
    source "$PROJECT_ROOT/slurm/cluster_specific_config.sh" # This will load the required environment dependencies
fi