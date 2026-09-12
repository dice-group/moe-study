#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "${REPO_ROOT}"

set +u
source /opt/software/pc2/EB-SW/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate merlin
set -u
export PYTHONUNBUFFERED=1

python -m pipeline_tools.routing_analysis.layerwise_lola \
  --routing-root outputs/routing_analysis/moe_study_final_top5_96_langs_initial_1000seq \
  --languages-csv result_analysis/l2v_nested_samples/96.csv \
  --granularity language \
  --granularity family

python -m pipeline_tools.routing_analysis.layerwise_lola \
  --routing-root outputs/routing_analysis/moe_study_final_top5_200_langs_initial_1000seq \
  --languages-csv result_analysis/l2v_nested_samples/200.csv \
  --granularity language \
  --granularity family
