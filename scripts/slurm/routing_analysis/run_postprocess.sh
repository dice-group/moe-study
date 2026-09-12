#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "${REPO_ROOT}"

set +u
source /opt/software/pc2/EB-SW/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate merlin
set -u
export PYTHONUNBUFFERED=1

CONFIG_96=configs/routing_analysis/final_top5_96_langs/initial_1000seq.json
CONFIG_200=configs/routing_analysis/final_top5_200_langs/initial_1000seq.json
ROOT_96=outputs/routing_analysis/moe_study_final_top5_96_langs_initial_1000seq
ROOT_200=outputs/routing_analysis/moe_study_final_top5_200_langs_initial_1000seq
COMPARISON=outputs/routing_analysis/moe_study_final_top5_96_vs_200_comparison

python -m pipeline_tools.routing_analysis --config "${CONFIG_96}" --aggregate-only
python -m pipeline_tools.routing_analysis --config "${CONFIG_200}" --aggregate-only

python -m pipeline_tools.routing_analysis.lola_style \
  --routing-root "${ROOT_96}" \
  --languages-csv result_analysis/l2v_nested_samples/96.csv \
  --granularity family \
  --granularity family_script \
  --granularity family_resource \
  --granularity language

python -m pipeline_tools.routing_analysis.lola_style \
  --routing-root "${ROOT_200}" \
  --languages-csv result_analysis/l2v_nested_samples/200.csv \
  --granularity family \
  --granularity family_script \
  --granularity family_resource \
  --granularity language

scripts/slurm/routing_analysis/run_layerwise_atlas.sh

python -m pipeline_tools.routing_analysis.compare \
  --root-96 "${ROOT_96}" \
  --root-200 "${ROOT_200}" \
  --metadata-96 result_analysis/l2v_nested_samples/96.csv \
  --metadata-200 result_analysis/l2v_nested_samples/200.csv \
  --output "${COMPARISON}"

jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  result_analysis/notebooks/routing_analysis/final_top5_96_vs_200_routing.ipynb
