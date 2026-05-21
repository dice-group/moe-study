# Routing Analysis Outputs

Routing analysis outputs are stored under:

`outputs/routing_analysis/<analysis_run_id>/`

For the current 96-language 1000-sequence run:

`outputs/routing_analysis/moe_study_final_winners_96_langs_initial_1000seq/`

The routing-analysis config that reproduces this run is:

`configs/routing_analysis/final_winners_96_langs/initial_1000seq.json`

The pipeline sampling profile that produced the 1000-sequence input set is:

`configs/pipeline/profiles/routing_analysis/96_initial_1000seq.json`

## Main Files

Each adapter has its own directory:

- `mola_tr64_e32_k4/`
- `moelora_tr64_e32_k4/`
- `hydralora_r2_heads32/`

The important files per adapter are:

- `routing_matrix.npz`: canonical numeric routing data.
- `metadata.json`: adapter config, language list, captured modules, and batch settings.
- `figures/`: quick default heatmaps and entropy plots.

The run-level file is:

- `summary.json`: list of completed adapter analyses and output paths.

## NPZ Data Format

Load with NumPy:

```python
from pathlib import Path
import numpy as np

path = Path("outputs/routing_analysis/moe_study_final_winners_96_langs_initial_1000seq/mola_tr64_e32_k4/routing_matrix.npz")
data = np.load(path, allow_pickle=True)

counts = data["routing_counts"]                 # language x module x expert
ratios = data["routing_count_normalized"]       # per-module routing ratios
weights = data["routing_weight_normalized"]     # normalized routing weights, if available
languages = data["languages"].tolist()
modules = data["layer_labels"].tolist()
experts = data["choice_labels"].tolist()
axis_name = str(data["axis_name"])
```

Expected shapes for this run:

- MoLA: `(102, 128, 32)`, axis `true_expert`
- MoE-LoRA: `(102, 1, 32)`, axis `expert`
- HydraLoRA: `(102, 224, 32)`, axis `head`

## Derived LOLA-Style Outputs

LOLA-style outputs are under:

`outputs/routing_analysis/<analysis_run_id>/lola_style/`

They include PNG visualizations and CSV summaries for `family`, `family_script`, and `language` granularities.

Useful CSVs:

- `top_experts_by_group.csv`: most-used experts per language/family group.
- `pairwise_similarity.csv`: cosine similarity between routing vectors.
- `group_members.csv`: languages included in each family/group.
- `by_base_layer/top_layer_experts_by_group.csv`: layerwise expert summaries where layer aggregation exists.

Recreate the LOLA-style outputs with:

```bash
MPLCONFIGDIR=/tmp/mplconfig-routing python3 -m pipeline_tools.routing_analysis.lola_style \
  --routing-root outputs/routing_analysis/moe_study_final_winners_96_langs_initial_1000seq \
  --languages-csv result_analysis/l2v_nested_samples/96.csv \
  --granularity family \
  --granularity family_script \
  --granularity language \
  --top-k 8 \
  --min-family-size 2
```

Use the `.npz` files as the reproducible source of truth; the CSVs and PNGs are derived views.
