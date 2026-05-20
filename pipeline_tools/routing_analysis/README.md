# Routing Analysis Tool

Standalone routing analysis for selected PEFT adapter checkpoints.

Dry-run validation:

```bash
python -m pipeline_tools.routing_analysis \
  --config configs/routing_analysis/final_winners_96_langs.json \
  --dry-run
```

Full analysis, intended for the GPU environment with the base model available:

```bash
python -m pipeline_tools.routing_analysis \
  --config configs/routing_analysis/final_winners_96_langs.json
```

The configured final-winner checkpoints are read-only inputs. Outputs are written under
`outputs/routing_analysis/moe_study_final_winners_96_langs/` as compressed routing
matrices, per-run `metadata.json`, optional figures, and a top-level `summary.json`.

Supported capture modes:

- `MOLA`: token-level true-expert routing plus null/bypass usage.
- `MOELORA`: shared language/task-id expert routing.
- `HYDRALORA`: head routing when `use_hydralora_experts=false`; expert routing otherwise.
