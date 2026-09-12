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

The completed 96-language, 1000-sequence analysis used:

```bash
python -m pipeline_tools.routing_analysis \
  --config configs/routing_analysis/final_winners_96_langs/initial_1000seq.json
```

The configured final-winner checkpoints are read-only inputs. Outputs are written under
`outputs/routing_analysis/moe_study_final_winners_96_langs/` as compressed routing
matrices, per-run `metadata.json`, optional figures, and a top-level `summary.json`.

Supported routing modes:

- `MOLA`: token-level true-expert routing plus null/bypass usage.
- `MOELORA`: shared language/task-id expert routing.
- `HYDRALORA`: head routing when `use_hydralora_experts=false`; expert routing otherwise.
- `MTLLORA`: language-conditioned softmax mixtures over shared up-projections.

`MOLA` and `HYDRALORA` are input-dependent and are measured from forward
passes. Padding positions are excluded with the tokenizer attention mask.
`MOELORA` and `MTLLORA` are input-independent for a fixed language id, so their
routing distributions are extracted directly from the checkpoint and can be
verified with `--validate-static`.

## Final Top-5 Analysis

```bash
python -m pipeline_tools.routing_analysis \
  --config configs/routing_analysis/final_top5_200_langs/initial_1000seq.json \
  --prepare-sample-manifest

python -m pipeline_tools.routing_analysis \
  --config configs/routing_analysis/final_top5_96_langs/initial_1000seq.json \
  --dry-run

python scripts/slurm/routing_analysis/submit_final_top5.py --dry-run
python scripts/slurm/routing_analysis/submit_final_top5.py
```

The shared manifest freezes the deterministic first 1,000 valid documents in
sorted-file/source-record order for all 215 routing ids. Token runs write
atomic per-language partials and resume automatically. Concurrent jobs write
only their own `run_summary.json`; the dependent post-processing job creates
the panel summaries, grouped views, aligned comparison, and notebook.

## Layerwise LoLA/t-SNE Atlas

The standard LoLA export embeds a whole normalized routing vector after
flattening its captured modules. The optional layerwise atlas instead writes a
routing heatmap and an independently fitted cosine t-SNE for each real
base-model layer at language and family granularity:

```bash
scripts/slurm/routing_analysis/run_layerwise_atlas.sh
```

The direct command for one panel is:

```bash
python -m pipeline_tools.routing_analysis.layerwise_lola \
  --routing-root outputs/routing_analysis/moe_study_final_top5_200_langs_initial_1000seq \
  --languages-csv result_analysis/l2v_nested_samples/200.csv \
  --granularity language \
  --granularity family
```

Outputs live below each run's
`lola_style/<run_id>/layerwise_atlas/` directory and include an HTML index,
per-view t-SNE coordinates, and `atlas_manifest.json`. The panel-level
`lola_style/layerwise_atlas_index.html` links all five approaches.

Hydra-LoRA, MOLA, and MTL-LoRA routing modules are averaged within real
base-model layers. MoE-LoRA exposes one shared language-conditioned router, so
the atlas labels it `global_shared` and renders it once rather than duplicating
the same distribution across artificial layers. Choice axes retain their
method-specific meanings (expert, head, true expert, or up-projection).
Individual t-SNE fits support within-view neighborhood interpretation only;
their absolute coordinates are not comparable across layers or approaches.
