# MoE Study

This repo contains a MoE study comparing multiple fine‑tuning approaches while scaling
both **language coverage** (10 -> 200 languages) and **model size** (1B -> 8B).

Training metrics (W&B): `https://wandb.ai/dice-nlp/moe-study-comparison`  
Eval metrics (Belebele + FLORES): `https://wandb.ai/dice-nlp/moe-study-comparison_eval`  
Metric explanations: `docs/training_metrics.md`

## Main entry points
- Compare approaches on the cluster: `scripts/slurm/comparison/run_comparison.py`  
  (configs in `scripts/slurm/comparison/*.json`)
- Data tokenization pipeline: `data_prep/merlin_data_prep/distributed_data_processor/scripts/tokenize_and_merge_pipeline.sh`

## Approaches in this study
Implemented in LLaMA‑Factory:
- **AdaMoLE** — `LLaMA-Factory/src/peft/tuners/adamole`
- **MoLA** — `LLaMA-Factory/src/peft/tuners/adamoe_1`
- **HydraLoRA** — `LLaMA-Factory/src/peft/tuners/hydralora`
- **CoLA** — `LLaMA-Factory/src/peft/tuners/cola`
- **MoE‑LPR** — `LLaMA-Factory/src/peft/tuners/moelpr`

Training entry code lives under `LLaMA-Factory/src/llamafactory/train`.

## Setup
Follow: `LLaMA-Factory/setup_conda_env.md`

Quick checklist:
- Create and activate the main conda env (`merlin`)
- Install requirements and `flash-attn`
- Install repo in editable mode
- (Optional) create separate `lm_eval` env for evaluations
