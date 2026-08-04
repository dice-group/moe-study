# MoE Study

This repo contains the implementation of a MoE-LoRA study comparing multiple
fine-tuning approaches for massively multilingual continual pretraining while
scaling language coverage from 10 to 200 languages.

> **Note:** We use a custom fork of [LLaMA-Factory](https://anonymous.4open.science/r/LLaMA-Factory-D6EA) for the MoE-LoRA method implementations. Because it is added as a submodule, it is not directly supported by `anonymous.4open.science`.

## Main entry points
- Main study pipeline executor: `python -m pipeline.run --config configs/pipeline --profile <profile>`
- Pipeline profiles and generated study configs: `configs/pipeline/profiles`
- Cluster comparison launcher: `scripts/slurm/comparison/run_comparison.py`
- Result analysis notebooks: `result_analysis/notebooks`
- Data tokenization pipeline: `data_prep/merlin_data_prep/distributed_data_processor/scripts/tokenize_and_merge_pipeline.sh`

## Approaches in this study
Implemented in LLaMA‑Factory:
- **LoRA** — `LLaMA-Factory/src/peft/tuners/lora`
- **AdaMoLE** — `LLaMA-Factory/src/peft/tuners/adamole`
- **AdaMoE** — `LLaMA-Factory/src/peft/tuners/adamoe_1`
- **HydraLoRA** — `LLaMA-Factory/src/peft/tuners/hydralora`
- **HMoRA** — `LLaMA-Factory/src/peft/tuners/hmora`
- **MTL-LoRA** — `LLaMA-Factory/src/peft/tuners/mtllora`
- **CoLA** — `LLaMA-Factory/src/peft/tuners/cola`
- **MoE‑LoRA** — `LLaMA-Factory/src/peft/tuners/moelora`
- **MoV‑LoRA** — `LLaMA-Factory/src/peft/tuners/movlora`

Training entry code lives under `LLaMA-Factory/src/llamafactory/train`.

## Setup
Set up the conda environment with `LLaMA-Factory/setup_conda_env.md`.

Quick checklist:
- Create and activate the main conda env (`merlin`)
- Install requirements and `flash-attn`
- Install repo in editable mode
- (Optional) create separate `lm_eval` env for evaluations

## Study artifacts
[paper_artifact/README.md](paper_artifact/README.md)

