# MoE Study

This repo contains the implementation of a MoE-LoRA study comparing multiple
fine-tuning approaches for massively multilingual continual pretraining while
scaling language coverage from 10 to 200 languages.

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
- Main study notes and details: `https://docs.google.com/document/d/1sSwH7xHYJsKHvBT7nrPL2CDcKXo9vIiIb-4YMeuNLPc/edit?tab=t.6ddn6ncjl3yf`
- Result table Excel workbooks are also backed up in Sciebo: `https://uni-paderborn.sciebo.de/apps/files/files/964369178?dir=/MoE%20%2B%20LoRA%20Empirical%20Study/MoE%20Study`
- Configuration space spreadsheet: `https://docs.google.com/spreadsheets/d/1w53Qei57JmuF7Kk8X7uVzHTrbkC2Ri2kVV1rCTro0cc/edit?gid=0#gid=0`
- Final adapter-only BACKUP on PC2DATA: `/pc2/groups/hpc-prf-merlin/joeldag/moe-study-final-results-adapter-only-2026-07-05`

