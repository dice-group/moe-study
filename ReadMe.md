# MoE–LoRA Methods for Multilingual Adaptation: A Systematic Empirical Study

Code for our study comparing LoRA and routed-LoRA (MoE–LoRA) methods for
massively multilingual continued pretraining, scaling language coverage from 10
to 200 languages. Paper: [arXiv:2611.23158](https://arxiv.org/abs/2611.23158).

## News

- **Nov 2026:** The paper has been accepted to the Findings of EACL 2027.

## Repository overview

- **Paper results:** [`paper_artifact/`](paper_artifact/README.md) holds the
  analysis notebooks, scripts, tables, and figures behind every number in the
  paper.
- **Methods:** implemented in our fork of
  [LLaMA-Factory](https://github.com/dice-group/LLaMA-Factory-Merlin), included
  as a submodule, under `LLaMA-Factory/src/peft/tuners/`: LoRA (`lora`),
  AdaMoLE (`adamole`), AdaMoE (`adamoe_1`), HydraLoRA (`hydralora`), HMoRA
  (`hmora`), MTL-LoRA (`mtllora`), MOELoRA (`moelora`), and MoV (`movlora`).
  CoLA (`cola`) is also implemented but excluded from the paper.
- **Study pipeline:** `python -m pipeline.run --config configs/pipeline --profile <profile>`,
  with profiles in `configs/pipeline/profiles`.
- **Data preparation:** `data_prep/merlin_data_prep/distributed_data_processor/scripts/tokenize_and_merge_pipeline.sh`.

## Setup

Clone with submodules (`git clone --recurse-submodules`), then follow
`LLaMA-Factory/setup_conda_env.md` to create the conda environment, install the
requirements and `flash-attn`, and install the repository in editable mode. A
separate `lm_eval` environment is optional for evaluation.

## Citation

If you use this work, please cite:

```bibtex
@misc{srivastava2026moeloramethodsmultilingualadaptation,
      title={MoE--LoRA Methods for Multilingual Adaptation: A Systematic Empirical Study}, 
      author={Nikit Srivastava and Joel Dag and René Speck and Axel-Cyrille Ngonga Ngomo},
      year={2026},
      eprint={2611.23158},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2611.23158}, 
}
```
