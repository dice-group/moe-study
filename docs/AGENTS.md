# AGENTS: Repository Alignment Guide

This file is the long-term onboarding and alignment document for agents working in this repository.

It replaces the older working note in `papers/further/AGENTS.md`.

## Mission

The repository exists to study and operationalize parameter-efficient fine-tuning methods for massively multilingual language model use cases.

The core goal is not just to implement PEFT methods in isolation, but to make them usable in a shared experimental system that is:

- reliable
- reproducible
- configurable
- easy to run and extend

That is why the repo has two equally important layers:

- method implementations in `LLaMA-Factory`
- orchestration, data prep, training, and evaluation in `pipeline/`

## Two Projects

There are two main project tracks in this repo.

### MOE Study

`MOE Study` is the empirical comparison project.

Its purpose is to compare different PEFT and MoE-style PEFT approaches under shared multilingual training conditions.

This is the broad comparison track. It includes baseline LoRA and multiple expertized or routed variants.

### HALA

`HALA` is the focused architecture-study track.

Its purpose is to study LoRA-family methods with more structure, especially around:

- LoRA
- CoLA
- HydraLoRA
- hierarchical or expertized extensions of those ideas

In practice, HALA is the more architecture-driven project, while MOE Study is the broader empirical comparison project.

## Source of Truth

Treat these as the main sources of truth when working here:

- pipeline entrypoint: `pipeline/run.py`
- pipeline orchestration: `pipeline/`
- config loading and profile resolution: `pipeline/config_loader.py`, `pipeline/runtime.py`
- training command synthesis: `pipeline/training.py`
- train-stage verification: `pipeline/checks.py`
- backend PEFT dispatch: `LLaMA-Factory/src/llamafactory/model/adapter.py`
- backend finetuning args: `LLaMA-Factory/src/llamafactory/hparams/finetuning_args.py`
- PEFT tuner implementations: `LLaMA-Factory/src/peft/tuners/`

## Runtime Environments

There are generally three execution environments.

### 1. Otus University cluster

This is the main cluster environment.

Current assumptions:

- multi-GPU cluster
- H100 GPUs
- use the `merlin` conda environment

Typical profile examples:

- `moe_study_profile_otus_cluster`
- `hala_otus`

### 2. Lola VM

This is the GPU VM environment used for local-but-real GPU runs.

Current assumptions:

- 2x H100 GPUs
- use the `merlin` conda environment where applicable

Typical profile examples:

- `moe_study_profile_lola_gpuvm`
- `hala_lola`

### 3. Local agent environments

These are sandbox or local development environments used by agents for implementation, smoke tests, and debugging.

These are not the main benchmark environments. They are for:

- editing code
- validating pipeline wiring
- running tiny or fixture-based tests

## Environment Rule

Unless there is a strong reason not to, use the `merlin` environment for cluster and VM work.

You can see this assumption throughout the repo in:

- pipeline Slurm executor defaults
- profile executor configs
- cluster job scripts

## Repository Structure

### `pipeline/`

This is the orchestration layer.

It is responsible for:

- data sampling
- tokenizer extension
- embedding initialization
- tokenization
- audit and verification
- training
- benchmarking
- summaries and reports

Common edit points:

- run-kind support: `pipeline/config_validator.py`
- training CLI mapping: `pipeline/training.py`
- executor behavior: `pipeline/executor.py`
- naming/runtime helpers: `pipeline/runtime.py`
- train verification: `pipeline/checks.py`

### `LLaMA-Factory/`

This is the backend training layer.

Important:

- this is a nested git repo
- changes in the root repo and changes in `LLaMA-Factory` are staged and committed separately

All custom approaches are implemented here as PEFT tuners.

### `scripts/slurm/comparison/`

This is the larger comparison and cluster-launch path.

Use it for:

- matrix runs
- multi-GPU comparison runs
- large study jobs

Main files:

- `run_comparison.py`
- `matrix_runner.sh`
- `comparison_config*.json`
- per-approach `*_job.sh` scripts
- `accelerate_launch.sh`

## Pipeline Stage Order

The canonical stage order is defined in `pipeline/run.py`.

1. `sample`
   Collect or subsample multilingual raw data into a controlled dataset slice.
2. `tokenizer_extend`
   Optionally extend the tokenizer with new vocabulary before training.
3. `embedding_init`
   Initialize embeddings for tokenizer-added vocabulary or embedding-related changes.
4. `tokenize`
   Convert the sampled corpus into tokenized training/eval artifacts.
5. `audit`
   Verify language maps, routing assumptions, dataset structure, and other pre-train invariants.
6. `train`
   Launch method-specific PEFT training runs.
7. `embedding_verify`
   Check embedding-related outputs and consistency when embedding training is enabled.
8. `benchmark`
   Run downstream evaluation tasks such as lm-eval-style benchmarking.
9. `baseline_eval`
   Run baseline evaluation workflows when configured.
10. `summary`
   Aggregate run outputs into shared summaries and gate files.
11. `report`
   Build report-style artifacts such as spreadsheets.

## Tiny Local Profiles

Yes, there are local tiny-model profiles, and agents should use them for quick implementation checks.

Important profiles:

- `local_mini`
- `local_mini_quick`

They use the tiny model family configured through:

- training default model: `hf-internal-testing/tiny-random-LlamaForCausalLM`
- tokenization default model: `hf-internal-testing/tiny-random-LlamaForCausalLM`

These profiles are appropriate for:

- pipeline smoke tests
- wiring validation
- fast method integration checks
- local agent iteration before using real GPUs

## Approach Inventory

All of the approaches below are implemented in `LLaMA-Factory` as PEFT tuners.

The descriptions below are intentionally brief and operational.

| Approach | What it does | Paper | Tuner |
|---|---|---|---|
| `lora` | Baseline low-rank additive adapters on selected linear layers. | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | `LLaMA-Factory/src/peft/tuners/lora/` |
| `cola` | Collaborative low-rank adapters using structured A/B composition; in this repo it can also be used with expert routing extensions. | [CoLA Collaborative Low-Rank Adaptation.pdf](/home/agy/repos/moe-study/papers/CoLA%20Collaborative%20Low-Rank%20Adaptation.pdf) | `LLaMA-Factory/src/peft/tuners/cola/` |
| `hydralora` | Asymmetric LoRA with shared A and multiple B-style components; supports routing/expert behavior in this repo. | [HydraLoRA An Asymmetric LoRA Architecture for.pdf](/home/agy/repos/moe-study/papers/HydraLoRA%20An%20Asymmetric%20LoRA%20Architecture%20for.pdf) | `LLaMA-Factory/src/peft/tuners/hydralora/` |
| `moelora` | Mixture-of-LoRA experts with task or routing signals selecting between multiple LoRA experts. | [MoE-LoRA.pdf](/home/agy/repos/moe-study/papers/MoE-LoRA.pdf) | `LLaMA-Factory/src/peft/tuners/moelora/` |
| `mixlora` | Standard LoRA on attention plus routed LoRA experts in the FFN/MLP path with auxiliary router loss. | [MixLoRA Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts.pdf](/home/agy/repos/moe-study/papers/MixLoRA%20Enhancing%20Large%20Language%20Models%20Fine-Tuning%20with%20LoRA-based%20Mixture%20of%20Experts.pdf) | `LLaMA-Factory/src/peft/tuners/mixlora/` |
| `adamole` | Adaptive mixture of LoRA experts with threshold-based expert activation instead of a fixed top-k. | [AdaMoLE.pdf](/home/agy/repos/moe-study/papers/AdaMoLE.pdf) | `LLaMA-Factory/src/peft/tuners/adamole/` |
| `mola` | MoLA/AdaMoe-style LoRA experts with null-expert and top-k style controls; implemented here under `adamoe_1`. | [AdaMoe.pdf](/home/agy/repos/moe-study/papers/AdaMoe.pdf) | `LLaMA-Factory/src/peft/tuners/adamoe_1/` |
| `movlora` | Routed IA3-style vector scaling experts (MoV), not classic additive LoRA A/B experts; more parameter-frugal than LoRA-style experts. | [Pushing Mixture of Experts to the Limit: Extremely Parameter Efficient MoE for Instruction Tuning](https://arxiv.org/abs/2309.05444) | `LLaMA-Factory/src/peft/tuners/movlora/` |
| `hmora` | Hierarchical MoE LoRA with token routers and optional task-level structure; the most explicitly hierarchical tuner in the repo. | [ICLR-2025-hmora-making-llms-more-effective-with-hierarchical-mixture-of-lora-experts-Paper-Conference.pdf](/home/agy/repos/moe-study/papers/ICLR-2025-hmora-making-llms-more-effective-with-hierarchical-mixture-of-lora-experts-Paper-Conference.pdf) | `LLaMA-Factory/src/peft/tuners/hmora/` |
| `moelpr` | Two-stage multilingual MoE expansion with language-prior routing; this remains implemented but is considered obsolete for the current planned experiments. | [MoE-LPR: Multilingual Extension of Large Language Models through Mixture-of-Experts with Language Priors Routing](https://arxiv.org/abs/2408.11396) | `LLaMA-Factory/src/peft/tuners/moelpr/` |

## Important Note on `moelpr`

`moelpr` still exists in the repository and the implementation remains available.

However:

- it is considered obsolete for the current planned experiments
- it was decided that it will not be part of the main experiment set going forward

So agents should treat it as:

- implemented and still useful for legacy context or debugging
- not part of the active planned comparison scope

## Project-to-Approach Alignment

### MOE Study

MOE Study is the broad comparison track.

Its active scope is centered on comparing methods such as:

- `lora`
- `cola`
- `hydralora`
- `moelora`
- `mixlora`
- `adamole`
- `mola`
- `movlora`
- `hmora`

`moelpr` may still appear in local or legacy configs, but should not be treated as part of the current intended experiment set.

### HALA

HALA is the focused architecture track.

Its conceptual center is:

- `lora`
- `cola`
- `hydralora`
- hierarchical or expertized extensions around those ideas

In other words:

- MOE Study asks: which multilingual PEFT method works better under shared conditions?
- HALA asks: how should we structure LoRA-family methods, especially CoLA/Hydra-style methods, for multilingual adaptation?

## Current Profiles

Current profile keys in the repo:

- `hala_otus`
- `hala_lola`
- `hala_lola_12_alpha_100kdocs`
- `hala_lola_12_alpha_smoke_200docs`
- `local_mini`
- `local_mini_quick`
- `moe_study_profile_otus_cluster`
- `moe_study_profile_lola_gpuvm`
- `moe_trainable_params_sweep_otus_cluster`

Important operational profiles:

- quick local smoke tests: `local_mini`, `local_mini_quick`
- MOE Study cluster: `moe_study_profile_otus_cluster`
- MOE Study GPU VM: `moe_study_profile_lola_gpuvm`
- HALA cluster: `hala_otus`
- HALA GPU VM: `hala_lola`

## Execution Guidance

### Use the generic pipeline when:

- you are validating repo-integrated behavior
- you need stage-by-stage reproducibility
- you are iterating locally
- you are running a controlled single-job profile

Entrypoint:

```bash
python -m pipeline.run --config configs/pipeline --profile <profile>
```

### Use the comparison SLURM path when:

- you are launching many runs
- you want multi-GPU comparison jobs
- you want the more mature cluster launcher path

Main entrypoints:

- `scripts/slurm/comparison/run_comparison.py`
- `scripts/slurm/comparison/matrix_runner.sh`

## Benchmarking Status

The benchmarking layer exists, but it is not fully complete relative to the broader research goals.

Current caveat:

- `SIB200` and `TAXI150` are still not fully integrated into the pipeline benchmark flow

Agents should check the current benchmark wiring before assuming those tasks are already part of the standard pipeline outputs.

## Known Caveats

### `configs/moelpr_lang_map.json`

This path is still missing from the repo. Be careful with any legacy code or profile path that still assumes it exists.

### Standalone method scripts are not always the source of truth

The authoritative execution paths are:

- the pipeline
- the comparison SLURM launcher path

Standalone convenience scripts may exist, but agents should not assume they are the best-maintained workflow.

## Agent Workflow

When working on a method or experiment in this repository:

1. Identify whether the work belongs to `MOE Study` or `HALA`.
2. Check the relevant profile first.
3. Confirm whether the method is active scope or legacy-only.
4. If you add or change a method, wire all required layers:
   - tuner implementation
   - finetuning arguments
   - adapter dispatch
   - pipeline command mapping
   - pipeline verification
   - profiles
   - SLURM comparison config if applicable
   - at least one local smoke or backend test
5. Validate locally with the tiny-model profiles before using real GPUs.

## Recommended Follow-Up Docs

These additional docs would be useful over time:

- a single `docs/approach_matrix.md` with active vs legacy methods per project
- a `docs/cluster_runbook.md` for Otus and Lola operational details
- a `docs/data_lineage.md` for sample, tokenization, tokenizer-extension, and embedding lineage
