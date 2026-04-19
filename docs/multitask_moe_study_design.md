# Multitask MoE Study Design

This document captures the current design for a separate empirical paper on general multitask MoE / routed PEFT fine-tuning. It is intentionally separate from the multilingual MoE study already documented elsewhere in this repo.

## Goal

Build a clean, peer-reviewed-backed empirical study that answers when MoE-style or routed PEFT methods help for general multitask fine-tuning under one shared implementation stack.

The paper should prioritize:
- fair cross-method comparison
- public, reproducible text-only benchmarks
- clean separation between headline results and broader appendix coverage
- research questions that can be answered directly from the benchmark design

## Literature-Derived Benchmark Families

The paper corpus in `papers/` and `papers/further/` supports four distinct experimental regimes:

1. Broad instruction-mixture multitask.
2. Controlled public multitask benchmark suites.
3. Domain-specific multitask training.
4. Adapter-merging studies.

The study design here uses:
- two core tracks for the main paper
- one appendix-only domain-transfer track
- no image-text track
- no adapter-merging track

## Final Study Shape

### Track 1: Broad Joint Multitask

This is the main headline benchmark.

- Public text-only `FLAN-v2`-style multitask mixture.
- One strong shared backbone: `meta-llama/Llama-3.1-8B`.
- No explicit `task_id` input.
- Balanced mixture across task families rather than naive concatenation.
- Report both seen-task and held-out unseen-task generalization.

Planned evaluation set:
- Seen-task: `MMLU`, `MMLU-Pro`, `ARC-E`, `ARC-C`, `OpenBookQA`, `SWAG`, `CommonsenseQA`
- Held-out unseen-task: `ANLI`, `CB`, `RTE`, `WSC`, `WiC`, `COPA`, `WinoGrande`, `HellaSwag`

This track is mainly justified by:
- [PUSHING MIXTURE OF EXPERTS TO THE LIMIT.pdf](../papers/PUSHING%20MIXTURE%20OF%20EXPERTS%20TO%20THE%20LIMIT.pdf) and [mov KB](../papers/further/mov/structured.yaml)
- [ICLR-2025-hmora-making-llms-more-effective-with-hierarchical-mixture-of-lora-experts-Paper-Conference.pdf](../papers/ICLR-2025-hmora-making-llms-more-effective-with-hierarchical-mixture-of-lora-experts-Paper-Conference.pdf) and [hmora KB](../papers/further/hmora/structured.yaml)
- [HydraLoRA An Asymmetric LoRA Architecture for.pdf](../papers/HydraLoRA%20An%20Asymmetric%20LoRA%20Architecture%20for.pdf) and [hydralora KB](../papers/further/hydralora/structured.yaml)
- [CoLA Collaborative Low-Rank Adaptation.pdf](../papers/CoLA%20Collaborative%20Low-Rank%20Adaptation.pdf) and [cola KB](../papers/further/cola/structured.yaml)

### Track 2: Controlled Public Multitask

This is the controlled secondary benchmark.

- Public text-only joint training on `GLUE + commonsense`.
- Explicit `task_id` support is allowed.
- Task-aware methods may consume task identity.
- Task-agnostic methods only receive prompt text.
- The purpose is to test whether task-aware routing helps under a more structured multitask setup.

Planned task groups:
- `GLUE`: `CoLA`, `SST-2`, `MRPC`, `QQP`, `MNLI`, `QNLI`, `RTE`, `STS-B`
- Commonsense: `BoolQ`, `PIQA`, `Social IQA`, `HellaSwag`, `WinoGrande`, `ARC-E`, `ARC-C`, `OpenBookQA`

This track is mainly justified by:
- [MTL-LoRA Low-Rank Adaptation for Multi-Task Learning.pdf](../papers/MTL-LoRA%20Low-Rank%20Adaptation%20for%20Multi-Task%20Learning.pdf) and [mtllora KB](../papers/further/mtllora/structured.yaml)
- [MixLoRA Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts.pdf](../papers/MixLoRA%20Enhancing%20Large%20Language%20Models%20Fine-Tuning%20with%20LoRA-based%20Mixture%20of%20Experts.pdf) and [mixlora KB](../papers/further/mixlora/structured.yaml)
- [AdaMoe.pdf](../papers/AdaMoe.pdf) and [adamoe KB](../papers/further/adamoe/structured.yaml)

### Appendix Track: Domain Transfer

This track is appendix-only and must not be folded into the main aggregate scores.

- Text-only specialist domains plus code.
- Intended domains: `medical`, `legal`, `finance`, `code`.
- Purpose: test whether the same methods remain effective outside the two core general-domain multitask tracks.

Likely anchor tasks:
- medical: public medical multitask setup inspired by `MoE-LoRA`
- legal: legal instruction tuning + legal evaluation inspired by `HydraLoRA` and `CoLA`
- finance: `fingpt-fineval`-style text tasks from `CoLA`
- code: `CodeAlpaca` training plus `HumanEval`-style evaluation from `HydraLoRA` and related papers

This track is mainly justified by:
- [MoE-LoRA.pdf](../papers/MoE-LoRA.pdf) and [moe_lora KB](../papers/further/moe_lora/structured.yaml)
- [HydraLoRA An Asymmetric LoRA Architecture for.pdf](../papers/HydraLoRA%20An%20Asymmetric%20LoRA%20Architecture%20for.pdf) and [hydralora KB](../papers/further/hydralora/structured.yaml)
- [CoLA Collaborative Low-Rank Adaptation.pdf](../papers/CoLA%20Collaborative%20Low-Rank%20Adaptation.pdf) and [cola KB](../papers/further/cola/structured.yaml)

## Methods In Scope

Headline comparison set:
- `lora_r8`
- `lora_r64`
- `cola`
- `hydralora`
- `moelora`
- `adamole`
- `adamoe_1`
- `mov`
- `hmora`
- `mtllora`
- `mixlora`

Dense reference:
- one smaller dense full fine-tuning proxy, reported separately as a dense reference rather than as a parameter-matched PEFT baseline

Excluded from this study benchmark:
- `FlyLoRA`
- `Adaptive LoRA Merge with Parameter Pruning`

Reason for exclusion:
- these are primarily adapter-merging studies rather than shared joint-training multitask comparisons

## Research Questions

The benchmark is designed to answer the following RQs.

1. Under a broad text-only multitask setup, which routed or MoE-style PEFT methods outperform LoRA at matched trainable-parameter budgets?
2. Do those gains transfer to held-out unseen tasks rather than only seen-task joint training?
3. Under a controlled multitask benchmark, when does explicit task identity help?
4. What are the tradeoffs between quality, trainable parameters, throughput, memory, and variance?
5. Do the same conclusions remain stable on specialist text domains and code in the appendix track?

## Paper Basis

| Paper | Raw PDF | Structured KB | Contribution To Design |
| --- | --- | --- | --- |
| AdaMoLE | [PDF](../papers/AdaMoLE.pdf) | [KB](../papers/further/adamole/structured.yaml) | Method in scope; supports adaptive expert routing inside the shared comparison matrix. |
| AdaMOE / daMOE | [PDF](../papers/AdaMoe.pdf) | [KB](../papers/further/adamoe/structured.yaml) | Method in scope; relevant to task-aware or adaptive expert selection in controlled multitask settings. |
| CoLA | [PDF](../papers/CoLA%20Collaborative%20Low-Rank%20Adaptation.pdf) | [KB](../papers/further/cola/structured.yaml) | Supports cross-domain text evaluation, including legal, medical, finance, and multi-task `OpenOrca -> BBH`. |
| HydraLoRA | [PDF](../papers/HydraLoRA%20An%20Asymmetric%20LoRA%20Architecture%20for.pdf) | [KB](../papers/further/hydralora/structured.yaml) | Supports broad multitask and domain-transfer framing, including `Flan-v2 -> BBH`, law, medicine, math, and code. |
| MoV | [PDF](../papers/PUSHING%20MIXTURE%20OF%20EXPERTS%20TO%20THE%20LIMIT.pdf) | [KB](../papers/further/mov/structured.yaml) | Main justification for held-out unseen-task evaluation using a `P3` instruction-mixture logic. |
| HMoRA | [PDF](../papers/ICLR-2025-hmora-making-llms-more-effective-with-hierarchical-mixture-of-lora-experts-Paper-Conference.pdf) | [KB](../papers/further/hmora/structured.yaml) | Main justification for `Flan-v2`-style multitask training and for evaluating task-router generalization to unseen tasks. |
| MTL-LoRA | [PDF](../papers/MTL-LoRA%20Low-Rank%20Adaptation%20for%20Multi-Task%20Learning.pdf) | [KB](../papers/further/mtllora/structured.yaml) | Main justification for the controlled `GLUE + commonsense` public text-only track with explicit task identities. |
| MixLoRA | [PDF](../papers/MixLoRA%20Enhancing%20Large%20Language%20Models%20Fine-Tuning%20with%20LoRA-based%20Mixture%20of%20Experts.pdf) | [KB](../papers/further/mixlora/structured.yaml) | Supports commonsense-style multitask joint training and catastrophic-forgetting style comparisons against LoRA and DoRA. |
| MoE-LoRA | [PDF](../papers/MoE-LoRA.pdf) | [KB](../papers/further/moe_lora/structured.yaml) | Justifies a domain-specific appendix track; not suitable as the main general-domain benchmark because it is medical-specific. |
| FlyLoRA | Raw PDF not present in current checkout | [KB](../papers/further/flylora/structured.yaml) | Relevant background, but excluded because it is centered on model merging and merge robustness rather than the core joint-training regime. |
| Adaptive LoRA Merge with Parameter Pruning | Raw PDF not present in current checkout | [KB](../papers/further/adaptive_lora_merge_with_parameter_pruning/structured.yaml) | Relevant background, but excluded because it studies LoRA merging rather than the shared multitask training setup targeted here. |

## Explicit Exclusions

These exclusions are intentional and part of the study design.

- `image-text` is excluded entirely because it changes modality and makes the core comparison less controlled.
- multilingual benchmarks are excluded because this is a separate paper from the multilingual MoE study.
- `medical`, `legal`, `finance`, and `code` are not part of the main aggregate headline scores; they belong only to the appendix domain-transfer track.
- adapter-merging papers are excluded from the study benchmark because they answer a different empirical question than shared joint-training multitask PEFT.

## Notes

- This document is a planning note, not the implementation itself.
- Exact dataset inventories and sampling rules still need to be finalized during implementation.
- When citing exact numbers from the papers, prefer the structured KB first, then confirm in the raw PDF if the KB marks a claim as needing review.
- Two exclusion/background papers referenced above are currently available only through `papers/further` in this checkout; their raw PDFs are not present under `papers/`.
