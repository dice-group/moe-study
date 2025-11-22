# Comparison Pipeline Notes

This file summarizes how the newly added CoLA and HydraLoRA runs are configured inside
`scripts/slurm/run_cluster_comparison.sh` and highlights remaining gaps relative to the original papers.

## CoLA

- run with `num_A=2` and `num_B=3`, PiSSA init (same as other CoLA usage), fully collaborative strategy (`CoLA⊺`), which is the best-performing configuration reported in the paper.
- `use_cola_experts` stays disabled, becasue this is for PG
- The same dataset/tokenization/settings used for the other baselines are reused to keep the comparison controlled.

Future considerations:
- Explore CoLA†/‡ variants or enable the MoE experts when comparing energy/computation budgets.
- Add dataset-specific language priors through `language_map` once multilingual benchmarks are incorporated. (but this is primarly PG)

## HydraLoRA

- We configure a shared `A` with rank 16, four `B` heads per adapter, and enable the Hydra MoE router (`use_hydralora_experts=true`, `hydralora_num_experts=4`, `top_k=2`) to mirror the asymmetrical design in the paper.
- Current implementation still requires manual selection of the number of intrinsic components (i.e., the number of `B` heads). The paper proposes TF–IDF + k-means clustering to discover these components automatically. We have added a TODO in `peft/tuners/hydralora/layer.py` to track this missing automation.

Future considerations:
- Implement the clustering/initialization pipeline so HydraLoRA can infer intrinsic components instead of relying on a fixed `lora_num`.
- Benchmark non-MoE Hydra runs (single adapter with multiple `B`s) to disentangle the gains from the asymmetric structure vs. routing.

## General

- All new jobs inherit the same core hyper-parameters (dataset, optimizer, schedule, logging cadence) as the existing comparisons to keep wall-clock and compute costs comparable.
- WANDB run names are suffixed with timestamps, which makes it straightforward to align metrics across AdaMoLE, MoLA, MoE-LPR, CoLA, and HydraLoRA.
