# MoE-LPR Loss Investigation (2026-02-09)

## Scope
Controlled A/B training on the same model and tokenized dataset to investigate why MoE-LPR loss differs from other approaches.

- Date: 2026-02-09
- Hardware: 2x NVIDIA H100 80GB
- Model: `meta-llama/Llama-3.2-1B`
- Data: `/data/project_data/moe_study/tokenized/preview_subset_llama32_1b`
- Attention backend: `sdpa` (FlashAttention was unavailable on this host due GLIBC mismatch)
- Trainer setup (both runs): 1 epoch, per-device batch size 4, grad accumulation 2, eval every 20 steps, save every 40 steps
- W&B project: `dice-nlp/moe-study-loss-investigation-20260209`

## Runs

| Approach | Run | Final train/loss | Final eval/loss | Runtime (s) | Train tokens/s |
|---|---|---:|---:|---:|---:|
| CoLA | https://wandb.ai/dice-nlp/moe-study-loss-investigation-20260209/runs/withs3z8 | 1.8704 | 1.8158 | 242 | 9828.6 |
| MoE-LPR (Stage 1) | https://wandb.ai/dice-nlp/moe-study-loss-investigation-20260209/runs/ivz30ltp | 2.4080 | 1.9451 | 515 | 4870.2 |

## Observations

1. MoE-LPR starts with much higher training loss.
- CoLA early points: `(step 1: 3.788)`, `(step 10: 1.6815)`
- MoE-LPR early points: `(step 1: 10.2714)`, `(step 10: 3.7545)`

2. MoE-LPR converges slower and remains higher on eval/loss at every checkpoint.
- CoLA eval trajectory: `1.9352 -> 1.8158`
- MoE-LPR eval trajectory: `2.5887 -> 1.9451`

3. MoE-LPR stage-1 throughput is ~2x lower in this setup.
- CoLA: `~9828 tokens/s`
- MoE-LPR: `~4870 tokens/s`

4. `train/moelpr_aux_loss` is frequently zero during eval windows.
- In the MoE-LPR run history, many `moelpr_aux_loss=0` events occur at eval boundaries (e.g. around global steps `20, 40, 60, ...`).
- Count observed in this run: `48` zero-valued aux entries out of `330` aux entries.
- This can make summary values look misleading if interpreted as “final training aux loss”.

## Code Evidence (why losses are not directly comparable)

1. MoE-LPR explicitly adds an extra auxiliary term to training loss.
- `LLaMA-Factory/src/llamafactory/train/sft/trainer.py:145`
- `LLaMA-Factory/src/llamafactory/train/sft/trainer.py:152`
- `LLaMA-Factory/src/llamafactory/train/sft/trainer.py:350`

2. In MoE-LPR model code, stage-1 aux loss is load-balancing loss scaled by `aux_loss_coef`.
- `LLaMA-Factory/src/peft/tuners/moelpr/model.py:111`
- `LLaMA-Factory/src/peft/tuners/moelpr/model.py:130`

3. MoE-LPR returns zero aux loss whenever model is not in training mode.
- `LLaMA-Factory/src/peft/tuners/moelpr/model.py:100`
- This explains zero-valued aux logs around eval sections.

4. Comparison pipeline uses a different stage-2 dataset for MoE-LPR (`stage2_english`) than generic datasets used by other approaches.
- `scripts/slurm/comparison/run_comparison.py:635`
- `scripts/slurm/comparison/run_comparison.py:637`
- `scripts/slurm/comparison/moelpr_job.sh:49`
- `scripts/slurm/comparison/moelpr_job.sh:241`

5. MoE-LPR stage-2 is router-only (most parameters frozen), changing optimization dynamics.
- `LLaMA-Factory/src/llamafactory/model/adapter.py:1092`
- `LLaMA-Factory/src/llamafactory/model/adapter.py:1096`
- `LLaMA-Factory/src/llamafactory/model/adapter.py:1100`

## Conclusion
The observed MoE-LPR “different loss” is real, but the largest causes are methodological and objective-level:

- MoE-LPR stage-1 optimizes `LM loss + routing aux loss`, while CoLA/Hydra-style baselines typically optimize different objective compositions.
- In the default comparison pipeline, MoE-LPR stage-2 is not dataset-equivalent (`stage2_english`) and parameter-update-equivalent (router-only) to baseline runs.
- Aux-loss logging semantics can create misleading summary values (zeros during eval windows).

These factors make direct raw-loss comparisons across methods (especially including MoE-LPR stage-2) non-equivalent unless objectives and data are aligned.

## Runtime Note
- The MoE-LPR run ended with an NCCL async teardown warning after final eval/logging, but W&B marked the run `finished` and all target metrics were recorded.
