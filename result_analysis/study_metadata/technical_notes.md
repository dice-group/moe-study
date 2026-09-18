# Study metadata: definitions and special cases

## Scope and units

The tables cover 159 Llama training runs: 55 rank-sweep, 68 expert-sweep,
22 mixed-follow-up, and seven runs each for final-96 and final-200.
They include the original failed/excluded CoLA runs as experiment records,
not as valid comparison results. Later CoLA retries and the Gemma/Falcon
extension are not included. Baseline evaluations are not training runs.

All runs use one node with four A100 GPUs. `world_size=4` means four
training processes, one per GPU. Sequence length is 1,024.

Join the tables on `dataset`; identify a run by (`stage`, `run_id`). The same
run ID can occur in more than one stage. `N/A` is intentional for MoV's
rank/scaling fields and the active-parameter point estimates replaced by bounds.

- Global batch = microbatch × gradient accumulation × world size.
- Optimizer steps count parameter updates, not microbatches.
- Tokens processed are recorded input positions.
- Token throughput = tokens processed / trainer runtime in seconds,
  aggregated across all four GPUs—not per GPU.
- Trainer runtime includes training-loop overhead; it is not GPU-busy time
  or total campaign cost including calibration, failed attempts, and evaluation.
- Counts use actual completed steps and recorded runtime, not planned-step
  throughput estimates.

## Budgets and comparison conditions

`training_budget_hours` must be read together with `budget_basis`:

| Stage | Budget (hours) | Clock |
|---|---:|---|
| Rank/expert sweeps | 3.6 | `job_clock`: includes setup |
| Mixed follow-up | 7.2 | `active_training` |
| Final-96 | 34 | `active_training` |
| Final-200 | 72 | `active_training` |

Use these explicit budget fields rather than inferring a duration from a
historical dataset or directory name (for example, `34p5h`).

Early experiments were initially step-calibrated toward 3.6 hours, then moved
to signal-based stopping. All 123 retained sweep submission scripts specify
a 3h51m Slurm limit and a stop signal 15 minutes before it: nominally 3.6 hours
from job start. Step/epoch limits may stop training earlier; shutdown and
checkpoint saving can affect measured duration. A configured deadline does
not establish which stop condition actually fired.

The early sweeps are **approximately budget-matched**, not identical in active
training time. Rank-sweep runtimes range from 206.75 to 217.06 minutes;
expert-sweep runtimes range from 197.86 to 216.59 minutes. The largest shortfall
from the nominal 216 minutes is **18.14 minutes**. These ranges include the
failed/excluded CoLA runs. Eight expert runs reached epoch 1: three original
CoLA configurations and all five MTL-LoRA configurations.

Early calibration maximized throughput without fixing a common global batch.
All mixed/final runs use global batch 48; sweep batch sizes vary and are
reported per run.

Some configurations could not fit all attention/FFN targets in GPU memory,
so their targets were reduced to attention only. Other methods, such as MoV,
have method-specific targets. Use `target_modules` for the actual settings:
when targets or batch sizes change alongside rank/expert count, the comparison
does not isolate rank/expert count alone.

Throughput differences between regimes are measured observations, not proof
of intrinsic architecture cost. Cluster variability was suggested as a cause,
but was not independently established.

## Rank, alpha, and active adapter parameters

`implementation_rank` is the rank stored in the adapter configuration.
For MoELoRA it is total rank; the selection analysis used per-expert configured
rank. All applicable saved alpha values are twice their implementation rank.
HMoRA uses its native alpha/sqrt(rank) scaling, so alpha=2×rank does not imply
the same multiplier as alpha/rank. MoV has no applicable LoRA rank/alpha.

Active parameters mean **configured trainable adapter capacity in one token's
routed path**, not total active model parameters. The frozen backbone is
reported separately. Counts come from checkpoint tensor shapes and saved
routing settings; total adapter counts match logged trainable counts.

| Method | Active adapter counting rule |
|---|---|
| LoRA | All A/B weights. |
| MoELoRA | Top-k expert weights, shared gate, and one task-embedding row. Numerical gate thresholds can reduce actual execution. |
| MoLA | All router weights, including null-slot outputs, plus bounds on selected real-expert weights. Null experts add no expert parameters. |
| AdaMoLE | Router/threshold weights plus conservative zero-to-all-experts bounds; selection is input-dependent. |
| HydraLoRA | Shared A, all softly mixed B heads, and router. |
| CoLA | All collaborative A/B components in the saved non-expert configuration. |
| MoV | All vector banks and routers in the saved dense configuration. |
| MTL-LoRA | Shared A, all B projections, and one task's lambda/mixing row. |
| HMoRA | Shared A, top-k B blocks, unique routers, shared task encoder, and one task-embedding row. |

For equal-sized experts, configured active capacity is shared parameters plus
top-k × parameters per expert. For MoLA, with E real experts, N null slots,
K selected slots, P parameters per real expert, and S router/shared parameters:

- Minimum = S + max(0, K − N) × P.
- Maximum = S + min(K, E) × P.

MoLA's saved router dimensions include null slots even where
`mola_use_null_expert` is false; the positive null-expert count and implementation
determine routing. AdaMoLE bounds are S to S + all expert weights, with S
including routers and thresholds.

There are 123 configured point counts and 36 bounds (32 MoLA, four AdaMoLE).
For bounds, `active_adapter_parameters=N/A`; use the minimum/maximum columns.
These are not measured routing averages or FLOP estimates. HMoRA can compute
unselected blocks, and MTL-LoRA can form mixtures for multiple tasks.
Shared weights and duplicate saved aliases count once; non-parameter buffers
are excluded.

## Dataset composition and English policy

Documents are source texts; examples are tokenized training sequences.
Language shares use the **prepared training split** as denominator and sum to
one; multiply by 100 for percentages. They are not measured per-run consumption:
multiplying a run's token count by these shares does not establish exact exposure.

| Dataset | Language-script labels | Construction recipe |
|---|---:|---|
| Original sweeps | 13 | Source sampling fraction 0.00025 |
| MoV expert sweep | 13 | Source sampling fraction 0.0005; larger prepared pool |
| Mixed follow-up | 13 | Expanded to 400,000 training examples |
| Final-96 | 102 | Source sampling fraction 0.0025 |
| Final-200 | 215 | Source sampling fraction 0.005 |

These fractions describe dataset construction, not a language's share.
Use frozen construction plans rather than stale training-profile defaults.
The regime names “96” and “200” differ from script-specific label counts;
the exact language-to-script reconciliation is not supplied by these tables.

English uses the **rounded median sampled document target of selected
non-English language entries**, not their mean or token share. This gives
19 English documents for final-96 (18 train, one validation) and 32 for
final-200 (31 train, one validation). English therefore does not retain its
natural source-corpus proportion. This balances document targets, not tokens.
In `languages.csv`, the English entry is labelled `english`, not `eng_Latn`.

## Evaluation panels

Evaluation is restricted to available benchmark tasks for the regime's training
language set. Configurations within a regime share a panel; different regimes
need not. Counts below are primary task/language/metric units, not unique languages.

| Regime | Belebele | FLORES | SIB200 | Taxi1500 |
|---|---:|---:|---:|---:|
| Sweeps / mixed | 9 | 18 | 9 | 6 |
| Final-96 | 40 | 100 | 51 | 58 |
| Final-200 | 65 | 166 | 84 | 93 |

FLORES includes both directions. Cross-regime claims about the same languages
require a common evaluation panel; the existing rankings compare within regimes.
