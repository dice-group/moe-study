# Parameter Matrix (MoE-Study)

This is the consolidated parameter ledger for the multilingual comparison runs. It records:
- What we set in code/configs.
- Whether each setting is "paper-best", "unified for fairness", or "script default".
- Notes where the implementation diverges from the paper or needs review.

Sources:
- `scripts/slurm/comparison/comparison_config.json`
- `scripts/slurm/comparison/comparison_config_smoke.json`
- `scripts/slurm/comparison/run_comparison.py`
- `scripts/slurm/comparison/adamole_job.sh`
- `scripts/slurm/comparison/adamoe_1_mola_job.sh`
- `scripts/slurm/comparison/cola_job.sh`
- `scripts/slurm/comparison/hydralora_job.sh`
- `scripts/slurm/comparison/moelpr_job.sh`
- Papers: `papers/further/*.txt` (AdaMoLE, AdaMoE/daMOE, CoLA, HydraLoRA, MoE-LPR)
- Design context: `papers/further/AGENTS.md`, `docs/token_budgets_moe_study.md`, `docs/onboarding_status.md`

Legend:
- Paper-best: explicitly described in the paper (or clearly implied).
- Unified: intentionally aligned across methods for fairness.
- Script default: set in scripts/configs, not stated in paper.
- Extension: behavior not in the paper (implementation-specific).
- Unclear: could not find explicit value in the extracted text.

## Decision log (fairness vs paper-best)

Goal: keep comparisons fair and tractable while staying as close as possible to each paper's best run. The default policy is:
- Use paper-best when it does not break cross-method fairness.
- Unify only when the parameter directly controls capacity or compute cost.
- Avoid ablations unless explicitly requested.

Decision key:
- Makes sense? Yes / Maybe / No, from a fairness-first viewpoint.
- Review: mark if it should be reconsidered later.

| Decision | Current choice | Paper reference | Rationale | Makes sense? | Review |
| --- | --- | --- | --- | --- | --- |
| Expert count (AdaMoLE, MoLA, MoE-LPR) | 4 total experts | AdaMoLE: 8 experts @ rank 4; daMOE: 8 true + nulls; MoE-LPR: 6 total | Capacity parity across methods | Maybe | Yes |
| LoRA rank/alpha (AdaMoLE/MoLA/CoLA/Hydra) | r=8, alpha=16 | CoLA default r=8; AdaMoLE uses total rank 32; Hydra varies | Keep comparable update size | Maybe | Yes |
| MoE-LPR top-k | 2 | Paper: K=2 | Paper-aligned routing | Yes | No |
| MoLA top-k | 3 | Paper: k=3 in main config | Paper-aligned, but fewer experts in our config | Maybe | Yes |
| MoLA aux loss coef | 0.02 | Paper: alpha=0.02 (epoch 1) | Paper-aligned magnitude | Yes | No |
| MoLA null expert penalty | 0 | Not specified | Stability in practice | Maybe | Yes |
| AdaMoLE aux loss coef | 1e-3 | Paper: 1e-3 | Paper-aligned | Yes | No |
| AdaMoLE max threshold | 0.125 | Paper: tau_max not explicit | Heuristic to avoid zero routing | Maybe | Yes |
| Learning rate | 2e-4 (most), 1e-4 (MoE-LPR stage2) | AdaMoLE: 1e-4; daMOE: 5e-5; MoE-LPR: 5e-5 | Shared optimizer schedule to limit search | Maybe | Yes |
| LR scheduler | cosine | AdaMoLE: constant; MoE-LPR: cosine | Shared schedule for comparability | Maybe | Yes |
| Batch size (per device) | AdaMoLE/CoLA/Hydra=12, MoLA=42, MoE-LPR=18 | Papers report total batch sizes on different hardware | Fit memory + throughput; not matched | Maybe | Yes |
| Precision | fp16 | MoE-LPR uses bf16 | Cluster default, reduce config surface | Maybe | Yes |
| Seq length | dataset default (likely 1024) | AdaMoLE uses 256 | Match multilingual data pipeline | Maybe | Yes |
| Shared datasets per tier | Yes | Not paper-specific | Fair comparison across methods | Yes | No |
| Wall-time per tier | From `docs/token_budgets_moe_study.md` | Not paper-specific | Equal time budget across methods | Yes | No |
| GPU scaling by model | 1/2/4 GPUs for 1B/3B/8B | Not paper-specific | Cluster efficiency, consistent budgets | Yes | No |
| MoE router use in CoLA/Hydra | Enabled | Not in paper | Compare MoE variants under shared routing | Maybe | Yes |

## Global comparison settings (shared unless noted)

Data and routing:
- `tokenized_base = /scratch/anonymous-project/project_data/moe_study/tokenized/fw_l2v_samples_moe_study_comparison` (Unified; `scripts/slurm/comparison/comparison_config.json`)
- `dataset_dir = ./LLaMA-Factory/data` (Unified; `scripts/slurm/comparison/comparison_config.json`)
- `template = llama3` (Unified; all job scripts)
- `tokenized_path = <tokenizer>/<tier>` (Unified; `run_comparison.py` sets `GENERIC_TOKENIZED_PATH`)
- MoE-LPR stage paths:
  - stage1: `<tokenized_base>/moelpr/<tokenizer>/<tier>/stage1_newlangs`
  - stage2: `<tokenized_base>/moelpr/<tokenizer>/<tier>/stage2_english`
  (Unified; `run_comparison.py`)
- `language_column = language`, `language_map = ./configs/moelpr_lang_map.json`, `moelpr_original_language = english` (Paper-aligned for MoE-LPR; `comparison_config.json`)

Models and tiers:
- Models: `Llama-3.2-1B`, `Llama-3.2-3B`, `Llama-3.1-8B` (Unified; `comparison_config.json`)
- Tiers: `5_langs`, `10_langs`, `22_langs`, `46_langs`, `96_langs`, `200_langs` (Unified; `comparison_config.json`)

GPU scaling:
- 1B: `gpus_per_job = 1`, 3B: `gpus_per_job = 2`, 8B: `gpus_per_job = 4` (Unified; `comparison_config.json`)
- Accelerate FSDP:
  - 2 GPUs uses `LLaMA-Factory/examples/accelerate/fsdp_2gpu_config.yaml`
  - 4 GPUs uses `LLaMA-Factory/examples/accelerate/fsdp_4gpu_config.yaml`
  (Unified; `run_comparison.py`)
- 1 GPU uses DeepSpeed (`configs/deepspeed/comparison_ds_config.json`) (Unified; job scripts)

Wall-time schedule (tier hours):
- 5: 24.00, 10: 27.69, 22: 36.55, 46: 54.28, 96: 91.20, 200: 168.00
  (Unified; `docs/token_budgets_moe_study.md`, enforced in `run_comparison.py` via `--time`)

Evaluation:
- `eval_strategy = steps`, `eval_steps = 200`, `save_steps = 200` (Unified; job scripts + `comparison_config.json`)
- `lm_eval.enabled = true`, tasks loaded from `./configs/lm_eval_tasks.txt` (Unified; `comparison_config.json`)

Logging:
- W&B project + group naming in `run_comparison.py`
- Per-run names append hyperparams + random id in each job script (Script default)

## Shared training defaults (set in job scripts)

These are common unless overridden per-approach in `comparison_config.json`:
- `num_train_epochs = 1` (Unified; job scripts)
- `learning_rate = 2e-4` (Unified; job scripts; MoE-LPR stage2 uses 1e-4)
- `lr_scheduler_type = cosine` (Unified; job scripts)
- `warmup_ratio = 0.03` (Unified; job scripts)
- `gradient_accumulation_steps = 2` (Unified; job scripts)
- `logging_steps = 10` (Unified; job scripts)
- `bf16 = False`, `fp16 = True` (Unified; job scripts)
- `lora_dropout = 0.05` (Unified; job scripts; LoRA-based only)
- `lora_target = q_proj,k_proj,v_proj,o_proj` (Unified; job scripts; LoRA-based only)
- `max_steps = 0` (Script default; job scripts)
- `disable_gradient_checkpointing = True` (Script default; in AdaMoLE, CoLA, HydraLoRA)

Notes:
- MoE-LPR explicitly sets `--seed` (default 42). Other approaches do not set seed explicitly (uses LLaMA-Factory defaults).

## AdaMoLE (adamole)

Paper notes (from `AdaMoLE.txt`):
- Dynamic threshold routing, not fixed top-k.
- Threshold network with max threshold `tau_max` (paper defines formula, does not give numeric values in extracted text).
- Uses Switch-style load balancing loss.
- Experiment settings in paper (explicit):
  - Base model: Llama-2-7B.
  - LoRA baseline: rank 32, alpha 16, dropout 0.05.
  - MoE variants (MoLE/AdaMoLE): 8 LoRA experts, each rank 4 (total rank 32).
  - Target modules: Wq, Wk, Wv, Wo (self-attention).
  - Max sequence length: 256.
  - Optimizer: AdamW.
  - Total batch size: 16.
  - Learning rate: 1e-4 with constant schedule + 200 warmup steps.
  - Epochs tuned between 1 and 20.
  - Load-balancing aux loss coef: 1e-3.

Current config (`comparison_config.json` + `adamole_job.sh`):
- `adamole_num_experts = 4` (Unified; to match MoLA and MoE-LPR)
- `adamole_max_threshold = 0.125` (Script default; paper does not give numeric `tau_max` in extracted text)
- `adamole_aux_loss_coef = 0.001` (Paper-aligned; paper uses 1e-3)
- `lora_rank = 8`, `lora_alpha = 16` (Unified across LoRA-based methods)
- `per_device_train_batch_size = 12` (Unified; paper uses total batch size 16)

Paper-vs-current deltas:
- Experts: paper uses 8 experts @ rank 4 (total rank 32); current uses 4 experts @ rank 8 (total rank 32).
- LR/schedule: paper uses 1e-4 with constant schedule + 200 warmup steps; current uses 2e-4 with cosine + warmup_ratio 0.03.
- Sequence length: paper uses 256; current uses dataset defaults (likely 1024).
- Batch size: paper total batch size 16; current per-device 12 with GA=2.

Implementation notes:
- Load balancing uses `adapted_gate_logits = softmax(gate) - tau` (see `LLaMA-Factory/src/peft/tuners/adamole/layer.py`).
- Divergence noted in `papers/further/AGENTS.md`: load-balancing uses adapted logits rather than raw softmax.

## MoLA / AdaMoE (adamoe_1)

Paper notes (from `AdaMoe.txt`):
- daMOE adds null experts and increases `k` to allow adaptive number of true experts.
- Load-balancing encourages average null usage. Exact `k`/null penalty values are not in extracted text.
- Explicit settings in paper (for Mixtral + QLoRA fine-tuning):
  - Base model: Mixtral-8x7B (8 FFN experts, top-2 router).
  - QLoRA 4-bit.
  - LoRA target modules: gate, gate2, w1, w2, w3.
  - LoRA rank: 8.
  - Learning rate: 5e-5.
  - 1000 samples per dataset, 2 epochs.
  - Reported main config: m = 8 null experts, k = 3 (top-3 selection).
  - Two-phase load-balancing strength: alpha = 0.02 then 0.0001.

Current config (`comparison_config.json` + `adamoe_1_mola_job.sh`):
- `mola_num_experts = 4` (Unified)
- `mola_top_k = 3` (Paper-aligned with daMOE main config k=3; note expert counts differ)
- `mola_use_null_expert = True` (Script default; not set in config)
- `mola_router_aux_loss_coef = 0.02` (Paper-aligned with alpha=0.02 epoch-1 in daMOE)
- `mola_null_expert_penalty = 0` (Unified? set for stability; configured)
- `mola_aux_loss_annealing = True` (Configured, but currently unused in implementation)
- `lora_rank = 8`, `lora_alpha = 16` (Unified)
- `per_device_train_batch_size = 42` (Script default; paper uses small sample count, not batch size per device)

Paper-vs-current deltas:
- Expert counts: paper uses 8 true experts + 8 null (m=8, k=3); current uses 4 true + 1 null.
- LR: paper uses 5e-5; current uses 2e-4.
- QLoRA 4-bit in paper; current uses standard fp16.

Implementation notes:
- Only a single null expert is implemented (see `LLaMA-Factory/src/peft/tuners/adamoe_1/layer.py`).
- Aux-loss annealing is defined but not used (see `AGENTS.md` divergence note).

## CoLA

Paper notes (from `CoLA Collaborative Low-Rank Adaptation.txt`):
- Flexible #A/#B: CoLA allows `#A = M`, `#B = N`.
- Collaborative strategies: fully (CoLA^T), random (CoLA^dagger), heuristic (CoLA^ddagger).
- PiSSA initialization is emphasized in the paper for faster convergence.
- LoRA rank default: "unless stated otherwise, LoRA rank is set to 8" (paper default).

Current config (`comparison_config.json` + `cola_job.sh`):
- `num_A = 1`, `num_B = 3` (Configured; closest to asymmetric structure)
- `cola_strategy = fully` (Paper-aligned; matches CoLA^T)
- `use_cola_experts = True` (Extension; enables router + experts)
- `cola_num_experts = 3`, `cola_top_k = 1` (Extension; MoE routing)
- `lora_rank = 8`, `lora_alpha = 16` (Unified)
- `per_device_train_batch_size = 12` (Script default; paper does not give explicit batch size in extracted text)

Paper-vs-current deltas:
- CoLA paper does not specify `num_A`/`num_B` values in extracted text; current uses 1x3.
- MoE router use is an extension; paper focuses on collaborative A/B without MoE routing.

Implementation notes:
- MoE-style router and language guidance are extensions (see `AGENTS.md`).
- PiSSA init is supported in code but not enabled in these scripts.

## HydraLoRA

Paper notes (from `HydraLoRA ... .txt`):
- Shared A with multiple B heads.
- Paper proposes automatic intrinsic-component discovery (TF-IDF + k-means).
- Ranks vary (r=8/16/32 in experiments).
- Default LoRA rank referenced as r=8 in comparisons.

Current config (`comparison_config.json` + `hydralora_job.sh`):
- `use_hydralora_experts = True` (Extension; MoE routing)
- `hydralora_num_experts = 3`, `hydralora_top_k = 1` (Configured)
- `lora_num (num_B) = 3` (Configured)
- `lora_rank = 8`, `lora_alpha = 16` (Unified)
- `per_device_train_batch_size = 12` (Script default; paper does not give explicit batch size in extracted text)

Paper-vs-current deltas:
- Paper does not use MoE routing; current uses router (extension).
- Paper does not specify fixed number of B heads in extracted text; current uses 3.

Implementation notes:
- Intrinsic-component clustering is not implemented (see `AGENTS.md`).

## MoE-LPR

Paper notes (from `MoE-LPR.txt`):
- Upcycle dense FFN to MoE, add new experts; paper example uses 5 new experts (6 total).
- `top_k = 2` is used in routing.
- Stage 1 uses load-balancing loss (weight 0.01).
- Stage 2 uses LPR loss; original languages routed to expert 0.
- Explicit training settings:
  - Base model: Qwen-1.5-1.8B.
  - Batch size: 512, sequence length: 1024.
  - Learning rate: 5e-5, cosine scheduler.
  - bf16 + flash attention.
  - Stage 2 uses 50K docs per original language, 100K per expanded language.

Current config (`comparison_config.json` + `moelpr_job.sh`):
- `moelpr_num_experts = 4` (Unified for fairness; deviates from paper example of 6)
- `moelpr_top_k = 2` (Paper-aligned)
- `moelpr_stage1_aux_loss_coef = 0.01` (Paper-aligned)
- `moelpr_stage2_lpr_loss_coef = 0.1` (Configured; paper does not give explicit value in extracted text)
- `per_device_train_batch_size = 18` (Script default; not controlled by `comparison_config.json`)
- `stage1_lr = 2e-4`, `stage2_lr = 1e-4` (Script default)

Paper-vs-current deltas:
- Expert count: paper uses 6 total experts; current uses 4.
- LR: paper uses 5e-5; current uses 2e-4 / 1e-4.
- Batch size: paper uses 512; current uses 18 per device with GA=2.
- Precision: paper uses bf16; current uses fp16.

Implementation notes:
- Stage 2 passes `moelpr_aux_loss_coef` but code only uses `lpr_loss_coef` for stage 2.
- Stage 1/2 batch sizes are not controlled by `PER_DEVICE_TRAIN_BATCH_SIZE` from `run_comparison.py`.

## Unified-for-fairness choices (explicit)

These are chosen to keep comparisons fair even if they are not paper defaults:
- `num_experts = 4` for AdaMoLE, MoLA, MoE-LPR (capacity parity).
- `lora_rank = 8`, `lora_alpha = 16`, `lora_dropout = 0.05`, `lora_target = q_proj,k_proj,v_proj,o_proj`
  across AdaMoLE, MoLA, CoLA, HydraLoRA.
- Shared dataset per (model x tier) and identical language proportions.
- Same wall-time per tier across model sizes (`docs/token_budgets_moe_study.md`).
- Shared scheduler, warmup, eval cadence, and template.

## Paper-best choices (explicit or closest match)

Where we intentionally align with papers (or their strongest reported configs):
- AdaMoLE: dynamic threshold router, load-balancing loss (implementation); aux loss coef = 1e-3.
- CoLA: `cola_strategy = fully` (CoLA^T) and default LoRA rank 8.
- MoE-LPR: `top_k = 2`, load-balancing weight 0.01, Stage 2 LPR routing to original language expert.
- HydraLoRA: asymmetric A/B structure (shared A + multiple B), though clustering not implemented.

## Open items / review targets

These are candidates for a second look:
- AdaMoLE: `adamole_max_threshold` (tau_max) not given in extracted text; current value is heuristic.
- AdaMoLE: LR/schedule/batch/seq_len differ from paper (paper: 1e-4, constant, batch 16, seq_len 256).
- MoLA: expert counts and null expert counts differ (paper uses m=8, k=3); LR differs (paper: 5e-5).
- CoLA/HydraLoRA: MoE router usage is an extension; consider disabling for paper-faithful runs.
- CoLA: num_A/num_B not specified in extracted text; current 1x3 is heuristic.
- HydraLoRA: number of B heads not specified in extracted text; current 3 is heuristic.
- MoE-LPR: expert count unified to 4 (paper example uses 6 total experts).
- MoE-LPR: LR/batch/precision differ from paper (paper: 5e-5, batch 512, bf16).
- Explicit seed control is only in MoE-LPR.
