# Domain Verification Scripts

Purpose:
We fine-tuned LLaMA 3 (1B, 3B, and 8B) models for multilinguality using different MoE-based PEFT and MoE-LoRA methods. However, all trained variants showed decreased multilingual performance.
We want to investigate whether:
 - The issue is caused by an error in our training pipeline, or
 - The degradation is specific to multilingual fine-tuning.

To test this, we plan to fine-tune the same LLaMA models on domain-specific tasks (e.g., mathematics, law, and medicine), where prior research shows performance improvements are typically achievable.
This will help us determine whether the performance degradation is:
 - A multilingual-specific issue, or
 - A general problem affecting all domains within our training pipeline.

Goal: check whether performance drops are caused by MoE-PEFT implementation/routing issues vs multilingual data/task effects.

## I Ran on LoLA (2 H100 GPUs)

Completed overnight run:
- base model: `meta-llama/Llama-3.2-1B`
- methods: `full`, `lora`, `adamole`, `mola`
- tracks (tiers):
- `gsm8k` (math)
- `dolly` (general instruction)
- `code` (code generation)
- train matrix: `3 tracks x 4 methods = 12 runs` (all `rc=0`)
- eval matrix: `12 runs` (all finished successfully)

### Why These Tracks

- `gsm8k -> gsm8k`: tight train/eval coupling for implementation sanity checks.
- `dolly -> mmlu`: distribution shift to test generalization.
- `code -> humaneval`: code-specific behavior under strict functional metric.

## Scripts Used

- `scripts/domain_verification/prepare_assets.py`
  - checks model/dataset/task availability.
- `scripts/domain_verification/train_domain_track.sh`
  - single training job per `(method, track)`.
- `scripts/domain_verification/eval_domain_suite.sh`
  - single benchmark eval for base model or trained adapter/checkpoint.
- `scripts/domain_verification/run_nightly_queue_2gpu.sh`
  - queue launcher across 2 GPUs, auto-start next job when one finishes.

Idea is, you only have to restart 'run_nightly_queue_2gpu.sh' and will get the results after some hours and can check whether you improved or not.

## Datasets, Model, Evals

- training datasets:
- `gsm8k_train`
- `dolly_15k_en` (Databricks Dolly)
- `codealpaca`
- evaluation benchmarks:
- `gsm8k` (metric: `exact_match,strict-match`)
- `mmlu` (metric: `acc,none`)
- `humaneval` (metric: `pass@1,create_test`)

## Where Outputs Are

(e.g base path on lola = '/anonymous-institution/users/j/anonymous-user/profiles/unix/cs/MERLIN/moe-study')
- train/eval status:
- `outputs/domain_verification_night_20260210/nightly_queue_logs/train_status.tsv`
- `outputs/domain_verification_night_20260210/nightly_queue_logs/eval_status.tsv`
- per-job logs:
- `outputs/domain_verification_night_20260210/nightly_queue_logs/train_*.log`
- `outputs/domain_verification_night_20260210/nightly_queue_logs/eval_*.log`
- eval json results:
- `outputs/domain_verification_night_20260210/eval/gsm8k/<method>/`
- `outputs/domain_verification_night_20260210/eval/dolly/<method>/`
- `outputs/domain_verification_night_20260210/eval/code/<method>/`
- comparison workbook:
- `outputs/domain_verification_night_20260210/domain_verification_comparison.xlsx`

## Results (With Base, Final)

Metric direction:
- `gsm8k exact_match,strict-match`: higher is better
- `mmlu acc,none`: higher is better
- `humaneval pass@1,create_test`: higher is better

Training budget (important context):
- this was a short sanity run, not a long convergence sweep
- training used `1 epoch` (`--num_train_epochs 1.0`), with no `--max_steps` and no `--max_samples` override
- queue settings used in this run: `nproc_per_node=1`, `train_batch_size=8`, `grad_accum=2`
- observed train wall-time from `train_status.tsv`: `2.2` to `25.5` minutes per run (avg `12.4` min, 12 runs total)
- observed optimization steps by track were roughly: `gsm8k ~444`, `dolly ~892`, `code ~1189`

| Track | Metric | base | full | lora | adamole | mola | winner |
|---|---|---:|---:|---:|---:|---:|---|
| gsm8k | exact_match,strict-match | 0.0637 | 0.0190 | 0.1145 | 0.1190 | 0.1236 | mola |
| dolly->mmlu | acc,none | 0.3774 | 0.2291 | 0.3638 | 0.3566 | 0.3655 | base |
| code->humaneval | pass@1,create_test | 0.1951 | 0.0305 | 0.2134 | 0.2012 | 0.1890 | lora |

Hard truth:
- Fine-tuning is not a universal win vs base in this setup.
- `full` is worst on all 3 suites.
- `lora` is most reliable; MoE variants are mixed.

## Not Included Yet

- Multilingual benchmark track is not included in this specific overnight folder.

See `docs/domain_verification_runbook.md` for full procedure details.
