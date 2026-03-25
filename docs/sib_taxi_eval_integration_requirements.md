# SIB200 / Taxi1500 Eval Integration Requirements

## Goal
- Add SIB200 and Taxi1500 as reproducible multilingual eval datasets in this repo's `lm-evaluation-harness` + `pipeline/*` flow.

## Dataset Facts
- SIB-200:
  - Multilingual topic classification dataset for 200+ languages/dialects.
  - 7 labels: `science/technology`, `travel`, `politics`, `sports`, `health`, `entertainment`, `geography`.
  - Public Hugging Face dataset exists: `Davlan/sib200`.
  - This is the easier integration target because the data is already loadable by language code.
- Taxi1500:
  - Multilingual sentence classification dataset with 6 labels: `Recommendation`, `Faith`, `Description`, `Sin`, `Grace`, `Violence`.
  - Covers 1502 languages, but the public release is only a distributable corpus subset, not a ready-to-load benchmark dataset for all languages.
  - The public repo states that 1871 editions in 823 languages are distributable at publication time; full access depends on Parallel Bible Corpus access.
  - This is the harder integration target because we need a local preprocessing step and a pinned released subset.

## Current Repo State
- The pipeline already expects a local harness at `scripts/eval/latest_checkpoint/lm-evaluation-harness`.
- That path is currently a nested Git repo, but it is not registered in the parent repo's `.gitmodules`.
- The wrapper already supports external task packs through `LM_EVAL_INCLUDE_PATH` / `--include-path`.
- Current harness contents:
  - no repo-native per-language `sib200_<lang>` tasks
  - no Taxi1500 tasks
  - only some Afrobench SIB prompt assets, which are not the same thing as the task layout we need here
- The local harness docs mention multi-directory `--include_path`, but this checkout's `TaskManager` originally treated the full colon-joined string as one path. That had to be patched in the harness.

## Implemented Design
- Task definitions now live directly in the harness task tree:
  - `scripts/eval/latest_checkpoint/lm-evaluation-harness/lm_eval/tasks/sib200`
  - `scripts/eval/latest_checkpoint/lm-evaluation-harness/lm_eval/tasks/taxi1500`
- Generated per-language YAMLs live under ignored `configs/` subdirectories, matching the FLORES pattern more closely.
- The tracked files mirror the FLORES pattern:
  - generator script
  - common include template
  - helper utils
  - README / `.gitignore`
- The generated files are not meant to be tracked:
  - `scripts/eval/latest_checkpoint/lm-evaluation-harness/lm_eval/tasks/sib200/configs/sib200_<lang>.yaml`
  - `scripts/eval/latest_checkpoint/lm-evaluation-harness/lm_eval/tasks/taxi1500/configs/taxi1500_<lang>.yaml`
- FLORES remains where it was.
- The wrappers and `pipeline/stages/benchmark.py` now build a combined include path for:
  - FLORES configs
  - SIB200 generated configs
  - Taxi1500 generated configs
- A harness bootstrap script now regenerates the task configs automatically before eval:
  - `scripts/eval/latest_checkpoint/lm-evaluation-harness/lm_eval/tasks/generate_local_task_configs.py`
- The harness `TaskManager` was patched to split a string `include_path` on `os.pathsep`, so the documented multi-directory behavior actually works in this checkout.

## Task Behavior
- SIB200:
  - one generated YAML per HF config: `sib200_<lang_script>`
  - loads directly from `Davlan/sib200`
  - fixed 0-shot multiple-choice classification prompt
  - reports `acc` and weighted `f1`
- Taxi1500:
  - one generated YAML per repo-supported language: `taxi1500_<lang_script>`
  - uses a `custom_dataset` loader
  - downloads and caches:
    - official English labels pinned to GitHub commit `ce17cf7ac0cb2e6c57ab7d9da4cbf5924ad313c3`
    - `Taxi1500-c_v3.0.zip`
  - rebuilds train/validation/test by verse ID
  - chooses one released edition per ISO by:
    - maximum overlap with the 1,077 labeled verse IDs
    - largest total verse count
    - filename tie-break
  - requires at least 900 labeled verses by default
  - reports `acc` and weighted `f1`

## What To Consider
- Taxi1500 is Bible-domain data. That is fine for multilingual transfer evaluation, but we should document the domain bias and not present it as a general-domain classifier benchmark.
- Coverage mismatch is expected:
  - SIB200 is broad and should align well with current eval tiers.
  - Taxi1500 coverage is constrained by the released distributable corpus. The repo task catalogs were pruned to the subset we can rebuild from `Taxi1500-c_v3.0` with at least 900 labeled verses.
- Comparison fairness depends on fixing one prompt protocol. Because these datasets are not currently native tasks in our harness flow, prompt drift would make results hard to compare.

## Verification Plan
- Harness-level checks:
  - `lm-eval ls tasks --include_path <sib200-configs>:<taxi1500-configs> | rg 'sib200_|taxi1500_'`
  - `lm-eval validate --tasks sib200_eng_Latn,taxi1500_eng_Latn --include_path <sib200-configs>:<taxi1500-configs>`
- Smoke eval checks:
  - run one SIB200 language and one Taxi1500 language on a base model with `--limit`
  - verify outputs contain the expected metric and task name
- Pipeline dry-run checks:
  - use a profile with `benchmark.enabled=true`
  - ensure benchmark language resolution is explicit via `sample.languages` or an existing sample manifest
  - run `python3 -m pipeline.run --config configs/pipeline --profile <profile> --start-at benchmark --stop-after benchmark --dry-run`
  - verify `selected_tasks_summary.json` contains non-zero SIB200 / Taxi1500 counts
- Pipeline real checks:
  - evaluate one known checkpoint with a tiny tasks file containing one SIB200 task and one Taxi1500 task
  - confirm benchmark outputs are written
  - confirm the report stage groups them correctly and surfaces per-language metrics

## Verification Results
- Discovery:
  - `TaskManager(include_path=...)` now sees `206` SIB200 tasks and `60` Taxi1500 tasks from the generated `configs/` directories.
- Validation:
  - `lm-eval validate --tasks sib200_eng_Latn,taxi1500_deu_Latn ...` passes with include paths pointing at the generated `configs/` directories.
- Real smoke evals:
  - `sib200_eng_Latn` runs end to end with `distilgpt2` on CPU and writes results.
  - `taxi1500_deu_Latn` runs end to end with `distilgpt2` on CPU, builds and caches the reconstructed dataset, and writes results.
- Pipeline selector:
  - the 10-language catalog now selects `42` tasks:
    - `9` Belebele
    - `9` SIB200
    - `6` Taxi1500
    - `18` FLORES
- Remaining gap:
  - no full `pipeline/stages/benchmark.py` checkpoint run was executed in this pass because there was no disposable adapter checkpoint staged for benchmark execution inside this repo checkout.

## Sources
- SIB-200 repo: https://github.com/dadelani/sib-200
- SIB-200 dataset card: https://huggingface.co/datasets/Davlan/sib200
- SIB-200 paper: https://arxiv.org/abs/2309.07445
- Taxi1500 repo: https://github.com/cisnlp/Taxi1500
- Taxi1500 paper: https://arxiv.org/abs/2305.08487
- Harness task docs:
  - https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md
  - https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
