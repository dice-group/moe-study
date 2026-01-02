# Data Subsampling and Language Rates (MoE-Study)

This note documents how we subsample the massive sharded L2V data and derive
language keep rates so each model-size x tier stays within the token budgets
in `docs/token_budgets_moe_study.md`.

## Inputs and layout
- Sharded L2V data: `/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/sharded_samples/`
- Tier language lists: `/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fineweb2_metadata/l2v_nested_samples/{5,10,22,46,96,200}.csv`
- Per-language doc counts: `/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/lang_counts/language_sample_counts.tsv`
- Tokenization probes: `/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/lang_counts/token_probe_tokens_*.tsv`

## Step 1: Count documents per language
We count documents across shards (not per-language single-job, to balance very
large languages like English).

Scripts:
- `data_prep/merlin_data_prep/distributed_data_processor/scripts/count_language_samples.py`
- `data_prep/merlin_data_prep/distributed_data_processor/scripts/count_language_samples.sbatch`

Run the Slurm array (0-199). It outputs `part_###.tsv` files; merge them into
`language_sample_counts.tsv`.

## Step 2: Tokenization probe (avg tokens per doc)
We estimate average tokens per document for each language using a small probe:
1k docs/lang, three tokenizers (1B/3B/8B).

Script:
- `data_prep/merlin_data_prep/distributed_data_processor/scripts/tokenization_probe.py`

Outputs:
- `token_probe_tokens_llama3.2-1b.tsv`
- `token_probe_tokens_llama3.2-3b.tsv`
- `token_probe_tokens_llama3.1-8b.tsv`

These provide average tokens/doc per language for each tokenizer.

## Step 3: Build language keep-rates
We compute a **single global keep-rate** per tier (per tokenizer) to preserve
the original distribution across non-English languages, and we **always include
English** but cap it so it does not dominate.

Script:
- `data_prep/merlin_data_prep/distributed_data_processor/scripts/build_language_rates.py`

Logic:
- Use tier CSVs to select languages, then add `english`.
- Use counts + probe averages to estimate token mass per language.
- Set a global keep-rate for non-English languages to fit the token budget.
- Cap English to the **median non-English token mass**; compute an English
  keep-rate that hits this cap.

Outputs:
- `/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/lang_counts/language_rates/<tokenizer>/<tier>_langs.json`

## Step 4: Subsample + tokenize
We tokenize and subsample using the distributed data processor with the rates
above. The pipeline uses tier CSVs to build language lists and applies the
rates JSON.

Script:
- `data_prep/merlin_data_prep/distributed_data_processor/scripts/tokenize_all_tokenizers_lang_combinations.sh`

This produces the tokenized datasets per tokenizer x tier, constrained by the
token budgets and the rate logic above.
