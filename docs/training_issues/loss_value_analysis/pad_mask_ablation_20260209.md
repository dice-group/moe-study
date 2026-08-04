# Pad Label Masking Ablation (CoLA, Llama-3.2-1B, 10-lang sampled subset)

Background:
- I (anonymous-user) observed relatively high training losses depending on the base model: ~4.0 for the 1B model and ~5.4–6.0 for the 8B model. This was initially not concerning. However, we later noticed that we were not outperforming the respective base models during training, which suggested a potential issue in the pipeline.
- After extensive debugging, I examined the tokenized dataset to investigate whether the construction of the 'labels' field in the tokenized data could be responsible for the unexpectedly high losses and lack of improvement over the base models.
- Date: 2026-02-09  

## Objective
Verify whether the pad-label masking behavior during tokenization explains large eval-loss differences.

A/B setup:
- A (masked): padded positions in `labels` are set to `-100` (ignored by CE loss)
- B (unmasked): legacy behavior `labels = input_ids` (padded positions also contribute to loss)

## Code Change Used
Tokenizer toggle was added in:
- `data_prep/merlin_data_prep/distributed_data_processor/src/tokenize_slurm.py:235`
- `data_prep/merlin_data_prep/distributed_data_processor/src/tokenize_slurm.py:392`

New arg:
- `--mask_pad_labels true|false` (default `true`)

Behavior:
- `true`: `labels[i] = token if attention_mask[i] == 1 else -100`
- `false`: `labels = input_ids` (legacy ablation mode)

## Data Used
Input sampled data:
- `sampled_data/llama-3.2-1B/10_langs`

Tokenized outputs:
- Masked: `/tmp/tokenized_10langs_masked`
- Unmasked: `/tmp/tokenized_10langs_unmasked`

Language map file used:
- `/tmp/lang_map_10langs_identity.json`

Languages in that map:
- `arb_Arab, arb_Latn, azj_Cyrl, azj_Latn, cmn_Hani, fas_Arab, hun_Latn, ind_Latn, jpn_Jpan, kor_Hang, rus_Cyrl, vie_Latn`

## Tokenization Commands (A/B)
Masked:
```bash
python data_prep/merlin_data_prep/distributed_data_processor/src/tokenize_slurm.py \
  --shard_dir sampled_data/llama-3.2-1B/10_langs \
  --save_tokenized_data_dir /tmp/tokenized_10langs_masked \
  --model_name meta-llama/Llama-3.2-1B \
  --max_length 1024 \
  --num_proc 2 \
  --eval_fraction 0.1 \
  --sample_seed 123 \
  --max_samples 256 \
  --language-map /tmp/lang_map_10langs_identity.json \
  --mask_pad_labels true
```

Unmasked:
```bash
python data_prep/merlin_data_prep/distributed_data_processor/src/tokenize_slurm.py \
  --shard_dir sampled_data/llama-3.2-1B/10_langs \
  --save_tokenized_data_dir /tmp/tokenized_10langs_unmasked \
  --model_name meta-llama/Llama-3.2-1B \
  --max_length 1024 \
  --num_proc 2 \
  --eval_fraction 0.1 \
  --sample_seed 123 \
  --max_samples 256 \
  --language-map /tmp/lang_map_10langs_identity.json \
  --mask_pad_labels false
```

## Dataset Integrity Check
Both variants are identical except labels:
- rows: `256`
- `input_ids`: identical (`diff_input = 0`)
- `attention_mask`: identical (`diff_attention = 0`)
- `labels`: differ (`diff_labels = 175` rows)

Pad-label audit:
- Masked: `pad_labels_not_ignored = 0 / 111,930` (ratio `0.0`)
- Unmasked: `pad_labels_not_ignored = 111,930 / 111,930` (ratio `1.0`)

This is the expected and intended A/B difference.

## Short Training/Eval Setup
Training script:
- `scripts/slurm/comparison/cola_job.sh`

Runs:
- Masked run dir: `/tmp/cola_padmask_ablation_masked_v2`
- Unmasked run dir: `/tmp/cola_padmask_ablation_unmasked_v2`

Common key settings:
- `NUM_TRAIN_EPOCHS=0.3`
- `PER_DEVICE_TRAIN_BATCH_SIZE=4`
- `PER_DEVICE_EVAL_BATCH_SIZE=4`
- `GRADIENT_ACCUMULATION_STEPS=1`
- `EVAL_STEPS=3`
- `WANDB_MODE=offline`
- 2x H100

## Results
Final eval losses:
- Masked: `2.7240467071533203` (`/tmp/cola_padmask_ablation_masked_v2/eval_results.json`)
- Unmasked: `1.5301992893218994` (`/tmp/cola_padmask_ablation_unmasked_v2/eval_results.json`)

Intermediate eval-loss trajectory:
- Masked: `2.7959 -> 2.7452 -> 2.7240`
- Unmasked: `1.6422 -> 1.5458 -> 1.5302`

## Interpretation
This experiment shows pad-label handling significantly shifts reported loss scale:
- With unmasked padding, padded tokens contribute to CE objective.
- This can artificially lower or otherwise distort loss values, making runs incomparable to properly masked runs.
- Therefore, inconsistent pad-label treatment is a plausible root cause for previously observed unexpected loss magnitudes.

## Important Caveat
Because `max_samples=256` and deterministic shard order was used, this tiny run effectively pulled one language-id bucket in practice (`language_ids` top count `[(0, 256)]`).

So this is a valid **mechanism check** (mask vs unmask), but not yet a multilingual-balanced conclusion.

## Recommended Next Step
Repeat the same A/B with balanced per-language sampling (fixed per-language caps) and same short eval protocol. That isolates pad masking while removing single-language sampling bias.
