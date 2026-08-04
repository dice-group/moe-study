## Tier wall-clock hours (1 GPU baseline)

- 10_langs: 27.69 hours
- 96_langs: 91.20 hours
- 200_langs: 168.00 hours

Below: GPU-hour budgets with resulting wall-clock times for specific GPU counts.

## 1B runs (GPU-hour budget fixed; wall-time scales with GPUs)

- 1B 200_langs: 4 GPUs, 168 GPU-h -> 42:00:00
- 1B 96_langs: 4 GPUs, 91.20 GPU-h-> 22:48:00
- 1B 10_langs: 1 GPU, 27.69 GPU-h -> 27:42:00

## 8B runs (GPU-hour budget fixed; wall-time scales with GPUs)

- 8B 200_langs: 4 GPUs, 168 GPU-h -> 42:00:00
- 8B 96_langs: 4 GPUs, 91.20 GPU-h -> 22:48:00

## Language distribution verification (tokenized data)

- Tokenized dataset checked: `/scratch/anonymous-project/project_data/moe_study/tokenized/fw_l2v_samples_moe_study_comparison/llama-3.2-1B_tokenizer/200_langs`
- Original counts: `/cluster/user/repos/moe-study/data_dir/fw_l2v_samples/lang_counts/language_sample_counts.tsv`
- Realized counts: `/cluster/user/moe-study/data_prep/merlin_data_prep/distributed_data_processor/scripts/extras/lang_stats.tsv`
- Comparison script: `data_prep/merlin_data_prep/distributed_data_processor/scripts/extras/compare_language_distribution.py`
- Method: `tokenize_slurm.py` samples by **language rates (probabilities)**, not fixed counts.
- Results (English excluded on purpose):
  - TV distance (samples): **0.001237**
  - Max per-language proportional deviation: **~0.000411** (~0.041 percentage points)
  - Conclusion: distribution preserved within very small error; English is intentionally downsampled.
