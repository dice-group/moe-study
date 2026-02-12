# 1B High-Loss Recreation Forensics (2026-02-09)

## Goal
Recreate the historical 1B CoLA high-loss regime (~4 train/eval) locally and isolate why it appears.

## Key result
The effect is reproducible on the cluster full 10-language dataset, but not on the local `rank_00047` slice.

## Verified evidence

### A) Historical 1B cluster run (paper-best)
- W&B run: `dice-nlp/moe-study-comparison/b928n5al`
- Dataset size in log: `Num examples = 2,939,418`
- Loss behavior:
  - `train/loss` around ~4.0 very early and stable (e.g., step 50: 4.1086, step 600: 4.0882)
  - `eval/loss` stable around ~4.05-4.07 (e.g., step 9000: 4.0691)

### B) Current cluster A/B run (full 10_langs, same regime)
Project: `dice-nlp/moe-study-cola-padmask-ablation-10langs`
- Unmasked run: `0p0eaae9`
  - step 1000: `train/loss = 4.0752`
- Masked run: `wq8baks7`
  - step 1000: `train/loss = 2.7293`

This directly confirms pad-label masking is a major driver of the high-loss regime on full data.

### C) Local slice recreation (`rank_00047`) on this 2-GPU machine
Available local data:
- `tokenized_data/10_langs_ranks/rank_00047` only (full root missing other ranks)
- Full cluster path `/scratch/.../10_langs` is not mounted here.

Dataset audit (`rank_00047`):
- Non-ignored pad labels ratio: `1.0` (unmasked)
- `language_ids` all `-1`
- Rows: `57,430` (train split from loader ~56,258)

Local run (unmasked, SDPA, bs=12, ga=2):
- W&B: `dice-nlp/moe-study-rank47-forensics-local/jelohyat`
- step 1: `loss=4.9850`
- step 20: `loss=2.1542`
- step 40: `loss=1.7203`
- step 100: `loss=1.7001`

Local run (masked, same settings):
- W&B: `dice-nlp/moe-study-rank47-forensics-local/y5f3guls`
- step 1: `loss=2.9216`
- step 20: `loss=2.7611`
- step 40: `loss=2.7436`
- step 60: `loss=2.7100`

Interpretation:
- On the slice, unmasked starts high but quickly drops below masked.
- On full dataset, unmasked stays around ~4 while masked stays around ~2.7.

## Why local slice fails to match cluster-stable ~4
1. **Data volume mismatch**
- Full run: ~2.94M examples.
- Local slice: ~56k train examples (~52x smaller).

2. **Data composition mismatch**
- Slice is one rank shard with a specific language mix and `language_ids=-1` everywhere.
- It is not representative of full 10-language distribution.

3. **Optimization dynamics differ on slice**
- Unmasked objective can overfit rapidly to frequent pad/eos behavior in a small shard.
- On full corpus, this high-loss pressure persists much longer.

## Additional environment findings
- `FLASH_ATTN=fa2` is unavailable locally due binary/glibc mismatch:
  - `ImportError: ... libc.so.6: version 'GLIBC_2.32' not found ... flash_attn_2_cuda`
- Local runs used `sdpa` backend.

## Practical conclusion
- The issue is **real and recreated** on the full cluster dataset: unmasked pad labels yield ~4 train-loss regime in 1B CoLA.
- The inability to keep ~4 on local `rank_00047` is expected: this slice is too small/non-representative for that exact dynamics.

## What is needed for exact local recreation
To reproduce cluster-stable ~4 on this machine, one of these is required:
1. Mount/copy full dataset path:
   - `/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/fw_l2v_samples_moe_study_comparison/llama-3.2-1B_tokenizer/10_langs`
2. Or copy enough representative ranks (not one shard) to approximate full distribution and scale.

## Useful files and logs
- Cluster submit script: `scripts/slurm/comparison/submit_cola_padmask_ablation_cluster.sh`
- Masking utility: `scripts/slurm/comparison/mask_pad_labels_dataset.py`
- Local unmasked log: `/tmp/cola_rank47_forensics_unmasked_bs12ga2/malformed_rank47.log`
- Local masked log: `/tmp/cola_rank47_forensics_masked_bs12ga2/healthy_masked.log`
