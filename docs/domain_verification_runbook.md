# Domain Verification Runbook

This runbook prepares and executes the domain-verification experiment matrix that separates:

1. MoE implementation/routing issues
2. multilingual data/task effects

## 1. Scope

Compared methods:

- `base` (no fine-tuning, eval only)
- `full` (full fine-tuning)
- `lora` (standard LoRA)
- `adamole` (MoE-LoRA)
- `mola` (MoE-LoRA)

Tracks:

- `gsm8k`: train `gsm8k_train` -> eval `gsm8k`
- `dolly`: train `dolly_15k_en` -> eval `mmlu`
- `multilingual10`: pre-tokenized multilingual 10-language data -> eval `belebele_10`

## 2. Asset Validation

```bash
/upb/users/j/joeldag/profiles/unix/cs/.conda/envs/merlin/bin/python \
  scripts/domain_verification/prepare_assets.py
```

Optional full download into HF cache:

```bash
/upb/users/j/joeldag/profiles/unix/cs/.conda/envs/merlin/bin/python \
  scripts/domain_verification/prepare_assets.py --download-datasets
```

## 3. Training Commands

Single run:

```bash
scripts/domain_verification/train_domain_track.sh \
  --method adamole \
  --track gsm8k \
  --nproc-per-node 2 \
  --model meta-llama/Llama-3.2-1B \
  --output-root ./outputs/domain_verification
```

Multilingual track (pre-tokenized load path required):

```bash
scripts/domain_verification/train_domain_track.sh \
  --method mola \
  --track multilingual10 \
  --nproc-per-node 2 \
  --tokenized-path /scratch/.../tokenized/.../10_langs \
  --model meta-llama/Llama-3.2-1B \
  --output-root ./outputs/domain_verification
```

Smoke run (no long training):

```bash
scripts/domain_verification/train_domain_track.sh \
  --method full \
  --track dolly \
  --nproc-per-node 1 \
  --max-steps 1 \
  --max-samples 64
```

## 4. Evaluation Commands

Base model:

```bash
scripts/domain_verification/eval_domain_suite.sh --suite gsm8k --model meta-llama/Llama-3.2-1B
scripts/domain_verification/eval_domain_suite.sh --suite mmlu --model meta-llama/Llama-3.2-1B
scripts/domain_verification/eval_domain_suite.sh --suite humaneval --model meta-llama/Llama-3.2-1B
```

Adapter checkpoint:

```bash
scripts/domain_verification/eval_domain_suite.sh \
  --suite gsm8k \
  --checkpoint ./outputs/domain_verification/gsm8k/adamole/seed_42
```

## 5. Tokenization and Data Loading Strategy

- `gsm8k`, `dolly`, `code`: use regular SFT data loading from `dataset_info.json`.
  - Tokenization is done by LlamaFactory pre-processing.
  - Validation split is created with `--val_size 0.05`.
- `multilingual10`: use `--tokenized_path` to load pre-tokenized data.
  - If tokenized records include `split`, LlamaFactory auto-builds train/validation.
  - This follows existing multilingual pipeline behavior in `LLaMA-Factory/src/llamafactory/data/loader.py`.

## 6. Overnight Plan Execution

Preview complete matrix (default dry-run):

```bash
scripts/domain_verification/plan_overnight.sh \
  --model meta-llama/Llama-3.2-1B \
  --output-root ./outputs/domain_verification \
  --nproc-per-node 2 \
  --multi-tokenized-path /scratch/.../10_langs
```

Execute full plan:

```bash
scripts/domain_verification/plan_overnight.sh --execute \
  --model meta-llama/Llama-3.2-1B \
  --output-root ./outputs/domain_verification \
  --nproc-per-node 2 \
  --multi-tokenized-path /scratch/.../10_langs
```
