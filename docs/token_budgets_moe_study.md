# Token budgets (MoE-Study)

this explains why I chose the current budgets and how to recompute them. It is meant to be followed step by
step by anyone onboarding who reviews / verifies this.

General setup: We run an empirical study on massively multilingual continued pretraining / finetuning, scaling across model size (1B/3B/8B) and language tier size (5 -> 200
langs). We compare five state‑of‑the‑art parameter‑efficient finetuning approaches (LoRA‑based and MoE‑style variants, specifically: adamole, adamoe_1/mola, moelpr, hydralora, cola). For each (model × tier), we build one shared tokenized dataset (cap set by the fastest method, currently CoLA) and give all approaches the same wall‑clock budget so performance differences reflect method
efficiency under equal time.

Why using shared datasets: We use shared datasets to ensure a fair comparison between approaches. Each method is given the same GPU budget, and all are trained on the same data–language distribution. Because some models have higher throughput (e.g., CoLA, which has no experts), we choose dataset sizes that are slightly larger than what can be fully processed within the allotted compute time. This ensures that compute, not data availability, is the limiting factor, while also allowing smooth handling of evaluation, data shuffling, and tokenization. 

---

## Goal (what we optimize for)

We want **comparable wall‑clock time** across approaches while we scale the
**number of languages** and **model size**. The data is huge, so we set **max
token budgets** per run and **subsample** while keeping the **original language
proportions**.

Why keeping original language proportions: We dont want to bias the distribution, we keep it as available, only caping really high resource langs such as english (decided in discussion with Prof. Axel)

---

## Throughputs Inputs we trust (measured on H100)

Throughput numbers come from `scripts/slurm/throughput_sft_bench.sh` with:
- model: Llama‑3.x
- seq_len=1024, bs=1
- effective tokens/sec

We measure 3B/8B directly (no scaling from 1B).

**Measured throughput (1B):**
- AdaMoLE ≈ **1.33k tok/s**
- CoLA ≈ **8.63k tok/s**
- HydraLoRA ≈ **5.41k tok/s**
- MoLA ≈ **1.14k tok/s**
- MoE‑LPR ≈ **3.48k tok/s**

**Measured throughput (3B):**
- AdaMoLE ≈ **0.782k tok/s**
- CoLA ≈ **4.750k tok/s**
- HydraLoRA ≈ **3.453k tok/s**
- MoLA ≈ **0.648k tok/s**
- MoE‑LPR ≈ **1.856k tok/s (partial log, confirm)**

**Measured throughput (8B):**
- AdaMoLE ≈ **0.557k tok/s**
- CoLA ≈ **2.775k tok/s**
- HydraLoRA ≈ **2.394k tok/s**
- MoLA ≈ **0.523k tok/s**
- MoE‑LPR ≈ **1.43k tok/s (estimate, pending benchmark)**

If these numbers change, **all budgets below must be recomputed**.

---

## Key decisions (with reasoning)

### 1) Hard cap
We cap the **largest run** at:

- **8B @ 200 languages = 7 days (168h)**

This prevents runaway cost and defines the overall scale.

### 2) GPU allocation
We scale GPUs with model size:

- 1B: **1 GPU**
- 3B: **2 GPUs**
- 8B: **4 GPUs**
If we need to argue why this was selected, we might argue that this is most
effective for our given cluster and hardware resources, since 1 node has 4 GPUs
and distributing over two nodes makes training less effective while requiring
more hardware.

### 3) Time scaling with number of languages
We use a **linear tier schedule** with a fixed minimum and maximum:

- **5 languages = 24h minimum**
- **200 languages = 168h maximum**
- linear interpolation by number of languages

We keep the **same wall‑time** for all model sizes at a given tier.

Language tiers: **5, 10, 22, 46, 96, 200**.

---

## Wall‑time schedule (hours)

Same hours for **all sizes** at a given tier:

| langs | hours |
|---:|---:|
| 5 | 24.00 |
| 10 | 27.69 |
| 22 | 36.55 |
| 46 | 54.28 |
| 96 | 91.20 |
| 200 | 168.00 |

---

## Token budget formula (what we actually compute)

For each approach `a`, model size `s`, tier `L`:

```
Time(L)       = tier_hours[L]
Tokens(a,s,L) = throughput(a) * GPUs(s) * Time(L)
```

Budgets below are **billions of tokens (B)**.

---

## Token budgets (B tokens)

### 1B (1 GPU)
| approach | 5 | 10 | 22 | 46 | 96 | 200 |
|---|---|---|---|---|---|---|
| AdaMoLE | 0.115 | 0.133 | 0.175 | 0.260 | 0.437 | 0.804 |
| CoLA | 0.746 | 0.860 | 1.136 | 1.686 | 2.833 | 5.219 |
| HydraLoRA | 0.467 | 0.539 | 0.712 | 1.057 | 1.776 | 3.272 |
| MoLA | 0.098 | 0.114 | 0.150 | 0.223 | 0.374 | 0.689 |
| MoE-LPR | 0.301 | 0.347 | 0.458 | 0.680 | 1.143 | 2.105 |

### 3B (2 GPUs)
| approach | 5 | 10 | 22 | 46 | 96 | 200 |
|---|---|---|---|---|---|---|
| AdaMoLE | 0.135 | 0.156 | 0.206 | 0.306 | 0.513 | 0.946 |
| CoLA | 0.821 | 0.947 | 1.250 | 1.856 | 3.119 | 5.746 |
| HydraLoRA | 0.597 | 0.688 | 0.909 | 1.349 | 2.267 | 4.177 |
| MoLA | 0.112 | 0.129 | 0.171 | 0.253 | 0.426 | 0.784 |
| MoE-LPR | 0.321 | 0.370 | 0.488 | 0.725 | 1.219 | 2.245 |

### 8B (4 GPUs)
| approach | 5 | 10 | 22 | 46 | 96 | 200 |
|---|---|---|---|---|---|---|
| AdaMoLE | 0.192 | 0.222 | 0.293 | 0.435 | 0.731 | 1.347 |
| CoLA | 0.959 | 1.106 | 1.461 | 2.169 | 3.644 | 6.713 |
| HydraLoRA | 0.827 | 0.955 | 1.260 | 1.871 | 3.144 | 5.792 |
| MoLA | 0.181 | 0.209 | 0.275 | 0.409 | 0.687 | 1.265 |
| MoE-LPR | 0.494 | 0.570 | 0.753 | 1.118 | 1.878 | 3.459 |

---

## Single dataset per (model size × tier)

We will produce **exactly 24 tokenized datasets**:

- 3 model sizes (1B / 3B / 8B)
- 6 language tiers (5 / 10 / 22 / 46 / 96 / 200)

To keep a **single dataset per (size, tier)** that works for *all approaches*, we
set the **max token budget** using the **fastest throughput** (currently **CoLA**).
Slower approaches will simply consume fewer tokens within the same time budget.

### Max token budgets (CoLA, B tokens)

These are the **dataset caps** to use when subsampling for each (size, tier).

**1B (1 GPU):** 5 → **0.746**, 10 → **0.860**, 22 → **1.136**, 46 → **1.686**, 96 → **2.833**, 200 → **5.219**

**3B (2 GPUs):** 5 → **0.821**, 10 → **0.947**, 22 → **1.250**, 46 → **1.856**, 96 → **3.119**, 200 → **5.746**

**8B (4 GPUs):** 5 → **0.959**, 10 → **1.106**, 22 → **1.461**, 46 → **2.169**, 96 → **3.644**, 200 → **6.713**

---

## How to subsample while preserving original language distribution

We do **no smoothing**. We keep the **original language proportions** and just
scale the total down to the tier budget.

Let `T_l` be tokens available for language `l` in the tier, and
`T_total = sum_l T_l`. For a tier budget `B`:

```
scale = min(1, B / T_total)
T_l_target = floor(T_l * scale)
```

That gives the **max tokens per language** while preserving the original
distribution.

---

## Tokenizer subset (default)

We reserve **10% of each tier budget** to train/extend tokenizers. The remaining
90% is tokenized with both base and extended tokenizers for training.

---

## If we need to update this file

Update in this order:
1) Replace throughput numbers (Section: Inputs we trust).
2) Decide the cap (Section: Hard cap).
3) Decide min/max tier hours (Section: Time scaling).
4) Recompute the wall‑time table.
5) Recompute the token budget tables.

---

## Notes

- All approaches get the **same wall‑time** per (model size, tier). Token budgets
  differ only because throughput differs.
- If any throughput numbers or tier hours change, recompute all tables.
