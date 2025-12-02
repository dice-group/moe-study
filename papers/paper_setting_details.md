# Further Paper Hyperparameters

Best-performing hyperparameter settings distilled from each paper referenced in `papers/further`.

## AdaMoLE (COLM 2024)
- **Backbone & adapters**: Llama-2-7B with LoRA applied to `{Wq, Wk, Wv, Wo}`; LoRA baseline rank 32, α = 16, dropout 0.05, while AdaMoLE/MoLE split that rank across 8 experts of rank 4 each so the effective rank stays 32 (papers/further/extracted/AdaMoLE/text.txt:321-335).
- **Routing & losses**: Adaptive gating with τ∈[0,1/N] and an auxiliary load-balancing loss scaled by 1e-3; best gating choice is the learnable 0–1/N threshold which outperforms fixed top-k (papers/further/extracted/AdaMoLE/text.txt:323-336, papers/further/extracted/AdaMoLE/text.txt:518-534).
- **Optimization**: AdamW, lr 1e-4, batch size 16, constant schedule with 200-step warm-up, sequence length 256, epochs tuned per dataset (1–20) (papers/further/extracted/AdaMoLE/text.txt:327-335).
- **Best N×r sweep**: CommonsenseQA accuracy peaks at the 8 experts × rank‑4 configuration (78.71%), keeping AdaMoLE ahead of LoRA/MoLE across the tested ranks (papers/further/extracted/AdaMoLE/text.txt:529-534).

## AdaMOE (Findings of EMNLP 2024)
- **Mo-LoRA setup**: Llama2‑7B with LoRA rank 8 adapters on `{Wq,Wk,Wv,Wo,Wgate,Wdown,Wup}`, AdamW lr 3e-4, 2 epochs per dataset, three random seeds; each MoE layer uses 4 true experts and AdaMOE augments with m null experts to enable adaptive loads (papers/further/extracted/AdaMoe/text.txt:643-676).
- **Best Mo-LoRA routing**: Highest accuracies on ARC-C/HELLA/PIQA/WINO arise with `m=8` null experts and `k=3` active selections (load≈1.66), while OpenBookQA peaks at `m=16,k=4`; all outperform fixed top‑k baselines under equal compute (papers/further/extracted/AdaMoe/text.txt:1134-1166).
- **MoE-LLM (Mixtral-8x7B) fine-tuning**: QLoRA w/4-bit quantization, LoRA rank 8 on `{gate,gate2,w1,w2,w3}`, lr 5e-5, 2 epochs over 1 000-sample subsets per dataset (papers/further/extracted/AdaMoe/text.txt:720-741).
- **Load-balancing schedule**: Auxiliary loss weights annealed with α₁ = 0.02 for epoch 1, α₂ = 1e-4 thereafter to keep router diversity without destabilizing small-data fine-tunes (papers/further/extracted/AdaMoe/text.txt:743-845).

## CoLA (Collaborative Low-Rank Adaptation, 2025)
- **Best adapter graphs**: Fully collaborative CoLA / CoLA⊺ variants with rank 8 adapters and #A<#B (e.g., `#A=1,#B=3` or `#A=2,#B=3`) consistently give the top zero-shot accuracy across domains and remain stable even with limited samples (papers/further/extracted/CoLA Collaborative Low-Rank Adaptation/text.txt:610-632, papers/further/extracted/CoLA Collaborative Low-Rank Adaptation/text.txt:744-759, papers/further/extracted/CoLA Collaborative Low-Rank Adaptation/text.txt:1554-1600).
- **Training recipe**: Batch size 8, 5 epochs, lr 5e-5 with cosine decay, gradient accumulation 8, fp16 on dual A800‑80G GPUs; evaluation every few steps (papers/further/extracted/CoLA Collaborative Low-Rank Adaptation/text.txt:1563-1575).
- **Targets per method**: LoRA targets `{down_proj, k_proj, v_proj, q_proj, up_proj, gate_proj, o_proj}` with `lora_alpha = 2r`; CoLA search space covers configurations `#A|#B ∈ {1|3, 2|3, 1|14, 4|10}` but best results favor the smallest #A with more B matrices (papers/further/extracted/CoLA Collaborative Low-Rank Adaptation/text.txt:1588-1593).

## HydraLoRA (2024)
- **Architecture**: Llama2-7B adapters with one shared rank‑8 A matrix and routed B experts (average #B = 3) fine-tuned via an MoE router; this asymmetric LoRA hits the best single-domain scores at only 0.124% trainable params (papers/further/extracted/HydraLoRA An Asymmetric LoRA Architecture for/text.txt:560-599).
- **Training insights**: Router learns softmax weights Wg for each expert, enabling adaptive merging during inference; HydraLoRA w/ MoE + gate outperforms ablations lacking gating/MoE or replacing shared A with multiple standalone LoRAs (papers/further/extracted/HydraLoRA An Asymmetric LoRA Architecture for/text.txt:520-660, papers/further/extracted/HydraLoRA An Asymmetric LoRA Architecture for/text.txt:700-760).
- **Energy/latency profile**: Compared to LoRA rank 32, HydraLoRA rank 8 cuts training time ~1.96× and reduces energy 49.6% without sacrificing accuracy, making the `#A=1,#B=3` setup the most compute-efficient (papers/further/extracted/HydraLoRA An Asymmetric LoRA Architecture for/text.txt:700-760).

## MoE-LPR (2024)
- **Stage‑1 post-pretraining**: Upcycle Qwen-1.5-1.8B into a 6-expert top‑2 MoE, train only the 5 new FFNs + router on 24 B tokens (8 B each for Greek/Hungarian/Turkish) with AdamW, lr 5e-5, batch size 512, seq length 1024, cosine schedule, bf16, flash attention, and load-balancing loss weight 0.01 (papers/further/extracted/MoE-LPR/text.txt:400-458).
- **Stage‑2 review (LPR)**: Router-only finetuning with γ=0.1 on the LPR loss, lr 5e-5, batch size 512, seq length 512, cosine schedule, bf16, flash attention, and no load-balancing term; best replay budget is 50 K docs/original language (≈0.138 B tokens total, <1% of stage‑1 data) paired with 100 K docs per expanded language (papers/further/extracted/MoE-LPR/text.txt:520-560).
- **Baselines & best settings**: Dense full fine-tuning stabilizes only at lr 5e-6; LoRA rank 64 helps expanded languages but still trails full FT, whereas MoE-LPR with 6 experts maintains top scores while keeping active parameters at 2.6 B (papers/further/extracted/MoE-LPR/text.txt:953-979, papers/further/extracted/MoE-LPR/text.txt:423-449).
