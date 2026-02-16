# Comparison Hyperparameters (10/96/200 langs, 1B/3B/8B; MoE-LPR excluded)

> **Note**: MoE-LoRA and MoV-LoRA are newer additions. Their comparison runs have not been
> configured in the SLURM comparison pipeline yet. See `configs/train_moelora.yaml` and
> `configs/train_movlora.yaml` for standalone configs.

This document derives the effective hyperparameters/parameters for the comparison runs started via
`scripts/slurm/comparison/run_comparison.py`.

Sources used (in precedence order):
1) JSON config passed to `run_comparison.py` (env overrides per approach/variant)
2) `run_comparison.py` base env values (global + dataset/model fields)
3) Job script defaults in `scripts/slurm/comparison/*_job.sh`
4) LLaMA-Factory defaults for flags not explicitly set in the job scripts

Configs covered (200/96/10 focus):
- `scripts/slurm/comparison/comparison_config_llama3_1_8b.json`
- `scripts/slurm/comparison/comparison_config_llama3_2_1b.json`
- `scripts/slurm/comparison/comparison_config_llama3_2_3b.json`

--------------------------------------------------------------------------------

## Global config values by model size

These are the global values from each config JSON. All approach runs under that config inherit them
unless a variant env overrides.

| Setting | 8B config | 1B config | 3B config | Notes |
| --- | --- | --- | --- | --- |
| flash_attn | `fa2` | same | same | Passed as `--flash_attn` |


## Tiered GPU-hour scheduling (run_comparison.py)

If no explicit `time` is set in resources/sbatch args, `run_comparison.py` computes a time budget
using `_tier_gpu_hours` and divides by total GPUs. For 10/96/200:

| dataset | gpu_hours | time if 1 GPU | time if 4 GPUs |
| --- | --- | --- | --- |
| `10_langs` | 27.69 | `1-03:42:00` | `06:56:00` |
| `96_langs` | 91.20 | `3-19:12:00` | `22:48:00` |
| `200_langs` | 168.00 | `7-00:00:00` | `1-18:00:00` |

I usually trained 96 and 200 langs on 4 GPUs (1 node), and `10_langs` on 1 GPU across all approaches
and model sizes. MoE-LPR is excluded; when it was used, I always ran with 4 GPUs to fit the model.
I use flash attention 2 (`fa2` in configs) for all training for efficiency.

--------------------------------------------------------------------------------

## Shared parameters across approaches (AdaMoLE, MoLA, HydraLoRA, CoLA)

| Parameter | AdaMoLE (8B/1B/3B) | MoLA (8B/1B/3B) | HydraLoRA (8B/1B/3B) | CoLA (8B/1B/3B) | Same across approaches? |
| --- | --- | --- | --- | --- | --- |
| per_device_train_batch_size | 3 / 15 / 6 | 4 / 15 / 7 | 3 / 17 / 8 | 5 / 21 / 11 | No (varies by approach + model). Not 100% sure these are all correct since some were overwritten |
| learning_rate | 2e-4 / 2e-4 / 2e-4 | 2e-4 / 2e-4 / 2e-4 | 2e-4 / 2e-4 / 2e-4 | 2e-4 / 2e-4 / 2e-4 | Yes |
| lr_scheduler_type | cosine / cosine / cosine | cosine / cosine / cosine | cosine / cosine / cosine | cosine / cosine / cosine | Yes |
| warmup_ratio | 0.03 / 0.03 / 0.03 | 0.03 / 0.03 / 0.03 | 0.03 / 0.03 / 0.03 | 0.03 / 0.03 / 0.03 | Yes |
| gradient_accumulation_steps | 2 / 2 / 2 | 2 / 2 / 2 | 2 / 2 / 2 | 2 / 2 / 2 | Yes (MoE-LPR excluded) |
| eval_steps | 5000 / 3000 / 3000 | 5000 / 3000 / 3000 | 5000 / 3000 / 3000 | 5000 / 3000 / 3000 | Differs by model size / # langs |
| save_steps | 7000 / 1000 / 1000 | 7000 / 1000 / 1000 | 7000 / 1000 / 1000 | 7000 / 1000 / 1000 | Differs by model size / # langs |
| save_total_limit | 3 / 4 / 4 | 3 / 4 / 4 | 3 / 4 / 4 | 3 / 4 / 4 | Yes (differs by model) |
| disable_gradient_checkpointing | True / True / True | True / True / True | True / True / True | True / True / True | Yes, because it was slightly more efficient on our GPUs |
| bf16 | True / True / True | True / True / True | True / True / True | True / True / True | Yes, recommended on H100 GPUs + most efficient |
| fp16 | False / False / False | False / False / False | False / False / False | False / False / False | Yes |

### Shared LoRA params

| Parameter | AdaMoLE (8B/1B/3B) | MoLA (8B/1B/3B) | HydraLoRA (8B/1B/3B) | CoLA (8B/1B/3B) | Same across approaches? |
| --- | --- | --- | --- | --- | --- |
| lora_rank | 8 / 8 / 8 | 8 / 8 / 8 | 8 / 8 / 8 | 8 / 8 / 8 | Yes |
| lora_alpha | 16 / 16 / 16 | 16 / 16 / 16 | 16 / 16 / 16 | 16 / 16 / 16 | Yes |
| lora_dropout | 0.05 / 0.05 / 0.05 | 0.05 / 0.05 / 0.05 | 0.05 / 0.05 / 0.05 | 0.05 / 0.05 / 0.05 | Yes |
| lora_target | q_proj,k_proj,v_proj,o_proj | q_proj,k_proj,v_proj,o_proj | q_proj,k_proj,v_proj,o_proj | q_proj,k_proj,v_proj,o_proj | Yes |

Reasoning: 
-  rank = 8: Used by AdaMoe / CoLA / HydraLoRA paper, AdaMOLE uses r=32
-  alpha = 16: Used by AdaMoLE / CoLA paper not specified in others
   - specifically there is this common rule: aplha = rank * 2 some use, e.g CoLA
- lora_target: q, k, v, o - these are all attention modules
  - the original LoRA paper says: "In the Transformer architecture, there are four weight matrices in
the self-attention module (Wq, Wk, Wv, Wo) and two in the MLP module. We treat Wq (or Wk, Wv)
as a single matrix of dimension dmodel ×dmodel, even though the output dimension is usually sliced
into attention heads. We limit our study to only adapting the attention weights for downstream
tasks and freeze the MLP modules (so they are not trained in downstream tasks) both for simplicity
and parameter-efficiency"
  - Otherwise AdaMoE /CoLA also applied lora to W_gate, W_down, W_up (MLP modules), AdaMoLE also only focussed on attention weights

Notes:
- Job-script defaults differ for CoLA and HydraLoRA (`lora_rank=16`, `lora_alpha=32`), but all configs
  explicitly set `LORA_R=8` and `LORA_ALPHA=16` for those approaches, so the effective values
  are still 8/16 across all approaches and model sizes.

--------------------------------------------------------------------------------

## Approach-specific parameters

Below are parameters unique to each approach (i.e., not shared across all four approaches above).
Values are shown as `8B / 1B / 3B`.

### AdaMoLE

| Parameter | 8B / 1B / 3B | Notes |
| --- | --- | --- |
| adamole_num_experts | 4 / 4 / 4 | !!! This goes agains original PAPER, they use 8 EXPERTS !!! |
| adamole_max_threshold | 0.25 / 0.25 / 0.25 | in paper they say 0,1/2N meaning 0.25 bc 4 experts |
| adamole_aux_loss_coef | 0.001 / 0.001 / 0.001 | config |
| adamole_debug_mode | False / False / False | job script default |

### MoLA (adamoe_1)

| Parameter | 8B / 1B / 3B | Notes |
| --- | --- | --- |
| mola_num_experts | 4 / 4 / 4 | same as in Paper |
| mola_num_null_experts | 1 / 1 / 1 | in paper best setup uses m7k4 (appendix). I set this in config, but seems like this was not proeprly injected into the runs scripts |
| mola_top_k | 4 / 4 / 4 | config |
| mola_use_null_expert | True / True / True | job script default |
| mola_router_aux_loss_coef | 0.001 / 0.001 / 0.001 | config |
| mola_null_expert_penalty | 0 / 0 / 0 | config |
| mola_aux_loss_annealing | False / False / False | config |
| mola_debug_mode | False / False / False | job script default |

Config-only keys (present but not passed by job script):
- `MOLA_NUM_NULL_EXPERTS`
- `MOLA_OUTPUT_ROUTER_LOGITS`
These do not map to CLI flags in `adamoe_1_mola_job.sh` and therefore do not affect runs
unless LLaMA-Factory is reading them directly from the environment (not observed in the job script).

### HydraLoRA

| Parameter | 8B / 1B / 3B | Notes |
| --- | --- | --- |
| use_hydralora_experts | True / True / True | config |
| hydralora_num_experts | 3 / 3 / 3 | config |
| hydralora_top_k | 1 / 1 / 1 | config |
| lora_num (HYDRALORA_NUM_B) | 3 / 3 / 3 | config |
| hydralora_debug | False / False / False | job script default |

### CoLA

| Parameter | 8B / 1B / 3B | Notes |
| --- | --- | --- |
| num_A (COLA_NUM_A) | 1 / 1 / 1 | config |
| num_B (COLA_NUM_B) | 3 / 3 / 3 | config |
| cola_strategy | fully / fully / fully | config |
| use_cola_experts | False / False / False | job script default |
| cola_num_experts | 4 / 4 / 4 | job script default |
| cola_top_k | 2 / 2 / 2 | job script default |
| cola_debug | False / False / False | job script default |

### MoE-LoRA

Paper: "When MOE Meets LLMs: Parameter Efficient Fine-tuning for Multi-task Medical Applications" (Liu et al., SIGIR 2024)

| Parameter | Default | Notes |
| --- | --- | --- |
| moelora_num_experts | 8 | Paper default. rank must be divisible by num_experts |
| moelora_task_num | 1 | Auto-inferred from language_map when set to 1 |
| moelora_task_embedding_dim | 64 | Embedding dimension for task/language ID |
| moelora_sparse_top_k | 0 | 0 = dense gate. Paper finds top-2 optimal |
| moelora_gate_temperature | 1.0 | Softmax temperature |
| lora_rank | 16 | Split: each expert gets r/N = 2 per expert |
| lora_alpha | 32 | Standard 2× rank |

Key difference: Uses language/task IDs as routing signal (not learned per-token).

### MoV-LoRA

Paper: "Pushing Mixture of Experts to the Limit" (Zadouri et al., ICLR 2024)

| Parameter | Default | Notes |
| --- | --- | --- |
| movlora_num_experts | 8 | Paper default |
| movlora_top_k | 0 | 0 = soft/dense routing. Paper: soft > sparse |
| movlora_router_temperature | 1.0 | Standard softmax |
| movlora_router_jitter_noise | 0.0 | Multiplicative jitter during training |
| movlora_router_bias | False | Paper: no bias |
| movlora_router_init_std | 0.02 | Paper: 2e-2 |
| movlora_router_ignore_padding_tokens | False | Mask padding tokens in routing |
| lora_rank | 16 | Each expert gets FULL rank r (not r/N) |
| lora_alpha | 32 | Standard 2× rank |

Key difference: Per-layer learned router. No task/language IDs needed. Token-level routing.
