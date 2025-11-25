## Multilingual Comparison Sweep

`run_cluster_comparison.sh` now fires a small set of per-approach ablations, all on Llama‑3.2‑1B and the 22‑language tokenized subset. For each method we launch three runs:

| Approach   | Variant       | Notes                                                                 |
|------------|---------------|-----------------------------------------------------------------------|
| AdaMoLE    | `default`     | Paper baseline: 4 experts, τ<sub>max</sub>=0.25, aux=1e‑3.            |
|            | `low-aux`     | Slightly weaker aux (5e‑4) to test looser balancing.                 |
|            | `high-aux`    | Stronger aux (3e‑3) for sparser routing.                             |
| MoLA/AdaMOE| `default`     | Top‑1 routing, no null penalty (TODO revisit); aux=0.02.             |
|            | `low-aux`     | Aux=0.01 to observe softer routing.                                  |
|            | `high-aux`    | Aux=0.03 for more aggressive load balancing.                         |
| HydraLoRA  | `shared-lora` | Plain LoRA head (no experts) as control (bs=12).                     |
|            | `hydra-default`| Paper-like Hydra: 4 experts, top‑1 routing (bs=8).                  |
|            | `hydra-high`  | Higher capacity Hydra with 8 experts (top‑1, bs=6).                  |
| CoLA       | `flat-2x3`    | Collaborative baseline (2 A, 3 B, top‑1).                            |
|            | `collab-rand` | Random pairing (1 A, 4 B, top‑1).                                    |
|            | `dense-1x4`   | Fully shared 1×4 block, top‑1 router (kept disabled).                |
| MoE-LPR    | `default`     | Stage‑1 aux 0.05 / Stage‑2 LPR 0.1, top‑1.                           |
|            | `strong-prior`| Heavier LPR weighting (0.08 / 0.2).                                  |
|            | `low-aux`     | Lighter Stage‑1/2 weights (0.03 / 0.05).                             |

Notes:
- AdaMoLE always trains with batch size 12; MoLA/AdaMOE uses 42 to saturate the null-expert setup, HydraLoRA’s variants carry the batch sizes noted above, and CoLA sticks to 12.
- All runs pin the number of experts per AdaMoLE layer to 4 to stay within the paper’s LoRA-r32 budget for now.
- Every router/top‑k setting is forced to 1 to keep the first multilingual sweep simple; we’ll relax this later.
- AdaMOE null penalties are disabled in every variant (TODO: reintroduce and study their effect once base training is stable).

Every job extends the existing comparison scripts (AdaMoLE, MoLA, CoLA, HydraLoRA, MoE-LPR) with the environment overrides above and writes to `OUTPUT_ROOT/<approach>/<variant>_<timestamp>`. Seeds are randomized unless you override `SEED`, ensuring we see variation across runs.
