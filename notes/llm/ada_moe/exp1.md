**Overview**

The file defines a modified version of the original **LLaMA** (LLaMA‑1) transformer, adding a lightweight routing‑logits mechanism (similar to MoE/LoRA) and an auxiliary load‑balancing loss.  
The core architecture (the *base model*) is the same as the vanilla LLaMA transformer, while the modifications are injected at three main places:

1. **Linear layers** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) now optionally return a **router logit** together with the usual output when the module name appears in `config.lora_target_modules`.  
2. **Decoder layers** (`LlamaDecoderLayer`) collect those router logits, expose them through a new flag `output_router_logits`, and optionally concatenate them across layers.  
3. **Model heads** (`LlamaForCausalLM_d`) compute an extra **auxiliary loss** (`load_balancing_loss_func`) from the collected router logits, scale it by `router_aux_loss_coef`, and add it to the standard cross‑entropy loss.

Below is a layer‑by‑layer walk‑through of the architecture, followed by a description of each modification.

---

## 1. Base LLaMA Architecture (unchanged)

### 1.1 Embedding Layer
```python
self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx)
```
* Token IDs → dense vectors of size `hidden_size`.  
* For multimodal inputs, optional vision‑tower embeddings are added (outside the scope of the pure language model).

### 1.2 Stacked Decoder Layers (`LlamaDecoderLayer`)

Each decoder layer consists of **two sub‑blocks**:

| Sub‑block | Components | Shape transformations |
|-----------|------------|-----------------------|
| **Self‑Attention** | `LlamaAttention` | Input → Q, K, V (via three linear layers) → scaled‑dot‑product attention → output projection (`o_proj`). |
| **Feed‑Forward (MLP)** | `LlamaMLP` | Input → `gate_proj` → activation → element‑wise multiply with `up_proj` → `down_proj` (project back to `hidden_size`). |

Both sub‑blocks are surrounded by **RMSNorm** (`LlamaRMSNorm`) which normalises over the hidden dimension without learnable scaling per‑channel (only a weight vector).  

#### 1.2.1 Attention Details
* **Head dimension**: `head_dim = hidden_size // num_attention_heads`.  
* Queries, keys, values are reshaped to `[batch, num_heads, seq_len, head_dim]`.  
* **Rotary positional embeddings** (`LlamaRotaryEmbedding`) are applied to Q and K.  
* Causal mask (or bi‑directional mask for encoder‑decoder variants) is added before softmax.  
* Output from attention (`attn_output`) passes through the final linear projection `o_proj`.

#### 1.2.2 MLP Details
* `gate_proj` (hidden → intermediate) + `up_proj` (hidden → intermediate) → element‑wise product → `down_proj` (intermediate → hidden).  
* Activation is looked up from `ACT2FN` based on `config.hidden_act` (e.g., `gelu`, `silu`).  

### 1.3 Final RMSNorm
After the last decoder layer, a final RMSNorm (`self.norm`) is applied to the hidden states before they are returned.

### 1.4 Output Head (Causal LM)
```python
self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
```
* Projects the final hidden state of each token into vocabulary logits.

---

## 2. Added Routing / LoRA‑style Modifications

### 2.1 `config.lora_target_modules`
A list of linear‑module names (e.g., `["q_proj", "v_proj", "up_proj"]`) that should emit **router logits**.  
If a module is *not* in this list, it behaves exactly like the original LLaMA linear layer.

### 2.2 Linear Modules Return a Tuple
All three projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) are overridden to optionally return **two tensors**:

```python
output, router_logit = self.<linear>(x)   # when module name in lora_target_modules
output = self.<linear>(x)                # otherwise
```

* The router logit is a 1‑D tensor (usually the weight *gradient* or a learned “gate” per expert).  
* The code collects every router logit in a list `router_logits` inside each forward pass.

### 2.3 Propagation through `LlamaAttention`
* Inside `LlamaAttention.forward` the three projections (`q_proj`, `k_proj`, `v_proj`) each optionally contribute a router logit, which are concatenated into `router_logits`.  
* After the output projection (`o_proj`) another router logit may be appended.  
* The final return tuple becomes:
```python
(attn_output, attn_weights, past_key_value, router_logits)
```
If no module emits a logit, `router_logits` is an empty tensor.

### 2.4 Propagation through `LlamaMLP`
* Same pattern: `gate_proj`, `up_proj`, `down_proj` optionally emit logits, which are concatenated and returned together with the MLP output.

### 2.5 `LlamaDecoderLayer` Adjustments
* The `forward` method now accepts an extra flag `output_router_logits`.  
* After calling `self.self_attn` and `self.mlp`, the layer concatenates any newly‑produced logits with the logits collected from the attention sub‑block.  
* The layer returns a tuple where the **last element** is the accumulated router logits for that layer (if `output_router_logits=True`).

### 2.6 `LlamaModel` Adjustments
* The top‑level `forward` of `LlamaModel` receives the new argument `output_router_logits`.  
* It creates a container `all_router_logits` (tuple of per‑layer logits) when the flag is true.  
* For each decoder layer, the returned `router_logits` (the last element of `layer_outputs`) are stored in `all_router_logits`.  
* The method finally returns **two objects**:
  * `BaseModelOutputWithPast` – the regular output (hidden states, past key values, etc.).
  * `all_router_logits` – the list of router‑logit tensors (only when requested).

---

## 3. New Model Head `LlamaForCausalLM_d`

### 3.1 Extra Hyper‑parameters
```python
self.router_aux_loss_coef = 0.001   # scaling factor for the auxiliary loss
```
* `self.number_experts`, `self.top_k`, and `self.obalance` are set via `get_new_parameters()` – they control the MoE‑style routing (number of experts, top‑k selection, and whether the auxiliary loss is active).

### 3.2 Forward Pass
1. Calls `self.model(..., output_router_logits=output_router_logits, ...)`.  
2. Receives:
   * `outputs` – the usual `BaseModelOutputWithPast`.  
   * `router_logits` – the concatenated logits from all layers (only when `output_router_logits=True`).  

3. Computes **standard language‑model loss** (cross‑entropy) if `labels` are provided.

4. If `output_router_logits` is true:
   * Calls `load_balancing_loss_func(router_logits, self.number_experts, self.top_k)` – this implements the Switch‑Transformer load‑balancing loss (penalises unbalanced expert usage).  
   * When `self.obalance` is true, the auxiliary loss is multiplied by `router_aux_loss_coef` and **added** to the cross‑entropy loss:
     ```python
     loss = ce_loss + router_aux_loss_coef * aux_loss
     ```

5. Returns a `CausalLMOutputWithPast` (or a tuple if `return_dict=False`) that contains:
   * `loss` (now possibly a sum of CE + aux).  
   * `logits`.  
   * `past_key_values`, `hidden_states`, `attentions`.

### 3.3 Utility Methods
* `prepare_inputs_for_generation` – unchanged except it forwards the `images` argument.  
* `_reorder_cache` – same as the original implementation (used for beam search).

---

## 4. Auxiliary Functions

### 4.1 `load_balancing_loss_func`
* Implements equations (4)‑(6) from the **Switch Transformer** paper.  
* For each layer:
  * Softmax over the routing logits → `routing_weights`.  
  * Takes the top‑k experts (using `top_k[layer_i]`).  
  * Builds a **one‑hot** mask of selected experts, computes *tokens‑per‑expert* distribution, and multiplies it by the average routing probability per expert.  
  * Scales by the number of experts and sums across layers.  
* Returns a scalar loss (type `torch.float32`).

### 4.2 `detect_nan_inf`
A tiny helper used in the attention block to print warnings when NaNs or infinities appear.

---

## 5. Summary of Modifications

| Component | Original Behavior | Modified Behavior |
|-----------|-------------------|-------------------|
| **Linear layers (`q/k/v/o/gate/up/down_proj`)** | Return only `output`. | Optionally return `(output, router_logit)` when the layer name is listed in `config.lora_target_modules`. |
| **LlamaAttention** | Returns `(attn_output, attn_weights, past_key_value)`. | Returns `(attn_output, attn_weights, past_key_value, router_logits)` where `router_logits` aggregates any logits from the three input projections and the output projection. |
| **LlamaMLP** | Returns `down_proj`. | Returns `(down_proj, router_logits)` when any of its three sub‑layers are targeted. |
| **LlamaDecoderLayer** | Returns `(hidden, attn_weights?, past_key_value?)`. | Adds a fourth element `router_logits` (concatenated from attention + MLP) when `output_router_logits=True`. |
| **LlamaModel** | Returns `BaseModelOutputWithPast`. | Returns a tuple `(...), all_router_logits` when the flag is true, and builds a container `all_router_logits`. |
| **LlamaForCausalLM_d** | Standard causal LM head. | Stores `router_aux_loss_coef`, exposes `get_new_parameters` for MoE settings, computes an **auxiliary load‑balancing loss**, scales it, and adds it to the main loss. |
| **Config** | No `lora_target_modules`, no `output_router_logits`. | Must contain `lora_target_modules` (list of strings) and optionally `output_router_logits` (bool) and `output_hidden_states`/`output_attentions` as in the base config. |

All the **core transformer computation** (self‑attention, feed‑forward, RMSNorm, causal masking) stays identical to the vanilla LLaMA. The added routing‑logits machinery is lightweight: it merely forwards an extra 1‑D tensor from each targeted linear projection, allowing the model to be trained with a Switch‑Transformer‑style load‑balancing regulariser without changing the forward arithmetic or the number of parameters (the logits are derived from existing weight matrices, or from a small additional “router” head that the user can inject via LoRA‑style adapters).  

Thus, the model can be used exactly like a normal LLaMA for inference, while optionally exposing the internal routing signals and auxiliary loss for research on sparse‑expert or LoRA‑enhanced LLMs.