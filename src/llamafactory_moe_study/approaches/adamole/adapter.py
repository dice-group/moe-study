import os
import torch.nn as nn

from typing import Iterable, Optional
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as HF_LORA_MAP
from llamafactory_moe_study.core.generic_adapter import AdapterConfig, MethodAdapter
from llamafactory_moe_study.approaches.adamole.adamole.layer import LinearAdaMoleLayer

def _default_targets(model) -> Iterable[str]:
    mt = getattr(getattr(model, "config", None), "model_type", None)
    return HF_LORA_MAP.get(mt, {"q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"})

def _get_parent_and_attr(model: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

class AdaMoLEAdapter(MethodAdapter):
    name = "ADAMOLE"

    def inject(self, model: nn.Module, knobs: AdapterConfig,
               target_modules: Optional[Iterable[str]] = None) -> nn.Module:
        n_experts = int(os.getenv("ADAMOLE_NUM_EXPERTS", "4"))
        max_thr   = os.getenv("ADAMOLE_MAX_THRESHOLD", None)
        max_thr_f = float(max_thr) if max_thr is not None else None

        targets = set(target_modules or _default_targets(model))
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(name.endswith(t) for t in targets):
                continue
            parent, attr = _get_parent_and_attr(model, name)
            setattr(parent, attr, LinearAdaMoleLayer(
                base_linear=module,
                num_experts=n_experts,
                r=knobs.r,
                alpha=knobs.alpha,
                dropout=knobs.dropout,
                max_threshold=max_thr_f,
            ))
        return model

    def aux_loss(self, model: nn.Module):
        import torch
        device = next(model.parameters()).device
        total = torch.tensor(0.0, device=device)
        for _, m in model.named_modules():
            if hasattr(m, "moe") and hasattr(m.moe, "layer_loss") and m.moe.layer_loss is not None:
                total = total + m.moe.layer_loss.to(device)
        coef = float(os.getenv("ADAMOLE_AUX_COEF", "0.01"))
        return coef * total
