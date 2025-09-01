import torch
import torch.nn as nn

from typing import Iterable, Optional
from llamafactory_moe_study.core.generic_adapter import AdapterConfig, MethodAdapter
from llamafactory_moe_study.approaches.adamole.adamole.layer import LinearAdaMoleLayer

def _get_parent_and_attr(model: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

class AdaMoLEAdapter(MethodAdapter):
    name = "ADAMOLE"

    def inject(
            self,
            model: nn.Module,
            knobs: AdapterConfig,
            target_modules: Optional[Iterable[str]] = None,
            num_experts: int = 4,
            max_threshold: Optional[float] = None,
    ) -> nn.Module:
        targets = set(target_modules)
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(name.endswith(t) for t in targets):
                continue
            parent, attr = _get_parent_and_attr(model, name)
            setattr(parent, attr, LinearAdaMoleLayer(
                base_layer=module,
                adapter_name="default",
                lora_rank=knobs.r,
                lora_alpha=knobs.alpha,
                lora_dropout=knobs.dropout,
                ininit_lora_weights=True,
                num_experts=num_experts,
                max_threshold=max_threshold,
            ))
        return model

    def aux_loss(self, model: nn.Module, coef: float = 0.01):
        device = next(model.parameters()).device
        total = torch.tensor(0.0, device=device)
        for _, m in model.named_modules():
            if hasattr(m, "moe") and hasattr(m.moe, "layer_loss") and m.moe.layer_loss is not None:
                total = total + m.moe.layer_loss.to(device)
        return coef * total
