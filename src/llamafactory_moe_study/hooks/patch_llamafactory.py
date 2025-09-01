import os

from typing import Set, Optional
from transformers import AutoModelForCausalLM

from llamafactory_moe_study.approaches.registry import get_selected_method
from llamafactory_moe_study.core.generic_adapter import AdapterConfig
from llamafactory_moe_study.core.wrappers import AuxLossWrapper

def _freeze_backbone_except(model, keep_pred):
    for n, p in model.named_parameters():
        p.requires_grad = bool(keep_pred(n))

def apply_patch(
    use_method: str = "adamole",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_targets: Optional[Set[str]] = None,
    adamole_aux_coef: float = 0.01,
):
    adapter = get_selected_method(use_method)
    if adapter is None:
        raise ValueError(f"The selected method {use_method} is not valid. Please choose a valid approach.")

    orig_from_pretrained = AutoModelForCausalLM.from_pretrained

    def patched_from_pretrained(*args, **kwargs):
        model = orig_from_pretrained(*args, **kwargs)
        knobs = AdapterConfig(
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_targets or set(),
        )

        # Inject approaches adapter
        model = adapter.inject(model, knobs)

        # Freeze backbone; keep experts/gate/threshold trainable
        keep = lambda name: any(tok in name for tok in (".moe.", "experts.", ".gate.", ".thr."))
        # TODO: check if freezed properly keep = lambda name: ("lora_" in name) or ("moe_layer" in name)
        _freeze_backbone_except(model, keep)

        # use approaches loss for model
        def _get_aux_loss():
            return adapter.aux_loss(model)
        model.get_aux_loss = _get_aux_loss

        return AuxLossWrapper(model, coef=adamole_aux_coef)

    AutoModelForCausalLM.from_pretrained = patched_from_pretrained
