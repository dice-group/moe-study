import os
from llamafactory_moe_study.core.generic_adapter import AdapterConfig
from llamafactory_moe_study.approaches.adamole.adapter import AdaMoLEAdapter

_REGISTRY = {
    "adamole": AdaMoLEAdapter(),
    "adamoe_1": "", #tbd
    "adamoe_2": "", #tbd
}

def get_selected_method():
    m = os.getenv("USE_METHOD")
    if m:
        return _REGISTRY.get(m.lower())
    if os.getenv("USE_ADAMOLE", "0") == "1":
        return _REGISTRY["adamole"]
    return None

def build_knobs_from_lf(peft_like_cfg) -> AdapterConfig:
    # d
    r = getattr(peft_like_cfg, "r", None) or getattr(peft_like_cfg, "lora_rank")
    alpha = getattr(peft_like_cfg, "lora_alpha")
    dropout = getattr(peft_like_cfg, "lora_dropout", 0.0)
    targets = getattr(peft_like_cfg, "target_modules", None) or getattr(peft_like_cfg, "lora_target", None) or ()

    if isinstance(targets, (list, tuple, set)):
        targets = set(targets)
    elif targets is None:
        targets = set()
    else:
        targets = {targets}
    return AdapterConfig(r=int(r), alpha=int(alpha), dropout=float(dropout), target_modules=targets)
