import os
import types

from typing import Set
from transformers import AutoModelForCausalLM

from llamafactory_moe_study.approaches.registry import get_selected_method
from llamafactory_moe_study.core.generic_adapter import AdapterConfig
from llamafactory_moe_study.core.wrappers import AuxLossWrapper

def _freeze_backbone_except(model, keep_pred):
    for n, p in model.named_parameters():
        p.requires_grad = bool(keep_pred(n))

def _targets_from_env() -> Set[str]:
    raw = os.getenv("LORA_TARGETS", "")
    if not raw:
        return set()
    return set([t.strip() for t in raw.split(",") if t.strip()])

def _knobs_from_env() -> AdapterConfig:
    r = int(os.getenv("LORA_R", os.getenv("LORA_RANK", "8")))
    alpha = int(os.getenv("LORA_ALPHA", "16"))
    dropout = float(os.getenv("LORA_DROPOUT", "0.0"))
    targets = _targets_from_env()
    return AdapterConfig(r=r, alpha=alpha, dropout=dropout, target_modules=targets)

def apply_patch():
    adapter = get_selected_method()
    if adapter is None:
        return  # no-op

    orig_from_pretrained = AutoModelForCausalLM.from_pretrained

    def patched_from_pretrained(*args, **kwargs):
        model = orig_from_pretrained(*args, **kwargs)

        # Build knobs: prefer env for first smoke run
        knobs = _knobs_from_env()
        if not knobs:
            # If env didn’t provide targets, adapter will fall back to HF mapping
            knobs = _knobs_from_env()
        lf_knobs = _knobs_from_env()
        knobs = _knobs_from_env()  # keep simple for first run
        knobs = _knobs_from_env() if lf_knobs else _knobs_from_env()
        knobs = _knobs_from_env()  # (env-driven for v0)

        knobs = _knobs_from_env()  # final
        knobs = _knobs_from_env()
        # Actually we just call once and ignore the above dance:
        knobs = _knobs_from_env()
        knobs = AdapterConfig(
            r=int(os.getenv("LORA_R", "8")),
            alpha=int(os.getenv("LORA_ALPHA", "16")),
            dropout=float(os.getenv("LORA_DROPOUT", "0.0")),
            target_modules=_targets_from_env(),
        )

        # Inject adamole
        model = adapter.inject(model, knobs)

        # Freeze backbone; keep experts/gate/threshold trainable
        keep = lambda name: any(tok in name for tok in (".moe.", "experts.", ".gate.", ".thr."))
        _freeze_backbone_except(model, keep)

        # Expose aux-loss getter for wrapper
        def _get_aux_loss():
            return adapter.aux_loss(model)
        model.get_aux_loss = _get_aux_loss

        # Wrap to add aux loss during training
        coef = float(os.getenv("ADAMOLE_AUX_COEF", "0.01"))
        return AuxLossWrapper(model, coef=coef)

    AutoModelForCausalLM.from_pretrained = patched_from_pretrained
