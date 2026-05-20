from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def layer_sort_key(label: str) -> tuple[int, str]:
    match = _LAYER_RE.search(label)
    if match:
        return int(match.group(1)), label
    return 10**9, label


def module_layer_label(module_name: str) -> str:
    match = _LAYER_RE.search(module_name)
    if match:
        suffix = module_name.split(f"layers.{match.group(1)}.", 1)[-1]
        return f"L{match.group(1)}:{suffix}"
    return module_name or "shared_gate"


@dataclass
class RoutingRecord:
    counts: np.ndarray
    weights: np.ndarray
    tokens: int = 0
    assignments: int = 0
    null_assignments: int = 0
    bypass_tokens: int = 0


@dataclass
class RoutingAccumulator:
    axis_name: str
    num_choices: int
    method: str
    layers: dict[str, RoutingRecord] = field(default_factory=dict)

    def add(
        self,
        layer: str,
        counts: np.ndarray,
        weights: np.ndarray,
        *,
        tokens: int,
        assignments: int,
        null_assignments: int = 0,
        bypass_tokens: int = 0,
    ) -> None:
        record = self.layers.setdefault(
            layer,
            RoutingRecord(
                counts=np.zeros(self.num_choices, dtype=np.float64),
                weights=np.zeros(self.num_choices, dtype=np.float64),
            ),
        )
        record.counts += counts.astype(np.float64)
        record.weights += weights.astype(np.float64)
        record.tokens += int(tokens)
        record.assignments += int(assignments)
        record.null_assignments += int(null_assignments)
        record.bypass_tokens += int(bypass_tokens)


class RoutingCollector:
    def __init__(self, method: str, adapter_config: dict[str, Any]) -> None:
        self.method = method
        self.adapter_config = adapter_config
        self.current_language = ""
        self.hooks: list[Any] = []
        self.by_language: dict[str, RoutingAccumulator] = {}
        self.axis_name = "expert"
        self.num_choices = self._num_choices()

    def _num_choices(self) -> int:
        if self.method == "MOLA":
            self.axis_name = "true_expert"
            return int(self.adapter_config.get("mola_num_experts", 0) or 0)
        if self.method == "MOELORA":
            self.axis_name = "expert"
            return int(self.adapter_config.get("expert_num", 0) or 0)
        if self.method == "HYDRALORA" and not bool(self.adapter_config.get("use_hydralora_experts", False)):
            self.axis_name = "head"
            return int(self.adapter_config.get("lora_num", 0) or 0)
        self.axis_name = "expert"
        return int(self.adapter_config.get("num_experts", 0) or 0)

    def _acc(self) -> RoutingAccumulator:
        if not self.current_language:
            raise RuntimeError("collector current_language is unset")
        if self.current_language not in self.by_language:
            self.by_language[self.current_language] = RoutingAccumulator(
                axis_name=self.axis_name,
                num_choices=self.num_choices,
                method=self.method,
            )
        return self.by_language[self.current_language]

    def set_language(self, language: str) -> None:
        self.current_language = language

    def attach(self, model: Any) -> None:
        if self.method == "MOLA":
            self._attach_mola(model)
        elif self.method == "MOELORA":
            self._attach_moelora(model)
        elif self.method == "HYDRALORA":
            self._attach_hydralora(model)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        if not self.hooks:
            raise RuntimeError(f"No routing hooks attached for {self.method}")

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _attach_mola(self, model: Any) -> None:
        for name, module in model.named_modules():
            if module.__class__.__name__ != "TopKMoeLayer" or not hasattr(module, "gate"):
                continue

            def hook(_gate, _inputs, output, *, parent=module, module_name=name):
                self._record_mola_logits(module_layer_label(module_name), output, parent)

            self.hooks.append(module.gate.register_forward_hook(hook))

    def _record_mola_logits(self, layer: str, logits: Any, parent: Any) -> None:
        import torch

        with torch.no_grad():
            probs = torch.softmax(logits.detach().to(torch.float32), dim=-1)
            top_k = int(getattr(parent, "top_k", 1))
            num_true = int(getattr(parent, "num_true_experts", self.num_choices))
            total = int(probs.shape[-1])
            weights, selected = torch.topk(probs, k=min(top_k, total), dim=-1)
            true_mask = selected < num_true
            true_selected = selected[true_mask]
            counts = torch.bincount(true_selected.reshape(-1), minlength=num_true)[:num_true].cpu().numpy()
            true_weights = torch.zeros(num_true, dtype=torch.float32, device=probs.device)
            if true_mask.any():
                true_weights.scatter_add_(0, selected[true_mask].reshape(-1), weights[true_mask].reshape(-1))
            tokens = int(probs.shape[0])
            null_assignments = int((~true_mask).sum().item())
            bypass_tokens = int((true_mask.sum(dim=-1) == 0).sum().item())
            self._acc().add(
                layer,
                counts,
                true_weights.cpu().numpy(),
                tokens=tokens,
                assignments=int(selected.numel()),
                null_assignments=null_assignments,
                bypass_tokens=bypass_tokens,
            )

    def _attach_moelora(self, model: Any) -> None:
        gate_dict = getattr(model, "lora_gate", None)
        if gate_dict is None and hasattr(model, "base_model"):
            gate_dict = getattr(model.base_model, "lora_gate", None)
        if gate_dict is not None:
            for name, gate in gate_dict.items():
                self.hooks.append(gate.register_forward_hook(lambda _m, _i, out, n=name: self._record_prob_logits(f"shared_gate:{n}", out)))
            return
        for name, module in model.named_modules():
            if name.endswith("lora_gate.default") or ".lora_gate." in name:
                self.hooks.append(module.register_forward_hook(lambda _m, _i, out, n=name: self._record_prob_logits(module_layer_label(n), out)))

    def _attach_hydralora(self, model: Any) -> None:
        want_heads = self.axis_name == "head"
        for name, module in model.named_modules():
            classes = {cls.__name__ for cls in module.__class__.__mro__}
            if "HydraLoraLayer" not in classes:
                continue

            def hook(mod, _inputs, _output, *, module_name=name):
                caches = getattr(mod, "_caches", {})
                if not isinstance(caches, dict):
                    return
                for key, value in list(caches.items()):
                    if not key.endswith("_router_logits"):
                        continue
                    if want_heads and not key.startswith("hydra_head_"):
                        continue
                    if (not want_heads) and key != "hydra_expert_router_logits":
                        continue
                    self._record_prob_logits(module_layer_label(f"{module_name}.{key}"), value)

            self.hooks.append(module.register_forward_hook(hook))

    def _record_prob_logits(self, layer: str, logits: Any) -> None:
        import torch

        with torch.no_grad():
            probs = torch.softmax(logits.detach().to(torch.float32), dim=-1)
            flat = probs.reshape(-1, probs.shape[-1])
            if flat.shape[-1] != self.num_choices:
                return
            choices = torch.argmax(flat, dim=-1)
            counts = torch.bincount(choices, minlength=self.num_choices)[: self.num_choices].cpu().numpy()
            weights = flat.sum(dim=0).cpu().numpy()
            self._acc().add(
                layer,
                counts,
                weights,
                tokens=int(flat.shape[0]),
                assignments=int(flat.shape[0]),
            )


def build_matrices(
    collector: RoutingCollector,
    languages: list[str],
) -> dict[str, Any]:
    layer_names = sorted(
        {layer for acc in collector.by_language.values() for layer in acc.layers},
        key=layer_sort_key,
    )
    counts = np.zeros((len(languages), len(layer_names), collector.num_choices), dtype=np.float64)
    weights = np.zeros_like(counts)
    tokens = np.zeros((len(languages), len(layer_names)), dtype=np.int64)
    assignments = np.zeros_like(tokens)
    null_assignments = np.zeros_like(tokens)
    bypass_tokens = np.zeros_like(tokens)

    layer_index = {layer: idx for idx, layer in enumerate(layer_names)}
    for lang_idx, language in enumerate(languages):
        acc = collector.by_language.get(language)
        if acc is None:
            continue
        for layer, record in acc.layers.items():
            idx = layer_index[layer]
            counts[lang_idx, idx] = record.counts
            weights[lang_idx, idx] = record.weights
            tokens[lang_idx, idx] = record.tokens
            assignments[lang_idx, idx] = record.assignments
            null_assignments[lang_idx, idx] = record.null_assignments
            bypass_tokens[lang_idx, idx] = record.bypass_tokens

    count_norm = np.divide(
        counts,
        np.maximum(assignments[:, :, None], 1),
        out=np.zeros_like(counts, dtype=np.float64),
        where=assignments[:, :, None] > 0,
    )
    weight_norm = np.divide(
        weights,
        np.maximum(tokens[:, :, None], 1),
        out=np.zeros_like(weights, dtype=np.float64),
        where=tokens[:, :, None] > 0,
    )
    totals = weight_norm.sum(axis=2, keepdims=True)
    probs = np.divide(weight_norm, totals, out=np.zeros_like(weight_norm), where=totals > 0)
    entropy = -(probs * np.log(probs + 1e-10)).sum(axis=2)

    return {
        "routing_counts": counts,
        "routing_count_normalized": count_norm,
        "routing_weight_sums": weights,
        "routing_weight_normalized": weight_norm,
        "tokens": tokens,
        "assignments": assignments,
        "null_assignments": null_assignments,
        "bypass_tokens": bypass_tokens,
        "layer_entropy": entropy,
        "layer_labels": np.array(layer_names, dtype=object),
        "choice_labels": np.array([f"{collector.axis_name}_{idx}" for idx in range(collector.num_choices)], dtype=object),
        "axis_name": collector.axis_name,
    }
