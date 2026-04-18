#!/usr/bin/env python3
"""Benchmark MixLoRA-based sparse execution variants.

This harness is intentionally standalone and Torch-only. It focuses on FFN-style
MoE+LoRA execution and compares:

- lora_ffn_dense: plain dense LoRA FFN control
- hmora_ffn_dense_packed: HMoRA-style dense-packed expert mixing over FFN LoRA deltas
- mixlora_current: current MixLoRA execution shape from the repo
- mixlora_sparse_grouped_v2: grouped sparse dispatch prototype

Two routing modes are supported:

- controlled_routing: matched externally generated top-k assignments
- native_routing: per-method router logits and top-k selection
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _str2bool(raw: str) -> bool:
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {raw!r}")


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise SystemExit(f"Unsupported dtype: {name!r}") from exc


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _safe_peak_memory(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _seeded_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _randn(
    *shape: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.randn(*shape, generator=generator, dtype=torch.float32)
    return tensor.to(device=device, dtype=dtype)


def _count_trainable_params(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def _masked_softmax_from_logits(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    top_k = max(1, min(int(top_k), logits.shape[-1]))
    probs = torch.softmax(logits, dim=-1)
    top_values, top_indices = torch.topk(probs, top_k, dim=-1)
    top_values = top_values / top_values.sum(dim=-1, keepdim=True)
    full_gate = torch.zeros_like(probs)
    full_gate.scatter_(-1, top_indices, top_values)
    return probs, top_indices, top_values.to(logits.dtype), full_gate.to(logits.dtype)


def _controlled_routing(
    *,
    num_positions: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    top_k = max(1, min(int(top_k), num_experts))
    generator = _seeded_generator(seed)
    logits = torch.randn(num_positions, num_experts, generator=generator, dtype=torch.float32)
    top_values, top_indices = torch.topk(logits, top_k, dim=-1)
    weights = torch.softmax(top_values, dim=-1).to(device=device, dtype=dtype)
    selected = top_indices.to(device=device)
    full_gate = torch.zeros(num_positions, num_experts, device=device, dtype=dtype)
    full_gate.scatter_(1, selected, weights)
    return selected, weights, full_gate


def _stats_from_full_gate(full_gate: torch.Tensor) -> dict[str, float]:
    gate = full_gate.float()
    active_mask = gate > 1e-6
    logical_active = active_mask.sum(dim=-1).float()
    expert_usage = active_mask.float().sum(dim=0)
    mean_usage = expert_usage.mean()
    load_cv = 0.0
    if float(mean_usage) > 0:
        load_cv = float((expert_usage.std(unbiased=False) / (mean_usage + 1e-6)).item())
    entropy = -(gate * torch.log(gate + 1e-9)).sum(dim=-1).mean()
    return {
        "logical_active_experts": float(logical_active.mean().item()),
        "active_expert_frac": float((expert_usage > 0).float().mean().item()),
        "expert_load_cv": load_cv,
        "route_entropy": float(entropy.item()),
    }


def _dispatch_stats(
    *,
    selected_experts: torch.Tensor,
    num_experts: int,
    executed_fraction: float,
    small_load_threshold: int,
) -> dict[str, float]:
    counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts).float()
    active_counts = counts[counts > 0]
    avg_packed = float(active_counts.mean().item()) if active_counts.numel() > 0 else 0.0
    small_frac = (
        float((active_counts < small_load_threshold).float().mean().item()) if active_counts.numel() > 0 else 0.0
    )
    return {
        "executed_expert_fraction": float(executed_fraction),
        "avg_packed_tokens_per_active_expert": avg_packed,
        "small_load_expert_frac": small_frac,
    }


def _mixlora_aux_loss(router_probs: torch.Tensor, selected_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    positions, top_k = selected_experts.shape
    expert_fraction = torch.bincount(selected_experts.reshape(-1), minlength=num_experts).float()
    expert_fraction = expert_fraction / float(max(1, positions * top_k))
    prob_fraction = router_probs.float().mean(dim=0)
    return num_experts * torch.sum(expert_fraction * prob_fraction)


class FrozenFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.act_fn = F.silu
        self.reset_parameters()
        for param in self.parameters():
            param.requires_grad_(False)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02)


class LoRADelta(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.lora_a = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_b = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.scaling = float(alpha) / float(rank)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_b(self.lora_a(x.to(self.lora_a.weight.dtype))) * self.scaling


class PackedExpertDelta(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        num_experts: int,
        alpha: float,
        device: torch.device,
        dtype: torch.dtype,
        use_hydra_lora: bool,
    ) -> None:
        super().__init__()
        a_rows = rank if use_hydra_lora else rank * num_experts
        self.lora_a = nn.Parameter(torch.empty(a_rows, in_features, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.empty(out_features, rank * num_experts, device=device, dtype=dtype))
        self.rank = rank
        self.num_experts = num_experts
        self.use_hydra_lora = use_hydra_lora
        self.scaling = float(alpha) / math.sqrt(rank)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor, full_gate: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(x.to(self.lora_a.dtype), self.lora_a)
        target_shape = hidden.shape[:-1] + (self.num_experts, self.rank)
        if self.use_hydra_lora:
            hidden = hidden.unsqueeze(-2).expand(target_shape)
        else:
            hidden = hidden.view(target_shape)
        hidden = (hidden * full_gate.unsqueeze(-1)).reshape(hidden.shape[:-2] + (-1,))
        delta = F.linear(hidden, self.lora_b) * self.scaling
        return delta.to(x.dtype)


class LoRAFFNDense(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        rank: int,
        alpha: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.base = FrozenFFN(hidden_size, intermediate_size, device=device, dtype=dtype)
        self.gate_delta = LoRADelta(hidden_size, intermediate_size, rank=rank, alpha=alpha, device=device, dtype=dtype)
        self.up_delta = LoRADelta(hidden_size, intermediate_size, rank=rank, alpha=alpha, device=device, dtype=dtype)
        self.down_delta = LoRADelta(intermediate_size, hidden_size, rank=rank, alpha=alpha, device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
        gate_states = self.base.gate_proj(hidden_states) + self.gate_delta(hidden_states)
        up_states = self.base.up_proj(hidden_states) + self.up_delta(hidden_states)
        act_result = self.base.act_fn(gate_states) * up_states
        output = self.base.down_proj(act_result) + self.down_delta(act_result)
        stats = {
            "logical_active_experts": 1.0,
            "active_expert_frac": 1.0,
            "expert_load_cv": 0.0,
            "route_entropy": 0.0,
            "executed_experts_per_position": 1.0,
            "executed_expert_fraction": 1.0,
            "avg_packed_tokens_per_active_expert": 0.0,
            "small_load_expert_frac": 0.0,
        }
        return output, stats, None


class HMoRAFFNDensePacked(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        rank: int,
        alpha: float,
        num_experts: int,
        top_k: int,
        jitter_noise: float,
        use_hydra_lora: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.base = FrozenFFN(hidden_size, intermediate_size, device=device, dtype=dtype)
        self.router = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
        self.gate_delta = PackedExpertDelta(
            hidden_size, intermediate_size, rank=rank, num_experts=num_experts, alpha=alpha,
            device=device, dtype=dtype, use_hydra_lora=use_hydra_lora
        )
        self.up_delta = PackedExpertDelta(
            hidden_size, intermediate_size, rank=rank, num_experts=num_experts, alpha=alpha,
            device=device, dtype=dtype, use_hydra_lora=use_hydra_lora
        )
        self.down_delta = PackedExpertDelta(
            intermediate_size, hidden_size, rank=rank, num_experts=num_experts, alpha=alpha,
            device=device, dtype=dtype, use_hydra_lora=use_hydra_lora
        )
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

    def _native_routing(
        self, flat_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        router_inputs = flat_hidden
        if self.training and self.jitter_noise > 0:
            jitter = torch.empty_like(flat_hidden).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
            router_inputs = router_inputs * jitter
        router_logits = self.router(router_inputs.to(self.router.weight.dtype))
        _, selected_experts, routing_weights, full_gate = _masked_softmax_from_logits(router_logits, self.top_k)
        return router_logits, selected_experts, routing_weights, full_gate

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        *,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        full_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        gate_states = self.base.gate_proj(flat_hidden) + self.gate_delta(flat_hidden, full_gate)
        up_states = self.base.up_proj(flat_hidden) + self.up_delta(flat_hidden, full_gate)
        act_result = self.base.act_fn(gate_states) * up_states
        output = self.base.down_proj(act_result) + self.down_delta(act_result, full_gate)
        routing_stats = _stats_from_full_gate(full_gate)
        dispatch_stats = _dispatch_stats(
            selected_experts=selected_experts,
            num_experts=self.num_experts,
            executed_fraction=1.0,
            small_load_threshold=32,
        )
        stats = {
            **routing_stats,
            **dispatch_stats,
            "executed_experts_per_position": float(self.num_experts),
        }
        return output.reshape_as(hidden_states), stats, None

    def forward_controlled(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        full_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
        return self._forward_impl(hidden_states, selected_experts=selected_experts, routing_weights=routing_weights, full_gate=full_gate)

    def forward_native(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_logits, selected_experts, routing_weights, full_gate = self._native_routing(flat_hidden)
        return self._forward_impl(hidden_states, selected_experts=selected_experts, routing_weights=routing_weights, full_gate=full_gate)


class MixLoRASharedBase(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        rank: int,
        alpha: float,
        num_experts: int,
        top_k: int,
        jitter_noise: float,
        router_init_range: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.base = FrozenFFN(hidden_size, intermediate_size, device=device, dtype=dtype)
        self.router = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
        self.gate_experts = nn.ModuleList(
            [LoRADelta(hidden_size, intermediate_size, rank=rank, alpha=alpha, device=device, dtype=dtype) for _ in range(num_experts)]
        )
        self.up_experts = nn.ModuleList(
            [LoRADelta(hidden_size, intermediate_size, rank=rank, alpha=alpha, device=device, dtype=dtype) for _ in range(num_experts)]
        )
        self.down_experts = nn.ModuleList(
            [LoRADelta(intermediate_size, hidden_size, rank=rank, alpha=alpha, device=device, dtype=dtype) for _ in range(num_experts)]
        )
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        nn.init.normal_(self.router.weight, mean=0.0, std=router_init_range)

    def _native_routing(self, flat_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        router_inputs = flat_hidden
        if self.training and self.jitter_noise > 0:
            jitter = torch.empty_like(flat_hidden).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
            router_inputs = router_inputs * jitter
        router_logits = self.router(router_inputs.to(self.router.weight.dtype))
        router_probs, selected_experts, routing_weights, full_gate = self._native_topk(router_logits)
        return router_logits, router_probs, selected_experts, routing_weights, full_gate

    def _native_topk(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        router_probs, selected_experts, routing_weights, full_gate = _masked_softmax_from_logits(router_logits, self.top_k)
        return router_probs, selected_experts, routing_weights, full_gate

    def _common_stats(
        self,
        *,
        router_probs: torch.Tensor | None,
        selected_experts: torch.Tensor,
        full_gate: torch.Tensor,
        executed_fraction: float,
        small_load_threshold: int = 32,
    ) -> tuple[dict[str, float], torch.Tensor | None]:
        routing_stats = _stats_from_full_gate(full_gate)
        dispatch_stats = _dispatch_stats(
            selected_experts=selected_experts,
            num_experts=self.num_experts,
            executed_fraction=executed_fraction,
            small_load_threshold=small_load_threshold,
        )
        stats = {**routing_stats, **dispatch_stats}
        stats["executed_experts_per_position"] = routing_stats["logical_active_experts"]
        aux_loss = None
        if router_probs is not None:
            aux_loss = _mixlora_aux_loss(router_probs, selected_experts, self.num_experts)
        return stats, aux_loss


class MixLoRACurrent(MixLoRASharedBase):
    def _compute_outputs(
        self,
        flat_hidden: torch.Tensor,
        *,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        full_gate: torch.Tensor,
        router_probs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
        common_gate = self.base.gate_proj(flat_hidden)
        common_up = self.base.up_proj(flat_hidden)
        final_hidden = torch.zeros(flat_hidden.shape[0], flat_hidden.shape[1], device=flat_hidden.device, dtype=flat_hidden.dtype)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            slot_idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            token_inputs = flat_hidden.index_select(0, top_x)
            gate_states = common_gate.index_select(0, top_x) + self.gate_experts[expert_idx](token_inputs).to(common_gate.dtype)
            up_states = common_up.index_select(0, top_x) + self.up_experts[expert_idx](token_inputs).to(common_up.dtype)
            act_result = self.base.act_fn(gate_states) * up_states
            down_states = self.base.down_proj(act_result) + self.down_experts[expert_idx](act_result).to(flat_hidden.dtype)
            weighted = down_states * routing_weights[top_x, slot_idx].unsqueeze(-1).to(down_states.dtype)
            final_hidden.index_add_(0, top_x, weighted.to(final_hidden.dtype))

        stats, aux_loss = self._common_stats(
            router_probs=router_probs,
            selected_experts=selected_experts,
            full_gate=full_gate,
            executed_fraction=float(full_gate.count_nonzero().item()) / float(full_gate.numel()),
        )
        return final_hidden, stats, aux_loss

    def forward_controlled(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        full_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        output, stats, _ = self._compute_outputs(
            flat_hidden,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            full_gate=full_gate,
            router_probs=None,
        )
        return output.reshape_as(hidden_states), stats, None

    def forward_native(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        _, router_probs, selected_experts, routing_weights, full_gate = self._native_routing(flat_hidden)
        output, stats, aux_loss = self._compute_outputs(
            flat_hidden,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            full_gate=full_gate,
            router_probs=router_probs,
        )
        return output.reshape_as(hidden_states), stats, aux_loss


class MixLoRASparseGroupedV2(MixLoRASharedBase):
    def _compute_outputs(
        self,
        flat_hidden: torch.Tensor,
        *,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        full_gate: torch.Tensor,
        router_probs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
        common_gate = self.base.gate_proj(flat_hidden)
        common_up = self.base.up_proj(flat_hidden)
        num_positions, top_k = selected_experts.shape

        token_indices = torch.arange(num_positions, device=flat_hidden.device).unsqueeze(1).expand(num_positions, top_k).reshape(-1)
        expert_indices = selected_experts.reshape(-1)
        packed_weights = routing_weights.reshape(-1)

        order = torch.argsort(expert_indices, stable=True)
        sorted_experts = expert_indices.index_select(0, order)
        sorted_tokens = token_indices.index_select(0, order)
        sorted_weights = packed_weights.index_select(0, order)

        packed_hidden = flat_hidden.index_select(0, sorted_tokens)
        packed_gate = common_gate.index_select(0, sorted_tokens)
        packed_up = common_up.index_select(0, sorted_tokens)
        packed_output = torch.zeros_like(flat_hidden)
        weighted_output = torch.zeros_like(packed_hidden)

        counts = torch.bincount(sorted_experts, minlength=self.num_experts)
        offsets = counts.cumsum(dim=0)
        start = 0
        for expert_idx, end in enumerate(offsets.tolist()):
            if end == start:
                continue
            token_inputs = packed_hidden[start:end]
            gate_states = packed_gate[start:end] + self.gate_experts[expert_idx](token_inputs).to(packed_gate.dtype)
            up_states = packed_up[start:end] + self.up_experts[expert_idx](token_inputs).to(packed_up.dtype)
            act_result = self.base.act_fn(gate_states) * up_states
            down_states = self.base.down_proj(act_result) + self.down_experts[expert_idx](act_result).to(flat_hidden.dtype)
            weighted_output[start:end] = down_states * sorted_weights[start:end].unsqueeze(-1).to(down_states.dtype)
            start = end

        packed_output.index_add_(0, sorted_tokens, weighted_output.to(packed_output.dtype))
        executed_fraction = float(full_gate.count_nonzero().item()) / float(full_gate.numel())
        stats, aux_loss = self._common_stats(
            router_probs=router_probs,
            selected_experts=selected_experts,
            full_gate=full_gate,
            executed_fraction=executed_fraction,
        )
        active_counts = counts[counts > 0].float()
        stats["avg_packed_tokens_per_active_expert"] = float(active_counts.mean().item()) if active_counts.numel() > 0 else 0.0
        stats["small_load_expert_frac"] = (
            float((active_counts < 32).float().mean().item()) if active_counts.numel() > 0 else 0.0
        )
        return packed_output, stats, aux_loss

    def forward_controlled(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        full_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        output, stats, _ = self._compute_outputs(
            flat_hidden,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            full_gate=full_gate,
            router_probs=None,
        )
        return output.reshape_as(hidden_states), stats, None

    def forward_native(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        _, router_probs, selected_experts, routing_weights, full_gate = self._native_routing(flat_hidden)
        output, stats, aux_loss = self._compute_outputs(
            flat_hidden,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            full_gate=full_gate,
            router_probs=router_probs,
        )
        return output.reshape_as(hidden_states), stats, aux_loss


def _run_timed_benchmark(
    *,
    model: nn.Module,
    step_fn,
    pass_mode: str,
    steps: int,
    warmup_steps: int,
    device: torch.device,
    router_aux_loss_coef: float,
) -> tuple[float, int | None, dict[str, float], bool, float]:
    model.train()
    last_stats: dict[str, float] = {}
    finite_pass = True
    last_aux_value = 0.0
    step_times: list[float] = []
    total_steps = warmup_steps + steps

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for step_idx in range(total_steps):
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = None

        _sync(device)
        start = time.perf_counter()
        output, last_stats, aux_loss = step_fn()
        if not torch.isfinite(output).all():
            finite_pass = False
        if pass_mode == "forward_backward":
            loss = output.float().square().mean()
            if aux_loss is not None and router_aux_loss_coef > 0:
                loss = loss + aux_loss.float() * float(router_aux_loss_coef)
                last_aux_value = float(aux_loss.detach().item())
            if not torch.isfinite(loss):
                finite_pass = False
            loss.backward()
            for param in model.parameters():
                if param.requires_grad and param.grad is not None and not torch.isfinite(param.grad).all():
                    finite_pass = False
                    break
        _sync(device)
        elapsed = time.perf_counter() - start
        if step_idx >= warmup_steps:
            step_times.append(elapsed)

    mean_step = float(sum(step_times) / len(step_times))
    peak_memory = _safe_peak_memory(device)
    return mean_step, peak_memory, last_stats, finite_pass, last_aux_value


def _write_results(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _result_row(
    *,
    variant: str,
    routing_mode: str,
    pass_mode: str,
    hidden_size: int,
    intermediate_size: int,
    rank: int,
    num_experts: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype_name: str,
    mean_step_seconds: float,
    peak_memory_bytes: int | None,
    trainable_params: int,
    stats: dict[str, float],
    finite_pass: bool,
    router_aux_loss: float,
    controlled_active_experts: int | None = None,
    native_top_k: int | None = None,
    use_hydra_lora: bool | None = None,
) -> dict[str, Any]:
    tokens_per_step = batch_size * seq_len
    return {
        "variant": variant,
        "routing_mode": routing_mode,
        "pass_mode": pass_mode,
        "device": str(device),
        "dtype": dtype_name,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "rank": rank,
        "num_experts": num_experts,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_step": tokens_per_step,
        "mean_step_seconds": mean_step_seconds,
        "tokens_per_second": float(tokens_per_step / mean_step_seconds),
        "peak_memory_bytes": peak_memory_bytes,
        "trainable_params": trainable_params,
        "logical_active_experts": float(stats["logical_active_experts"]),
        "executed_experts_per_position": float(stats["executed_experts_per_position"]),
        "executed_expert_fraction": float(stats["executed_expert_fraction"]),
        "active_expert_frac": float(stats["active_expert_frac"]),
        "expert_load_cv": float(stats["expert_load_cv"]),
        "route_entropy": float(stats["route_entropy"]),
        "avg_packed_tokens_per_active_expert": float(stats["avg_packed_tokens_per_active_expert"]),
        "small_load_expert_frac": float(stats["small_load_expert_frac"]),
        "finite_pass": bool(finite_pass),
        "router_aux_loss": float(router_aux_loss),
        "controlled_active_experts": controlled_active_experts,
        "native_top_k": native_top_k,
        "use_hydra_lora": use_hydra_lora,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default="lora_ffn_dense,hmora_ffn_dense_packed,mixlora_current,mixlora_sparse_grouped_v2",
    )
    parser.add_argument("--routing-modes", default="controlled_routing,native_routing")
    parser.add_argument("--pass-mode", choices=["forward", "forward_backward", "both"], default="forward_backward")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--hidden-sizes", default="4096")
    parser.add_argument("--ffn-multipliers", default="4")
    parser.add_argument("--ranks", default="8")
    parser.add_argument("--num-experts-values", default="8,16,32")
    parser.add_argument("--batch-size-values", default="1,8")
    parser.add_argument("--seq-len-values", default="256,1024")
    parser.add_argument("--controlled-active-experts-values", default="1,2,4")
    parser.add_argument("--native-top-k-values", default="1,2,4")
    parser.add_argument("--router-init-range", type=float, default=0.02)
    parser.add_argument("--jitter-noise", type=float, default=0.0)
    parser.add_argument("--router-aux-loss-coef", type=float, default=1e-2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hmora-use-hydra-lora", type=_str2bool, default=False)
    parser.add_argument("--output-dir", default="outputs/mixlora_sparse_v2_study")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    variants = _parse_str_list(args.variants)
    routing_modes = _parse_str_list(args.routing_modes)
    hidden_sizes = _parse_int_list(args.hidden_sizes)
    ffn_multipliers = _parse_int_list(args.ffn_multipliers)
    ranks = _parse_int_list(args.ranks)
    num_experts_values = _parse_int_list(args.num_experts_values)
    batch_sizes = _parse_int_list(args.batch_size_values)
    seq_lens = _parse_int_list(args.seq_len_values)
    controlled_active_values = _parse_int_list(args.controlled_active_experts_values)
    native_top_ks = _parse_int_list(args.native_top_k_values)
    pass_modes = ["forward", "forward_backward"] if args.pass_mode == "both" else [args.pass_mode]
    device = _device(args.device)
    dtype = _torch_dtype(args.dtype)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    rows: list[dict[str, Any]] = []
    combo_seed = args.seed

    for hidden_size in hidden_sizes:
        for ffn_multiplier in ffn_multipliers:
            intermediate_size = int(hidden_size * ffn_multiplier)
            for rank in ranks:
                alpha = float(rank * 2)
                for batch_size in batch_sizes:
                    for seq_len in seq_lens:
                        hidden_generator = _seeded_generator(combo_seed)
                        hidden_states = _randn(batch_size, seq_len, hidden_size, generator=hidden_generator, device=device, dtype=dtype)
                        num_positions = batch_size * seq_len

                        for variant in variants:
                            if variant == "lora_ffn_dense":
                                model = LoRAFFNDense(
                                    hidden_size=hidden_size,
                                    intermediate_size=intermediate_size,
                                    rank=rank,
                                    alpha=alpha,
                                    device=device,
                                    dtype=dtype,
                                )
                                for routing_mode in routing_modes:
                                    for pass_mode in pass_modes:
                                        mean_step, peak_mem, stats, finite_pass, aux_loss = _run_timed_benchmark(
                                            model=model,
                                            step_fn=lambda m=model, h=hidden_states: m(h),
                                            pass_mode=pass_mode,
                                            steps=args.steps,
                                            warmup_steps=args.warmup_steps,
                                            device=device,
                                            router_aux_loss_coef=args.router_aux_loss_coef,
                                        )
                                        rows.append(
                                            _result_row(
                                                variant=variant,
                                                routing_mode=routing_mode,
                                                pass_mode=pass_mode,
                                                hidden_size=hidden_size,
                                                intermediate_size=intermediate_size,
                                                rank=rank,
                                                num_experts=1,
                                                batch_size=batch_size,
                                                seq_len=seq_len,
                                                device=device,
                                                dtype_name=args.dtype,
                                                mean_step_seconds=mean_step,
                                                peak_memory_bytes=peak_mem,
                                                trainable_params=_count_trainable_params(model),
                                                stats=stats,
                                                finite_pass=finite_pass,
                                                router_aux_loss=aux_loss,
                                            )
                                        )
                                combo_seed += 1
                                continue

                            for num_experts in num_experts_values:
                                if variant == "hmora_ffn_dense_packed":
                                    model = HMoRAFFNDensePacked(
                                        hidden_size=hidden_size,
                                        intermediate_size=intermediate_size,
                                        rank=rank,
                                        alpha=alpha,
                                        num_experts=num_experts,
                                        top_k=native_top_ks[0],
                                        jitter_noise=args.jitter_noise,
                                        use_hydra_lora=args.hmora_use_hydra_lora,
                                        device=device,
                                        dtype=dtype,
                                    )
                                elif variant == "mixlora_current":
                                    model = MixLoRACurrent(
                                        hidden_size=hidden_size,
                                        intermediate_size=intermediate_size,
                                        rank=rank,
                                        alpha=alpha,
                                        num_experts=num_experts,
                                        top_k=native_top_ks[0],
                                        jitter_noise=args.jitter_noise,
                                        router_init_range=args.router_init_range,
                                        device=device,
                                        dtype=dtype,
                                    )
                                elif variant == "mixlora_sparse_grouped_v2":
                                    model = MixLoRASparseGroupedV2(
                                        hidden_size=hidden_size,
                                        intermediate_size=intermediate_size,
                                        rank=rank,
                                        alpha=alpha,
                                        num_experts=num_experts,
                                        top_k=native_top_ks[0],
                                        jitter_noise=args.jitter_noise,
                                        router_init_range=args.router_init_range,
                                        device=device,
                                        dtype=dtype,
                                    )
                                else:
                                    raise SystemExit(f"Unsupported variant: {variant!r}")

                                trainable_params = _count_trainable_params(model)

                                for routing_mode in routing_modes:
                                    if routing_mode == "controlled_routing":
                                        for active_experts in controlled_active_values:
                                            if active_experts > num_experts:
                                                continue
                                            selected_experts, routing_weights, full_gate = _controlled_routing(
                                                num_positions=num_positions,
                                                num_experts=num_experts,
                                                top_k=active_experts,
                                                device=device,
                                                dtype=dtype,
                                                seed=combo_seed + active_experts,
                                            )
                                            for pass_mode in pass_modes:
                                                mean_step, peak_mem, stats, finite_pass, aux_loss = _run_timed_benchmark(
                                                    model=model,
                                                    step_fn=(
                                                        lambda m=model, h=hidden_states, se=selected_experts, rw=routing_weights, fg=full_gate:
                                                        m.forward_controlled(h, se, rw, fg)
                                                    ),
                                                    pass_mode=pass_mode,
                                                    steps=args.steps,
                                                    warmup_steps=args.warmup_steps,
                                                    device=device,
                                                    router_aux_loss_coef=args.router_aux_loss_coef,
                                                )
                                                rows.append(
                                                    _result_row(
                                                        variant=variant,
                                                        routing_mode=routing_mode,
                                                        pass_mode=pass_mode,
                                                        hidden_size=hidden_size,
                                                        intermediate_size=intermediate_size,
                                                        rank=rank,
                                                        num_experts=num_experts,
                                                        batch_size=batch_size,
                                                        seq_len=seq_len,
                                                        device=device,
                                                        dtype_name=args.dtype,
                                                        mean_step_seconds=mean_step,
                                                        peak_memory_bytes=peak_mem,
                                                        trainable_params=trainable_params,
                                                        stats=stats,
                                                        finite_pass=finite_pass,
                                                        router_aux_loss=aux_loss,
                                                        controlled_active_experts=active_experts,
                                                        use_hydra_lora=args.hmora_use_hydra_lora if variant == "hmora_ffn_dense_packed" else None,
                                                    )
                                                )
                                    elif routing_mode == "native_routing":
                                        for top_k in native_top_ks:
                                            if top_k > num_experts:
                                                continue
                                            if variant == "hmora_ffn_dense_packed":
                                                model.top_k = top_k
                                                step_fn = lambda m=model, h=hidden_states: m.forward_native(h)
                                            else:
                                                model.top_k = top_k
                                                step_fn = lambda m=model, h=hidden_states: m.forward_native(h)
                                            for pass_mode in pass_modes:
                                                mean_step, peak_mem, stats, finite_pass, aux_loss = _run_timed_benchmark(
                                                    model=model,
                                                    step_fn=step_fn,
                                                    pass_mode=pass_mode,
                                                    steps=args.steps,
                                                    warmup_steps=args.warmup_steps,
                                                    device=device,
                                                    router_aux_loss_coef=args.router_aux_loss_coef,
                                                )
                                                rows.append(
                                                    _result_row(
                                                        variant=variant,
                                                        routing_mode=routing_mode,
                                                        pass_mode=pass_mode,
                                                        hidden_size=hidden_size,
                                                        intermediate_size=intermediate_size,
                                                        rank=rank,
                                                        num_experts=num_experts,
                                                        batch_size=batch_size,
                                                        seq_len=seq_len,
                                                        device=device,
                                                        dtype_name=args.dtype,
                                                        mean_step_seconds=mean_step,
                                                        peak_memory_bytes=peak_mem,
                                                        trainable_params=trainable_params,
                                                        stats=stats,
                                                        finite_pass=finite_pass,
                                                        router_aux_loss=aux_loss,
                                                        native_top_k=top_k,
                                                        use_hydra_lora=args.hmora_use_hydra_lora if variant == "hmora_ffn_dense_packed" else None,
                                                    )
                                                )
                                    else:
                                        raise SystemExit(f"Unsupported routing mode: {routing_mode!r}")

                                combo_seed += 1

    if not rows:
        raise SystemExit("No benchmark rows were produced.")

    output_dir = Path(args.output_dir)
    _write_results(rows, output_dir)
    print(f"Wrote {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
