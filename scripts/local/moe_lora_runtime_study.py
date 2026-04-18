#!/usr/bin/env python3
"""Benchmark runtime behavior of LoRA-family MoE adapters.

This benchmark is intentionally standalone and Torch-only. It reproduces the
execution shape of the checked-in tuner implementations without importing the
full training stack.

Methods:
- lora: one dense low-rank path
- hmora: token-routed, packed dense execution
- adamole: token-routed, sparse expert dispatch
- moelora: batch/task-routed, sparse expert dispatch

Routing modes:
- controlled_routing: matched externally generated routing density
- native_routing: each method uses its own router rule

Artifacts:
- results.jsonl
- results.csv
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


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


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


def _count_trainable_params(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def _seeded_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _randn(
    *shape: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    scale: float = 1.0,
) -> torch.Tensor:
    tensor = torch.randn(*shape, generator=generator, dtype=torch.float32)
    if scale != 1.0:
        tensor = tensor * scale
    return tensor.to(device=device, dtype=dtype)


def _randint(
    low: int,
    high: int,
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.randint(low, high, shape, generator=generator, dtype=torch.long)
    return tensor.to(device=device)


def _masked_softmax_from_logits(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k <= 0 or top_k >= logits.shape[-1]:
        return torch.softmax(logits, dim=-1)
    top_k = min(top_k, logits.shape[-1])
    top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
    sparse_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
    sparse_logits.scatter_(-1, top_k_indices, top_k_values)
    return torch.softmax(sparse_logits, dim=-1)


def _normalize_selected_weights(weights: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    weighted = weights * active_mask.to(weights.dtype)
    denom = weighted.sum(dim=-1, keepdim=True)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return weighted / denom


def _dedup_floats(values: list[float]) -> list[float]:
    seen: set[float] = set()
    result: list[float] = []
    for value in values:
        rounded = round(float(value), 8)
        if rounded in seen:
            continue
        seen.add(rounded)
        result.append(float(value))
    return result


def _token_gate_stats(gate: torch.Tensor) -> dict[str, float]:
    flat_gate = gate.reshape(-1, gate.shape[-1]).float()
    active = flat_gate > 1e-6
    logical_active = active.sum(dim=-1).float()
    expert_usage = active.float().sum(dim=0)
    mean_usage = expert_usage.mean()
    entropy = -(flat_gate * torch.log(flat_gate + 1e-9)).sum(dim=-1).mean()
    load_cv = 0.0
    if float(mean_usage) > 0:
        load_cv = float((expert_usage.std(unbiased=False) / (mean_usage + 1e-6)).item())
    return {
        "logical_active_experts": float(logical_active.mean().item()),
        "active_expert_frac": float((expert_usage > 0).float().mean().item()),
        "expert_load_cv": load_cv,
        "route_entropy": float(entropy.item()),
    }


def _sample_gate_stats(sample_gate: torch.Tensor) -> dict[str, float]:
    gate = sample_gate.float()
    active = gate > 1e-6
    logical_active = active.sum(dim=-1).float()
    expert_usage = active.float().sum(dim=0)
    mean_usage = expert_usage.mean()
    entropy = -(gate * torch.log(gate + 1e-9)).sum(dim=-1).mean()
    load_cv = 0.0
    if float(mean_usage) > 0:
        load_cv = float((expert_usage.std(unbiased=False) / (mean_usage + 1e-6)).item())
    return {
        "logical_active_experts": float(logical_active.mean().item()),
        "active_expert_frac": float((expert_usage > 0).float().mean().item()),
        "expert_load_cv": load_cv,
        "route_entropy": float(entropy.item()),
    }


def _expand_sample_gate(sample_gate: torch.Tensor, seq_len: int) -> torch.Tensor:
    return sample_gate.unsqueeze(1).expand(sample_gate.shape[0], seq_len, sample_gate.shape[-1])


def _controlled_sample_gate(
    *,
    batch_size: int,
    num_experts: int,
    active_experts: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    active_experts = max(1, min(active_experts, num_experts))
    generator = _seeded_generator(seed)
    logits = torch.randn(batch_size, num_experts, generator=generator, dtype=torch.float32)
    _, top_idx = torch.topk(logits, active_experts, dim=-1)
    gate = torch.zeros(batch_size, num_experts, dtype=torch.float32)
    selected_logits = logits.gather(1, top_idx)
    selected_weights = torch.softmax(selected_logits, dim=-1)
    gate.scatter_(1, top_idx, selected_weights)
    return gate.to(device=device, dtype=dtype)


def _default_adamole_thresholds(num_experts: int) -> list[float]:
    return _dedup_floats([1.0 / float(num_experts), 0.25])


class LoRAAdapter(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        rank: int,
        alpha: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.base_weight = nn.Parameter(torch.empty(hidden_size, hidden_size, device=device, dtype=dtype), requires_grad=False)
        self.base_bias = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype), requires_grad=False)
        self.lora_a = nn.Parameter(torch.empty(rank, hidden_size, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.empty(hidden_size, rank, device=device, dtype=dtype))
        self.scaling = float(alpha) / float(rank)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.base_weight, std=0.02)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        result = F.linear(x, self.base_weight, self.base_bias)
        delta = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
        stats = {
            "logical_active_experts": 1.0,
            "executed_experts_per_position": 1.0,
            "executed_expert_fraction": 1.0,
            "active_expert_frac": 1.0,
            "expert_load_cv": 0.0,
            "route_entropy": 0.0,
        }
        return result + delta, stats


class HMoRAAdapter(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        rank: int,
        alpha: float,
        num_experts: int,
        use_hydra_lora: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        a_rows = rank if use_hydra_lora else rank * num_experts
        self.base_weight = nn.Parameter(torch.empty(hidden_size, hidden_size, device=device, dtype=dtype), requires_grad=False)
        self.base_bias = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype), requires_grad=False)
        self.router = nn.Linear(hidden_size, num_experts, bias=True, device=device, dtype=dtype)
        self.lora_a = nn.Parameter(torch.empty(a_rows, hidden_size, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.empty(hidden_size, rank * num_experts, device=device, dtype=dtype))
        self.rank = rank
        self.num_experts = num_experts
        self.use_hydra_lora = use_hydra_lora
        self.scaling = float(alpha) / math.sqrt(rank)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.base_weight, std=0.02)
        nn.init.kaiming_uniform_(self.router.weight, a=math.sqrt(5))
        nn.init.zeros_(self.router.bias)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def _apply_with_gate(self, x: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        result = F.linear(x, self.base_weight, self.base_bias)
        hidden_states = F.linear(x, self.lora_a)
        target_shape = hidden_states.shape[:-1] + (self.num_experts, self.rank)
        if self.use_hydra_lora:
            hidden_states = hidden_states.unsqueeze(-2).expand(target_shape)
        else:
            hidden_states = hidden_states.view(target_shape)
        hidden_states = (hidden_states * gate.unsqueeze(-1)).reshape(hidden_states.shape[:-2] + (-1,))
        delta = F.linear(hidden_states, self.lora_b) * self.scaling
        logical_stats = _token_gate_stats(gate)
        logical_stats["executed_experts_per_position"] = float(self.num_experts)
        logical_stats["executed_expert_fraction"] = 1.0
        return result + delta.to(result.dtype), logical_stats

    def forward_controlled(self, x: torch.Tensor, token_gate: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        return self._apply_with_gate(x, token_gate)

    def forward_native(self, x: torch.Tensor, top_k: int) -> tuple[torch.Tensor, dict[str, float]]:
        gate = _masked_softmax_from_logits(self.router(x), top_k)
        return self._apply_with_gate(x, gate)


class LoraExpert(nn.Module):
    def __init__(self, hidden_size: int, rank: int, scaling: float, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.lora_a = nn.Linear(hidden_size, rank, bias=False, device=device, dtype=dtype)
        self.lora_b = nn.Linear(rank, hidden_size, bias=False, device=device, dtype=dtype)
        self.scaling = scaling
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_b(self.lora_a(x)) * self.scaling


class AdaMoLEAdapter(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        rank: int,
        alpha: float,
        num_experts: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.base_weight = nn.Parameter(torch.empty(hidden_size, hidden_size, device=device, dtype=dtype), requires_grad=False)
        self.base_bias = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype), requires_grad=False)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
        self.threshold = nn.Linear(hidden_size, 1, bias=True, device=device, dtype=dtype)
        scaling = float(alpha) / float(rank)
        self.experts = nn.ModuleList(
            [LoraExpert(hidden_size, rank, scaling, device=device, dtype=dtype) for _ in range(num_experts)]
        )
        self.num_experts = num_experts
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.base_weight, std=0.02)
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.threshold.weight, a=math.sqrt(5))
        nn.init.zeros_(self.threshold.bias)

    def _forward_selected(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        selected: torch.Tensor,
        entropy_source: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        result = F.linear(x, self.base_weight, self.base_bias)
        flat_x = x.reshape(-1, x.shape[-1])
        flat_weights = weights.reshape(-1, weights.shape[-1])
        flat_selected = selected.reshape(-1, selected.shape[-1])
        flat_delta = torch.zeros_like(flat_x)

        for expert_idx, expert in enumerate(self.experts):
            active = flat_selected[:, expert_idx]
            if not bool(active.any()):
                continue
            delta = expert(flat_x[active])
            gate = flat_weights[active, expert_idx].unsqueeze(-1).to(delta.dtype)
            flat_delta[active] += delta * gate

        token_gate = flat_weights.view(*x.shape[:-1], -1)
        stats = _token_gate_stats(token_gate)
        stats["executed_experts_per_position"] = stats["logical_active_experts"]
        stats["executed_expert_fraction"] = stats["logical_active_experts"] / float(self.num_experts)
        stats["route_entropy"] = float(
            (-(entropy_source.float() * torch.log(entropy_source.float() + 1e-9)).sum(dim=-1).mean()).item()
        )
        return result + flat_delta.view_as(result), stats

    def forward_controlled(self, x: torch.Tensor, token_gate: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        selected = token_gate > 1e-6
        return self._forward_selected(x, token_gate, selected, token_gate)

    def forward_native(self, x: torch.Tensor, max_threshold: float) -> tuple[torch.Tensor, dict[str, float]]:
        flat_x = x.reshape(-1, x.shape[-1])
        gate_probs = torch.softmax(self.gate(flat_x), dim=-1)
        thresholds = torch.sigmoid(self.threshold(flat_x)) * max_threshold
        adapted = gate_probs - thresholds
        selected = adapted >= 0
        weights = _normalize_selected_weights(adapted, selected)
        return self._forward_selected(
            x,
            weights.view(*x.shape[:-1], -1),
            selected.view(*x.shape[:-1], -1),
            gate_probs.view(*x.shape[:-1], -1),
        )


class MoELoRAAdapter(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        rank: int,
        alpha: float,
        num_experts: int,
        task_num: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if rank % num_experts != 0:
            raise ValueError(f"MoELoRA requires rank divisible by num_experts, got rank={rank}, experts={num_experts}")
        rank_per_expert = rank // num_experts
        self.base_weight = nn.Parameter(torch.empty(hidden_size, hidden_size, device=device, dtype=dtype), requires_grad=False)
        self.base_bias = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype), requires_grad=False)
        self.task_embedding = nn.Embedding(task_num, hidden_size, device=device, dtype=dtype)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
        self.lora_a = nn.ModuleList(
            [nn.Linear(hidden_size, rank_per_expert, bias=False, device=device, dtype=dtype) for _ in range(num_experts)]
        )
        self.lora_b = nn.ModuleList(
            [nn.Linear(rank_per_expert, hidden_size, bias=False, device=device, dtype=dtype) for _ in range(num_experts)]
        )
        self.num_experts = num_experts
        self.rank = rank
        self.rank_per_expert = rank_per_expert
        self.scaling = float(alpha) / float(rank)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.base_weight, std=0.02)
        nn.init.normal_(self.task_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.gate.weight)
        for lora_a, lora_b in zip(self.lora_a, self.lora_b):
            nn.init.kaiming_uniform_(lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(lora_b.weight)

    def _apply_routing(self, x: torch.Tensor, sample_gate: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        result = F.linear(x, self.base_weight, self.base_bias)
        x_cast = x.to(self.lora_a[0].weight.dtype)
        gate = sample_gate.to(result.dtype)
        active_experts = torch.nonzero((gate > 1e-6).any(dim=0), as_tuple=False).flatten()
        for expert_idx in active_experts.tolist():
            expert_gate = gate[:, expert_idx]
            active_batch = expert_gate > 1e-6
            if not bool(active_batch.any()):
                continue

            if bool(torch.all(active_batch)):
                delta = self.lora_b[expert_idx](self.lora_a[expert_idx](x_cast))
                gate_view = expert_gate.view(gate.shape[0], *([1] * (delta.dim() - 1))).to(delta.dtype)
                result = result + (delta * gate_view * self.scaling).to(result.dtype)
                continue

            x_sub = x_cast[active_batch]
            delta = self.lora_b[expert_idx](self.lora_a[expert_idx](x_sub))
            gate_sub = expert_gate[active_batch].view(-1, *([1] * (delta.dim() - 1))).to(delta.dtype)
            result_sub = result[active_batch] + (delta * gate_sub * self.scaling).to(result.dtype)
            result[active_batch] = result_sub

        stats = _sample_gate_stats(gate)
        stats["executed_experts_per_position"] = stats["logical_active_experts"]
        stats["executed_expert_fraction"] = stats["logical_active_experts"] / float(self.num_experts)
        return result, stats

    def forward_controlled(self, x: torch.Tensor, sample_gate: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        return self._apply_routing(x, sample_gate)

    def forward_native(self, x: torch.Tensor, task_ids: torch.Tensor, top_k: int) -> tuple[torch.Tensor, dict[str, float]]:
        task_embeds = self.task_embedding(task_ids)
        logits = self.gate(task_embeds)
        routing = _masked_softmax_from_logits(logits, top_k)
        return self._apply_routing(x, routing)


def _run_timed_benchmark(
    *,
    model: nn.Module,
    step_fn,
    pass_mode: str,
    steps: int,
    warmup_steps: int,
    device: torch.device,
) -> tuple[float, int | None, dict[str, float], bool]:
    model.train()
    last_stats: dict[str, float] = {}
    finite_pass = True
    total_steps = steps + warmup_steps
    step_times: list[float] = []

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for step_idx in range(total_steps):
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = None

        _sync(device)
        start = time.perf_counter()
        output, last_stats = step_fn()
        if not torch.isfinite(output).all():
            finite_pass = False
        if pass_mode == "forward_backward":
            loss = output.float().square().mean()
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

    mean_step_seconds = float(sum(step_times) / len(step_times))
    peak_memory = _safe_peak_memory(device)
    return mean_step_seconds, peak_memory, last_stats, finite_pass


def _result_row(
    *,
    method: str,
    routing_mode: str,
    pass_mode: str,
    hidden_size: int,
    rank: int,
    num_experts: int,
    batch_size: int,
    seq_len: int,
    dtype_name: str,
    device: torch.device,
    trainable_params: int,
    mean_step_seconds: float,
    peak_memory_bytes: int | None,
    stats: dict[str, float],
    finite_pass: bool,
    controlled_active_experts: int | None = None,
    hmora_top_k: int | None = None,
    moelora_top_k: int | None = None,
    adamole_max_threshold: float | None = None,
    task_num: int | None = None,
    use_hydra_lora: bool | None = None,
) -> dict[str, Any]:
    tokens_per_step = batch_size * seq_len
    row = {
        "method": method,
        "routing_mode": routing_mode,
        "pass_mode": pass_mode,
        "device": str(device),
        "dtype": dtype_name,
        "hidden_size": hidden_size,
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
        "finite_pass": bool(finite_pass),
        "controlled_active_experts": controlled_active_experts,
        "hmora_top_k": hmora_top_k,
        "moelora_top_k": moelora_top_k,
        "adamole_max_threshold": adamole_max_threshold,
        "task_num": task_num,
        "use_hydra_lora": use_hydra_lora,
    }
    return row


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="lora,hmora,adamole,moelora")
    parser.add_argument("--routing-modes", default="controlled_routing,native_routing")
    parser.add_argument("--pass-mode", choices=["forward", "forward_backward", "both"], default="forward_backward")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--hidden-sizes", default="4096")
    parser.add_argument("--ranks", default="64")
    parser.add_argument("--num-experts-values", default="8,16,32")
    parser.add_argument("--batch-size-values", default="1,8")
    parser.add_argument("--seq-len-values", default="256,1024")
    parser.add_argument("--controlled-active-experts-values", default="1,2,4")
    parser.add_argument("--hmora-top-k-values", default="1,2,4")
    parser.add_argument("--moelora-top-k-values", default="1,2,4")
    parser.add_argument("--adamole-max-threshold-values", default="")
    parser.add_argument("--task-num", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--use-hydra-lora", type=_str2bool, default=True)
    parser.add_argument("--output-dir", default="outputs/moe_lora_runtime_study")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    methods = _parse_str_list(args.methods)
    routing_modes = _parse_str_list(args.routing_modes)
    hidden_sizes = _parse_int_list(args.hidden_sizes)
    ranks = _parse_int_list(args.ranks)
    num_experts_values = _parse_int_list(args.num_experts_values)
    batch_sizes = _parse_int_list(args.batch_size_values)
    seq_lens = _parse_int_list(args.seq_len_values)
    controlled_active_values = _parse_int_list(args.controlled_active_experts_values)
    hmora_top_ks = _parse_int_list(args.hmora_top_k_values)
    moelora_top_ks = _parse_int_list(args.moelora_top_k_values)
    adamole_threshold_values = _parse_float_list(args.adamole_max_threshold_values) if args.adamole_max_threshold_values else []
    pass_modes = ["forward", "forward_backward"] if args.pass_mode == "both" else [args.pass_mode]
    device = _device(args.device)
    dtype = _torch_dtype(args.dtype)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    rows: list[dict[str, Any]] = []
    combo_seed = args.seed

    for method in methods:
        if method not in {"lora", "hmora", "adamole", "moelora"}:
            raise SystemExit(f"Unsupported method: {method!r}")

        for hidden_size in hidden_sizes:
            for rank in ranks:
                alpha = float(rank * 2)
                for batch_size in batch_sizes:
                    for seq_len in seq_lens:
                        if method == "lora":
                            model = LoRAAdapter(
                                hidden_size=hidden_size,
                                rank=rank,
                                alpha=alpha,
                                device=device,
                                dtype=dtype,
                            )
                            input_gen = _seeded_generator(combo_seed)
                            x = _randn(batch_size, seq_len, hidden_size, generator=input_gen, device=device, dtype=dtype)
                            for routing_mode in routing_modes:
                                for pass_mode in pass_modes:
                                    mean_step_seconds, peak_memory, stats, finite_pass = _run_timed_benchmark(
                                        model=model,
                                        step_fn=lambda m=model, x=x: m(x),
                                        pass_mode=pass_mode,
                                        steps=args.steps,
                                        warmup_steps=args.warmup_steps,
                                        device=device,
                                    )
                                    rows.append(
                                        _result_row(
                                            method=method,
                                            routing_mode=routing_mode,
                                            pass_mode=pass_mode,
                                            hidden_size=hidden_size,
                                            rank=rank,
                                            num_experts=1,
                                            batch_size=batch_size,
                                            seq_len=seq_len,
                                            dtype_name=args.dtype,
                                            device=device,
                                            trainable_params=_count_trainable_params(model),
                                            mean_step_seconds=mean_step_seconds,
                                            peak_memory_bytes=peak_memory,
                                            stats=stats,
                                            finite_pass=finite_pass,
                                        )
                                    )
                            combo_seed += 1
                            continue

                        for num_experts in num_experts_values:
                            if method == "moelora" and rank % num_experts != 0:
                                continue

                            input_gen = _seeded_generator(combo_seed)
                            x = _randn(batch_size, seq_len, hidden_size, generator=input_gen, device=device, dtype=dtype)
                            sample_gate = _controlled_sample_gate(
                                batch_size=batch_size,
                                num_experts=num_experts,
                                active_experts=min(controlled_active_values[0], num_experts),
                                device=device,
                                dtype=dtype,
                                seed=combo_seed,
                            )
                            task_ids = _randint(0, args.task_num, (batch_size,), generator=input_gen, device=device)

                            if method == "hmora":
                                model = HMoRAAdapter(
                                    hidden_size=hidden_size,
                                    rank=rank,
                                    alpha=alpha,
                                    num_experts=num_experts,
                                    use_hydra_lora=args.use_hydra_lora,
                                    device=device,
                                    dtype=dtype,
                                )
                            elif method == "adamole":
                                model = AdaMoLEAdapter(
                                    hidden_size=hidden_size,
                                    rank=rank,
                                    alpha=alpha,
                                    num_experts=num_experts,
                                    device=device,
                                    dtype=dtype,
                                )
                            else:
                                model = MoELoRAAdapter(
                                    hidden_size=hidden_size,
                                    rank=rank,
                                    alpha=alpha,
                                    num_experts=num_experts,
                                    task_num=args.task_num,
                                    device=device,
                                    dtype=dtype,
                                )

                            trainable_params = _count_trainable_params(model)

                            for routing_mode in routing_modes:
                                if routing_mode == "controlled_routing":
                                    for active_experts in controlled_active_values:
                                        if active_experts > num_experts:
                                            continue
                                        sample_gate = _controlled_sample_gate(
                                            batch_size=batch_size,
                                            num_experts=num_experts,
                                            active_experts=active_experts,
                                            device=device,
                                            dtype=dtype,
                                            seed=combo_seed + active_experts,
                                        )
                                        token_gate = _expand_sample_gate(sample_gate, seq_len)
                                        if method == "hmora":
                                            step_fn = lambda m=model, x=x, gate=token_gate: m.forward_controlled(x, gate)
                                        elif method == "adamole":
                                            step_fn = lambda m=model, x=x, gate=token_gate: m.forward_controlled(x, gate)
                                        else:
                                            step_fn = lambda m=model, x=x, gate=sample_gate: m.forward_controlled(x, gate)

                                        for pass_mode in pass_modes:
                                            mean_step_seconds, peak_memory, stats, finite_pass = _run_timed_benchmark(
                                                model=model,
                                                step_fn=step_fn,
                                                pass_mode=pass_mode,
                                                steps=args.steps,
                                                warmup_steps=args.warmup_steps,
                                                device=device,
                                            )
                                            rows.append(
                                                _result_row(
                                                    method=method,
                                                    routing_mode=routing_mode,
                                                    pass_mode=pass_mode,
                                                    hidden_size=hidden_size,
                                                    rank=rank,
                                                    num_experts=num_experts,
                                                    batch_size=batch_size,
                                                    seq_len=seq_len,
                                                    dtype_name=args.dtype,
                                                    device=device,
                                                    trainable_params=trainable_params,
                                                    mean_step_seconds=mean_step_seconds,
                                                    peak_memory_bytes=peak_memory,
                                                    stats=stats,
                                                    finite_pass=finite_pass,
                                                    controlled_active_experts=active_experts,
                                                    task_num=args.task_num if method == "moelora" else None,
                                                    use_hydra_lora=args.use_hydra_lora if method == "hmora" else None,
                                                )
                                            )
                                elif routing_mode == "native_routing":
                                    if method == "hmora":
                                        for top_k in hmora_top_ks:
                                            if top_k > num_experts:
                                                continue
                                            step_fn = lambda m=model, x=x, top_k=top_k: m.forward_native(x, top_k)
                                            for pass_mode in pass_modes:
                                                mean_step_seconds, peak_memory, stats, finite_pass = _run_timed_benchmark(
                                                    model=model,
                                                    step_fn=step_fn,
                                                    pass_mode=pass_mode,
                                                    steps=args.steps,
                                                    warmup_steps=args.warmup_steps,
                                                    device=device,
                                                )
                                                rows.append(
                                                    _result_row(
                                                        method=method,
                                                        routing_mode=routing_mode,
                                                        pass_mode=pass_mode,
                                                        hidden_size=hidden_size,
                                                        rank=rank,
                                                        num_experts=num_experts,
                                                        batch_size=batch_size,
                                                        seq_len=seq_len,
                                                        dtype_name=args.dtype,
                                                        device=device,
                                                        trainable_params=trainable_params,
                                                        mean_step_seconds=mean_step_seconds,
                                                        peak_memory_bytes=peak_memory,
                                                        stats=stats,
                                                        finite_pass=finite_pass,
                                                        hmora_top_k=top_k,
                                                        use_hydra_lora=args.use_hydra_lora,
                                                    )
                                                )
                                    elif method == "adamole":
                                        thresholds = adamole_threshold_values or _default_adamole_thresholds(num_experts)
                                        for max_threshold in thresholds:
                                            step_fn = lambda m=model, x=x, threshold=max_threshold: m.forward_native(x, threshold)
                                            for pass_mode in pass_modes:
                                                mean_step_seconds, peak_memory, stats, finite_pass = _run_timed_benchmark(
                                                    model=model,
                                                    step_fn=step_fn,
                                                    pass_mode=pass_mode,
                                                    steps=args.steps,
                                                    warmup_steps=args.warmup_steps,
                                                    device=device,
                                                )
                                                rows.append(
                                                    _result_row(
                                                        method=method,
                                                        routing_mode=routing_mode,
                                                        pass_mode=pass_mode,
                                                        hidden_size=hidden_size,
                                                        rank=rank,
                                                        num_experts=num_experts,
                                                        batch_size=batch_size,
                                                        seq_len=seq_len,
                                                        dtype_name=args.dtype,
                                                        device=device,
                                                        trainable_params=trainable_params,
                                                        mean_step_seconds=mean_step_seconds,
                                                        peak_memory_bytes=peak_memory,
                                                        stats=stats,
                                                        finite_pass=finite_pass,
                                                        adamole_max_threshold=float(max_threshold),
                                                    )
                                                )
                                    elif method == "moelora":
                                        for top_k in moelora_top_ks:
                                            if top_k > num_experts:
                                                continue
                                            step_fn = lambda m=model, x=x, task_ids=task_ids, top_k=top_k: m.forward_native(x, task_ids, top_k)
                                            for pass_mode in pass_modes:
                                                mean_step_seconds, peak_memory, stats, finite_pass = _run_timed_benchmark(
                                                    model=model,
                                                    step_fn=step_fn,
                                                    pass_mode=pass_mode,
                                                    steps=args.steps,
                                                    warmup_steps=args.warmup_steps,
                                                    device=device,
                                                )
                                                rows.append(
                                                    _result_row(
                                                        method=method,
                                                        routing_mode=routing_mode,
                                                        pass_mode=pass_mode,
                                                        hidden_size=hidden_size,
                                                        rank=rank,
                                                        num_experts=num_experts,
                                                        batch_size=batch_size,
                                                        seq_len=seq_len,
                                                        dtype_name=args.dtype,
                                                        device=device,
                                                        trainable_params=trainable_params,
                                                        mean_step_seconds=mean_step_seconds,
                                                        peak_memory_bytes=peak_memory,
                                                        stats=stats,
                                                        finite_pass=finite_pass,
                                                        moelora_top_k=top_k,
                                                        task_num=args.task_num,
                                                    )
                                                )
                                    else:
                                        # LoRA already handled above.
                                        pass
                                else:
                                    raise SystemExit(f"Unsupported routing mode: {routing_mode!r}")

                            combo_seed += 1

    if not rows:
        raise SystemExit("No benchmark rows were produced. Check ranks / expert counts / method filters.")

    output_dir = Path(args.output_dir)
    _write_results(rows, output_dir)
    print(f"Wrote {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
