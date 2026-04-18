#!/usr/bin/env python3
"""Benchmark dense-packed vs true-sparse HMoRA execution.

This study is intentionally independent from the full training stack so it can
run even when local `transformers` / PEFT versions do not match the repo's
training environment.

Artifacts:
- JSONL: one row per benchmark result
- CSV: flat table mirror of the JSONL rows

The benchmark supports two tiers:
- `adapter`: one adapted linear layer with residual base projection
- `block`: a small transformer-like block that applies multiple HMoRA adapters

The two variants use the same weights and gates:
- `dense_packed`: current repo behavior, packed expert dimensions and dense matmuls
- `true_sparse`: execute only selected experts via explicit token/expert dispatch
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


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _safe_peak_memory(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _apply_topk_mask(weights: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return weights
    top_k = min(top_k, weights.shape[-1])
    top_k_values, top_k_indices = torch.topk(weights, top_k, dim=-1)
    sparse = torch.full_like(weights, torch.finfo(weights.dtype).min)
    sparse.scatter_(-1, top_k_indices, top_k_values)
    return torch.softmax(sparse, dim=-1)


def _generate_gate(
    *,
    batch_size: int,
    seq_len: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    dtype: torch.dtype,
    gate_source: str,
    seed: int,
    gate_path: str,
) -> torch.Tensor:
    if gate_source == "file":
        if not gate_path:
            raise SystemExit("--gate-path is required when --gate-source=file")
        loaded = torch.load(gate_path, map_location="cpu")
        if not isinstance(loaded, torch.Tensor):
            raise SystemExit(f"Expected a tensor in gate file: {gate_path}")
        gate = loaded.to(device=device, dtype=dtype)
        expected = (batch_size, seq_len, num_experts)
        if tuple(gate.shape) != expected:
            raise SystemExit(f"Gate tensor shape mismatch. Expected {expected}, got {tuple(gate.shape)}")
        return gate

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if gate_source == "uniform_topk":
        logits = torch.randn(batch_size, seq_len, num_experts, generator=generator, dtype=torch.float32)
        top_k = max(1, min(top_k, num_experts))
        _, top_k_indices = torch.topk(logits, top_k, dim=-1)
        gate = torch.zeros(batch_size, seq_len, num_experts, dtype=torch.float32)
        gate.scatter_(-1, top_k_indices, 1.0 / float(top_k))
        return gate.to(device=device, dtype=dtype)

    if gate_source != "random_logits":
        raise SystemExit(f"Unsupported gate source: {gate_source!r}")

    logits = torch.randn(batch_size, seq_len, num_experts, generator=generator, dtype=torch.float32)
    weights = torch.softmax(logits, dim=-1)
    gate = _apply_topk_mask(weights, top_k)
    return gate.to(device=device, dtype=dtype)


def _gate_stats(gate: torch.Tensor) -> dict[str, float]:
    flat_gate = gate.reshape(-1, gate.shape[-1])
    active = flat_gate > 0
    experts_per_token = active.sum(dim=-1).float()
    expert_usage = active.float().sum(dim=0)
    active_experts = (expert_usage > 0).float().mean()
    mean_load = expert_usage.mean()
    if float(mean_load) > 0:
        load_cv = float((expert_usage.std(unbiased=False) / (mean_load + 1e-6)).item())
    else:
        load_cv = 0.0
    entropy = -(flat_gate * torch.log(flat_gate + 1e-9)).sum(dim=-1).mean()
    return {
        "experts_per_token": float(experts_per_token.mean().item()),
        "topk_density": float((experts_per_token.mean() / gate.shape[-1]).item()),
        "active_expert_frac": float(active_experts.item()),
        "expert_load_cv": load_cv,
        "router_entropy": float(entropy.item()),
    }


class DensePackedHmoraAdapter(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        rank: int,
        num_experts: int,
        lora_alpha: float,
        use_hydra_lora: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        a_rows = rank if use_hydra_lora else rank * num_experts
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype), requires_grad=False)
        self.base_bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype), requires_grad=False)
        self.mora_a = nn.Parameter(torch.empty(a_rows, in_features, device=device, dtype=dtype))
        self.mora_b = nn.Parameter(torch.empty(out_features, rank * num_experts, device=device, dtype=dtype))
        self.rank = rank
        self.num_experts = num_experts
        self.use_hydra_lora = use_hydra_lora
        self.scaling = float(lora_alpha) / math.sqrt(rank)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.base_weight, std=0.02)
        nn.init.kaiming_uniform_(self.mora_a, a=math.sqrt(5))
        nn.init.zeros_(self.mora_b)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.base_weight, self.base_bias)
        hidden_states = F.linear(x, self.mora_a)
        target_shape = hidden_states.shape[:-1] + (self.num_experts, self.rank)
        if self.use_hydra_lora:
            hidden_states = hidden_states.unsqueeze(-2).expand(target_shape)
        else:
            hidden_states = hidden_states.view(target_shape)
        hidden_states = (hidden_states * gate.unsqueeze(-1)).reshape(hidden_states.shape[:-2] + (-1,))
        delta = F.linear(hidden_states, self.mora_b) * self.scaling
        return result + delta.to(result.dtype)


class TrueSparseHmoraAdapter(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        rank: int,
        num_experts: int,
        lora_alpha: float,
        use_hydra_lora: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        a_rows = rank if use_hydra_lora else rank * num_experts
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype), requires_grad=False)
        self.base_bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype), requires_grad=False)
        self.mora_a = nn.Parameter(torch.empty(a_rows, in_features, device=device, dtype=dtype))
        self.mora_b = nn.Parameter(torch.empty(out_features, rank * num_experts, device=device, dtype=dtype))
        self.rank = rank
        self.num_experts = num_experts
        self.use_hydra_lora = use_hydra_lora
        self.scaling = float(lora_alpha) / math.sqrt(rank)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.base_weight, std=0.02)
        nn.init.kaiming_uniform_(self.mora_a, a=math.sqrt(5))
        nn.init.zeros_(self.mora_b)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        output_shape = x.shape[:-1] + (self.base_weight.shape[0],)
        flat_x = x.reshape(-1, x.shape[-1])
        flat_gate = gate.reshape(-1, gate.shape[-1])
        flat_result = F.linear(flat_x, self.base_weight, self.base_bias)
        flat_delta = torch.zeros_like(flat_result)

        if self.use_hydra_lora:
            shared_hidden = F.linear(flat_x, self.mora_a)
            expert_a = None
        else:
            shared_hidden = None
            expert_a = self.mora_a.view(self.num_experts, self.rank, self.mora_a.shape[-1])
        expert_b = self.mora_b.view(self.mora_b.shape[0], self.num_experts, self.rank)

        for expert_idx in range(self.num_experts):
            weights = flat_gate[:, expert_idx]
            active = weights > 0
            if not bool(active.any()):
                continue
            if shared_hidden is not None:
                hidden = shared_hidden[active]
            else:
                hidden = F.linear(flat_x[active], expert_a[expert_idx])
            delta = F.linear(hidden, expert_b[:, expert_idx, :]) * self.scaling
            flat_delta[active] += delta * weights[active].unsqueeze(-1)

        return (flat_result + flat_delta).view(output_shape)


class ToyHmoraBlock(nn.Module):
    def __init__(
        self,
        *,
        adapter_cls: type[nn.Module],
        hidden_size: int,
        intermediate_size: int,
        rank: int,
        num_experts: int,
        lora_alpha: float,
        use_hydra_lora: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        adapter_kwargs = {
            "rank": rank,
            "num_experts": num_experts,
            "lora_alpha": lora_alpha,
            "use_hydra_lora": use_hydra_lora,
            "device": device,
            "dtype": dtype,
        }
        self.q_proj = adapter_cls(in_features=hidden_size, out_features=hidden_size, **adapter_kwargs)
        self.k_proj = adapter_cls(in_features=hidden_size, out_features=hidden_size, **adapter_kwargs)
        self.v_proj = adapter_cls(in_features=hidden_size, out_features=hidden_size, **adapter_kwargs)
        self.o_proj = adapter_cls(in_features=hidden_size, out_features=hidden_size, **adapter_kwargs)
        self.gate_proj = adapter_cls(in_features=hidden_size, out_features=intermediate_size, **adapter_kwargs)
        self.up_proj = adapter_cls(in_features=hidden_size, out_features=intermediate_size, **adapter_kwargs)
        self.down_proj = adapter_cls(in_features=intermediate_size, out_features=hidden_size, **adapter_kwargs)

    def forward(self, x: torch.Tensor, gates: dict[str, torch.Tensor]) -> torch.Tensor:
        q = self.q_proj(x, gates["q_proj"])
        k = self.k_proj(x, gates["k_proj"])
        v = self.v_proj(x, gates["v_proj"])
        attention_mix = torch.tanh((q + k + v) / 3.0)
        h = x + self.o_proj(attention_mix, gates["o_proj"])
        gate_hidden = self.gate_proj(h, gates["gate_proj"])
        up_hidden = self.up_proj(h, gates["up_proj"])
        mlp = F.silu(gate_hidden) * up_hidden
        return h + self.down_proj(mlp, gates["down_proj"])


def _build_block_gates(
    *,
    batch_size: int,
    seq_len: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    dtype: torch.dtype,
    gate_source: str,
    seed: int,
    gate_path: str,
    share_router_qkv: bool,
    share_router_wi: bool,
) -> dict[str, torch.Tensor]:
    def one_gate(offset: int) -> torch.Tensor:
        return _generate_gate(
            batch_size=batch_size,
            seq_len=seq_len,
            num_experts=num_experts,
            top_k=top_k,
            device=device,
            dtype=dtype,
            gate_source=gate_source,
            seed=seed + offset,
            gate_path=gate_path,
        )

    gates: dict[str, torch.Tensor] = {}
    if share_router_qkv:
        shared = one_gate(11)
        gates["q_proj"] = shared
        gates["k_proj"] = shared
        gates["v_proj"] = shared
    else:
        gates["q_proj"] = one_gate(11)
        gates["k_proj"] = one_gate(12)
        gates["v_proj"] = one_gate(13)

    gates["o_proj"] = one_gate(14)

    if share_router_wi:
        shared = one_gate(21)
        gates["gate_proj"] = shared
        gates["up_proj"] = shared
    else:
        gates["gate_proj"] = one_gate(21)
        gates["up_proj"] = one_gate(22)

    gates["down_proj"] = one_gate(23)
    return gates


def _clone_for_compare(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


def _compare_modules(
    *,
    dense: nn.Module,
    sparse: nn.Module,
    dense_inputs: tuple[Any, ...],
    sparse_inputs: tuple[Any, ...],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    dense.zero_grad(set_to_none=True)
    sparse.zero_grad(set_to_none=True)

    dense_out = dense(*dense_inputs)
    sparse_out = sparse(*sparse_inputs)
    output_max_abs_diff = float((dense_out - sparse_out).abs().max().item())

    dense_loss = dense_out.float().square().mean()
    sparse_loss = sparse_out.float().square().mean()
    dense_loss.backward()
    sparse_loss.backward()

    dense_x = dense_inputs[0]
    sparse_x = sparse_inputs[0]
    input_grad_max_abs_diff = float((dense_x.grad - sparse_x.grad).abs().max().item())

    grad_diffs: list[float] = []
    dense_named = dict(dense.named_parameters())
    sparse_named = dict(sparse.named_parameters())
    for name, dense_param in dense_named.items():
        if not dense_param.requires_grad:
            continue
        sparse_param = sparse_named[name]
        grad_diffs.append(float((dense_param.grad - sparse_param.grad).abs().max().item()))

    grad_max_abs_diff = max(grad_diffs) if grad_diffs else 0.0
    passed = torch.allclose(dense_out, sparse_out, atol=atol, rtol=rtol) and grad_max_abs_diff <= max(atol, rtol)
    return {
        "equiv_pass": bool(passed),
        "equiv_output_max_abs_diff": output_max_abs_diff,
        "equiv_input_grad_max_abs_diff": input_grad_max_abs_diff,
        "equiv_param_grad_max_abs_diff": grad_max_abs_diff,
    }


def _benchmark_single(
    *,
    module: nn.Module,
    inputs_factory,
    steps: int,
    warmup_steps: int,
    measure: str,
    device: torch.device,
) -> dict[str, Any]:
    if steps <= 0:
        raise SystemExit("--steps must be positive")

    peak_memory = 0

    def one_step() -> None:
        module.zero_grad(set_to_none=True)
        prepared = inputs_factory()
        out = module(*prepared)
        if measure == "forward_backward":
            loss = out.float().square().mean()
            loss.backward()

    module.train(measure == "forward_backward")

    for _ in range(warmup_steps):
        one_step()
    _synchronize(device)

    total = 0.0
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        one_step()
        _synchronize(device)
        total += time.perf_counter() - start
        current_peak = _safe_peak_memory(device)
        if current_peak is not None:
            peak_memory = max(peak_memory, current_peak)

    return {
        "mean_step_seconds": total / float(steps),
        "peak_memory_bytes": peak_memory if device.type == "cuda" else None,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _benchmark_adapter_config(
    *,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    hidden_size: int,
    rank: int,
    num_experts: int,
    top_k: int,
    batch_size: int,
    seq_len: int,
    pass_mode: str,
    seed: int,
) -> list[dict[str, Any]]:
    dense = DensePackedHmoraAdapter(
        in_features=hidden_size,
        out_features=hidden_size,
        rank=rank,
        num_experts=num_experts,
        lora_alpha=args.lora_alpha,
        use_hydra_lora=args.use_hydra_lora,
        device=device,
        dtype=dtype,
    )
    sparse = TrueSparseHmoraAdapter(
        in_features=hidden_size,
        out_features=hidden_size,
        rank=rank,
        num_experts=num_experts,
        lora_alpha=args.lora_alpha,
        use_hydra_lora=args.use_hydra_lora,
        device=device,
        dtype=dtype,
    )
    sparse.load_state_dict(dense.state_dict())
    gate = _generate_gate(
        batch_size=batch_size,
        seq_len=seq_len,
        num_experts=num_experts,
        top_k=top_k,
        device=device,
        dtype=dtype,
        gate_source=args.gate_source,
        seed=seed,
        gate_path=args.gate_path,
    )
    stats = _gate_stats(gate)
    compare_x_dense = _clone_for_compare(torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype))
    compare_x_sparse = compare_x_dense.detach().clone().requires_grad_(True)
    equiv = _compare_modules(
        dense=dense,
        sparse=sparse,
        dense_inputs=(compare_x_dense, gate),
        sparse_inputs=(compare_x_sparse, gate),
        atol=args.atol,
        rtol=args.rtol,
    )

    rows: list[dict[str, Any]] = []
    variants = ("dense_packed", "true_sparse") if args.variant == "both" else (args.variant,)
    for variant in variants:
        module = dense if variant == "dense_packed" else sparse

        def factory():
            x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=(pass_mode == "forward_backward"))
            return (x, gate)

        metrics = _benchmark_single(
            module=module,
            inputs_factory=factory,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            measure=pass_mode,
            device=device,
        )
        tokens_per_step = batch_size * seq_len
        rows.append(
            {
                "tier": "adapter",
                "variant": variant,
                "pass_mode": pass_mode,
                "hidden_size": hidden_size,
                "intermediate_size": None,
                "rank": rank,
                "num_experts": num_experts,
                "top_k": top_k,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "tokens_per_step": tokens_per_step,
                "tokens_per_second": tokens_per_step / metrics["mean_step_seconds"],
                "use_hydra_lora": args.use_hydra_lora,
                "share_router_qkv": None,
                "share_router_wi": None,
                "gate_source": args.gate_source,
                "device": str(device),
                "dtype": args.dtype,
                **stats,
                **equiv,
                **metrics,
            }
        )
    return rows


def _benchmark_block_config(
    *,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    hidden_size: int,
    intermediate_size: int,
    rank: int,
    num_experts: int,
    top_k: int,
    batch_size: int,
    seq_len: int,
    pass_mode: str,
    seed: int,
) -> list[dict[str, Any]]:
    dense = ToyHmoraBlock(
        adapter_cls=DensePackedHmoraAdapter,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        rank=rank,
        num_experts=num_experts,
        lora_alpha=args.lora_alpha,
        use_hydra_lora=args.use_hydra_lora,
        device=device,
        dtype=dtype,
    )
    sparse = ToyHmoraBlock(
        adapter_cls=TrueSparseHmoraAdapter,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        rank=rank,
        num_experts=num_experts,
        lora_alpha=args.lora_alpha,
        use_hydra_lora=args.use_hydra_lora,
        device=device,
        dtype=dtype,
    )
    sparse.load_state_dict(dense.state_dict())
    gates = _build_block_gates(
        batch_size=batch_size,
        seq_len=seq_len,
        num_experts=num_experts,
        top_k=top_k,
        device=device,
        dtype=dtype,
        gate_source=args.gate_source,
        seed=seed,
        gate_path=args.gate_path,
        share_router_qkv=args.share_router_qkv,
        share_router_wi=args.share_router_wi,
    )
    gate_stats = _gate_stats(torch.cat([gate.reshape(-1, gate.shape[-1]) for gate in gates.values()], dim=0))
    compare_x_dense = _clone_for_compare(torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype))
    compare_x_sparse = compare_x_dense.detach().clone().requires_grad_(True)
    equiv = _compare_modules(
        dense=dense,
        sparse=sparse,
        dense_inputs=(compare_x_dense, gates),
        sparse_inputs=(compare_x_sparse, gates),
        atol=args.atol,
        rtol=args.rtol,
    )

    rows: list[dict[str, Any]] = []
    variants = ("dense_packed", "true_sparse") if args.variant == "both" else (args.variant,)
    for variant in variants:
        module = dense if variant == "dense_packed" else sparse

        def factory():
            x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=(pass_mode == "forward_backward"))
            return (x, gates)

        metrics = _benchmark_single(
            module=module,
            inputs_factory=factory,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            measure=pass_mode,
            device=device,
        )
        tokens_per_step = batch_size * seq_len
        rows.append(
            {
                "tier": "block",
                "variant": variant,
                "pass_mode": pass_mode,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "rank": rank,
                "num_experts": num_experts,
                "top_k": top_k,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "tokens_per_step": tokens_per_step,
                "tokens_per_second": tokens_per_step / metrics["mean_step_seconds"],
                "use_hydra_lora": args.use_hydra_lora,
                "share_router_qkv": args.share_router_qkv,
                "share_router_wi": args.share_router_wi,
                "gate_source": args.gate_source,
                "device": str(device),
                "dtype": args.dtype,
                **gate_stats,
                **equiv,
                **metrics,
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense-packed vs true-sparse HMoRA study.")
    parser.add_argument("--variant", choices=["dense_packed", "true_sparse", "both"], default="both")
    parser.add_argument("--tier", choices=["adapter", "block", "both"], default="both")
    parser.add_argument("--pass-mode", choices=["forward", "forward_backward", "both"], default="both")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--hidden-sizes", default="512,1024")
    parser.add_argument("--ranks", default="4,8")
    parser.add_argument("--num-experts-values", default="2,4,8,16")
    parser.add_argument("--top-k-values", default="1,2,4,8")
    parser.add_argument("--batch-size-values", default="2,4")
    parser.add_argument("--seq-len-values", default="128,512")
    parser.add_argument("--ffn-multipliers", default="4")
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--use-hydra-lora", type=_str2bool, default=True)
    parser.add_argument("--share-router-qkv", type=_str2bool, default=True)
    parser.add_argument("--share-router-wi", type=_str2bool, default=True)
    parser.add_argument("--gate-source", choices=["random_logits", "uniform_topk", "file"], default="random_logits")
    parser.add_argument("--gate-path", default="")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="outputs/hmora_sparse_study")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = _device(args.device)
    dtype = _torch_dtype(args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        raise SystemExit("CPU benchmarks currently require --dtype=float32")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    hidden_sizes = _parse_int_list(args.hidden_sizes)
    ranks = _parse_int_list(args.ranks)
    num_experts_values = _parse_int_list(args.num_experts_values)
    top_k_values = _parse_int_list(args.top_k_values)
    batch_size_values = _parse_int_list(args.batch_size_values)
    seq_len_values = _parse_int_list(args.seq_len_values)
    ffn_multipliers = _parse_int_list(args.ffn_multipliers)
    if not hidden_sizes or not ranks or not num_experts_values or not top_k_values or not batch_size_values or not seq_len_values:
        raise SystemExit("All sweep lists must be non-empty")

    pass_modes = ("forward", "forward_backward") if args.pass_mode == "both" else (args.pass_mode,)
    tiers = ("adapter", "block") if args.tier == "both" else (args.tier,)

    rows: list[dict[str, Any]] = []
    config_idx = 0

    for hidden_size in hidden_sizes:
        for rank in ranks:
            for num_experts in num_experts_values:
                valid_top_ks = [top_k for top_k in top_k_values if 0 < top_k <= num_experts]
                if not valid_top_ks:
                    continue
                for top_k in valid_top_ks:
                    for batch_size in batch_size_values:
                        for seq_len in seq_len_values:
                            for pass_mode in pass_modes:
                                seed = args.seed + config_idx * 97
                                config_idx += 1
                                if "adapter" in tiers:
                                    rows.extend(
                                        _benchmark_adapter_config(
                                            args=args,
                                            device=device,
                                            dtype=dtype,
                                            hidden_size=hidden_size,
                                            rank=rank,
                                            num_experts=num_experts,
                                            top_k=top_k,
                                            batch_size=batch_size,
                                            seq_len=seq_len,
                                            pass_mode=pass_mode,
                                            seed=seed,
                                        )
                                    )
                                if "block" in tiers:
                                    for ffn_multiplier in ffn_multipliers:
                                        rows.extend(
                                            _benchmark_block_config(
                                                args=args,
                                                device=device,
                                                dtype=dtype,
                                                hidden_size=hidden_size,
                                                intermediate_size=hidden_size * ffn_multiplier,
                                                rank=rank,
                                                num_experts=num_experts,
                                                top_k=top_k,
                                                batch_size=batch_size,
                                                seq_len=seq_len,
                                                pass_mode=pass_mode,
                                                seed=seed + ffn_multiplier,
                                            )
                                        )

    output_dir = Path(args.output_dir)
    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"
    _write_jsonl(jsonl_path, rows)
    _write_csv(csv_path, rows)

    summary = {
        "rows": len(rows),
        "output_jsonl": str(jsonl_path),
        "output_csv": str(csv_path),
        "device": str(device),
        "dtype": args.dtype,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
