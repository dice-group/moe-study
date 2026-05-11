import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader, load

REPO_ROOT = Path(__file__).resolve().parents[1]
LLAMAFACTORY_SRC = REPO_ROOT / "LLaMA-Factory" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LLAMAFACTORY_SRC) not in sys.path:
    sys.path.insert(0, str(LLAMAFACTORY_SRC))

from peft import TaskType
from peft.tuners.hmora import HMoRaConfig
from peft.tuners.moelora import MoELoraConfig

from pipeline.config_loader import load_pipeline_config
from pipeline.runtime import _deep_merge, _resolve_profile


RECOVERABLE_KINDS = {"hmora", "moelora"}
ADAPTER_MARKERS = (
    ".lora_",
    ".lora_task_embedding.",
    ".lora_gate.",
    ".lora_router_pool.",
    ".lora_task_encoder.",
)


def _split_targets(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]
    return items or None


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except Exception:
        return -1


def _latest_plain_checkpoint(run_dir: Path) -> Path | None:
    candidates = [
        child
        for child in run_dir.glob("checkpoint-*")
        if child.is_dir() and child.name.split("-", 1)[-1].isdigit()
    ]
    return max(candidates, key=_checkpoint_step, default=None)


def _strip_saved_key(key: str) -> str:
    if key.startswith("model."):
        key = key[len("model.") :]
    return key.replace(".default", "")


def _adapter_keys(keys: list[str]) -> list[str]:
    return [key for key in keys if any(marker in key for marker in ADAPTER_MARKERS)]


def _load_adapter_state(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    fsdp_dir = checkpoint_dir / "pytorch_model_fsdp_0"
    if not (fsdp_dir / ".metadata").exists():
        raise RuntimeError(f"missing FSDP metadata: {fsdp_dir / '.metadata'}")

    reader = FileSystemReader(str(fsdp_dir))
    metadata = reader.read_metadata()
    keys = _adapter_keys(list(metadata.state_dict_metadata.keys()))
    if not keys:
        raise RuntimeError(f"no adapter-like keys found in {fsdp_dir}")

    state: dict[str, torch.Tensor] = {}
    for key in keys:
        tensor_meta = metadata.state_dict_metadata[key]
        state[key] = torch.empty(tuple(tensor_meta.size), dtype=tensor_meta.properties.dtype)

    load(state_dict=state, storage_reader=reader)
    return {_strip_saved_key(key): tensor.detach().cpu().contiguous() for key, tensor in state.items()}


def _build_config(
    *,
    run: dict[str, Any],
    train_cfg: dict[str, Any],
    base_model: str,
) -> Any:
    kind = str(run["kind"])
    common_lora = train_cfg.get("lora", {})
    lora = _deep_merge(common_lora, run.get("lora", {}))
    params = dict(run.get("params", {}))
    targets = _split_targets(lora.get("target"))
    rank = int(lora.get("rank", 8))
    alpha = int(lora.get("alpha", 16))
    dropout = float(lora.get("dropout", 0.05))

    if kind == "moelora":
        expert_num = int(params["moelora_num_experts"])
        return MoELoraConfig(
            task_type=TaskType.CAUSAL_LM,
            base_model_name_or_path=base_model,
            inference_mode=True,
            r=rank,
            target_modules=targets,
            lora_alpha=alpha,
            lora_dropout=dropout,
            expert_num=expert_num,
            task_num=int(params.get("moelora_task_num", 12)),
            task_embedding_dim=int(params.get("moelora_task_embedding_dim", 64)),
            gate_top_k=int(params.get("moelora_sparse_top_k", 0)),
            gate_temperature=float(params.get("moelora_gate_temperature", 1.0)),
        )

    if kind == "hmora":
        target_modules_lora = _split_targets(params.get("hmora_target_modules_lora"))
        return HMoRaConfig(
            task_type=TaskType.CAUSAL_LM,
            base_model_name_or_path=base_model,
            inference_mode=True,
            r=rank,
            target_modules=targets,
            lora_alpha=alpha,
            lora_dropout=dropout,
            hidden_size=int(params.get("hidden_size", 4096)),
            model_type=str(params.get("model_type", "llama")),
            torch_dtype=str(params.get("torch_dtype", "bfloat16")),
            dropout=float(params.get("hmora_dropout", 0.1)),
            num_experts=int(params["hmora_num_experts"]),
            use_hydra_lora=bool(params.get("hmora_use_hydra_lora", False)),
            top_k_routing_strategy=bool(params.get("hmora_top_k_routing_strategy", False)),
            top_k=int(params.get("hmora_top_k", 2)),
            use_task_router=bool(params.get("hmora_use_task_router", False)),
            task_router_only=bool(params.get("hmora_task_router_only", False)),
            share_router_for_qkv=bool(params.get("hmora_share_router_for_qkv", False)),
            share_router_for_w_i=bool(params.get("hmora_share_router_for_w_i", False)),
            num_router_mlp_layers=int(params.get("hmora_num_router_mlp_layers", 1)),
            router_hidden_dim=int(params.get("hmora_router_hidden_dim", 32)),
            epsilon_alpha=float(params.get("hmora_epsilon_alpha", 2.0)),
            alpha_shift=float(params.get("hmora_alpha_shift", 0.0)),
            alpha_low_bound=float(params.get("hmora_alpha_low_bound", 0.0)),
            alpha_up_bound=float(params.get("hmora_alpha_up_bound", 1.0)),
            use_load_balancing_loss=bool(params.get("hmora_use_load_balancing_loss", False)),
            use_div_loss=bool(params.get("hmora_use_div_loss", False)),
            gamma_div_certain_t=float(params.get("hmora_gamma_div_certain_t", 0.0)),
            gamma_div_balance_t=float(params.get("hmora_gamma_div_balance_t", 1.0)),
            gamma_div_certain_s=float(params.get("hmora_gamma_div_certain_s", 0.0)),
            gamma_div_balance_s=float(params.get("hmora_gamma_div_balance_s", 1.0)),
            lambda_auxiliary=float(params.get("hmora_lambda_auxiliary", 0.01)),
            lambda_lm=float(params.get("hmora_lambda_lm", 1.0)),
            eta_b=float(params.get("hmora_eta_b", 1.0)),
            target_modules_lora=target_modules_lora,
            use_language_ids_as_task_ids=bool(params.get("hmora_use_language_ids_as_task_ids", False)),
            num_task_embeddings=int(params.get("hmora_num_task_embeddings", 12)),
            task_token=str(params.get("hmora_task_token", "?")),
            task_token_id=params.get("hmora_task_token_id"),
            num_encoder_layer=int(params.get("hmora_num_encoder_layer", 1)),
        )

    raise RuntimeError(f"unsupported kind: {kind}")


def _recover_run(
    *,
    root: Path,
    run: dict[str, Any],
    train_cfg: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    run_id = str(run["id"])
    run_dir = root / "runs" / run_id
    checkpoint = _latest_plain_checkpoint(run_dir)
    if checkpoint is None:
        return {"run_id": run_id, "status": "missing_checkpoint"}

    adapter_dir = run_dir / f"{checkpoint.name}_adapter"
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_weights_path = adapter_dir / "adapter_model.safetensors"
    if adapter_config_path.exists() and adapter_weights_path.exists() and not force:
        return {"run_id": run_id, "status": "already_recovered", "adapter_dir": str(adapter_dir)}

    state = _load_adapter_state(checkpoint)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(adapter_weights_path), metadata={"format": "pt"})
    config = _build_config(run=run, train_cfg=train_cfg, base_model=str(train_cfg["model_name_or_path"]))
    config.save_pretrained(adapter_dir)
    readme = adapter_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "---",
                "library_name: peft",
                "---",
                "",
                f"Recovered from FSDP checkpoint `{checkpoint}`.",
                f"Recovered tensors: {len(state)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "status": "recovered",
        "checkpoint": str(checkpoint),
        "adapter_dir": str(adapter_dir),
        "tensors": len(state),
        "bytes": sum(t.numel() * t.element_size() for t in state.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover PEFT adapter artifacts from FSDP DCP checkpoints.")
    parser.add_argument("--config", default="configs/pipeline")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-json")
    args = parser.parse_args()

    profile = _resolve_profile(load_pipeline_config(Path(args.config)), args.profile)
    root = REPO_ROOT / str(profile["output_root"])
    train_cfg = profile["training"]
    selected = set(args.run_id or [])
    runs = [
        run
        for run in train_cfg.get("runs", [])
        if str(run.get("kind")) in RECOVERABLE_KINDS and (not selected or str(run["id"]) in selected)
    ]
    if selected:
        found = {str(run["id"]) for run in runs}
        missing = sorted(selected - found)
        if missing:
            raise SystemExit(f"unknown or unsupported run ids: {missing}")

    results = []
    for run in runs:
        try:
            result = _recover_run(root=root, run=run, train_cfg=train_cfg, force=bool(args.force))
        except Exception as exc:
            result = {"run_id": str(run.get("id", "")), "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, sort_keys=True))
        results.append(result)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    recovered = sum(1 for item in results if item.get("status") in {"recovered", "already_recovered"})
    failed = [item for item in results if item.get("status") not in {"recovered", "already_recovered"}]
    print(f"recovered_or_present={recovered} failed={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
