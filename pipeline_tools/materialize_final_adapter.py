import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
LLAMAFACTORY_SRC = REPO_ROOT / "LLaMA-Factory" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LLAMAFACTORY_SRC) not in sys.path:
    sys.path.insert(0, str(LLAMAFACTORY_SRC))

from peft import TaskType
from peft.tuners.cola import ColaConfig
from peft.tuners.movlora import MovLoraConfig
from peft.tuners.mtllora import MtlLoraConfig

from pipeline.config_loader import load_pipeline_config
from pipeline.runtime import _deep_merge, _resolve_profile


ADAPTER_KEY_MARKERS = (
    ".lora_",
    ".ia3_",
)


def _strip_adapter_name(key: str, adapter_name: str = "default") -> str:
    if key.startswith("_fsdp_wrapped_module."):
        key = key.removeprefix("_fsdp_wrapped_module.")
    if key.startswith("model."):
        key = key.removeprefix("model.")
    return key.replace(f".{adapter_name}", "")


def _split_targets(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]
    return items or None


def _language_list(language_map_path: Path) -> list[str] | None:
    if not language_map_path.exists():
        return None
    payload = json.loads(language_map_path.read_text(encoding="utf-8"))
    languages: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str):
                languages.add(str(key))
            elif isinstance(value, dict):
                for lang in value.get("languages") or value.get("language") or []:
                    languages.add(str(lang))
                metadata = value.get("metadata")
                if isinstance(metadata, dict):
                    languages.update(str(lang) for lang in metadata)
    return sorted(languages) or None


def _adapter_state(full_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        _strip_adapter_name(key): value
        for key, value in full_state.items()
        if any(marker in key for marker in ADAPTER_KEY_MARKERS)
    }


def _trainer_global_step(run_dir: Path) -> int | None:
    state_path = run_dir / "trainer_state.json"
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    step = payload.get("global_step")
    if isinstance(step, int):
        return step
    if isinstance(step, str) and step.strip().isdigit():
        return int(step)
    return None


def _write_manifest(
    *,
    path: Path,
    run_id: str,
    source_path: Path,
    output_dir: Path,
    adapter_state: dict[str, torch.Tensor],
) -> None:
    stat = source_path.stat()
    payload = {
        "run_id": run_id,
        "recovered_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": str(source_path),
        "source_size_bytes": stat.st_size,
        "source_mtime": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "num_adapter_tensors": len(adapter_state),
        "num_keys_with_default_suffix": sum(".default" in key for key in adapter_state),
        "method": "materialized_from_final_pytorch_model_bin_with_adapter_name_stripped",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_config(
    *,
    run: dict[str, Any],
    common_lora: dict[str, Any],
    base_model: str,
    language_map_path: Path,
) -> Any:
    kind = str(run["kind"])
    lora = _deep_merge(common_lora, run.get("lora", {}))
    params = dict(run.get("params", {}))
    targets = _split_targets(lora.get("target"))
    rank = int(lora.get("rank", 8))
    alpha = int(lora.get("alpha", 16))
    dropout = float(lora.get("dropout", 0.05))

    if kind == "cola":
        return ColaConfig(
            task_type=TaskType.CAUSAL_LM,
            base_model_name_or_path=base_model,
            inference_mode=True,
            r=rank,
            target_modules=targets,
            lora_alpha=alpha,
            lora_dropout=dropout,
            num_A=int(params.get("cola_num_A", 1)),
            num_B=int(params.get("cola_num_B", 1)),
            cola_strategy=str(params.get("cola_strategy", "fully")),
            use_cola_experts=bool(params.get("cola_use_experts", False)),
            init_lora_weights="pissa" if bool(params.get("use_cola_pissa_init", False)) else True,
        )

    if kind == "mtllora":
        languages = _language_list(language_map_path)
        use_language_ids = bool(params.get("mtllora_use_language_ids_as_task_ids", True))
        task_num = int(params.get("mtllora_task_num", len(languages or []) or 1))
        return MtlLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            base_model_name_or_path=base_model,
            inference_mode=True,
            r=rank,
            target_modules=targets,
            lora_alpha=alpha,
            lora_dropout=dropout,
            task_num=task_num,
            num_up_projections=int(params.get("mtllora_num_up_projections", 3)),
            temperature=float(params.get("mtllora_temperature", 0.1)),
            lambda_format=str(params.get("mtllora_lambda_format", "full")),
            use_language_ids_as_task_ids=use_language_ids,
            language_list=languages if use_language_ids else None,
        )

    if kind == "movlora":
        return MovLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            base_model_name_or_path=base_model,
            inference_mode=True,
            target_modules=targets,
            num_experts=int(params.get("movlora_num_experts", 30)),
            router_top_k=int(params.get("movlora_top_k", 0)),
            router_temperature=float(params.get("movlora_router_temperature", 1.0)),
            router_jitter_noise=float(params.get("movlora_router_jitter_noise", 0.0)),
            router_bias=bool(params.get("movlora_router_bias", False)),
            router_init_std=float(params.get("movlora_router_init_std", 0.02)),
            router_ignore_padding_tokens=bool(params.get("movlora_router_ignore_padding_tokens", False)),
            init_ia3_weights=True,
        )

    raise SystemExit(f"unsupported recovery kind for run {run.get('id')}: {kind}")


def _run(args: argparse.Namespace) -> int:
    profile = _resolve_profile(load_pipeline_config(Path(args.config)), args.profile)
    root = REPO_ROOT / str(profile["output_root"])
    train_cfg = profile["training"]
    common_lora = train_cfg.get("lora", {})
    base_model = str(train_cfg["model_name_or_path"])
    language_map_path = root / "language_map_from_csv.json"
    if not language_map_path.exists():
        language_map_path = REPO_ROOT / str(profile.get("language_map_path", ""))

    selected = set(args.run_id or [])
    runs = [
        run
        for run in train_cfg["runs"]
        if (not selected or str(run["id"]) in selected)
        and str(run["kind"]) in {"cola", "mtllora", "movlora"}
    ]
    if selected:
        found = {str(run["id"]) for run in runs}
        missing = sorted(selected - found)
        if missing:
            raise SystemExit(f"unknown or unsupported run ids: {missing}")

    materialized = []
    for run in runs:
        run_id = str(run["id"])
        run_dir = root / "runs" / run_id
        full_state_path = run_dir / "pytorch_model.bin"
        target_dir = run_dir
        if args.checkpoint_dir_from_trainer_state:
            step = _trainer_global_step(run_dir)
            if step is None:
                raise SystemExit(f"{run_id}: trainer_state.json missing or has no global_step")
            target_dir = run_dir / f"checkpoint-{step}_adapter"
        adapter_config_path = target_dir / "adapter_config.json"
        adapter_model_path = target_dir / "adapter_model.bin"
        if adapter_config_path.exists() and adapter_model_path.exists() and not args.force:
            print(f"[skip] {run_id}: adapter already materialized")
            continue
        if not full_state_path.exists():
            print(f"[skip] {run_id}: missing {full_state_path}")
            continue

        print(f"[load] {run_id}: {full_state_path}")
        full_state = torch.load(full_state_path, map_location="cpu", weights_only=True, mmap=True)
        adapter_state = _adapter_state(full_state)
        if not adapter_state:
            raise SystemExit(f"{run_id}: no adapter keys found in {full_state_path}")
        if any(".default" in key for key in adapter_state):
            raise SystemExit(f"{run_id}: adapter keys still contain '.default' after stripping")

        config = _build_config(
            run=run,
            common_lora=common_lora,
            base_model=base_model,
            language_map_path=language_map_path,
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        config.save_pretrained(target_dir)
        torch.save(adapter_state, adapter_model_path)
        _write_manifest(
            path=target_dir / "adapter_recovery_manifest.json",
            run_id=run_id,
            source_path=full_state_path,
            output_dir=target_dir,
            adapter_state=adapter_state,
        )
        materialized.append(run_id)
        print(f"[ok] {run_id}: keys={len(adapter_state)} dir={target_dir}")

    print(f"materialized={len(materialized)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover PEFT adapter artifacts from final pytorch_model.bin states.")
    parser.add_argument("--config", default="configs/pipeline")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--checkpoint-dir-from-trainer-state", action="store_true")
    return _run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
