import argparse
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
        key: value
        for key, value in full_state.items()
        if any(marker in key for marker in ADAPTER_KEY_MARKERS)
    }


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
        adapter_config_path = run_dir / "adapter_config.json"
        adapter_model_path = run_dir / "adapter_model.bin"
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

        config = _build_config(
            run=run,
            common_lora=common_lora,
            base_model=base_model,
            language_map_path=language_map_path,
        )
        config.save_pretrained(run_dir)
        torch.save(adapter_state, adapter_model_path)
        materialized.append(run_id)
        print(f"[ok] {run_id}: keys={len(adapter_state)} dir={run_dir}")

    print(f"materialized={len(materialized)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover PEFT adapter artifacts from final pytorch_model.bin states.")
    parser.add_argument("--config", default="configs/pipeline")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return _run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
