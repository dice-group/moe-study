from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_PEFT_TYPES = {"MOLA", "MOELORA", "HYDRALORA", "MTLLORA"}


@dataclass(frozen=True)
class AdapterSpec:
    run_id: str
    checkpoint: Path
    label: str
    batch_size: int | None = None


@dataclass(frozen=True)
class AnalysisConfig:
    repo_root: Path
    base_model: str
    sample_root: Path
    sample_manifest: Path | None
    language_map: Path | None
    languages: list[str]
    analysis_languages: list[str]
    output_root: Path
    adapters: list[AdapterSpec]
    batch_size: int
    max_sequences: int
    max_length: int
    text_field: str
    device: str
    torch_dtype: str
    use_language_ids: bool
    create_figures: bool
    auto_find_batch_size: bool
    length_sort_batches: bool
    strict_samples: bool
    static_validation_languages: list[str]
    checkpoint_hashes: bool


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def abs_path(repo_root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON config must be an object: {path}")
    return payload


def flatten_language_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    if all(isinstance(value, str) or value is None for value in payload.values()):
        return {str(k): str(v) for k, v in payload.items() if v is not None}

    flattened: dict[str, str] = {}
    for group_id, entry in payload.items():
        if not isinstance(entry, dict):
            continue
        family = str(entry.get("group") or group_id)
        languages = set(entry.get("languages") or entry.get("language") or [])
        subgroups = entry.get("subgroups") or {}
        if isinstance(subgroups, dict):
            for members in subgroups.values():
                if isinstance(members, list):
                    languages.update(members)
        metadata = entry.get("metadata") or {}
        if isinstance(metadata, dict):
            languages.update(metadata.keys())
        for language in languages:
            flattened.setdefault(str(language), family)
    return flattened


def language_ids_from_map(language_map: dict[str, str]) -> dict[str, int]:
    return {language: idx for idx, language in enumerate(sorted(language_map))}


def languages_from_csv(
    path: Path,
    *,
    column: str = "subset",
    split_column: str = "split",
    split_value: str = "train",
) -> list[str]:
    languages: list[str] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f"Language CSV missing column {column!r}: {path}")
        if split_value and split_column not in reader.fieldnames:
            raise ValueError(f"Language CSV missing split column {split_column!r}: {path}")
        for row in reader:
            if split_value and str(row.get(split_column, "")).strip() != split_value:
                continue
            language = str(row.get(column, "")).strip()
            if language and language not in seen:
                seen.add(language)
                languages.append(language)
    return languages


def resolve_languages(repo_root: Path, payload: dict[str, Any], language_map_path: Path | None) -> list[str]:
    raw_languages = payload.get("languages")
    if isinstance(raw_languages, list) and raw_languages:
        return [str(item).strip() for item in raw_languages if str(item).strip()]

    csv_path = abs_path(repo_root, payload.get("languages_csv"))
    languages: list[str] = []
    if csv_path is not None and csv_path.exists():
        languages = languages_from_csv(
            csv_path,
            column=str(payload.get("languages_csv_column", "subset")).strip() or "subset",
            split_column=str(payload.get("languages_csv_split_column", "split")).strip() or "split",
            split_value=str(payload.get("languages_csv_split", "train")).strip(),
        )
    elif language_map_path is not None and language_map_path.exists():
        languages = sorted(flatten_language_map(language_map_path))

    extras = payload.get("languages_extra", [])
    if isinstance(extras, list):
        seen = set(languages)
        for item in extras:
            language = str(item).strip()
            if language and language not in seen:
                seen.add(language)
                languages.append(language)
    return languages


def load_config(config_path: Path) -> AnalysisConfig:
    repo_root = repo_root_from_here()
    payload = load_json(config_path if config_path.is_absolute() else repo_root / config_path)

    language_map = abs_path(repo_root, payload.get("language_map"))
    languages = resolve_languages(repo_root, payload, language_map)
    if not languages:
        raise ValueError("No languages resolved from config.")

    adapter_specs: list[AdapterSpec] = []
    for index, item in enumerate(payload.get("adapters", [])):
        if not isinstance(item, dict):
            raise ValueError(f"adapters[{index}] must be an object")
        run_id = str(item.get("run_id", "")).strip()
        checkpoint = abs_path(repo_root, item.get("checkpoint"))
        if not run_id or checkpoint is None:
            raise ValueError(f"adapters[{index}] requires run_id and checkpoint")
        adapter_specs.append(
            AdapterSpec(
                run_id=run_id,
                checkpoint=checkpoint,
                label=str(item.get("label") or run_id),
                batch_size=int(item["batch_size"]) if item.get("batch_size") is not None else None,
            )
        )
    if not adapter_specs:
        raise ValueError("Config must include at least one adapter.")

    return AnalysisConfig(
        repo_root=repo_root,
        base_model=str(payload["base_model"]).strip(),
        sample_root=abs_path(repo_root, payload.get("sample_root")) or repo_root,
        sample_manifest=abs_path(repo_root, payload.get("sample_manifest")),
        language_map=language_map,
        languages=languages,
        analysis_languages=languages,
        output_root=abs_path(repo_root, payload.get("output_root")) or (repo_root / "outputs" / "routing_analysis"),
        adapters=adapter_specs,
        batch_size=int(payload.get("batch_size", 1)),
        max_sequences=int(payload.get("max_sequences", 100)),
        max_length=int(payload.get("max_length", 1024)),
        text_field=str(payload.get("text_field", "text")),
        device=str(payload.get("device", "cuda")),
        torch_dtype=str(payload.get("torch_dtype", "bfloat16")),
        use_language_ids=bool(payload.get("use_language_ids", True)),
        create_figures=bool(payload.get("create_figures", True)),
        auto_find_batch_size=bool(payload.get("auto_find_batch_size", False)),
        length_sort_batches=bool(payload.get("length_sort_batches", True)),
        strict_samples=bool(payload.get("strict_samples", True)),
        static_validation_languages=[
            str(item).strip()
            for item in payload.get("static_validation_languages", ["english", "arb_Arab", "jpn_Jpan"])
            if str(item).strip()
        ],
        checkpoint_hashes=bool(payload.get("checkpoint_hashes", True)),
    )


def read_adapter_config(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / "adapter_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json: {path}")
    return load_json(path)


def detect_method(adapter_config: dict[str, Any]) -> str:
    peft_type = str(adapter_config.get("peft_type", "")).upper()
    if peft_type not in SUPPORTED_PEFT_TYPES:
        raise ValueError(f"Unsupported peft_type {peft_type!r}; expected {sorted(SUPPORTED_PEFT_TYPES)}")
    return peft_type


def validate_config(config: AnalysisConfig) -> list[dict[str, Any]]:
    for name in ("batch_size", "max_sequences", "max_length"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if not config.sample_root.exists():
        raise FileNotFoundError(f"sample_root does not exist: {config.sample_root}")
    if config.sample_manifest is not None and not config.sample_manifest.exists():
        raise FileNotFoundError(f"sample_manifest does not exist: {config.sample_manifest}")
    if config.language_map is not None and not config.language_map.exists():
        raise FileNotFoundError(f"language_map does not exist: {config.language_map}")

    summaries: list[dict[str, Any]] = []
    for spec in config.adapters:
        if not spec.checkpoint.exists():
            raise FileNotFoundError(f"Adapter checkpoint does not exist: {spec.checkpoint}")
        adapter_cfg = read_adapter_config(spec.checkpoint)
        method = detect_method(adapter_cfg)
        summaries.append(
            {
                "run_id": spec.run_id,
                "checkpoint": str(spec.checkpoint),
                "peft_type": method,
                "base_model": adapter_cfg.get("base_model_name_or_path"),
                "target_modules": adapter_cfg.get("target_modules"),
            }
        )
    return summaries
