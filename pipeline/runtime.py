#!/usr/bin/env python3
"""Shared runtime helpers for local pipeline orchestrators.

This module intentionally stays lightweight and dependency-minimal so orchestration
scripts can reuse common logic without duplicating implementation details.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .config_loader import (
    deep_merge as _loader_deep_merge,
    load_json as _loader_load_json,
    load_pipeline_config as _loader_load_pipeline_config,
    load_pipeline_config_bundle as _loader_load_pipeline_config_bundle,
)


LOGGER = logging.getLogger("proof_pipeline")


def _load_json(path: Path) -> dict[str, Any]:
    return _loader_load_json(path)


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _deep_merge(base: Any, override: Any) -> Any:
    return _loader_deep_merge(base, override)


def _load_pipeline_config(path: Path) -> dict[str, Any]:
    try:
        return _loader_load_pipeline_config(path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _load_pipeline_config_bundle(bundle_dir: Path) -> dict[str, Any]:
    try:
        return _loader_load_pipeline_config_bundle(bundle_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _resolve_profile(config: dict[str, Any], profile_name: str) -> dict[str, Any]:
    defaults = config.get("defaults", {})
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        raise SystemExit(f"Unknown profile '{profile_name}'. Available: {sorted(profiles.keys())}")

    resolved_cache: dict[str, dict[str, Any]] = {}

    def _resolve_profile_chain(name: str, chain: list[str]) -> dict[str, Any]:
        if name in resolved_cache:
            return resolved_cache[name]
        if name in chain:
            cycle = " -> ".join(chain + [name])
            raise SystemExit(f"Profile inheritance cycle detected: {cycle}")

        override = profiles.get(name)
        if not isinstance(override, dict):
            raise SystemExit(f"Profile '{name}' must be an object.")

        extends_raw = override.get("extends")
        parent_name = ""
        if extends_raw is not None:
            if not isinstance(extends_raw, str) or not extends_raw.strip():
                raise SystemExit(f"Profile '{name}'.extends must be a non-empty string when provided.")
            parent_name = extends_raw.strip()
            if parent_name not in profiles:
                raise SystemExit(f"Profile '{name}'.extends references unknown profile '{parent_name}'.")

        current_override = dict(override)
        current_override.pop("extends", None)
        parent_payload: dict[str, Any] = {}
        if parent_name:
            parent_payload = _resolve_profile_chain(parent_name, chain + [name])

        merged = _deep_merge(parent_payload, current_override)
        if not isinstance(merged, dict):
            raise SystemExit(f"Profile '{name}' did not resolve to an object.")
        resolved_cache[name] = merged
        return merged

    return _deep_merge(defaults, _resolve_profile_chain(profile_name, []))


def _abs(repo_root: Path, maybe_rel: str) -> Path:
    path = Path(maybe_rel)
    return path if path.is_absolute() else (repo_root / path)


def _missing_modules(names: list[str]) -> list[str]:
    missing: list[str] = []
    for name in names:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    return missing


from .executor import Executor, LocalExecutor, build_executor  # noqa: F401


_DEFAULT_EXECUTOR = LocalExecutor()


def _run_with_tee(
    cmd: list[str],
    env: dict[str, str],
    dry_run: bool,
    log_path: Path,
    executor: Executor | None = None,
) -> None:
    (executor or _DEFAULT_EXECUTOR).run(cmd, env, dry_run, log_path)


def _run_stage_cmd(
    *,
    cmd: list[str],
    env: dict[str, str],
    dry_run: bool,
    logs_root: Path,
    stage: str,
    suffix: str = "",
    executor: Executor | None = None,
) -> None:
    log_path = _stage_log_path(logs_root, env, stage, suffix)
    _run_with_tee(cmd, env, dry_run, log_path, executor=executor)


def _stage_log_path(
    logs_root: Path,
    env: dict[str, str],
    stage: str,
    suffix: str = "",
) -> Path:
    """Build a deterministic log path for a stage invocation."""
    suffix_part = f".{suffix}" if suffix else ""
    run_timestamp = str(env.get("PROOF_PIPELINE_RUN_TIMESTAMP", "")).strip()
    log_name = f"{stage}{suffix_part}.{run_timestamp}.log" if run_timestamp else f"{stage}{suffix_part}.log"
    return logs_root / log_name


def _sanitize_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return cleaned.strip("-") or "na"


def _model_slug(model_name_or_path: str) -> str:
    value = model_name_or_path.strip()
    if "/" in value:
        value = value.split("/")[-1]
    value = value.replace("Llama", "l").replace("llama", "l").replace(".", "-")
    return _sanitize_slug(value)


def _build_run_label(base_run_label: str, run: dict[str, Any], default_num_experts: int) -> str:
    run_kind = str(run.get("kind", "run"))
    params = run.get("params", {})
    suffix_parts = [run_kind]
    if run_kind in {"cola", "hydralora"}:
        if "num_experts" in params:
            suffix_parts.append(f"nexp{int(params['num_experts'])}")
        elif "use_experts" in params and bool(params["use_experts"]):
            suffix_parts.append(f"nexp{int(default_num_experts)}")
        if "top_k" in params:
            suffix_parts.append(f"k{int(params['top_k'])}")
        if run_kind == "cola" and "strategy" in params:
            suffix_parts.append(_sanitize_slug(str(params["strategy"])))
    elif run_kind in {"adamole", "mola", "moelpr", "moelora", "movlora"}:
        if "num_experts" in params:
            suffix_parts.append(f"nexp{int(params['num_experts'])}")
        if "top_k" in params:
            suffix_parts.append(f"k{int(params['top_k'])}")
        elif "sparse_top_k" in params:
            suffix_parts.append(f"k{int(params['sparse_top_k'])}")
    return _sanitize_slug(f"{base_run_label}-{'-'.join(suffix_parts)}")


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"stages": {}}
    return _load_json(path)


def _save_stage_state(
    state_path: Path,
    stage: str,
    signature: str,
    artifacts: list[str],
) -> None:
    state = _load_state(state_path)
    stages = state.setdefault("stages", {})
    stages[stage] = {
        "signature": signature,
        "status": "completed",
        "artifacts": artifacts,
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    _dump_json(state_path, state)


def _stage_signature(payload: dict[str, Any]) -> str:
    code_sig = str(os.environ.get("PROOF_PIPELINE_CODE_SIG", "")).strip()
    if code_sig:
        payload = {"_code_sig": code_sig, **payload}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _compute_pipeline_code_signature(repo_root: Path) -> str:
    digest = hashlib.sha256()
    scan_paths: list[Path] = [
        repo_root / "pipeline",
    ]
    for candidate in scan_paths:
        if candidate.is_file():
            digest.update(str(candidate.relative_to(repo_root)).encode("utf-8"))
            digest.update(candidate.read_bytes())
            continue
        if candidate.is_dir():
            for file_path in sorted(candidate.rglob("*.py")):
                digest.update(str(file_path.relative_to(repo_root)).encode("utf-8"))
                digest.update(file_path.read_bytes())
    return digest.hexdigest()[:16]


def _artifacts_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def _should_skip_stage(
    *,
    state_path: Path,
    stage: str,
    signature: str,
    expected_artifacts: list[Path],
    force_stages: set[str],
    dry_run: bool,
) -> bool:
    if stage in force_stages or dry_run:
        return False
    state = _load_state(state_path)
    info = state.get("stages", {}).get(stage)
    if not isinstance(info, dict):
        return False
    if info.get("signature") != signature:
        return False
    return _artifacts_exist(expected_artifacts)


def _count_experts(language_map_path: Path) -> int:
    payload = _load_json(language_map_path)
    if isinstance(payload, dict):
        return len(payload)
    if isinstance(payload, list):
        return len(payload)
    raise ValueError(f"Unsupported language_map format in {language_map_path}")


def _latest_metric(history: list[dict[str, Any]], suffix: str) -> float | None:
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        if suffix in entry and isinstance(entry[suffix], (int, float)):
            return float(entry[suffix])
        for key, value in entry.items():
            if key.endswith("/" + suffix) and isinstance(value, (int, float)):
                return float(value)
    return None


def _latest_metric_any_suffix(history: list[dict[str, Any]], suffixes: list[str]) -> float | None:
    for suffix in suffixes:
        value = _latest_metric(history, suffix)
        if value is not None:
            return value
    return None


def _read_task_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _select_tasks_for_languages(
    *,
    tasks_file: Path,
    languages: list[str],
    include_belebele: bool,
    include_flores: bool,
    flores_mode: str,
) -> list[str]:
    lang_set = set(languages)
    selected: list[str] = []
    for task in _read_task_lines(tasks_file):
        if include_belebele and task.startswith("belebele_"):
            lang = task.replace("belebele_", "", 1)
            if lang in lang_set:
                selected.append(task)
                continue

        if include_flores and task.startswith("flores_en_perplexity_"):
            rest = task.replace("flores_en_perplexity_", "", 1)
            if "-" not in rest:
                continue
            src, tgt = rest.split("-", 1)
            include = False
            if flores_mode == "both":
                include = src in lang_set or tgt in lang_set
            elif flores_mode == "into_en":
                include = src in lang_set and tgt == "eng_Latn"
            elif flores_mode == "from_en":
                include = src == "eng_Latn" and tgt in lang_set
            if include:
                selected.append(task)
    return selected


def _resolve_profile_languages(
    *,
    sample_cfg: dict[str, Any],
    sample_manifest_path: Path,
    sampled_dir: Path,
) -> list[str]:
    explicit = [item for item in sample_cfg.get("languages", []) if isinstance(item, str) and item.strip()]
    if explicit:
        return sorted(set(explicit))

    if sample_manifest_path.exists():
        try:
            manifest = _load_json(sample_manifest_path)
            selection = manifest.get("selection", {})
            langs = selection.get("languages", [])
            if isinstance(langs, list) and langs:
                return sorted({str(item) for item in langs if str(item).strip()})
        except Exception:
            pass

    plan_csv = sampled_dir / "sampling_plan.csv"
    if plan_csv.exists():
        langs: list[str] = []
        with plan_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                subset = str(row.get("subset", "")).strip()
                if subset:
                    langs.append(subset)
        if langs:
            return sorted(set(langs))
    return []


def _parse_optional_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        parsed: list[int] = []
        for item in value:
            parsed.append(int(item))
        return parsed
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    return None


def _language_groupings(path: Path) -> tuple[dict[str, str], list[str], list[int]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"language_map must be a JSON object: {path}")

    language_to_family: dict[str, str] = {}
    subgroup_sizes_by_family: dict[str, int] = {}

    for group_id, entry in sorted(payload.items(), key=lambda kv: str(kv[0])):
        if not isinstance(entry, dict):
            continue
        family = str(entry.get("group") or group_id)
        subgroup_sizes_by_family[family] = 0
        subgroups = entry.get("subgroups")
        if isinstance(subgroups, dict) and subgroups:
            subgroup_sizes_by_family[family] = len([k for k, v in subgroups.items() if isinstance(v, list)])
            for members in subgroups.values():
                if isinstance(members, list):
                    for lang in members:
                        language_to_family[str(lang)] = family
        for lang in entry.get("languages", []) or []:
            language_to_family.setdefault(str(lang), family)
        metadata = entry.get("metadata")
        if isinstance(metadata, dict):
            for lang in metadata.keys():
                language_to_family.setdefault(str(lang), family)

    if not language_to_family:
        raise ValueError(f"No language entries found in language_map: {path}")

    families = sorted(set(language_to_family.values()))
    subgroup_sizes = [int(subgroup_sizes_by_family.get(fam, 0)) for fam in families]
    return language_to_family, families, subgroup_sizes


def _validate_lpr_contract(
    *,
    train_cfg: dict[str, Any],
    sample_cfg: dict[str, Any],
    language_map_path: Path,
) -> dict[str, Any]:
    language_to_family, families, subgroup_sizes = _language_groupings(language_map_path)
    expected_experts = len(families)
    if expected_experts <= 0:
        raise ValueError("No language families/groups resolved from language_map.")

    sampled_langs = [str(x) for x in sample_cfg.get("languages", []) if str(x).strip()]
    missing_langs = sorted([lang for lang in sampled_langs if lang not in language_to_family])
    if missing_langs:
        raise ValueError(f"Sample languages missing from language_map: {missing_langs}")

    runs = train_cfg.get("runs", [])
    run_checks: list[dict[str, Any]] = []
    for run in runs:
        run_id = str(run.get("id", "unknown"))
        kind = str(run.get("kind", ""))
        if kind not in {"cola", "hydralora"}:
            continue
        params = run.get("params", {})
        if not bool(params.get("use_experts", True)):
            continue

        if kind == "cola":
            num_experts = int(params.get("num_experts", train_cfg.get("num_experts", expected_experts)))
            if num_experts != expected_experts:
                raise ValueError(
                    f"LPR contract mismatch for run '{run_id}': cola_num_experts={num_experts} "
                    f"but language_map groups={expected_experts}."
                )
            explicit_heads = _parse_optional_int_list(params.get("expert_num_B"))
            if explicit_heads is not None:
                heads = explicit_heads
            elif subgroup_sizes and any(size > 0 for size in subgroup_sizes):
                heads = list(subgroup_sizes)
            else:
                heads = [int(params.get("num_B", 3))] * expected_experts
        else:
            num_experts = int(params.get("num_experts", train_cfg.get("num_experts", expected_experts)))
            if num_experts != expected_experts:
                raise ValueError(
                    f"LPR contract mismatch for run '{run_id}': hydralora_num_experts={num_experts} "
                    f"but language_map groups={expected_experts}."
                )
            explicit_heads = _parse_optional_int_list(params.get("expert_lora_nums"))
            if explicit_heads is not None:
                heads = explicit_heads
            elif subgroup_sizes and any(size > 0 for size in subgroup_sizes):
                heads = list(subgroup_sizes)
            else:
                heads = [int(params.get("lora_num", 3))] * expected_experts

        if len(heads) != expected_experts or any(int(h) <= 0 for h in heads):
            raise ValueError(
                f"LPR head contract mismatch for run '{run_id}': expected {expected_experts} positive entries, got {heads}."
            )

        router_mode = str(params.get("router_mode", "learned"))
        head_router_mode = str(params.get("head_router_mode", "learned"))
        guidance_scope = str(params.get("guidance_scope", "all"))
        if router_mode != "learned" or head_router_mode != "learned":
            raise ValueError(
                f"LPR routing contract mismatch for run '{run_id}': router_mode/head_router_mode must be learned "
                f"(got {router_mode}/{head_router_mode})."
            )
        if guidance_scope not in {"all", "router", "head"}:
            raise ValueError(f"LPR guidance scope invalid for run '{run_id}': {guidance_scope}")

        run_checks.append(
            {
                "run_id": run_id,
                "kind": kind,
                "num_experts": num_experts,
                "heads_per_expert": [int(x) for x in heads],
                "router_mode": router_mode,
                "head_router_mode": head_router_mode,
                "guidance_scope": guidance_scope,
                "prior_weight": float(params.get("prior_weight", 0.1)),
            }
        )

    return {
        "language_map_path": str(language_map_path),
        "expected_experts": expected_experts,
        "families": families,
        "subgroup_sizes": subgroup_sizes,
        "sampled_languages": sampled_langs,
        "runs": run_checks,
    }
