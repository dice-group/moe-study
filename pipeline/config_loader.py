import json
from pathlib import Path
from typing import Any

"""Shared loader for pipeline config JSON and config bundles."""

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def load_pipeline_config(path: Path) -> dict[str, Any]:
    if path.is_dir():
        return load_pipeline_config_bundle(path)

    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"proof config must be a JSON object: {path}")

    bundle_dir = payload.get("config_bundle_dir")
    if isinstance(bundle_dir, str) and bundle_dir.strip():
        bundle_path = Path(bundle_dir.strip())
        if not bundle_path.is_absolute():
            bundle_path = (path.parent / bundle_path).resolve()
        return load_pipeline_config_bundle(bundle_path)

    return payload


def load_pipeline_config_bundle(bundle_dir: Path) -> dict[str, Any]:
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        raise ValueError(f"config bundle directory does not exist: {bundle_dir}")

    schema_path = bundle_dir / "schema.json"
    schema_version = 1
    if schema_path.exists():
        schema_payload = load_json(schema_path)
        raw_version = schema_payload.get("schema_version") if isinstance(schema_payload, dict) else None
        if isinstance(raw_version, int):
            schema_version = raw_version

    defaults_dir = bundle_dir / "defaults"
    profiles_dir = bundle_dir / "profiles"
    if not defaults_dir.exists():
        raise ValueError(f"config bundle missing defaults directory: {defaults_dir}")
    if not profiles_dir.exists():
        raise ValueError(f"config bundle missing profiles directory: {profiles_dir}")

    defaults: dict[str, Any] = {}
    for file_path in sorted(defaults_dir.rglob("*.json")):
        fragment = load_json(file_path)
        if not isinstance(fragment, dict):
            raise ValueError(f"defaults fragment must be a JSON object: {file_path}")
        defaults = deep_merge(defaults, fragment)

    profiles: dict[str, Any] = {}
    for file_path in sorted(profiles_dir.rglob("*.json")):
        fragment = load_json(file_path)
        if not isinstance(fragment, dict):
            raise ValueError(f"profiles fragment must be a JSON object: {file_path}")
        for profile_name, override in fragment.items():
            if not isinstance(profile_name, str):
                raise ValueError(f"profile name must be a string in {file_path}")
            if profile_name in profiles:
                profiles[profile_name] = deep_merge(profiles[profile_name], override)
            else:
                profiles[profile_name] = override

    return {
        "schema_version": schema_version,
        "defaults": defaults,
        "profiles": profiles,
    }
