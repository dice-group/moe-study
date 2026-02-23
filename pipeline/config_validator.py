import argparse
from pathlib import Path
from typing import Any

from .config_loader import load_pipeline_config as _shared_load_pipeline_config

"""validator for proof pipeline config JSON, pointer JSON, or complete dirs"""

REQUIRED_TOP_LEVEL_KEYS = (
    "schema_version",
    "defaults",
    "profiles",
)

REQUIRED_MERGED_PROFILE_KEYS = (
    "output_root",
    "language_map_path",
    "sample",
    "tokenization",
    "training",
    "summary",
    "tokenizer_extension",
    "embedding_init",
    "embedding_training",
    "benchmark",
    "report",
)

ALLOWED_RUN_KINDS = {"lora", "cola", "hydralora", "adamole", "mola", "moelpr", "moelora", "movlora", "hmora"}
ALLOWED_SAMPLE_SOURCE_TYPES = {"hf", "json", "fineweb_datatrove"}
ALLOWED_FLORES_MODES = {"both", "into_en", "from_en"}
ALLOWED_PROFILE_EXTRA_KEYS = {"run_naming", "logs_dir", "extends"}


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_non_empty_string(value: Any, path: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path} must be a non-empty string")
        return ""
    return value.strip()


def _check_bool(value: Any, path: str, errors: list[str]) -> bool:
    if not isinstance(value, bool):
        errors.append(f"{path} must be a boolean")
        return False
    return value


def _check_dict(value: Any, path: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return {}
    return value


def _check_list(value: Any, path: str, errors: list[str]) -> list[Any]:
    if not isinstance(value, list):
        errors.append(f"{path} must be a list")
        return []
    return value


def _load_proof_pipeline_payload(path: Path) -> dict[str, Any]:
    return _shared_load_pipeline_config(path)


def _validate_training_runs(training: dict[str, Any], path: str, errors: list[str]) -> set[str]:
    run_ids: set[str] = set()
    runs = training.get("runs")
    if not isinstance(runs, list):
        errors.append(f"{path}.runs must be a list")
        return run_ids
    if not runs and bool(training.get("enabled", True)):
        errors.append(f"{path}.runs must not be empty when {path}.enabled is true")
        return run_ids

    for index, run in enumerate(runs):
        run_path = f"{path}.runs[{index}]"
        if not isinstance(run, dict):
            errors.append(f"{run_path} must be an object")
            continue
        run_id = run.get("id")
        kind = run.get("kind")
        run_id_value = _check_non_empty_string(run_id, f"{run_path}.id", errors)
        kind_value = _check_non_empty_string(kind, f"{run_path}.kind", errors)
        if run_id_value:
            if run_id_value in run_ids:
                errors.append(f"duplicate training run id '{run_id_value}' in {path}.runs")
            run_ids.add(run_id_value)
        if kind_value and kind_value not in ALLOWED_RUN_KINDS:
            errors.append(
                f"{run_path}.kind must be one of {sorted(ALLOWED_RUN_KINDS)} (got '{kind_value}')"
            )
    return run_ids


def _validate_profile(profile_name: str, merged_profile: dict[str, Any], errors: list[str]):
    prefix = f"profiles.{profile_name}"

    for key in REQUIRED_MERGED_PROFILE_KEYS:
        if key not in merged_profile:
            errors.append(f"{prefix} is missing merged required key '{key}'")

    _check_non_empty_string(merged_profile.get("output_root"), f"{prefix}.output_root", errors)
    _check_non_empty_string(merged_profile.get("language_map_path"), f"{prefix}.language_map_path", errors)

    sample = _check_dict(merged_profile.get("sample"), f"{prefix}.sample", errors)
    tokenization = _check_dict(merged_profile.get("tokenization"), f"{prefix}.tokenization", errors)
    training = _check_dict(merged_profile.get("training"), f"{prefix}.training", errors)
    summary = _check_dict(merged_profile.get("summary"), f"{prefix}.summary", errors)
    tokenizer_extension = _check_dict(
        merged_profile.get("tokenizer_extension"), f"{prefix}.tokenizer_extension", errors
    )
    embedding_init = _check_dict(merged_profile.get("embedding_init"), f"{prefix}.embedding_init", errors)
    embedding_training = _check_dict(
        merged_profile.get("embedding_training"), f"{prefix}.embedding_training", errors
    )
    benchmark = _check_dict(merged_profile.get("benchmark"), f"{prefix}.benchmark", errors)
    report = _check_dict(merged_profile.get("report"), f"{prefix}.report", errors)

    source_type = _check_non_empty_string(sample.get("source_type"), f"{prefix}.sample.source_type", errors)
    if source_type and source_type not in ALLOWED_SAMPLE_SOURCE_TYPES:
        errors.append(
            f"{prefix}.sample.source_type must be one of {sorted(ALLOWED_SAMPLE_SOURCE_TYPES)} "
            f"(got '{source_type}')"
        )
    _check_non_empty_string(sample.get("output_dir"), f"{prefix}.sample.output_dir", errors)

    _check_non_empty_string(tokenization.get("output_dir"), f"{prefix}.tokenization.output_dir", errors)
    _check_non_empty_string(tokenization.get("model_name"), f"{prefix}.tokenization.model_name", errors)

    training_enabled = _check_bool(training.get("enabled", True), f"{prefix}.training.enabled", errors)
    _check_non_empty_string(training.get("output_dir"), f"{prefix}.training.output_dir", errors)
    _check_non_empty_string(
        training.get("model_name_or_path"),
        f"{prefix}.training.model_name_or_path",
        errors,
    )
    num_experts = training.get("num_experts")
    if not _is_int(num_experts) or int(num_experts) <= 0:
        errors.append(f"{prefix}.training.num_experts must be a positive integer")
    run_ids = _validate_training_runs(training, f"{prefix}.training", errors)

    _check_non_empty_string(summary.get("summary_csv"), f"{prefix}.summary.summary_csv", errors)
    _check_non_empty_string(summary.get("gate_json"), f"{prefix}.summary.gate_json", errors)
    baseline_id = _check_non_empty_string(summary.get("baseline_id"), f"{prefix}.summary.baseline_id", errors)
    _check_non_empty_string(summary.get("primary_metric"), f"{prefix}.summary.primary_metric", errors)
    min_rel_impr = summary.get("min_relative_improvement")
    if not _is_number(min_rel_impr):
        errors.append(f"{prefix}.summary.min_relative_improvement must be numeric")

    if baseline_id and run_ids and baseline_id not in run_ids:
        errors.append(
            f"{prefix}.summary.baseline_id '{baseline_id}' is not present in {prefix}.training.runs ids"
        )

    tokenizer_extension_enabled = _check_bool(
        tokenizer_extension.get("enabled", False),
        f"{prefix}.tokenizer_extension.enabled",
        errors,
    )
    _check_non_empty_string(
        tokenizer_extension.get("output_dir"),
        f"{prefix}.tokenizer_extension.output_dir",
        errors,
    )

    embedding_init_enabled = _check_bool(
        embedding_init.get("enabled", False),
        f"{prefix}.embedding_init.enabled",
        errors,
    )
    _check_non_empty_string(embedding_init.get("output_dir"), f"{prefix}.embedding_init.output_dir", errors)

    embedding_training_enabled = _check_bool(
        embedding_training.get("enabled", False),
        f"{prefix}.embedding_training.enabled",
        errors,
    )
    _check_bool(
        embedding_training.get("verify_after_train", True),
        f"{prefix}.embedding_training.verify_after_train",
        errors,
    )
    _check_bool(
        embedding_training.get("require_weight_change", True),
        f"{prefix}.embedding_training.require_weight_change",
        errors,
    )
    _check_non_empty_string(
        embedding_training.get("additional_target"),
        f"{prefix}.embedding_training.additional_target",
        errors,
    )

    benchmark_enabled = _check_bool(benchmark.get("enabled", False), f"{prefix}.benchmark.enabled", errors)
    _check_non_empty_string(benchmark.get("tasks_file"), f"{prefix}.benchmark.tasks_file", errors)
    _check_non_empty_string(benchmark.get("output_dir"), f"{prefix}.benchmark.output_dir", errors)
    if bool(benchmark.get("include_flores", True)):
        flores_mode = _check_non_empty_string(benchmark.get("flores_mode"), f"{prefix}.benchmark.flores_mode", errors)
        if flores_mode and flores_mode not in ALLOWED_FLORES_MODES:
            errors.append(
                f"{prefix}.benchmark.flores_mode must be one of {sorted(ALLOWED_FLORES_MODES)} "
                f"(got '{flores_mode}')"
            )

    report_enabled = _check_bool(report.get("enabled", False), f"{prefix}.report.enabled", errors)
    _check_non_empty_string(report.get("output_xlsx"), f"{prefix}.report.output_xlsx", errors)

    if embedding_init_enabled and not tokenizer_extension_enabled:
        errors.append(
            f"{prefix} stage dependency failed: embedding_init.enabled requires tokenizer_extension.enabled"
        )
    if embedding_training_enabled and not tokenizer_extension_enabled:
        errors.append(
            f"{prefix} stage dependency failed: embedding_training.enabled requires tokenizer_extension.enabled"
        )
    if embedding_training_enabled and not embedding_init_enabled:
        errors.append(
            f"{prefix} stage dependency failed: embedding_training.enabled requires embedding_init.enabled"
        )
    if benchmark_enabled and not training_enabled:
        errors.append(f"{prefix} stage dependency failed: benchmark.enabled requires training.enabled")
    if benchmark_enabled and not run_ids:
        errors.append(f"{prefix} stage dependency failed: benchmark.enabled requires training.runs")
    if report_enabled and not run_ids:
        errors.append(f"{prefix} stage dependency failed: report.enabled requires training.runs")


def validate_proof_pipeline_config(payload: Any) -> list[str]:
    errors: list[str] = []

    if not isinstance(payload, dict):
        return ["proof config must be a JSON object"]

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in payload:
            errors.append(f"missing required top-level key: {key}")

    schema_version = payload.get("schema_version")
    if not _is_int(schema_version) or int(schema_version) <= 0:
        errors.append("schema_version must be a positive integer")

    defaults = payload.get("defaults")
    profiles = payload.get("profiles")
    if not isinstance(defaults, dict) or not defaults:
        errors.append("defaults must be a non-empty object")
        defaults = {}
    if not isinstance(profiles, dict) or not profiles:
        errors.append("profiles must be a non-empty object")
        profiles = {}

    if not defaults or not profiles:
        return errors

    allowed_profile_keys = set(defaults.keys()) | ALLOWED_PROFILE_EXTRA_KEYS
    resolved_profile_overrides: dict[str, dict[str, Any]] = {}
    failed_profiles: set[str] = set()

    def _resolve_profile_override(name: str, chain: list[str]) -> dict[str, Any] | None:
        if name in resolved_profile_overrides:
            return resolved_profile_overrides[name]
        if name in failed_profiles:
            return None
        if name in chain:
            errors.append(f"profiles.{name} has inheritance cycle: {' -> '.join(chain + [name])}")
            failed_profiles.add(name)
            return None

        profile_override_raw = profiles.get(name)
        if not isinstance(profile_override_raw, dict):
            errors.append(f"profiles.{name} must be an object")
            failed_profiles.add(name)
            return None

        extends_raw = profile_override_raw.get("extends")
        parent_name = ""
        if extends_raw is not None:
            if not isinstance(extends_raw, str) or not extends_raw.strip():
                errors.append(f"profiles.{name}.extends must be a non-empty string")
                failed_profiles.add(name)
                return None
            parent_name = extends_raw.strip()
            if parent_name not in profiles:
                errors.append(f"profiles.{name}.extends references unknown profile '{parent_name}'")
                failed_profiles.add(name)
                return None

        own_override = dict(profile_override_raw)
        own_override.pop("extends", None)
        parent_override: dict[str, Any] = {}
        if parent_name:
            parent_resolved = _resolve_profile_override(parent_name, chain + [name])
            if parent_resolved is None:
                failed_profiles.add(name)
                return None
            parent_override = parent_resolved

        merged_override = _deep_merge(parent_override, own_override)
        if not isinstance(merged_override, dict):
            errors.append(f"profiles.{name} did not merge to an object")
            failed_profiles.add(name)
            return None
        resolved_profile_overrides[name] = merged_override
        return merged_override

    for profile_name, profile_override in sorted(profiles.items()):
        profile_path = f"profiles.{profile_name}"
        if not isinstance(profile_override, dict):
            errors.append(f"{profile_path} must be an object")
            continue

        unknown = sorted(set(profile_override.keys()) - allowed_profile_keys)
        if unknown:
            errors.append(f"{profile_path} has unknown top-level keys: {unknown}")

        for key, value in profile_override.items():
            if key == "extends":
                continue
            default_value = defaults.get(key)
            if isinstance(default_value, dict) and not isinstance(value, dict):
                errors.append(
                    f"{profile_path}.{key} must be an object to merge with defaults.{key}"
                )

        resolved_override = _resolve_profile_override(profile_name, [])
        if resolved_override is None:
            continue
        merged_profile = _deep_merge(defaults, resolved_override)
        if not isinstance(merged_profile, dict):
            errors.append(f"{profile_path} did not merge to an object")
            continue

        _validate_profile(profile_name, merged_profile, errors)

    return errors


def validate_proof_pipeline_config_file(path: Path) -> list[str]:
    try:
        payload = _load_proof_pipeline_payload(path)
    except Exception as exc:
        return [f"invalid proof config ({path}): {exc}"]
    return validate_proof_pipeline_config(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate proof pipeline profile config.")
    parser.add_argument(
        "--config",
        default="configs/proof_pipeline",
        help="Path to proof pipeline bundle directory or config JSON.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    errors = validate_proof_pipeline_config_file(config_path)
    if errors:
        print("[FAIL] Proof pipeline config validation failed:")
        for err in errors:
            print(f" - {err}")
        return 1

    print(f"[OK] Proof pipeline config valid: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
