#!/usr/bin/env python3
"""Lightweight repo config validator for CI and local preflight."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _require_path(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing path: {path}")


def _require_json_object(path: Path, errors: list[str]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - direct diagnostics path
        errors.append(f"invalid json ({path}): {exc}")
        return
    if not isinstance(payload, dict) or not payload:
        errors.append(f"expected non-empty JSON object: {path}")


def _require_contains(path: Path, token: str, errors: list[str]) -> None:
    content = path.read_text(encoding="utf-8")
    if token not in content:
        errors.append(f"missing token '{token}' in {path}")


def _validate_proof_pipeline_config(repo_root: Path, errors: list[str]) -> None:
    config_path = repo_root / "configs" / "proof_pipeline"
    if not config_path.exists():
        return

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from pipeline.config_validator import (  # pylint: disable=import-outside-toplevel
            validate_proof_pipeline_config_file,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        errors.append(f"failed loading proof pipeline validator: {exc}")
        return

    for err in validate_proof_pipeline_config_file(config_path):
        errors.append(f"proof pipeline config: {err}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    errors: list[str] = []

    must_exist = [
        repo_root / "configs" / "run_manifest.yaml",
        repo_root / "scripts" / "comparison" / "run_multilingual_ablation.sh",
        repo_root / "docs" / "AGENT_RULES.md",
        repo_root / "docs" / "PR_CHECKLIST.md",
        repo_root / "data_prep" / "clustering" / "two_stage_clustering" / "12_tier_language_groupings.json",
        repo_root / "data_prep" / "clustering" / "two_stage_clustering" / "72_tier_language_groupings.json",
        repo_root / "data_prep" / "clustering" / "two_stage_clustering" / "200_tier_language_groupings.json",
    ]

    for path in must_exist:
        _require_path(path, errors)

    for tier_file in [
        repo_root / "data_prep" / "clustering" / "two_stage_clustering" / "12_tier_language_groupings.json",
        repo_root / "data_prep" / "clustering" / "two_stage_clustering" / "72_tier_language_groupings.json",
        repo_root / "data_prep" / "clustering" / "two_stage_clustering" / "200_tier_language_groupings.json",
    ]:
        if tier_file.exists():
            _require_json_object(tier_file, errors)

    ablation_script = repo_root / "scripts" / "comparison" / "run_multilingual_ablation.sh"
    if ablation_script.exists():
        _require_contains(ablation_script, "HYDRA_VARIANTS=(", errors)
        _require_contains(ablation_script, "COLA_VARIANTS=(", errors)
        _require_contains(ablation_script, "LORA_VARIANTS=(", errors)

    _validate_proof_pipeline_config(repo_root, errors)

    if errors:
        print("[FAIL] Repo config validation failed:")
        for err in errors:
            print(f" - {err}")
        return 1

    print("[OK] Repo config validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
