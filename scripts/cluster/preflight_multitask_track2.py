#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path


REQUIRED_MODULES = ("datasets", "transformers", "peft", "lm_eval", "unitxt")
EXPECTED_DATASETS_VERSION = "2.21.0"
MODEL_ID = "meta-llama/Llama-3.1-8B"
ASSET_PATH = Path("MTL-LoRA/commonsense_170k_taskid.json")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _check_imports() -> tuple[bool, list[dict[str, str]]]:
    ok = True
    rows: list[dict[str, str]] = []
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = str(getattr(module, "__version__", "<unknown>"))
            rows.append({"module": module_name, "status": "ok", "version": version})
        except Exception as exc:  # noqa: BLE001
            ok = False
            rows.append({"module": module_name, "status": "import_fail", "error": repr(exc)})
    return ok, rows


def _check_datasets_version() -> tuple[bool, str]:
    try:
        datasets = importlib.import_module("datasets")
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)
    version = str(getattr(datasets, "__version__", ""))
    return version == EXPECTED_DATASETS_VERSION, version


def _check_asset(repo_root: Path) -> tuple[bool, str]:
    path = repo_root / ASSET_PATH
    return path.exists(), str(path)


def _check_model_access() -> tuple[bool, str]:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(MODEL_ID)
        model_type = str(getattr(cfg, "model_type", "<unknown>"))
        return True, model_type
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


def main() -> int:
    repo_root = _repo_root()
    imports_ok, import_rows = _check_imports()
    datasets_ok, datasets_value = _check_datasets_version()
    asset_ok, asset_path = _check_asset(repo_root)
    model_ok, model_detail = _check_model_access()

    report = {
        "repo_root": str(repo_root),
        "required_asset": asset_path,
        "required_asset_exists": asset_ok,
        "imports_ok": imports_ok,
        "imports": import_rows,
        "datasets_version_expected": EXPECTED_DATASETS_VERSION,
        "datasets_version_ok": datasets_ok,
        "datasets_version_found": datasets_value,
        "model_id": MODEL_ID,
        "model_access_ok": model_ok,
        "model_access_detail": model_detail,
        "hf_home": os.environ.get("HF_HOME", ""),
        "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE", ""),
        "tmpdir": os.environ.get("TMPDIR", ""),
        "lm_eval_trust_remote_code": os.environ.get("LM_EVAL_TRUST_REMOTE_CODE", ""),
    }

    print(json.dumps(report, indent=2))

    failures: list[str] = []
    if not imports_ok:
        failures.append("required Python imports failed")
    if not datasets_ok:
        failures.append(f"datasets version must be {EXPECTED_DATASETS_VERSION}")
    if not asset_ok:
        failures.append("required task-id asset is missing")
    if not model_ok:
        failures.append("model access check failed")
    if os.environ.get("HF_HOME"):
        failures.append("HF_HOME is set; leave it unset for Track 2 cluster runs")

    if failures:
        for item in failures:
            print(f"[FAIL] {item}", file=sys.stderr)
        return 1

    print("[OK] Track 2 cluster preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
