#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path


REQUIRED_MODULES = ("datasets", "transformers", "peft", "lm_eval")
MODEL_ID = "meta-llama/Llama-3.1-8B"
ASSET_PATH = Path("MTL-LoRA/commonsense_170k_taskid.json")
TRACK2_MANIFEST_PATH = Path("configs/pipeline/benchmark_manifests/track2_glue_commonsense.json")
TRACK2_INCLUDE_PATH = Path("configs/pipeline/lm_eval_tasks/track2")


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


def _datasets_version() -> str:
    try:
        datasets = importlib.import_module("datasets")
    except Exception as exc:  # noqa: BLE001
        return repr(exc)
    return str(getattr(datasets, "__version__", ""))


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


def _check_track2_benchmark_tasks(repo_root: Path) -> tuple[bool, list[dict[str, str]]]:
    try:
        from lm_eval.tasks import TaskManager
    except Exception as exc:  # noqa: BLE001
        return False, [{"task": "<task-manager>", "status": "import_fail", "error": repr(exc)}]

    manifest = json.loads((repo_root / TRACK2_MANIFEST_PATH).read_text(encoding="utf-8"))
    include_path = repo_root / TRACK2_INCLUDE_PATH
    task_manager = TaskManager(include_path=str(include_path))
    rows: list[dict[str, str]] = []
    ok = True
    for task_cfg in manifest.get("tasks", []):
        if not isinstance(task_cfg, dict):
            continue
        task_name = str(task_cfg.get("lm_eval_task", task_cfg.get("task_name", ""))).strip()
        if not task_name:
            continue
        try:
            task_manager.load_task_or_group([task_name])
            rows.append({"task": task_name, "status": "ok"})
        except Exception as exc:  # noqa: BLE001
            ok = False
            rows.append({"task": task_name, "status": "load_fail", "error": repr(exc)})
    return ok, rows


def main() -> int:
    repo_root = _repo_root()
    imports_ok, import_rows = _check_imports()
    datasets_value = _datasets_version()
    asset_ok, asset_path = _check_asset(repo_root)
    model_ok, model_detail = _check_model_access()
    benchmark_tasks_ok, benchmark_task_rows = _check_track2_benchmark_tasks(repo_root)

    report = {
        "repo_root": str(repo_root),
        "required_asset": asset_path,
        "required_asset_exists": asset_ok,
        "imports_ok": imports_ok,
        "imports": import_rows,
        "datasets_version_found": datasets_value,
        "model_id": MODEL_ID,
        "model_access_ok": model_ok,
        "model_access_detail": model_detail,
        "track2_task_manifest": str((repo_root / TRACK2_MANIFEST_PATH).resolve()),
        "track2_include_path": str((repo_root / TRACK2_INCLUDE_PATH).resolve()),
        "track2_benchmark_tasks_ok": benchmark_tasks_ok,
        "track2_benchmark_tasks": benchmark_task_rows,
        "hf_home": os.environ.get("HF_HOME", ""),
        "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE", ""),
        "tmpdir": os.environ.get("TMPDIR", ""),
        "lm_eval_trust_remote_code": os.environ.get("LM_EVAL_TRUST_REMOTE_CODE", ""),
    }

    print(json.dumps(report, indent=2))

    failures: list[str] = []
    if not imports_ok:
        failures.append("required Python imports failed")
    if not asset_ok:
        failures.append("required task-id asset is missing")
    if not model_ok:
        failures.append("model access check failed")
    if not benchmark_tasks_ok:
        failures.append("track2 benchmark task loading failed")
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
