import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config_loader import load_pipeline_config
from pipeline.runtime import _deep_merge, _resolve_profile
from pipeline.training import adapter_checkpoint_provenance


DEFAULT_ROOTS = [
    "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_final_rescue_nosync",
    "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_ready",
    "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_working_calibrated",
    "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_deterministic_budget_filtered_save",
    "outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_a100_4gpu_failed_recovery_full",
]
DEFAULT_PROFILE = "moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_deterministic_budget_filtered_save"
CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)_adapter")


def _load_expected_steps(profile_name: str | None) -> dict[str, int]:
    if not profile_name:
        return {}
    config = load_pipeline_config(REPO_ROOT / "configs/pipeline")
    profile = _resolve_profile(config, profile_name)
    common_train = profile.get("training", {}).get("train", {})
    expected: dict[str, int] = {}
    for run in profile.get("training", {}).get("runs", []):
        if not isinstance(run, dict) or "id" not in run:
            continue
        train = _deep_merge(common_train, run.get("train", {}))
        try:
            steps = int(train.get("max_steps"))
        except (TypeError, ValueError):
            continue
        if steps > 0:
            expected[str(run["id"])] = steps
    return expected


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _recovered_run_ids(root: Path) -> set[str]:
    recovered: set[str] = set()
    for name in ("recovery_summary.json", "recovery_probe_summary.json"):
        payload = _read_json(root / name)
        rows = payload if isinstance(payload, list) else []
        for row in rows:
            if isinstance(row, dict) and str(row.get("status", "")).lower() == "recovered":
                run_id = str(row.get("run_id", "")).strip()
                if run_id:
                    recovered.add(run_id)
    return recovered


def _find_peft_paths(payload: Any) -> list[str]:
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "peft" and isinstance(value, str):
                found.append(value)
            elif isinstance(value, str) and "checkpoint-" in value and "_adapter" in value:
                found.append(value)
            else:
                found.extend(_find_peft_paths(value))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_find_peft_paths(item))
    return found


def _latest_result_adapter_paths(run_benchmark_dir: Path) -> list[str]:
    paths: list[str] = []
    for result_path in sorted(run_benchmark_dir.rglob("results_*.json")):
        payload = _read_json(result_path)
        if payload is None:
            continue
        paths.extend(_find_peft_paths(payload))
    return sorted(set(paths))


def _checkpoint_step(path: str) -> int | None:
    match = CHECKPOINT_RE.search(path)
    return int(match.group(1)) if match else None


def _classify(
    *,
    root: Path,
    run_id: str,
    stage: str,
    adapter_path: str,
    expected_step: int | None,
    recovered_run_ids: set[str],
    provenance: dict[str, Any] | None,
) -> str:
    root_name = root.name.lower()
    if run_id in recovered_run_ids or "recovery" in root_name or "final_rescue" in root_name:
        return "invalid_recovered"
    if provenance is None:
        return "stale_or_unknown"
    if expected_step is not None and provenance.get("valid") is True:
        return "valid_final"
    if expected_step is not None and provenance.get("valid") is False:
        return "invalid_wrong_checkpoint"
    if stage in {"benchmark", "final_eval_loss"} and not adapter_path:
        return "stale_or_unknown"
    return "stale_or_unknown"


def _row(
    *,
    root: Path,
    run_id: str,
    stage: str,
    adapter_path: str,
    expected_step: int | None,
    recovered_run_ids: set[str],
) -> dict[str, Any]:
    adapter = Path(adapter_path) if adapter_path else None
    provenance = None
    if adapter is not None and adapter.exists():
        provenance = adapter_checkpoint_provenance(
            adapter,
            expected_step=expected_step,
            require_direct_save_marker=expected_step is not None,
        )
    checkpoint_step = provenance.get("checkpoint_step") if provenance else _checkpoint_step(adapter_path)
    errors = provenance.get("errors", []) if provenance else []
    warnings = provenance.get("warnings", []) if provenance else []
    return {
        "root": str(root),
        "root_name": root.name,
        "run_id": run_id,
        "stage": stage,
        "adapter_path": adapter_path,
        "checkpoint_step": checkpoint_step,
        "expected_step": expected_step,
        "validity_class": _classify(
            root=root,
            run_id=run_id,
            stage=stage,
            adapter_path=adapter_path,
            expected_step=expected_step,
            recovered_run_ids=recovered_run_ids,
            provenance=provenance,
        ),
        "adapter_config_exists": provenance.get("adapter_config_exists") if provenance else "",
        "adapter_weight_file": provenance.get("adapter_weight_file") if provenance else "",
        "adapter_weight_size": provenance.get("adapter_weight_size") if provenance else "",
        "direct_save_marker_exists": provenance.get("direct_save_marker_exists") if provenance else "",
        "num_adapter_keys": provenance.get("num_adapter_keys") if provenance else "",
        "dirty_key_count": provenance.get("dirty_key_count") if provenance else "",
        "errors": "; ".join(str(item) for item in errors),
        "warnings": "; ".join(str(item) for item in warnings),
    }


def _collect_root_rows(root: Path, expected_steps: dict[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recovered = _recovered_run_ids(root)

    summary = _read_json(root / "final_eval_loss" / "summary.json")
    if isinstance(summary, dict):
        for item in summary.get("runs", []):
            if not isinstance(item, dict):
                continue
            run_id = str(item.get("run_id", "")).strip()
            adapter_path = str(item.get("adapter_name_or_path", "")).strip()
            rows.append(
                _row(
                    root=root,
                    run_id=run_id,
                    stage="final_eval_loss",
                    adapter_path=adapter_path,
                    expected_step=expected_steps.get(run_id),
                    recovered_run_ids=recovered,
                )
            )

    bench_summary = _read_json(root / "benchmarks" / "benchmark_summary.json")
    if isinstance(bench_summary, dict):
        for item in bench_summary.get("runs", []):
            if not isinstance(item, dict):
                continue
            run_id = str(item.get("run_id", "")).strip()
            adapter_path = str(item.get("evaluated_checkpoint", "")).strip()
            rows.append(
                _row(
                    root=root,
                    run_id=run_id,
                    stage="benchmark",
                    adapter_path=adapter_path,
                    expected_step=expected_steps.get(run_id),
                    recovered_run_ids=recovered,
                )
            )

    benchmark_root = root / "benchmarks"
    if benchmark_root.exists():
        for run_dir in sorted(path for path in benchmark_root.iterdir() if path.is_dir()):
            if any(row["stage"] == "benchmark" and row["run_id"] == run_dir.name for row in rows):
                continue
            for adapter_path in _latest_result_adapter_paths(run_dir):
                rows.append(
                    _row(
                        root=root,
                        run_id=run_dir.name,
                        stage="benchmark_result_scan",
                        adapter_path=adapter_path,
                        expected_step=expected_steps.get(run_dir.name),
                        recovered_run_ids=recovered,
                    )
                )

    runs_root = root / "runs"
    if runs_root.exists():
        for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            run_id = run_dir.name
            for adapter_dir in sorted(run_dir.glob("checkpoint-*_adapter")):
                rows.append(
                    _row(
                        root=root,
                        run_id=run_id,
                        stage="adapter_inventory",
                        adapter_path=str(adapter_dir),
                        expected_step=expected_steps.get(run_id),
                        recovered_run_ids=recovered,
                    )
                )

    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "root_name",
        "run_id",
        "stage",
        "validity_class",
        "checkpoint_step",
        "expected_step",
        "direct_save_marker_exists",
        "dirty_key_count",
        "adapter_weight_size",
        "adapter_path",
        "errors",
        "warnings",
        "root",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit checkpoint provenance for pipeline eval/benchmark outputs.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--root", action="append", dest="roots", default=[])
    parser.add_argument("--output-dir", default="outputs/pipeline_runs/checkpoint_provenance_audit")
    args = parser.parse_args()

    expected_steps = _load_expected_steps(args.profile)
    roots = [REPO_ROOT / root for root in (args.roots or DEFAULT_ROOTS)]
    rows: list[dict[str, Any]] = []
    for root in roots:
        if root.exists():
            rows.extend(_collect_root_rows(root, expected_steps))

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "profile": args.profile,
        "roots": [str(root) for root in roots],
        "summary": {
            key: sum(1 for row in rows if row["validity_class"] == key)
            for key in sorted({str(row["validity_class"]) for row in rows})
        },
        "rows": rows,
    }
    (out_dir / "checkpoint_provenance_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_dir / "checkpoint_provenance_audit.csv", rows)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(out_dir / "checkpoint_provenance_audit.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
