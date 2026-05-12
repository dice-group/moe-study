import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

RANK_PROFILE = "configs/pipeline/profiles/moe_study_rank_expert_sweep_noctua2_4gpu_alltargets_calibrated.generated.json"
SIGNAL_PROFILE = "configs/pipeline/profiles/moe_study_expert_head_sweep_r2_noctua2_4gpu_signal_3p6h.generated.json"

OUTPUT_ROOTS = {
    "rank": [
        "outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated",
        "outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_rescue_kvdown",
        "outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_rescue_moelora_vonly",
        "outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_a100_4gpu_failed_recovery_full",
    ],
    "expert_head": [
        "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_working_calibrated",
        "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_ready",
        "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_final_rescue_nosync",
        "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_deterministic_budget_filtered_save",
        "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_signal_3p6h",
    ],
}

CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)_adapter")
SIGNAL_REQUEST_RE = re.compile(r"Signal: requesting checkpoint save at step=(\d+)")
ADAPTER_SAVED_RE = re.compile(r"Adapter checkpoint saved at: .*?checkpoint-(\d+)_adapter")
RUNTIME_HMS_RE = re.compile(r"train_runtime\s+=\s+(\d+):(\d+):(\d+(?:\.\d+)?)")
RUNTIME_FLOAT_RE = re.compile(r"'train_runtime':\s*([0-9]+(?:\.[0-9]+)?)")


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_profile_runs(path: str) -> tuple[str, list[dict[str, Any]]]:
    payload = _read_json(REPO_ROOT / path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Could not load profile: {path}")
    name, profile = next(iter(payload.items()))
    runs = profile.get("training", {}).get("runs", [])
    if not isinstance(runs, list):
        runs = []
    return str(name), [run for run in runs if isinstance(run, dict) and run.get("id")]


def _observed_run_ids(label: str) -> list[str]:
    ids: set[str] = set()
    for root_name in OUTPUT_ROOTS[label]:
        run_root = REPO_ROOT / root_name / "runs"
        if run_root.exists():
            ids.update(path.name for path in run_root.iterdir() if path.is_dir())
    return sorted(ids)


def _approach(run_id: str, run: dict[str, Any] | None = None) -> str:
    if run and run.get("kind"):
        kind = str(run["kind"])
        return "adamole" if kind == "adamoe" else kind
    prefixes = [
        ("adamoe", "adamole"),
        ("adamole", "adamole"),
        ("moelora", "moelora"),
        ("mola", "mola"),
        ("movlora", "movlora"),
        ("mov", "movlora"),
        ("hmora", "hmora"),
        ("hydralora", "hydralora"),
        ("mtllora", "mtllora"),
        ("cola", "cola"),
        ("lora", "lora"),
    ]
    for prefix, label in prefixes:
        if run_id.startswith(prefix):
            return label
    return run_id.split("_", 1)[0]


def _weight_file(adapter_dir: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = adapter_dir / name
        if path.exists():
            return path
    return None


def _checkpoint_step(path: Path | str) -> int | None:
    match = CHECKPOINT_RE.search(str(path))
    return int(match.group(1)) if match else None


def _run_log_files(root: Path, run_id: str) -> list[Path]:
    candidates: list[Path] = []
    train_dir = root / "logs" / "train_runs" / run_id
    if train_dir.exists():
        candidates.extend(sorted(train_dir.glob("*.log")))
    chain_dir = root / "logs" / "chain" / "slurm"
    if chain_dir.exists():
        candidates.extend(sorted(chain_dir.glob(f"chain_train.{run_id}*.slurm.log")))
    return [path for path in candidates if path.is_file()]


def _runtime_seconds(text: str) -> float | None:
    values: list[float] = []
    for hours, minutes, seconds in RUNTIME_HMS_RE.findall(text):
        values.append(int(hours) * 3600 + int(minutes) * 60 + float(seconds))
    values.extend(float(value) for value in RUNTIME_FLOAT_RE.findall(text))
    return max(values) if values else None


def _log_evidence(root: Path, run_id: str, step: int | None) -> dict[str, Any]:
    best = {
        "signal_received": False,
        "signal_checkpoint_saved": False,
        "signal_requested_step": "",
        "adapter_saved_after_signal": False,
        "log_path": "",
        "train_runtime_seconds": "",
    }
    if step is None:
        return best

    for log_path in _run_log_files(root, run_id):
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        requested_steps = {int(value) for value in SIGNAL_REQUEST_RE.findall(text)}
        adapter_steps = {int(value) for value in ADAPTER_SAVED_RE.findall(text)}
        signal_received = "SIGUSR1" in text or "SLURM_SIGNAL" in text
        signal_checkpoint_saved = "Signal: checkpoint saved" in text
        adapter_saved_after_signal = step in adapter_steps
        if signal_received and step in requested_steps and signal_checkpoint_saved and adapter_saved_after_signal:
            return {
                "signal_received": True,
                "signal_checkpoint_saved": True,
                "signal_requested_step": step,
                "adapter_saved_after_signal": True,
                "log_path": str(log_path),
                "train_runtime_seconds": _runtime_seconds(text) or "",
            }
        if signal_received and not best["signal_received"]:
            best.update(
                {
                    "signal_received": True,
                    "signal_checkpoint_saved": signal_checkpoint_saved,
                    "signal_requested_step": ",".join(str(value) for value in sorted(requested_steps)),
                    "adapter_saved_after_signal": adapter_saved_after_signal,
                    "log_path": str(log_path),
                    "train_runtime_seconds": _runtime_seconds(text) or "",
                }
            )
    return best


def _benchmark_adapter_paths(root: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    summary = _read_json(root / "benchmarks" / "benchmark_summary.json")
    if isinstance(summary, dict):
        for item in summary.get("runs", []):
            if not isinstance(item, dict):
                continue
            run_id = str(item.get("run_id", "")).strip()
            adapter = str(item.get("evaluated_checkpoint", "")).strip()
            if run_id and adapter:
                found[run_id] = adapter
    return found


def _final_eval_adapter_paths(root: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    summary = _read_json(root / "final_eval_loss" / "summary.json")
    if isinstance(summary, dict):
        for item in summary.get("runs", []):
            if not isinstance(item, dict):
                continue
            run_id = str(item.get("run_id", "")).strip()
            adapter = str(item.get("adapter_name_or_path", "")).strip()
            if run_id and adapter:
                found[run_id] = adapter
    return found


def _trainer_completion(run_dir: Path) -> dict[str, Any]:
    state = _read_json(run_dir / "trainer_state.json")
    results = _read_json(run_dir / "train_results.json") or _read_json(run_dir / "all_results.json")
    state = state if isinstance(state, dict) else {}
    results = results if isinstance(results, dict) else {}
    global_step = state.get("global_step")
    max_steps = state.get("max_steps")
    epoch = state.get("epoch", results.get("epoch"))
    runtime = results.get("train_runtime")
    try:
        completed = int(global_step) == int(max_steps) and int(max_steps) > 0 and float(epoch) >= 1.0
    except (TypeError, ValueError):
        completed = False
    return {
        "trainer_global_step": global_step if global_step is not None else "",
        "trainer_max_steps": max_steps if max_steps is not None else "",
        "trainer_epoch": epoch if epoch is not None else "",
        "trainer_completed_epoch": completed,
        "trainer_train_runtime_seconds": runtime if runtime is not None else "",
    }


def _candidate_rows(label: str, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root_name in OUTPUT_ROOTS[label]:
        root = REPO_ROOT / root_name
        run_dir = root / "runs" / run_id
        if not run_dir.exists():
            continue

        bench_paths = _benchmark_adapter_paths(root)
        eval_paths = _final_eval_adapter_paths(root)
        benchmark_dir_exists = (root / "benchmarks" / run_id).is_dir()
        final_eval_present = (root / "final_eval_loss" / f"{run_id}.json").exists() or run_id in eval_paths

        adapter_dirs = sorted(run_dir.glob("checkpoint-*_adapter"), key=lambda path: _checkpoint_step(path) or -1)
        if (run_dir / "adapter_config.json").exists():
            adapter_dirs.append(run_dir)

        for adapter_dir in adapter_dirs:
            step = _checkpoint_step(adapter_dir)
            weight = _weight_file(adapter_dir)
            marker = (adapter_dir / ".adapter_direct_save_complete").exists()
            log = _log_evidence(root, run_id, step)
            trainer = _trainer_completion(run_dir)
            runtime_seconds = log["train_runtime_seconds"] or trainer["trainer_train_runtime_seconds"]
            basic_ok = (adapter_dir / "adapter_config.json").exists() and weight is not None
            clean = bool(basic_ok and (marker or log["adapter_saved_after_signal"]))
            if clean and marker:
                status = "clean_signal_final_marker"
            elif clean:
                status = "clean_signal_final_log"
            elif basic_ok and step is None and trainer["trainer_completed_epoch"]:
                status = "normal_epoch_final_save_not_signal"
            elif basic_ok and "recovery" in root.name.lower():
                status = "adapter_exists_recovery_unproven"
            elif basic_ok:
                status = "adapter_exists_no_signal_proof"
            else:
                status = "missing_or_incomplete_adapter"
            rows.append(
                {
                    "sweep": label,
                    "root_name": root.name,
                    "run_id": run_id,
                    "adapter_path": str(adapter_dir),
                    "checkpoint_step": step if step is not None else "",
                    "adapter_config_exists": (adapter_dir / "adapter_config.json").exists(),
                    "adapter_weight_file": str(weight) if weight else "",
                    "adapter_weight_size": weight.stat().st_size if weight else "",
                    "direct_save_marker_exists": marker,
                    "signal_received": log["signal_received"],
                    "signal_checkpoint_saved": log["signal_checkpoint_saved"],
                    "signal_requested_step": log["signal_requested_step"],
                    "adapter_saved_after_signal": log["adapter_saved_after_signal"],
                    "train_runtime_seconds": runtime_seconds,
                    "train_runtime_hours": round(float(runtime_seconds) / 3600, 4) if runtime_seconds else "",
                    "trainer_global_step": trainer["trainer_global_step"],
                    "trainer_max_steps": trainer["trainer_max_steps"],
                    "trainer_epoch": trainer["trainer_epoch"],
                    "trainer_completed_epoch": trainer["trainer_completed_epoch"],
                    "log_path": log["log_path"],
                    "benchmark_dir_exists": benchmark_dir_exists,
                    "benchmark_adapter_path": bench_paths.get(run_id, ""),
                    "final_eval_present": final_eval_present,
                    "final_eval_adapter_path": eval_paths.get(run_id, ""),
                    "clean_final_checkpoint": clean,
                    "checkpoint_status": status,
                }
            )
    return rows


def _candidate_score(row: dict[str, Any]) -> tuple[int, int, int]:
    status_score = {
        "clean_signal_final_marker": 5,
        "clean_signal_final_log": 4,
        "normal_epoch_final_save_not_signal": 3,
        "adapter_exists_no_signal_proof": 2,
        "adapter_exists_recovery_unproven": 1,
        "missing_or_incomplete_adapter": 0,
    }.get(str(row["checkpoint_status"]), 0)
    bench_score = 1 if row.get("benchmark_dir_exists") else 0
    step = int(row["checkpoint_step"]) if str(row.get("checkpoint_step", "")).isdigit() else -1
    return status_score, bench_score, step


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Final Checkpoint Clean Audit",
        "",
        f"Created: {summary['created_at']}",
        "",
        "A clean final checkpoint means adapter files exist and either the new direct-save marker is present, or logs prove that SLURM/SIGUSR1 requested the same checkpoint step and the adapter was saved after the signal.",
        "",
        "## Summary",
        "",
        "| sweep | approach | total | clean | unproven | missing |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["by_sweep_approach"]:
        lines.append(
            f"| {item['sweep']} | {item['approach']} | {item['total']} | {item['clean']} | {item['unproven']} | {item['missing']} |"
        )
    lines.extend(
        [
            "",
            "## Unclean Or Unproven Runs",
            "",
            "| sweep | approach | run_id | status | root | checkpoint | runtime_h |",
            "|---|---|---|---|---|---:|---:|",
        ]
    )
    for row in rows:
        if row["clean_final_checkpoint"]:
            continue
        lines.append(
            f"| {row['sweep']} | {row['approach']} | `{row['run_id']}` | {row['checkpoint_status']} | {row['root_name']} | {row['checkpoint_step']} | {row['train_runtime_hours']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rank_profile_name, rank_runs = _load_profile_runs(RANK_PROFILE)
    signal_profile_name, signal_runs = _load_profile_runs(SIGNAL_PROFILE)
    del signal_profile_name, signal_runs

    rank_run_map = {str(run["id"]): run for run in rank_runs}
    expected = {
        "rank": sorted(rank_run_map),
        "expert_head": _observed_run_ids("expert_head"),
    }

    all_attempts: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for sweep, run_ids in expected.items():
        for run_id in run_ids:
            run = rank_run_map.get(run_id) if sweep == "rank" else None
            attempts = _candidate_rows(sweep, run_id)
            all_attempts.extend(attempts)
            if attempts:
                selected = max(attempts, key=_candidate_score)
            else:
                selected = {
                    "sweep": sweep,
                    "root_name": "",
                    "run_id": run_id,
                    "adapter_path": "",
                    "checkpoint_step": "",
                    "adapter_config_exists": False,
                    "adapter_weight_file": "",
                    "adapter_weight_size": "",
                    "direct_save_marker_exists": False,
                    "signal_received": False,
                    "signal_checkpoint_saved": False,
                    "signal_requested_step": "",
        "adapter_saved_after_signal": False,
        "train_runtime_seconds": "",
        "train_runtime_hours": "",
        "trainer_global_step": "",
        "trainer_max_steps": "",
        "trainer_epoch": "",
        "trainer_completed_epoch": False,
        "log_path": "",
                    "benchmark_dir_exists": False,
                    "benchmark_adapter_path": "",
                    "final_eval_present": False,
                    "final_eval_adapter_path": "",
                    "clean_final_checkpoint": False,
                    "checkpoint_status": "missing_run_or_adapter",
                }
            selected = dict(selected)
            selected["approach"] = _approach(run_id, run)
            selected_rows.append(selected)

    by_sweep_approach: list[dict[str, Any]] = []
    for sweep in sorted({row["sweep"] for row in selected_rows}):
        approaches = sorted({row["approach"] for row in selected_rows if row["sweep"] == sweep})
        for approach in approaches:
            group = [row for row in selected_rows if row["sweep"] == sweep and row["approach"] == approach]
            clean = sum(1 for row in group if row["clean_final_checkpoint"])
            missing = sum(1 for row in group if row["checkpoint_status"] == "missing_run_or_adapter")
            by_sweep_approach.append(
                {
                    "sweep": sweep,
                    "approach": approach,
                    "total": len(group),
                    "clean": clean,
                    "unproven": len(group) - clean - missing,
                    "missing": missing,
                }
            )

    out_dir = REPO_ROOT / "outputs/pipeline_runs/final_checkpoint_clean_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rank_profile": rank_profile_name,
        "rank_runs": len(expected["rank"]),
        "expert_head_source": "observed output roots",
        "expert_head_runs": len(expected["expert_head"]),
        "overall": {
            "total": len(selected_rows),
            "clean": sum(1 for row in selected_rows if row["clean_final_checkpoint"]),
            "unclean_or_unproven": sum(1 for row in selected_rows if not row["clean_final_checkpoint"]),
        },
        "by_sweep_approach": by_sweep_approach,
    }
    _write_csv(out_dir / "checkpoint_attempts.csv", all_attempts)
    _write_csv(out_dir / "run_final_checkpoint_status.csv", selected_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_markdown(out_dir / "README.md", selected_rows, summary)

    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    print(out_dir / "run_final_checkpoint_status.csv")
    print(out_dir / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
