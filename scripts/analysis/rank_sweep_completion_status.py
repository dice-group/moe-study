import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROFILE = "moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated"
DEFAULT_ROOTS = [
    "outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated",
    "outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_rescue_kvdown",
    "outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_rescue_moelora_vonly",
    "outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_a100_4gpu_failed_recovery_full",
]


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _load_profile(config_dir: Path, profile_name: str) -> dict[str, Any]:
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from pipeline.config_loader import load_pipeline_config
    from pipeline.runtime import _resolve_profile

    config = load_pipeline_config(config_dir)
    return _resolve_profile(config, profile_name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _latest_adapter(run_id: str, roots: list[Path]) -> dict[str, Any]:
    candidates: list[tuple[int, Path, Path]] = []
    for root in roots:
        run_dir = root / "runs" / run_id
        for adapter in run_dir.glob("checkpoint-*_adapter"):
            candidates.append((_checkpoint_step(adapter), root, adapter))
    if not candidates:
        return {}
    step, root, adapter = max(candidates, key=lambda item: item[0])
    state_path = adapter.with_name(adapter.name.removesuffix("_adapter")) / "trainer_state.json"
    state = _read_json(state_path)
    return {
        "train_root": root.name,
        "adapter_path": str(adapter),
        "global_step": state.get("global_step", step),
        "max_steps": state.get("max_steps", ""),
        "epoch": state.get("epoch", ""),
    }


def _selected_tasks(root: Path, run_id: str) -> set[str]:
    path = root / "benchmarks" / f"selected_tasks.{run_id}.txt"
    if not path.exists():
        path = root / "benchmarks" / "selected_tasks.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}


def _tasks_seen_from_json(path: Path) -> set[str]:
    data = _read_json(path)
    results = data.get("results")
    if isinstance(results, dict):
        return {str(key) for key in results}
    return set()


def _tasks_seen_from_jsonl(path: Path) -> set[str]:
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if isinstance(data, dict):
                    results = data.get("results")
                    if isinstance(results, dict):
                        seen.update(str(key) for key in results)
                    task = data.get("task") or data.get("task_name")
                    if task:
                        seen.add(str(task))
    except Exception:
        return seen
    return seen


def _eval_status(run_id: str, roots: list[Path]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for root in roots:
        run_dir = root / "benchmarks" / run_id
        if not run_dir.exists():
            continue
        result_files = [
            path
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path.suffix in {".json", ".jsonl"} and not path.name.startswith("selected_tasks")
        ]
        if not result_files:
            best = best or {"eval_root": root.name, "eval_status": "partial", "eval_output_files": 0}
            continue
        selected = _selected_tasks(root, run_id)
        seen: set[str] = set()
        for path in result_files:
            if path.suffix == ".json":
                seen.update(_tasks_seen_from_json(path))
            elif path.suffix == ".jsonl":
                seen.update(_tasks_seen_from_jsonl(path))
        missing = sorted(selected - seen) if selected and seen else []
        status = "done"
        if selected and seen and missing:
            status = "partial"
        return {
            "eval_root": root.name,
            "eval_status": status,
            "eval_output_files": len(result_files),
            "eval_tasks_expected": len(selected),
            "eval_tasks_seen": len(seen),
            "eval_missing_tasks": ";".join(missing),
        }
    return best or {"eval_status": "missing"}


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], csv_path: Path, profile_name: str) -> None:
    total = len(rows)
    train_done = sum(1 for row in rows if row["train_status"] == "done")
    eval_done = sum(1 for row in rows if row["eval_status"] == "done")
    missing_eval = [row for row in rows if row["eval_status"] != "done"]
    partial_train = [
        row
        for row in rows
        if str(row.get("max_steps", "")).strip()
        and str(row.get("global_step", "")).strip()
        and int(float(row["global_step"])) < int(float(row["max_steps"]))
    ]
    lines = [
        "# anonymous-cluster Rank Sweep Status",
        "",
        f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        f"Profile: `{profile_name}`",
        f"CSV: `{csv_path}`",
        "",
        "## Summary",
        "",
        f"- Intended runs: {total}",
        f"- Training adapters present: {train_done}/{total}",
        f"- Eval outputs complete: {eval_done}/{total}",
        f"- Eval outputs missing or partial: {len(missing_eval)}",
        "",
        "## Missing Or Partial Eval",
        "",
    ]
    if missing_eval:
        lines.extend(["| Run ID | Kind | Eval status | Adapter root |", "|---|---|---|---|"])
        for row in missing_eval:
            lines.append(f"| `{row['run_id']}` | `{row['kind']}` | {row['eval_status']} | `{row.get('train_root', '')}` |")
    else:
        lines.append("All intended eval outputs are present.")
    lines.extend(["", "## Timeboxed Training Checkpoints", ""])
    if partial_train:
        lines.extend(["| Run ID | Kind | Step | Max steps | Epoch | Root |", "|---|---|---:|---:|---:|---|"])
        for row in partial_train:
            lines.append(
                f"| `{row['run_id']}` | `{row['kind']}` | {row['global_step']} | {row['max_steps']} | {row['epoch']} | `{row['train_root']}` |"
            )
    else:
        lines.append("No trainer_state file reports a checkpoint below configured max_steps.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write completion status for the anonymous-cluster rank/expert sweep.")
    parser.add_argument("--config", default="configs/pipeline", help="Pipeline config bundle path.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Resolved profile name for the 55-run study matrix.")
    parser.add_argument("--root", action="append", default=[], help="Output root to scan. May be repeated.")
    parser.add_argument(
        "--csv",
        default="outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated/rank_sweep_completion_status.csv",
        help="CSV output path.",
    )
    parser.add_argument("--markdown", default="docs/noctua2_rank_sweep_status.md", help="Markdown status output path.")
    args = parser.parse_args()

    profile = _load_profile(REPO_ROOT / args.config, args.profile)
    roots = [REPO_ROOT / value for value in (args.root or DEFAULT_ROOTS)]
    fields = [
        "run_id",
        "kind",
        "train_status",
        "train_root",
        "adapter_path",
        "global_step",
        "max_steps",
        "epoch",
        "eval_status",
        "eval_root",
        "eval_output_files",
        "eval_tasks_expected",
        "eval_tasks_seen",
        "eval_missing_tasks",
    ]
    rows: list[dict[str, Any]] = []
    for run in profile["training"]["runs"]:
        run_id = str(run["id"])
        train = _latest_adapter(run_id, roots)
        eval_info = _eval_status(run_id, roots)
        rows.append(
            {
                "run_id": run_id,
                "kind": str(run.get("kind", "")),
                "train_status": "done" if train else "missing",
                "train_root": train.get("train_root", ""),
                "adapter_path": train.get("adapter_path", ""),
                "global_step": train.get("global_step", ""),
                "max_steps": train.get("max_steps", ""),
                "epoch": train.get("epoch", ""),
                "eval_status": eval_info.get("eval_status", "missing"),
                "eval_root": eval_info.get("eval_root", ""),
                "eval_output_files": eval_info.get("eval_output_files", ""),
                "eval_tasks_expected": eval_info.get("eval_tasks_expected", ""),
                "eval_tasks_seen": eval_info.get("eval_tasks_seen", ""),
                "eval_missing_tasks": eval_info.get("eval_missing_tasks", ""),
            }
        )

    csv_path = REPO_ROOT / args.csv
    md_path = REPO_ROOT / args.markdown
    _write_csv(csv_path, rows, fields)
    _write_markdown(md_path, rows, csv_path.relative_to(REPO_ROOT), args.profile)
    print(f"[rank_sweep_completion_status] wrote {csv_path}")
    print(f"[rank_sweep_completion_status] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
