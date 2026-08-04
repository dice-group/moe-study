import argparse
import csv
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

PROFILE = "configs/pipeline/profiles/moe_study_expert_head_sweep_r2.generated.json"
ANALYSIS_ROOT = (
    "outputs/pipeline_runs/"
    "moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_ready/"
    "analysis/expert_status_tracker"
)
CHECKPOINT_AUDIT = "outputs/pipeline_runs/final_checkpoint_clean_audit/run_final_checkpoint_status.csv"
EXPERT_SELECTION_RUNS = (
    "outputs/pipeline_runs/"
    "moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_ready/"
    "analysis/expert_selection/expert_selection_run_scores.csv"
)

ROOTS = [
    "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_working_calibrated",
    "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_ready",
    "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_final_rescue_nosync",
    "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_deterministic_budget_filtered_save",
    "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_signal_3p6h",
    "outputs/pipeline_runs/moe_study_movlora_expert_rerun_sf0005",
    "outputs/pipeline_runs/moe_study_movlora_dense_expert_sweep_sf0005_calibrated",
    "outputs/pipeline_runs/moe_study_hmora_attention_calibrated",
    "outputs/pipeline_runs/moe_study_moelora_attention_sizefsdp_calibrated",
    "outputs/pipeline_runs/moe_study_hydralora_heads32_rerun",
    "outputs/pipeline_runs/moe_study_remaining_problematic_recovery",
]

TRUSTED_RERUN_ROOTS = {
    "moe_study_movlora_dense_expert_sweep_sf0005_calibrated",
    "moe_study_hmora_attention_calibrated",
    "moe_study_moelora_attention_sizefsdp_calibrated",
    "moe_study_hydralora_heads32_rerun",
}

DENSE_MOV_RUNS = [
    {"id": "mov_dense_e2", "kind": "movlora"},
    {"id": "mov_dense_e4", "kind": "movlora"},
    {"id": "mov_dense_e8", "kind": "movlora"},
    {"id": "mov_dense_e16", "kind": "movlora"},
    {"id": "mov_dense_e32", "kind": "movlora"},
]

PROBE_ROOTS = [
    "outputs/pipeline_runs/moe_study_adamole_fsdp_size_signal_probe",
    "outputs/pipeline_runs/moe_study_adamole_fsdp_size_e16_signal_probe",
    "outputs/pipeline_runs/moe_study_adamole_fsdp_size_e32_signal_probe",
    "outputs/pipeline_runs/moe_study_adamole_fsdp_size_e32_nogc_signal_probe",
]

EXPERT_RE = re.compile(r"_e(\d+)(?:_|$)")
TOPK_RE = re.compile(r"_k(\d+)(?:_|$)")
HEADS_RE = re.compile(r"_heads(\d+)$")
NUP_RE = re.compile(r"_nup(\d+)$")
COLA_B_RE = re.compile(r"_b(\d+)$")
CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)_adapter")
RESULT_TS_RE = re.compile(r"results_(.+)\.json$")


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_profile() -> tuple[str, list[dict[str, Any]]]:
    payload = read_json(REPO_ROOT / PROFILE)
    if not isinstance(payload, dict):
        raise SystemExit(f"Could not load profile {PROFILE}")
    name, profile = next(iter(payload.items()))
    runs = profile.get("training", {}).get("runs", [])
    profile_runs = [run for run in runs if isinstance(run, dict) and run.get("id")]
    profile_runs = [run for run in profile_runs if not str(run["id"]).startswith("mov_e")]
    profile_runs.extend(DENSE_MOV_RUNS)
    return str(name), profile_runs


def kind_from_id(run_id: str, run: dict[str, Any]) -> str:
    kind = str(run.get("kind") or "")
    if kind:
        return "mola" if run_id.startswith("adamoe_1") else kind
    if run_id.startswith("adamoe_1"):
        return "mola"
    if run_id.startswith("mov_"):
        return "movlora"
    return run_id.split("_", 1)[0]


def metadata(run: dict[str, Any]) -> dict[str, Any]:
    run_id = str(run["id"])
    expert_count = None
    axis = "experts"
    for regex, label in ((EXPERT_RE, "experts"), (HEADS_RE, "heads"), (NUP_RE, "up_projections"), (COLA_B_RE, "cola_B")):
        match = regex.search(run_id)
        if match:
            expert_count = int(match.group(1))
            axis = label
            break
    topk = TOPK_RE.search(run_id)
    return {
        "run_id": run_id,
        "kind": kind_from_id(run_id, run),
        "expert_count": expert_count if expert_count is not None else "",
        "top_k": int(topk.group(1)) if topk else "",
        "axis_label": axis,
    }


def task_family(task: str) -> str:
    if task.startswith("belebele_"):
        return "belebele"
    if task.startswith("flores_en_perplexity_"):
        return "flores"
    if task.startswith("sib200_"):
        return "sib200"
    if task.startswith("taxi1500_"):
        return "taxi1500"
    return ""


def result_timestamp(path: Path) -> str:
    match = RESULT_TS_RE.match(path.name)
    return match.group(1) if match else ""


def benchmark_payload_has_adapter(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return '"peft"' in text or "peft=" in text or "adapter_name_or_path" in text


def count_benchmark_tasks(root: Path, run_id: str) -> tuple[int, str, str, bool]:
    run_dir = root / "benchmarks" / run_id
    if not run_dir.exists():
        return 0, "", "", False

    combined: list[tuple[str, float, Path]] = []
    for path in run_dir.rglob("results_*.json"):
        combined.append((result_timestamp(path), path.stat().st_mtime, path))
    for _ts, _mtime, path in sorted(combined, reverse=True):
        data = read_json(path)
        results = data.get("results", {}) if isinstance(data, dict) else {}
        if isinstance(results, dict):
            tasks = {str(task) for task, metrics in results.items() if task_family(str(task)) and isinstance(metrics, dict)}
            if tasks:
                return len(tasks), str(path), "combined_results_json", benchmark_payload_has_adapter(path)

    tasks: set[str] = set()
    files = 0
    adapter_refs = 0
    for path in run_dir.rglob("*.json"):
        name = path.name
        if name.startswith("results_") or name.startswith("selected_tasks") or name in {"benchmark_summary.json", "summary.json"}:
            continue
        data = read_json(path)
        results = data.get("results", {}) if isinstance(data, dict) else {}
        if not isinstance(results, dict):
            continue
        before = len(tasks)
        tasks.update(str(task) for task, metrics in results.items() if task_family(str(task)) and isinstance(metrics, dict))
        if len(tasks) > before:
            files += 1
            adapter_refs += int(benchmark_payload_has_adapter(path))
    if tasks:
        return len(tasks), str(run_dir), f"per_task_json:{files}", adapter_refs == files
    return 0, "", "", False


def trainer_info(run_dir: Path) -> dict[str, Any]:
    state = read_json(run_dir / "trainer_state.json")
    results = read_json(run_dir / "train_results.json") or read_json(run_dir / "all_results.json")
    state = state if isinstance(state, dict) else {}
    results = results if isinstance(results, dict) else {}
    global_step = state.get("global_step", "")
    max_steps = state.get("max_steps", "")
    epoch = state.get("epoch", results.get("epoch", ""))
    runtime = results.get("train_runtime", "")
    complete_marker = (run_dir / ".training_complete").exists()
    completed_epoch = False
    try:
        completed_epoch = bool(int(global_step) == int(max_steps) and int(max_steps) > 0 and float(epoch) >= 1.0)
    except Exception:
        pass
    return {
        "has_train_results": (run_dir / "train_results.json").exists() or (run_dir / "all_results.json").exists(),
        "has_trainer_state": (run_dir / "trainer_state.json").exists(),
        "training_complete_marker": complete_marker,
        "trainer_global_step": global_step,
        "trainer_max_steps": max_steps,
        "trainer_epoch": epoch,
        "train_runtime_seconds": runtime,
        "completed_epoch": completed_epoch,
    }


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.search(str(path))
    return int(match.group(1)) if match else -1


def adapter_weight(path: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = path / name
        if candidate.exists():
            return candidate
    return None


def scan_root(root: Path, run_id: str) -> dict[str, Any] | None:
    run_dir = root / "runs" / run_id
    benchmark_tasks, benchmark_source, benchmark_payload_kind, benchmark_adapter_attached = count_benchmark_tasks(root, run_id)
    if not run_dir.exists() and benchmark_tasks == 0:
        return None

    info = trainer_info(run_dir) if run_dir.exists() else {}
    final_eval_dir = run_dir / "final_eval_loss"
    final_eval = (final_eval_dir / "eval_results.json").exists() or (final_eval_dir / "all_results.json").exists()
    adapters = sorted(run_dir.glob("checkpoint-*_adapter"), key=checkpoint_step) if run_dir.exists() else []
    if run_dir.exists() and (run_dir / "adapter_config.json").exists():
        adapters.append(run_dir)

    best_adapter = ""
    direct_marker = False
    adapter_exists = False
    adapter_size = ""
    if adapters:
        adapter = adapters[-1]
        weight = adapter_weight(adapter)
        adapter_exists = (adapter / "adapter_config.json").exists() and weight is not None
        best_adapter = str(adapter)
        direct_marker = (adapter / ".adapter_direct_save_complete").exists()
        adapter_size = weight.stat().st_size if weight else ""

    return {
        "root_name": root.name,
        "run_dir_exists": run_dir.exists(),
        **info,
        "adapter_exists": adapter_exists,
        "adapter_path": best_adapter,
        "adapter_weight_size": adapter_size,
        "direct_save_marker": direct_marker,
        "final_eval_done": final_eval,
        "benchmark_tasks": benchmark_tasks,
        "benchmark_source": benchmark_source,
        "benchmark_payload_kind": benchmark_payload_kind,
        "benchmark_adapter_attached": benchmark_adapter_attached,
    }


def load_checkpoint_audit() -> dict[str, dict[str, str]]:
    rows = [row for row in read_csv(REPO_ROOT / CHECKPOINT_AUDIT) if row.get("sweep") == "expert_head"]
    by_run: dict[str, dict[str, str]] = {}

    def score(row: dict[str, str]) -> tuple[int, float]:
        clean = row.get("clean_final_checkpoint") == "True"
        status = row.get("checkpoint_status", "")
        runtime = float(row.get("train_runtime_hours") or 0)
        priority = 0
        if clean and "signal" in status and runtime >= 3.4:
            priority = 5
        elif clean and "signal" in status:
            priority = 4
        elif status == "normal_epoch_final_save_not_signal" and runtime >= 3.24:
            priority = 3
        elif row.get("adapter_config_exists") == "True":
            priority = 2
        return priority, runtime

    for row in rows:
        run_id = row.get("run_id", "")
        if not run_id:
            continue
        if run_id not in by_run or score(row) > score(by_run[run_id]):
            by_run[run_id] = row
    return by_run


def load_analysis_scores() -> dict[str, dict[str, str]]:
    return {row["run_id"]: row for row in read_csv(REPO_ROOT / EXPERT_SELECTION_RUNS) if row.get("run_id")}


def squeue_jobs() -> dict[str, dict[str, str]]:
    try:
        proc = subprocess.run(
            ["squeue", "-u", "anonymous-user", "-h", "-o", "%i|%j|%T|%M|%D|%R"],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
            timeout=20,
        )
    except Exception:
        return {}
    if proc.returncode != 0:
        return {}
    jobs: dict[str, dict[str, str]] = {}
    for line in proc.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != 6:
            continue
        job_id, name, state, elapsed, nodes, reason = parts
        run_id = name
        if run_id.startswith("train10l_"):
            run_id = run_id[len("train10l_") :]
        jobs[run_id] = {
            "slurm_job_id": job_id,
            "slurm_job_name": name,
            "slurm_state": state,
            "slurm_elapsed": elapsed,
            "slurm_nodes": nodes,
            "slurm_reason": reason,
        }
    return jobs


def choose_observation(run_id: str) -> dict[str, Any]:
    observations = [scan_root(REPO_ROOT / root, run_id) for root in ROOTS]
    observations = [obs for obs in observations if obs]
    if not observations:
        return {}

    def obs_score(obs: dict[str, Any]) -> tuple[int, int, float]:
        benchmark_tasks = int(obs.get("benchmark_tasks") or 0)
        has_eval = bool(obs.get("final_eval_done"))
        has_marker = bool(obs.get("direct_save_marker"))
        has_adapter = bool(obs.get("adapter_exists"))
        runtime = float(obs.get("train_runtime_seconds") or 0)
        return (benchmark_tasks, int(has_eval) + int(has_marker) + int(has_adapter), runtime)

    return max(observations, key=obs_score)


def probe_note(run_id: str) -> str:
    if not run_id.startswith("adamole_r2_e"):
        return ""
    exp = run_id.rsplit("_e", 1)[-1]
    hits = []
    for root_name in PROBE_ROOTS:
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for path in (root / "runs").glob(f"*e{exp}*") if (root / "runs").exists() else []:
            if path.is_dir():
                hits.append(root.name)
    return "; ".join(hits)


def status_for(row: dict[str, Any]) -> tuple[str, str]:
    running = row.get("slurm_state") in {"RUNNING", "PENDING", "CONFIGURING"}
    checkpoint = row.get("checkpoint_clean_final")
    accepted_checkpoint = checkpoint in {"yes", "accepted_normal_final"}
    if running:
        return "running_recompute", "training currently active in SLURM"
    if checkpoint == "short_probe_not_valid":
        return "short_probe_not_valid", "checkpoint exists but runtime is far below the study budget"
    benchmark_tasks = int(row.get("benchmark_tasks") or 0)
    if 0 < benchmark_tasks < 42 and row.get("benchmark_adapter_attached") == "no":
        return "invalid_benchmark", "benchmark payload has no adapter reference; scores are not usable"
    if benchmark_tasks == 42 and accepted_checkpoint:
        return "complete_clean", "accepted final checkpoint plus full benchmark metrics"
    if benchmark_tasks > 0 and accepted_checkpoint:
        return "partial_benchmark", "clean checkpoint, but benchmark task set is partial"
    if row.get("final_eval_done") == "yes" and accepted_checkpoint:
        return "needs_benchmark", "training/final eval done; benchmark metrics missing"
    if accepted_checkpoint:
        return "needs_eval_benchmark", "clean training checkpoint exists; final eval/benchmark missing"
    if row.get("adapter_exists") == "yes":
        return "checkpoint_untrusted", "adapter exists but not proven clean final"
    if row.get("run_dir_exists") == "yes":
        return "failed_or_incomplete", "run directory exists but no usable final adapter"
    return "not_computed", "no official run output found in tracked roots"


def action_for(run_id: str, status: str, note: str) -> str:
    if status == "complete_clean":
        return "use"
    if status == "partial_benchmark":
        if run_id.startswith("cola_r2_a"):
            return "rerun_missing_flores_reverse_tasks"
        return "rerun_missing_benchmark_tasks"
    if status == "running_recompute":
        return "wait_for_training_signal_checkpoint_then_eval_benchmark"
    if status == "invalid_benchmark":
        return "rerun_benchmark_with_adapter"
    if status == "short_probe_not_valid":
        return "do_not_use_probe; rerun_full_budget_or_mark_infeasible"
    if run_id.startswith("mov_"):
        return "superseded_by_running_sf0005_rerun" if status != "complete_clean" else "use"
    if run_id.startswith("adamole_r2_e8"):
        return "official_rerun_with_size_fsdp_after_probe_success" if note else "fix_then_rerun"
    if run_id.startswith("adamole_r2_e16") or run_id.startswith("adamole_r2_e32"):
        return "mark_infeasible_current_a100_or_reduce_config"
    if status in {"needs_benchmark", "needs_eval_benchmark"}:
        return "submit_eval_benchmark"
    if status == "checkpoint_untrusted":
        return "do_not_use_until_clean_final_checkpoint"
    return "investigate_or_rerun"


def scored_task_count(score_row: dict[str, str], obs: dict[str, Any]) -> int:
    try:
        scored = int(score_row.get("num_scored_items") or 0)
    except ValueError:
        scored = 0
    return scored if scored > 0 else int(obs.get("benchmark_tasks") or 0)


def write_markdown(path: Path, rows: list[dict[str, Any]], profile_name: str) -> None:
    counts = Counter(row["status"] for row in rows)
    by_kind = defaultdict(Counter)
    for row in rows:
        by_kind[row["kind"]][row["status"]] += 1

    lines = [
        "# Expert Sweep Status Tracker",
        "",
        f"Profile: `{profile_name}`",
        f"Expected runs: `{len(rows)}`",
        "",
        "## Status Counts",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"| `{status}` | {count} |")
    lines.extend(["", "## By Approach", "", "| approach | complete_clean | partial_benchmark | running_recompute | needs_eval_benchmark | failed_or_incomplete | not_computed | other |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    known = {"complete_clean", "partial_benchmark", "running_recompute", "needs_eval_benchmark", "failed_or_incomplete", "not_computed"}
    for kind in sorted(by_kind):
        counter = by_kind[kind]
        other = sum(value for key, value in counter.items() if key not in known)
        lines.append(
            f"| `{kind}` | {counter['complete_clean']} | {counter['partial_benchmark']} | "
            f"{counter['running_recompute']} | {counter['needs_eval_benchmark']} | "
            f"{counter['failed_or_incomplete']} | {counter['not_computed']} | {other} |"
        )
    lines.extend(
        [
            "",
            "## Exact Matrix",
            "",
            "| run_id | kind | count | top_k | checkpoint | eval | benchmark | status | action |",
            "|---|---|---:|---:|---|---|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['run_id']}` | `{row['kind']}` | {row['expert_count']} | {row['top_k']} | "
            f"{row['checkpoint_clean_final']} | {row['final_eval_done']} | {row['benchmark_tasks']} | "
            f"`{row['status']}` | `{row['action']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=ANALYSIS_ROOT)
    args = parser.parse_args()

    profile_name, profile_runs = load_profile()
    audit = load_checkpoint_audit()
    scores = load_analysis_scores()
    jobs = squeue_jobs()
    rows: list[dict[str, Any]] = []

    for index, run in enumerate(profile_runs, start=1):
        meta = metadata(run)
        run_id = meta["run_id"]
        obs = choose_observation(run_id)
        audit_row = audit.get(run_id, {})
        score_row = scores.get(run_id, {})
        job = jobs.get(run_id, {})
        clean = audit_row.get("clean_final_checkpoint") == "True"
        checkpoint_status = audit_row.get("checkpoint_status", "")
        runtime_h = audit_row.get("train_runtime_hours", "")
        runtime_h_float = 0.0
        try:
            runtime_h_float = float(runtime_h or 0)
        except ValueError:
            runtime_h_float = 0.0
        obs_runtime_h = float(obs.get("train_runtime_seconds") or 0) / 3600 if obs.get("train_runtime_seconds") else 0.0
        observation_is_newer = obs_runtime_h > runtime_h_float
        effective_runtime_h = max(runtime_h_float, obs_runtime_h)
        long_enough = effective_runtime_h >= 3.24
        if obs.get("direct_save_marker") and obs.get("adapter_exists") and long_enough:
            clean_final = "yes"
        elif clean and checkpoint_status == "normal_epoch_final_save_not_signal" and long_enough:
            clean_final = "accepted_normal_final"
        elif clean and checkpoint_status == "normal_epoch_final_save_not_signal":
            clean_final = "short_probe_not_valid"
        elif clean and long_enough:
            clean_final = "yes"
        elif clean:
            clean_final = "short_probe_not_valid"
        elif (
            obs.get("root_name") in TRUSTED_RERUN_ROOTS
            and obs.get("adapter_exists")
            and obs.get("final_eval_done")
            and scored_task_count(score_row, obs) == 42
        ):
            clean_final = "yes"
        elif obs.get("direct_save_marker") and obs.get("adapter_exists"):
            clean_final = "yes" if long_enough else "short_probe_not_valid"
        else:
            clean_final = "no"

        benchmark_tasks = scored_task_count(score_row, obs)
        adapter_exists = obs.get("adapter_exists") or audit_row.get("adapter_config_exists") == "True"
        final_eval_done = obs.get("final_eval_done") or audit_row.get("final_eval_present") == "True"
        try:
            global_step_source = obs if observation_is_newer else audit_row
            fallback_source = audit_row if observation_is_newer else obs
            global_step = int(float(global_step_source.get("trainer_global_step", fallback_source.get("trainer_global_step", 0)) or 0))
            max_steps = int(float(global_step_source.get("trainer_max_steps", fallback_source.get("trainer_max_steps", 0)) or 0))
            trainer_epoch = float(global_step_source.get("trainer_epoch", fallback_source.get("trainer_epoch", 0)) or 0)
        except ValueError:
            global_step = 0
            max_steps = 0
            trainer_epoch = 0.0
        full_epoch_final_save = (
            checkpoint_status == "normal_epoch_final_save_not_signal"
            and max_steps > 0
            and global_step == max_steps
            and trainer_epoch >= 1.0
        )
        if (
            clean_final == "no"
            and full_epoch_final_save
            and adapter_exists
            and final_eval_done
            and benchmark_tasks == 42
        ):
            clean_final = "accepted_normal_final"
        if (
            obs.get("root_name") in TRUSTED_RERUN_ROOTS
            and obs.get("adapter_exists")
            and obs.get("final_eval_done")
            and benchmark_tasks == 42
        ):
            clean_final = "yes"
        note = probe_note(run_id)
        row = {
            "profile_index": index,
            **meta,
            "wanted": "yes",
            "root_name": obs.get("root_name", audit_row.get("root_name", "")),
            "run_dir_exists": "yes" if obs.get("run_dir_exists") else "no",
            "training_observed": "yes" if obs.get("has_train_results") or audit_row else "no",
            "training_complete_marker": "yes" if obs.get("training_complete_marker") else "no",
            "train_runtime_hours": round(effective_runtime_h, 4) if effective_runtime_h else "",
            "trainer_global_step": global_step or "",
            "trainer_max_steps": max_steps or "",
            "trainer_epoch": trainer_epoch or "",
            "adapter_exists": "yes" if adapter_exists else "no",
            "checkpoint_clean_final": clean_final,
            "checkpoint_status": "direct_save_marker" if observation_is_newer and obs.get("direct_save_marker") else checkpoint_status or ("direct_save_marker" if obs.get("direct_save_marker") else ""),
            "adapter_path": obs.get("adapter_path", "") if observation_is_newer else audit_row.get("adapter_path", obs.get("adapter_path", "")),
            "final_eval_done": "yes" if final_eval_done else "no",
            "benchmark_tasks": benchmark_tasks,
            "benchmark_adapter_attached": "yes" if obs.get("benchmark_adapter_attached") else ("no" if benchmark_tasks > 0 else ""),
            "benchmark_state": "full" if benchmark_tasks == 42 else ("partial" if benchmark_tasks > 0 else "missing"),
            "benchmark_source": score_row.get("result_json", obs.get("benchmark_source", "")),
            **job,
            "probe_note": note,
        }
        status, reason = status_for(row)
        row["status"] = status
        row["status_reason"] = reason
        row["action"] = action_for(run_id, status, note)
        rows.append(row)

    output_dir = REPO_ROOT / args.output_dir
    write_csv(output_dir / "expert_sweep_exact_status.csv", rows)
    write_markdown(output_dir / "expert_sweep_exact_status.md", rows, profile_name)

    summary_rows = []
    for kind in sorted({row["kind"] for row in rows}):
        subset = [row for row in rows if row["kind"] == kind]
        counter = Counter(row["status"] for row in subset)
        summary_rows.append({"kind": kind, "wanted": len(subset), **dict(sorted(counter.items()))})
    write_csv(output_dir / "expert_sweep_status_by_approach.csv", summary_rows)

    print(f"[expert_sweep_status_tracker] wrote {output_dir}")
    print(f"[expert_sweep_status_tracker] wanted={len(rows)} statuses={dict(sorted(Counter(row['status'] for row in rows).items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
