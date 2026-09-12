#!/usr/bin/env python3
"""Fail-closed completion audit for the promoted 96/200-language macro-3 campaign."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SELECTED = ("mtllora_r8_nup16", "moelora_tr32_e16_k8", "hydralora_r2")
REQUIRED_REPORT_SHEETS = {
    "benchmark_compact",
    "data_overview",
    "run_configs",
    "pipeline_profile",
    "benchmark_raw",
    "experiment_meta",
}
EXPECTED_STAGE_COUNTS = {
    "train": 3,
    "train_acceptance": 3,
    "final_eval_loss": 3,
    "benchmark": 3,
    "summary": 1,
    "report": 1,
}
CAMPAIGNS = {
    "96": {
        "profile": "moe_study_final_macro3_noctua2_a100_4gpu_96_langs_34p5h_v1",
        "output_root": ROOT
        / "outputs/pipeline_runs/moe_study_final_macro3_noctua2_a100_4gpu_96_langs_34p5h_v1",
        "graph": "logs/chain/job_graph_20260720_085626.json",
        "snapshot": "logs/chain/job_graph_20260720_085626.spool_verified.json",
        "task_count": 183,
        "task_breakdown": {"Belebele": 34, "FLORES": 92, "SIB200": 34, "Taxi1500": 23},
        "target_runtime": 122400,
    },
    "200": {
        "profile": "moe_study_final_macro3_noctua2_a100_4gpu_200_langs_72h_v1",
        "output_root": ROOT
        / "outputs/pipeline_runs/moe_study_final_macro3_noctua2_a100_4gpu_200_langs_72h_v1",
        "graph": "logs/chain/job_graph_20260720_085644.json",
        "snapshot": "logs/chain/job_graph_20260720_085644.spool_verified.json",
        "amendment": "logs/chain/scheduler_amendment_20260720_090707.json",
        "task_count": 306,
        "task_breakdown": {"Belebele": 57, "FLORES": 156, "SIB200": 57, "Taxi1500": 36},
        "target_runtime": 259200,
    },
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def slurm_states(job_ids: list[str]) -> dict[str, dict[str, str]]:
    result = subprocess.run(
        [
            "sacct",
            "-j",
            ",".join(job_ids),
            "-X",
            "-n",
            "-P",
            "-o",
            "JobIDRaw,State,ExitCode,NodeList",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    states: dict[str, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 4 and fields[0] in job_ids:
            states[fields[0]] = {
                "state": fields[1],
                "exit_code": fields[2],
                "node": fields[3],
            }
    return states


def audit_campaign(label: str, spec: dict[str, Any], *, check_slurm: bool) -> dict[str, Any]:
    errors: list[str] = []
    root = Path(spec["output_root"])

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    try:
        graph = read_json(root / str(spec["graph"]))
        jobs = [job for job in graph.get("jobs", []) if isinstance(job, dict)]
        job_ids = [str(job.get("job_id", "")) for job in jobs]
        stage_counts = Counter(str(job.get("stage", "")) for job in jobs)
        check(graph.get("profile") == spec["profile"], "job graph profile is incorrect")
        check(len(jobs) == 14 and len(set(job_ids)) == 14, "job graph is not 14 unique jobs")
        check(stage_counts == EXPECTED_STAGE_COUNTS, f"job graph stage counts changed: {dict(stage_counts)}")
        for stage in ("train", "train_acceptance", "final_eval_loss", "benchmark"):
            run_ids = {str(job.get("run_id", "")) for job in jobs if job.get("stage") == stage}
            check(run_ids == set(SELECTED), f"{stage} jobs do not cover exactly the selected runs")
    except Exception as exc:
        jobs = []
        job_ids = []
        stage_counts = Counter()
        errors.append(f"cannot validate job graph: {exc}")

    try:
        snapshot = read_json(root / str(spec["snapshot"]))
        snapshot_audit = snapshot.get("audit", {})
        check(snapshot.get("artifact_type") == "slurm_controller_spool_snapshot", "wrong snapshot type")
        check(snapshot_audit.get("all_jobs_accounted_for") is True, "spool snapshot missed jobs")
        check(snapshot_audit.get("source_manifest_preserved") is True, "source manifest was not preserved")
        check(snapshot_audit.get("captured_job_count") == 14, "spool snapshot is not 14 jobs")
        check(snapshot_audit.get("source_script_path_collision_count") == 0, "spool paths collided")
    except Exception as exc:
        errors.append(f"cannot validate controller-spool snapshot: {exc}")

    amendment_path = spec.get("amendment")
    if amendment_path:
        try:
            amendment = read_json(root / str(amendment_path))
            check(amendment.get("status") == "applied_and_preflight_verified", "scheduler amendment is unverified")
            allocations = amendment.get("observed_allocations", {})
            check(set(allocations) == {"33673846", "33673847"}, "unexpected amended job set")
            for job_id, allocation in allocations.items():
                check(allocation.get("nodes") == 1, f"amended job {job_id} did not use one node")
                check(allocation.get("a100_gpus") == 4, f"amended job {job_id} did not use four A100s")
                check(
                    allocation.get("gpu_preflight") == "PASS: ECC, NVLink, and four-rank NCCL",
                    f"amended job {job_id} lacks its GPU preflight proof",
                )
        except Exception as exc:
            errors.append(f"cannot validate scheduler amendment: {exc}")

    accepted_runs: list[str] = []
    finite_losses: dict[str, float] = {}
    strict_benchmarks: list[str] = []
    for run_id in SELECTED:
        try:
            acceptance = read_json(root / "acceptance" / run_id / "acceptance.json")
            checks = acceptance.get("checks", {})
            required_checks = {
                "checkpoint",
                "finite_values",
                "global_step",
                "logs",
                "runtime",
                "slurm",
                "train_loss",
                "train_results",
                "trainer_state",
            }
            valid = (
                acceptance.get("accepted") is True
                and acceptance.get("run_id") == run_id
                and required_checks.issubset(checks)
                and all(checks[name].get("pass") is True for name in required_checks)
                and checks["runtime"].get("target_seconds") == float(spec["target_runtime"])
            )
            check(valid, f"{run_id}: training acceptance is not strict/pass")
            if valid:
                accepted_runs.append(run_id)
        except Exception as exc:
            errors.append(f"{run_id}: cannot validate training acceptance: {exc}")

        try:
            final_run = read_json(root / "final_eval_loss" / "run_summaries" / f"{run_id}.json").get("run", {})
            loss = final_run.get("eval_loss") if isinstance(final_run, dict) else None
            provenance = final_run.get("adapter_provenance", {}) if isinstance(final_run, dict) else {}
            valid = (
                isinstance(final_run, dict)
                and final_run.get("run_id") == run_id
                and finite(loss)
                and provenance.get("valid") is True
                and provenance.get("direct_save_marker_exists") is True
            )
            check(valid, f"{run_id}: final eval loss/provenance is invalid")
            if valid:
                finite_losses[run_id] = float(loss)
        except Exception as exc:
            errors.append(f"{run_id}: cannot validate final evaluation: {exc}")

        try:
            benchmark_run = read_json(root / "benchmarks" / "run_summaries" / f"{run_id}.json").get("run", {})
            valid = (
                isinstance(benchmark_run, dict)
                and benchmark_run.get("run_id") == run_id
                and benchmark_run.get("strict_completeness") is True
                and benchmark_run.get("num_tasks_expected") == spec["task_count"]
                and benchmark_run.get("num_tasks_seen") == spec["task_count"]
                and benchmark_run.get("missing_tasks") == []
                and benchmark_run.get("extra_tasks_seen") == []
                and benchmark_run.get("expected_task_breakdown") == spec["task_breakdown"]
                and benchmark_run.get("seen_task_breakdown") == spec["task_breakdown"]
            )
            check(valid, f"{run_id}: benchmark panel is not strictly complete")
            if valid:
                strict_benchmarks.append(run_id)
        except Exception as exc:
            errors.append(f"{run_id}: cannot validate benchmark summary: {exc}")

    summary_path = root / "summary.csv"
    try:
        summary = pd.read_csv(summary_path)
        summary_runs = tuple(summary["run_id"].astype(str))
        check(len(summary) == 3 and set(summary_runs) == set(SELECTED), "summary.csv does not contain the selected three")
        for column in ("train_loss", "train_runtime", "eval_loss"):
            values = pd.to_numeric(summary[column], errors="coerce")
            check(len(values) == 3 and values.map(math.isfinite).all(), f"summary.csv {column} is incomplete/non-finite")
    except Exception as exc:
        errors.append(f"cannot validate summary.csv: {exc}")

    try:
        gate = read_json(root / "gate.json")
        check(gate.get("profile") == spec["profile"], "gate profile is incorrect")
        check(gate.get("baseline_id") == SELECTED[0], "gate baseline is not the macro-rank winner")
        check(finite(gate.get("baseline_value")), "gate baseline eval loss is non-finite")
        comparisons = gate.get("comparisons", [])
        check(
            {item.get("run_id") for item in comparisons if isinstance(item, dict)} == set(SELECTED[1:]),
            "gate comparisons do not cover the other two winners",
        )
        check(
            all(finite(item.get("metric_value")) for item in comparisons if isinstance(item, dict)),
            "gate comparison eval losses are non-finite",
        )
    except Exception as exc:
        errors.append(f"cannot validate gate.json: {exc}")

    report_path = root / "report.xlsx"
    try:
        workbook = pd.ExcelFile(report_path)
        check(REQUIRED_REPORT_SHEETS.issubset(workbook.sheet_names), "report workbook is missing required sheets")
        benchmark_raw = pd.read_excel(report_path, sheet_name="benchmark_raw")
        selected_raw = benchmark_raw[benchmark_raw["run_id"].astype(str).isin(SELECTED)].copy()
        check(set(selected_raw["run_id"].astype(str)) == set(SELECTED), "report omits selected runs")
        task_counts = selected_raw.groupby("run_id")["task"].nunique()
        check(
            len(task_counts) == 3 and task_counts.eq(int(spec["task_count"])).all(),
            "report benchmark_raw has incomplete task coverage",
        )
        check(
            pd.to_numeric(selected_raw["value"], errors="coerce").notna().all(),
            "report benchmark_raw contains non-numeric selected-run values",
        )
    except Exception as exc:
        errors.append(f"cannot validate report.xlsx: {exc}")

    states: dict[str, dict[str, str]] = {}
    if check_slurm and job_ids:
        try:
            states = slurm_states(job_ids)
            missing = sorted(set(job_ids) - set(states))
            bad = {
                job_id: state
                for job_id, state in states.items()
                if state.get("state") != "COMPLETED" or state.get("exit_code") != "0:0"
            }
            check(not missing and not bad, f"Slurm graph is not fully successful: missing={missing} bad={bad}")
        except Exception as exc:
            errors.append(f"cannot validate Slurm completion: {exc}")

    return {
        "label": label,
        "profile": spec["profile"],
        "output_root": str(root),
        "passed": not errors,
        "errors": errors,
        "selected_runs": list(SELECTED),
        "accepted_runs": accepted_runs,
        "finite_final_eval_losses": finite_losses,
        "strict_benchmark_runs": strict_benchmarks,
        "job_graph_stage_counts": dict(stage_counts),
        "slurm_states": states,
        "summary_csv": str(summary_path),
        "report_xlsx": str(report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=tuple(CAMPAIGNS), action="append")
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs/pipeline_runs/moe_study_final_macro3_completion_audit_v1.json"),
    )
    parser.add_argument("--skip-slurm", action="store_true", help="Skip live Slurm accounting (never use for final sign-off).")
    args = parser.parse_args()

    labels = args.campaign or list(CAMPAIGNS)
    campaigns = [audit_campaign(label, CAMPAIGNS[label], check_slurm=not args.skip_slurm) for label in labels]
    payload = {
        "schema_version": 1,
        "passed": all(campaign["passed"] for campaign in campaigns),
        "slurm_checked": not args.skip_slurm,
        "campaigns": campaigns,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)

    if payload["passed"]:
        print("PASS: both macro-3 campaigns have accepted training, finite final losses, strict benchmarks, reports, and successful Slurm graphs")
        return 0
    for campaign in campaigns:
        if campaign["passed"]:
            print(f"PASS: {campaign['label']}-language campaign")
            continue
        print(f"INCOMPLETE: {campaign['label']}-language campaign")
        for error in campaign["errors"]:
            print(f"  - {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
