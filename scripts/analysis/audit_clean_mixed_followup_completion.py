#!/usr/bin/env python3
"""Audit the consolidated clean mixed-follow-up result against its full 22-run contract."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / (
    "outputs/pipeline_runs/moe_study_mixed_rank_expert_followup_clean22_consolidated_v13"
)
REPORT_XLSX = OUTPUT_ROOT / "moe_study_mixed_rank_expert_followup_clean22_consolidated_v13.xlsx"
FINALIZER_MANIFEST = OUTPUT_ROOT / "finalizer_submission.json"
V14_GRAPH = REPO_ROOT / (
    "outputs/pipeline_runs/moe_study_mixed_rank_expert_followup_clean22_routed_final_eval_repair_v14/"
    "logs/chain/job_graph.json"
)
CANARY_GATE = V14_GRAPH.parent / "canary_gate.json"
CANARY_SUBMISSION = V14_GRAPH.parent / "canary_submission.json"
NODE_POOL_UPDATE = V14_GRAPH.parent / "node_pool_update_20260720.json"
EXPECTED_BREAKDOWN = {"Belebele": 9, "FLORES": 18, "SIB200": 9, "Taxi1500": 6}
REQUIRED_REPORT_SHEETS = {
    "benchmark_compact",
    "data_overview",
    "run_configs",
    "pipeline_profile",
    "benchmark_raw",
    "experiment_meta",
}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _slurm_states(job_ids: list[str]) -> dict[str, dict[str, str]]:
    result = subprocess.run(
        [
            "sacct", "-j", ",".join(job_ids), "-X", "-n", "-P",
            "-o", "JobIDRaw,State,ExitCode,NodeList",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT_ROOT / "completion_audit.json"))
    args = parser.parse_args()
    errors: list[str] = []
    evidence: dict[str, Any] = {}

    try:
        provenance = _read(OUTPUT_ROOT / "provenance.json")
        provenance_runs = provenance.get("runs", [])
        expected_runs = {
            str(record.get("run_id", ""))
            for record in provenance_runs
            if isinstance(record, dict)
        }
        if provenance.get("run_count") != 22 or len(expected_runs) != 22:
            errors.append("provenance does not contain exactly 22 unique runs")
        roles = {
            str(graph.get("role", ""))
            for graph in provenance.get("submission_graphs", [])
            if isinstance(graph, dict)
        }
        required_roles = {
            "training_and_acceptance",
            "initial_evaluation_repair",
            "routed_adapter_final_eval_correction",
            "routed_adapter_canary_gate",
            "evaluation_node_pool_health_update",
            "evaluation_code_validation",
        }
        if not required_roles.issubset(roles):
            errors.append(f"provenance graph roles are incomplete: {sorted(required_roles - roles)}")
    except Exception as exc:
        provenance = {}
        expected_runs = set()
        errors.append(f"cannot validate provenance: {exc}")

    finite_losses: dict[str, float] = {}
    strict_benchmarks: list[str] = []
    for run_id in sorted(expected_runs):
        try:
            final_run = _read(
                OUTPUT_ROOT / "final_eval_loss" / "run_summaries" / f"{run_id}.json"
            ).get("run", {})
            value = final_run.get("eval_loss") if isinstance(final_run, dict) else None
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"{run_id}: final eval loss is missing or non-finite")
            else:
                finite_losses[run_id] = float(value)
        except Exception as exc:
            errors.append(f"{run_id}: cannot validate final eval summary: {exc}")

        try:
            benchmark_run = _read(
                OUTPUT_ROOT / "benchmarks" / "run_summaries" / f"{run_id}.json"
            ).get("run", {})
            if not isinstance(benchmark_run, dict):
                raise ValueError("run payload is malformed")
            valid = (
                benchmark_run.get("strict_completeness") is True
                and benchmark_run.get("num_tasks_expected") == 42
                and benchmark_run.get("num_tasks_seen") == 42
                and benchmark_run.get("missing_tasks") == []
                and benchmark_run.get("extra_tasks_seen") == []
                and benchmark_run.get("expected_task_breakdown") == EXPECTED_BREAKDOWN
                and benchmark_run.get("seen_task_breakdown") == EXPECTED_BREAKDOWN
            )
            if not valid:
                errors.append(f"{run_id}: benchmark evidence is not strict 42/42")
            else:
                strict_benchmarks.append(run_id)
        except Exception as exc:
            errors.append(f"{run_id}: cannot validate benchmark summary: {exc}")

    summary_path = OUTPUT_ROOT / "summary.csv"
    try:
        summary = pd.read_csv(summary_path)
        summary_runs = set(summary["run_id"].astype(str))
        if len(summary) != 22 or summary_runs != expected_runs:
            errors.append("summary.csv does not contain exactly the frozen 22 runs")
        summary_losses = pd.to_numeric(summary["eval_loss"], errors="coerce")
        if len(summary_losses) != 22 or not summary_losses.map(math.isfinite).all():
            errors.append("summary.csv does not contain 22 finite eval losses")
    except Exception as exc:
        errors.append(f"cannot validate summary.csv: {exc}")

    report_path = REPORT_XLSX
    try:
        workbook = pd.ExcelFile(report_path)
        missing_sheets = REQUIRED_REPORT_SHEETS - set(workbook.sheet_names)
        if missing_sheets:
            errors.append(f"report workbook is missing sheets: {sorted(missing_sheets)}")
        benchmark_raw = pd.read_excel(report_path, sheet_name="benchmark_raw")
        selected_raw = benchmark_raw[benchmark_raw["run_id"].astype(str).isin(expected_runs)].copy()
        report_runs = set(selected_raw["run_id"].astype(str))
        if report_runs != expected_runs:
            errors.append("report benchmark_raw does not cover all frozen runs")
        task_counts = selected_raw.groupby("run_id")["task"].nunique()
        if len(task_counts) != 22 or not task_counts.eq(42).all():
            errors.append("report benchmark_raw does not contain 42 tasks for every frozen run")
        if pd.to_numeric(selected_raw["value"], errors="coerce").isna().any():
            errors.append("report benchmark_raw contains non-numeric selected-run values")
    except Exception as exc:
        errors.append(f"cannot validate report.xlsx: {exc}")

    try:
        graph = _read(V14_GRAPH)
        effective_jobs = graph.get("preserved_v13_jobs", []) + graph.get("jobs", [])
        effective_ids = [
            str(job.get("job_id", "")) for job in effective_jobs if isinstance(job, dict)
        ]
        effective_pairs = {
            (str(job.get("stage", "")), str(job.get("run_id", "")))
            for job in effective_jobs
            if isinstance(job, dict)
        }
        if len(effective_ids) != 44 or len(set(effective_ids)) != 44 or len(effective_pairs) != 44:
            errors.append("effective evaluation graph is not exactly 44 unique jobs/pairs")
        evaluation_states = _slurm_states(effective_ids)
        bad_jobs = {
            job_id: state
            for job_id, state in evaluation_states.items()
            if state.get("state") != "COMPLETED" or state.get("exit_code") != "0:0"
        }
        missing_jobs = sorted(set(effective_ids) - set(evaluation_states))
        if bad_jobs or missing_jobs:
            errors.append(
                f"effective evaluation jobs are not all successful: bad={bad_jobs} missing={missing_jobs}"
            )
    except Exception as exc:
        evaluation_states = {}
        errors.append(f"cannot validate effective Slurm jobs: {exc}")

    try:
        gate = _read(CANARY_GATE)
        if gate.get("passed") is not True or len(gate.get("canaries", [])) != 2:
            errors.append("routed-adapter canary gate did not pass with two canaries")
        canary_submission = _read(CANARY_SUBMISSION)
        canary_job_id = str(canary_submission.get("job_id", ""))
        canary_job_state = _slurm_states([canary_job_id]).get(canary_job_id, {})
        if canary_job_state.get("state") != "COMPLETED" or canary_job_state.get("exit_code") != "0:0":
            errors.append(f"routed-adapter canary gate job did not succeed: {canary_job_state}")
    except Exception as exc:
        gate = {}
        canary_job_state = {}
        errors.append(f"cannot validate routed-adapter canary gate: {exc}")

    try:
        node_pool_update = _read(NODE_POOL_UPDATE)
        probe_job_id = str(node_pool_update.get("probe_job_id", ""))
        probe_state = _slurm_states([probe_job_id]).get(probe_job_id, {})
        if probe_state.get("state") != "COMPLETED" or probe_state.get("exit_code") != "0:0":
            errors.append(f"expanded evaluation node probe did not succeed: {probe_state}")
    except Exception as exc:
        probe_state = {}
        errors.append(f"cannot validate expanded evaluation node probe: {exc}")

    try:
        finalizer = _read(FINALIZER_MANIFEST)
        finalizer_job_id = str(finalizer.get("job_id", ""))
        finalizer_state = _slurm_states([finalizer_job_id]).get(finalizer_job_id, {})
        if finalizer_state.get("state") != "COMPLETED" or finalizer_state.get("exit_code") != "0:0":
            errors.append(f"finalizer did not complete successfully: {finalizer_state}")
    except Exception as exc:
        finalizer_state = {}
        errors.append(f"cannot validate finalizer: {exc}")

    evidence.update(
        {
            "schema_version": 1,
            "passed": not errors,
            "errors": errors,
            "expected_run_count": 22,
            "provenance_run_count": len(expected_runs),
            "finite_final_loss_count": len(finite_losses),
            "strict_benchmark_count": len(strict_benchmarks),
            "effective_evaluation_job_count": len(evaluation_states),
            "canary_gate_job_state": canary_job_state,
            "node_probe_job_state": probe_state,
            "finalizer_state": finalizer_state,
            "summary_csv": str(summary_path),
            "report_xlsx": str(report_path),
            "provenance_json": str(OUTPUT_ROOT / "provenance.json"),
            "finite_final_losses": finite_losses,
        }
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)

    if errors:
        print(f"FAIL: completion audit found {len(errors)} issue(s)")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("PASS: 22 finite losses, 22 strict 42/42 benchmarks, report/provenance, and 44 Slurm jobs verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
