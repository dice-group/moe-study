#!/usr/bin/env python3
"""Fail closed while assembling the clean 22-run mixed-follow-up result.

The four valid v7 runs and the eighteen one-node v11 retries intentionally remain in
separate output roots.  This tool verifies each accepted run, then creates a
read-only symlink view plus merged per-run evaluation summaries for the final
summary/report profile.  Failed v7 preflight outputs are never considered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config_loader import load_pipeline_config
from pipeline.runtime import _resolve_profile
from pipeline.stages.per_run_summary import write_per_run_summary


SOURCE_V7 = Path("outputs/pipeline_runs/moe_study_mixed_rank_expert_followup_clean22_active7p2h_v7")
SOURCE_RECOVERY = Path("outputs/pipeline_runs/moe_study_mixed_rank_expert_followup_clean22_recovery18_active7p2h_one_node_v11")
V7_REPAIR_GRAPH = Path(
    "outputs/pipeline_runs/moe_study_mixed_rank_expert_followup_clean22_v7_completed4_acceptance_repair_v11/"
    "logs/chain/job_graph_20260718_115220.json"
)
RECOVERY_GRAPH = SOURCE_RECOVERY / "logs/chain/job_graph_20260718_114729.json"
EVALUATION_REPAIR_GRAPH_V13 = Path(
    "outputs/pipeline_runs/moe_study_mixed_rank_expert_followup_clean22_evaluation_repair_v13/"
    "logs/chain/evaluation_job_graph.json"
)
ROUTED_FINAL_EVAL_REPAIR_GRAPH_V14 = Path(
    "outputs/pipeline_runs/moe_study_mixed_rank_expert_followup_clean22_routed_final_eval_repair_v14/"
    "logs/chain/job_graph.json"
)
ROUTED_FINAL_EVAL_CANARY_GATE_V14 = ROUTED_FINAL_EVAL_REPAIR_GRAPH_V14.parent / "canary_gate.json"
EVALUATION_NODE_POOL_UPDATE_V14 = (
    ROUTED_FINAL_EVAL_REPAIR_GRAPH_V14.parent / "node_pool_update_20260720.json"
)
EVALUATION_CODE_VALIDATION_V14 = ROUTED_FINAL_EVAL_REPAIR_GRAPH_V14.parent / "code_validation.json"
CONSOLIDATION_PROFILE = "moe_study_mixed_rank_expert_followup_clean22_consolidated_v13"

# These are the only v7 training jobs that passed the distributed GPU preflight.
V7_RUNS = {
    "moelora_tr256_e32_k8",
    "moelora_tr16_e8_k2",
    "mola_tr128_e16_k1",
    "mtllora_r8_nup32",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_ids(profile_name: str) -> list[str]:
    config = load_pipeline_config(REPO_ROOT / "configs/pipeline")
    profile = _resolve_profile(config, profile_name)
    runs = profile.get("training", {}).get("runs", [])
    return [str(run.get("id", "")) for run in runs if isinstance(run, dict)]


def _source_for(run_id: str) -> Path:
    return SOURCE_V7 if run_id in V7_RUNS else SOURCE_RECOVERY


def _submission_train_jobs() -> tuple[dict[str, str], list[dict[str, Any]]]:
    train_jobs: dict[str, str] = {}
    graph_evidence: list[dict[str, Any]] = []
    for graph_path, expected_jobs in ((V7_REPAIR_GRAPH, 12), (RECOVERY_GRAPH, 72)):
        payload = _read_json(REPO_ROOT / graph_path)
        jobs = payload.get("jobs", [])
        if not isinstance(jobs, list) or len(jobs) != expected_jobs:
            raise ValueError(f"{graph_path}: expected {expected_jobs} job records")
        graph_evidence.append(
            {
                "path": str(graph_path),
                "sha256": _sha256(REPO_ROOT / graph_path),
                "role": "training_and_acceptance",
                "profile": payload.get("profile", ""),
                "code_signature": payload.get("code_signature", ""),
                "job_count": len(jobs),
            }
        )
        for job in jobs:
            if not isinstance(job, dict):
                continue
            run_id = str(job.get("run_id", ""))
            if graph_path == RECOVERY_GRAPH and job.get("stage") == "train":
                train_jobs[run_id] = str(job.get("job_id", ""))
            elif graph_path == V7_REPAIR_GRAPH and job.get("stage") == "train_acceptance":
                train_jobs[run_id] = str(job.get("source_train_job_id", ""))

    evaluation_graph_v13 = _read_json(REPO_ROOT / EVALUATION_REPAIR_GRAPH_V13)
    evaluation_jobs_v13 = evaluation_graph_v13.get("jobs", [])
    if not isinstance(evaluation_jobs_v13, list) or len(evaluation_jobs_v13) != 44:
        raise ValueError(f"{EVALUATION_REPAIR_GRAPH_V13}: expected 44 evaluation job records")

    repair_graph_v14 = _read_json(REPO_ROOT / ROUTED_FINAL_EVAL_REPAIR_GRAPH_V14)
    replacement_jobs = repair_graph_v14.get("jobs", [])
    preserved_jobs = repair_graph_v14.get("preserved_v13_jobs", [])
    invalid_jobs = repair_graph_v14.get("invalid_v13_final_eval_jobs", [])
    for label, jobs, expected in (
        ("replacement", replacement_jobs, 16),
        ("preserved", preserved_jobs, 28),
        ("invalid", invalid_jobs, 16),
    ):
        if not isinstance(jobs, list) or len(jobs) != expected or not all(isinstance(job, dict) for job in jobs):
            raise ValueError(
                f"{ROUTED_FINAL_EVAL_REPAIR_GRAPH_V14}: expected {expected} {label} job records"
            )

    v13_ids = {str(job.get("job_id", "")) for job in evaluation_jobs_v13 if isinstance(job, dict)}
    preserved_ids = {str(job.get("job_id", "")) for job in preserved_jobs}
    invalid_ids = {str(job.get("job_id", "")) for job in invalid_jobs}
    replacement_ids = {str(job.get("job_id", "")) for job in replacement_jobs}
    if len(v13_ids) != 44 or preserved_ids & invalid_ids or preserved_ids | invalid_ids != v13_ids:
        raise ValueError(
            f"{ROUTED_FINAL_EVAL_REPAIR_GRAPH_V14}: preserved and invalid jobs do not partition v13"
        )
    if len(replacement_ids) != 16 or not all(job_id.isdigit() for job_id in replacement_ids):
        raise ValueError(f"{ROUTED_FINAL_EVAL_REPAIR_GRAPH_V14}: replacement job IDs are incomplete")
    if not all(job_id.isdigit() for job_id in v13_ids):
        raise ValueError(f"{EVALUATION_REPAIR_GRAPH_V13}: evaluation job IDs are incomplete")
    if any(job.get("stage") != "final_eval_loss" for job in invalid_jobs + replacement_jobs):
        raise ValueError(f"{ROUTED_FINAL_EVAL_REPAIR_GRAPH_V14}: corrected jobs must be final_eval_loss")
    if {str(job.get("run_id", "")) for job in invalid_jobs} != {
        str(job.get("run_id", "")) for job in replacement_jobs
    }:
        raise ValueError(f"{ROUTED_FINAL_EVAL_REPAIR_GRAPH_V14}: replacement run set is incomplete")

    canary_gate = _read_json(REPO_ROOT / ROUTED_FINAL_EVAL_CANARY_GATE_V14)
    canary_records = canary_gate.get("canaries", [])
    if canary_gate.get("passed") is not True or not isinstance(canary_records, list) or len(canary_records) != 2:
        raise ValueError(f"{ROUTED_FINAL_EVAL_CANARY_GATE_V14}: routed-adapter canary gate did not pass")
    if {str(record.get("run_id", "")) for record in canary_records if isinstance(record, dict)} != {
        "moelora_tr128_e16_k8",
        "mola_tr128_e16_k8",
    }:
        raise ValueError(f"{ROUTED_FINAL_EVAL_CANARY_GATE_V14}: canary evidence is incomplete")

    effective_evaluation_jobs = preserved_jobs + replacement_jobs
    effective_pairs = [
        (str(job.get("stage", "")), str(job.get("run_id", "")))
        for job in effective_evaluation_jobs
    ]
    if len(effective_pairs) != 44 or len(set(effective_pairs)) != 44:
        raise ValueError("effective evaluation graph must contain 44 unique stage/run pairs")
    expected_run_ids = set(train_jobs)
    for stage in ("final_eval_loss", "benchmark"):
        stage_runs = {
            str(job.get("run_id", ""))
            for job in effective_evaluation_jobs
            if job.get("stage") == stage
        }
        if stage_runs != expected_run_ids:
            raise ValueError(f"effective evaluation graph: {stage} does not cover all 22 runs")
    graph_evidence.append(
        {
            "path": str(EVALUATION_REPAIR_GRAPH_V13),
            "sha256": _sha256(REPO_ROOT / EVALUATION_REPAIR_GRAPH_V13),
            "role": "initial_evaluation_repair",
            "profiles": evaluation_graph_v13.get("profiles", []),
            "code_signature": evaluation_graph_v13.get("code_signature", ""),
            "job_count": len(evaluation_jobs_v13),
            "preserved_job_count": len(preserved_jobs),
        }
    )
    graph_evidence.append(
        {
            "path": str(ROUTED_FINAL_EVAL_REPAIR_GRAPH_V14),
            "sha256": _sha256(REPO_ROOT / ROUTED_FINAL_EVAL_REPAIR_GRAPH_V14),
            "role": "routed_adapter_final_eval_correction",
            "code_signature": repair_graph_v14.get("code_signature", ""),
            "adapter_code_sha256": repair_graph_v14.get("adapter_code_sha256", ""),
            "root_cause": repair_graph_v14.get("root_cause", ""),
            "invalid_v13_job_count": len(invalid_jobs),
            "replacement_job_count": len(replacement_jobs),
            "effective_evaluation_job_count": len(effective_evaluation_jobs),
        }
    )
    graph_evidence.append(
        {
            "path": str(ROUTED_FINAL_EVAL_CANARY_GATE_V14),
            "sha256": _sha256(REPO_ROOT / ROUTED_FINAL_EVAL_CANARY_GATE_V14),
            "role": "routed_adapter_canary_gate",
            "passed": True,
            "canary_count": len(canary_records),
        }
    )
    node_pool_update = _read_json(REPO_ROOT / EVALUATION_NODE_POOL_UPDATE_V14)
    if node_pool_update.get("probe_state") != "COMPLETED" or node_pool_update.get("probe_exit_code") != "0:0":
        raise ValueError(f"{EVALUATION_NODE_POOL_UPDATE_V14}: expanded node did not pass its probe")
    graph_evidence.append(
        {
            "path": str(EVALUATION_NODE_POOL_UPDATE_V14),
            "sha256": _sha256(REPO_ROOT / EVALUATION_NODE_POOL_UPDATE_V14),
            "role": "evaluation_node_pool_health_update",
            "removed_node": node_pool_update.get("removed_node", ""),
            "added_node": node_pool_update.get("added_node", ""),
            "probe_job_id": node_pool_update.get("probe_job_id", ""),
        }
    )
    code_validation = _read_json(REPO_ROOT / EVALUATION_CODE_VALIDATION_V14)
    validated_files = code_validation.get("files", {})
    validation_tests = code_validation.get("tests", [])
    if not isinstance(validated_files, dict) or not isinstance(validation_tests, list):
        raise ValueError(f"{EVALUATION_CODE_VALIDATION_V14}: malformed validation evidence")
    for relative_path, expected_sha256 in validated_files.items():
        source_path = REPO_ROOT / str(relative_path)
        if not source_path.is_file() or _sha256(source_path) != expected_sha256:
            raise ValueError(f"{EVALUATION_CODE_VALIDATION_V14}: source hash changed for {relative_path}")
    if len(validation_tests) != 2 or any(
        not isinstance(test, dict) or test.get("failed") != 0 or int(test.get("passed", 0)) <= 0
        for test in validation_tests
    ):
        raise ValueError(f"{EVALUATION_CODE_VALIDATION_V14}: correction tests did not pass")
    graph_evidence.append(
        {
            "path": str(EVALUATION_CODE_VALIDATION_V14),
            "sha256": _sha256(REPO_ROOT / EVALUATION_CODE_VALIDATION_V14),
            "role": "evaluation_code_validation",
            "validated_file_count": len(validated_files),
            "test_count": len(validation_tests),
            "passed_test_count": sum(int(test.get("passed", 0)) for test in validation_tests),
        }
    )
    return train_jobs, graph_evidence


def _validate_run(run_id: str, root: Path, expected_train_job_id: str) -> dict[str, Any]:
    errors: list[str] = []
    acceptance_path = root / "acceptance" / run_id / "acceptance.json"
    final_eval_path = root / "final_eval_loss" / "run_summaries" / f"{run_id}.json"
    benchmark_path = root / "benchmarks" / "run_summaries" / f"{run_id}.json"
    run_dir = root / "runs" / run_id

    try:
        acceptance = _read_json(acceptance_path)
    except ValueError as exc:
        acceptance = {}
        errors.append(str(exc))
    if acceptance.get("accepted") is not True:
        errors.append(f"{run_id}: training acceptance is not true")
    accepted_train_job_id = str(
        acceptance.get("checks", {}).get("slurm", {}).get("job_id", "")
        if isinstance(acceptance.get("checks"), dict)
        else ""
    )
    if not expected_train_job_id or accepted_train_job_id != expected_train_job_id:
        errors.append(
            f"{run_id}: accepted train job {accepted_train_job_id!r} does not match submission provenance "
            f"{expected_train_job_id!r}"
        )

    for required in (run_dir / "trainer_state.json", run_dir / "train_results.json"):
        if not required.is_file():
            errors.append(f"{run_id}: missing training artifact {required}")

    try:
        final_eval_payload = _read_json(final_eval_path)
        final_eval = final_eval_payload.get("run", {})
        value = final_eval.get("eval_loss") if isinstance(final_eval, dict) else None
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"{run_id}: final eval loss is missing or non-finite")
    except ValueError as exc:
        final_eval_payload = {}
        final_eval = {}
        errors.append(str(exc))

    try:
        benchmark_payload = _read_json(benchmark_path)
        benchmark = benchmark_payload.get("run", {})
        if not isinstance(benchmark, dict):
            errors.append(f"{run_id}: benchmark per-run payload is malformed")
            benchmark = {}
        if benchmark.get("num_tasks_expected") != 42 or benchmark.get("num_tasks_seen") != 42:
            errors.append(f"{run_id}: benchmark coverage is not exactly 42/42")
        if benchmark.get("missing_tasks") not in ([], None):
            errors.append(f"{run_id}: benchmark has missing tasks")
        if benchmark.get("strict_completeness") is not True:
            errors.append(f"{run_id}: benchmark strict-completeness evidence is missing")
    except ValueError as exc:
        benchmark_payload = {}
        benchmark = {}
        errors.append(str(exc))

    checkpoint_path = str(
        acceptance.get("checks", {}).get("checkpoint", {}).get("adapter_path", "")
        if isinstance(acceptance.get("checks"), dict)
        else ""
    )
    if not checkpoint_path or not Path(checkpoint_path).is_dir():
        errors.append(f"{run_id}: accepted direct adapter checkpoint is unavailable")

    return {
        "run_id": run_id,
        "source_root": str(root),
        "acceptance_path": str(acceptance_path),
        "final_eval_summary": str(final_eval_path),
        "benchmark_summary": str(benchmark_path),
        "adapter_path": checkpoint_path,
        "slurm_train_job_id": accepted_train_job_id,
        "final_eval": final_eval,
        "benchmark": benchmark,
        "errors": errors,
        "final_eval_payload": final_eval_payload,
        "benchmark_payload": benchmark_payload,
    }


def _link(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        raise ValueError(f"refusing to replace existing consolidation path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source.resolve(), destination, target_is_directory=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=CONSOLIDATION_PROFILE)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--plan", action="store_true", help="Show the frozen source mapping without requiring completed artifacts.")
    args = parser.parse_args()

    run_ids = _run_ids(args.profile)
    if len(run_ids) != 22 or len(set(run_ids)) != 22:
        raise SystemExit(f"{args.profile} must resolve to the frozen 22-run manifest (got {len(run_ids)} runs)")
    if not V7_RUNS.issubset(set(run_ids)):
        raise SystemExit("the v7 source set is not a subset of the frozen manifest")

    profile = _resolve_profile(load_pipeline_config(REPO_ROOT / "configs/pipeline"), args.profile)
    configured_root = Path(str(profile.get("output_root", "")))
    output_root = REPO_ROOT / (Path(args.output_root) if args.output_root else configured_root)
    if output_root.resolve() in {(REPO_ROOT / SOURCE_V7).resolve(), (REPO_ROOT / SOURCE_RECOVERY).resolve()}:
        raise SystemExit("consolidation output root must differ from both source roots")

    mapping = [{"run_id": run_id, "source_root": str(_source_for(run_id))} for run_id in run_ids]
    if args.plan:
        print(json.dumps({"profile": args.profile, "output_root": str(output_root), "sources": mapping}, indent=2))
        return 0

    try:
        train_jobs, graph_evidence = _submission_train_jobs()
    except ValueError as exc:
        print(f"FAIL: submission provenance is incomplete: {exc}")
        return 1
    if set(train_jobs) != set(run_ids):
        print("FAIL: submission provenance does not cover the frozen 22-run manifest")
        return 1

    records = [
        _validate_run(run_id, REPO_ROOT / _source_for(run_id), train_jobs[run_id])
        for run_id in run_ids
    ]
    errors = [error for record in records for error in record["errors"]]
    if errors:
        print("FAIL: consolidation prerequisites are incomplete")
        for error in errors:
            print(f"  - {error}")
        return 1

    for record in records:
        run_id = str(record["run_id"])
        source = Path(str(record["source_root"]))
        _link(source / "runs" / run_id, output_root / "runs" / run_id)
        _link(source / "benchmarks" / run_id, output_root / "benchmarks" / run_id)

        final_eval_run = dict(record["final_eval"])
        final_eval_run["source_root"] = str(source)
        benchmark_run = dict(record["benchmark"])
        benchmark_run["source_root"] = str(source)
        write_per_run_summary(
            per_run_path=output_root / "final_eval_loss" / "run_summaries" / f"{run_id}.json",
            aggregate_path=output_root / "final_eval_loss" / "summary.json",
            metadata={"consolidation_profile": args.profile, "source_roots": [str(SOURCE_V7), str(SOURCE_RECOVERY)]},
            run=final_eval_run,
        )
        write_per_run_summary(
            per_run_path=output_root / "benchmarks" / "run_summaries" / f"{run_id}.json",
            aggregate_path=output_root / "benchmarks" / "benchmark_summary.json",
            metadata={"consolidation_profile": args.profile, "source_roots": [str(SOURCE_V7), str(SOURCE_RECOVERY)]},
            run=benchmark_run,
        )

    provenance = {
        "schema_version": 1,
        "profile": args.profile,
        "source_roots": [str(SOURCE_V7), str(SOURCE_RECOVERY)],
        "submission_graphs": graph_evidence,
        "run_count": len(records),
        "runs": [{key: value for key, value in record.items() if key not in {"errors", "final_eval_payload", "benchmark_payload"}} for record in records],
    }
    _write_json(output_root / "provenance.json", provenance)
    print(f"PASS: consolidated {len(records)} accepted runs into {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
