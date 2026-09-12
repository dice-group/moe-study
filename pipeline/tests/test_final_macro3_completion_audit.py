from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.analysis.audit_final_macro3_completion import SELECTED, audit_campaign, finite


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_finite_rejects_booleans_and_nonfinite_values() -> None:
    assert finite(1.25)
    assert not finite(True)
    assert not finite(float("nan"))
    assert not finite(float("inf"))


def test_audit_campaign_accepts_complete_artifacts(tmp_path: Path) -> None:
    stage_jobs: list[dict[str, str | None]] = []
    counter = 100
    for stage in ("train", "train_acceptance", "final_eval_loss", "benchmark"):
        for run_id in SELECTED:
            counter += 1
            stage_jobs.append(
                {
                    "job_id": str(counter),
                    "stage": stage,
                    "run_id": run_id,
                    "dependency": None,
                }
            )
    for stage in ("summary", "report"):
        counter += 1
        stage_jobs.append(
            {"job_id": str(counter), "stage": stage, "run_id": "all", "dependency": None}
        )

    write_json(
        tmp_path / "job_graph.json",
        {"profile": "test_profile", "jobs": stage_jobs},
    )
    write_json(
        tmp_path / "snapshot.json",
        {
            "artifact_type": "slurm_controller_spool_snapshot",
            "audit": {
                "all_jobs_accounted_for": True,
                "source_manifest_preserved": True,
                "captured_job_count": 14,
                "source_script_path_collision_count": 0,
            },
        },
    )

    required_checks = (
        "checkpoint",
        "finite_values",
        "global_step",
        "logs",
        "runtime",
        "slurm",
        "train_loss",
        "train_results",
        "trainer_state",
    )
    breakdown = {"Belebele": 1, "FLORES": 1, "SIB200": 0, "Taxi1500": 0}
    for offset, run_id in enumerate(SELECTED):
        checks = {name: {"pass": True} for name in required_checks}
        checks["runtime"]["target_seconds"] = 123.0
        write_json(
            tmp_path / "acceptance" / run_id / "acceptance.json",
            {"accepted": True, "run_id": run_id, "checks": checks},
        )
        write_json(
            tmp_path / "final_eval_loss" / "run_summaries" / f"{run_id}.json",
            {
                "run": {
                    "run_id": run_id,
                    "eval_loss": 2.0 + offset / 10,
                    "adapter_provenance": {
                        "valid": True,
                        "direct_save_marker_exists": True,
                    },
                }
            },
        )
        write_json(
            tmp_path / "benchmarks" / "run_summaries" / f"{run_id}.json",
            {
                "run": {
                    "run_id": run_id,
                    "strict_completeness": True,
                    "num_tasks_expected": 2,
                    "num_tasks_seen": 2,
                    "missing_tasks": [],
                    "extra_tasks_seen": [],
                    "expected_task_breakdown": breakdown,
                    "seen_task_breakdown": breakdown,
                }
            },
        )

    pd.DataFrame(
        {
            "run_id": SELECTED,
            "train_loss": [2.0, 2.1, 2.2],
            "train_runtime": [123.0, 123.0, 123.0],
            "eval_loss": [2.0, 2.1, 2.2],
        }
    ).to_csv(tmp_path / "summary.csv", index=False)
    write_json(
        tmp_path / "gate.json",
        {
            "profile": "test_profile",
            "baseline_id": SELECTED[0],
            "baseline_value": 2.0,
            "comparisons": [
                {"run_id": SELECTED[1], "metric_value": 2.1},
                {"run_id": SELECTED[2], "metric_value": 2.2},
            ],
        },
    )

    benchmark_rows = [
        {"run_id": run_id, "task": task, "value": 0.5}
        for run_id in SELECTED
        for task in ("task_a", "task_b")
    ]
    with pd.ExcelWriter(tmp_path / "report.xlsx") as writer:
        pd.DataFrame(benchmark_rows).to_excel(writer, sheet_name="benchmark_raw", index=False)
        for sheet in (
            "benchmark_compact",
            "data_overview",
            "run_configs",
            "pipeline_profile",
            "experiment_meta",
        ):
            pd.DataFrame({"value": [1]}).to_excel(writer, sheet_name=sheet, index=False)

    result = audit_campaign(
        "test",
        {
            "profile": "test_profile",
            "output_root": tmp_path,
            "graph": "job_graph.json",
            "snapshot": "snapshot.json",
            "task_count": 2,
            "task_breakdown": breakdown,
            "target_runtime": 123,
        },
        check_slurm=False,
    )

    assert result["passed"] is True
    assert result["errors"] == []
    assert result["accepted_runs"] == list(SELECTED)
    assert result["strict_benchmark_runs"] == list(SELECTED)
