from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline.checks import check_benchmark
from pipeline.executor import LocalExecutor, SlurmExecutor, _normalize_preflight_config
from pipeline.stages.benchmark import _validate_strict_task_selection
from pipeline.stages.per_run_summary import write_per_run_summary


def test_slurm_script_contains_bounded_preflight_and_job_scoped_copy(tmp_path: Path) -> None:
    executor = SlurmExecutor(
        modules=[],
        conda_base="none",
        conda_env="none",
        preflight=_normalize_preflight_config(
            {
                "enabled": True,
                "required_gpu_count": 4,
                "required_gpu_name": "A100",
                "check_ecc": True,
                "check_nvlink": True,
                "check_torch_distributed": True,
                "timeout_seconds": 123,
            }
        ),
    )
    token = "__PROOF_PIPELINE_SLURM_JOB_ID__"
    script = executor._build_job_script(
        ["python", "train.py", f"/dev/shm/data-{token}"],
        {
            "PROOF_PIPELINE_TOKENIZED_LOCAL_COPY": "1",
            "PROOF_PIPELINE_TOKENIZED_SRC": "/shared/data",
            "PROOF_PIPELINE_TOKENIZED_DST": f"/dev/shm/data-{token}",
            "PROOF_PIPELINE_TOKENIZED_COPY_TIMEOUT_SECONDS": "600",
            "PROOF_PIPELINE_TOKENIZED_VERIFY_FILES": "dataset_info.json,state.json",
            "PROOF_PIPELINE_TOKENIZED_CLEANUP": "1",
            "PROOF_PIPELINE_TOKENIZED_JOB_SCOPED": "1",
        },
        "test",
        str(tmp_path / "slurm.log"),
    )

    syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr
    assert "--required-gpu-count 4" in script
    assert "--required-gpu-name A100" in script
    assert "--check-torch-distributed" in script
    assert "timeout --signal=TERM --kill-after=30 153s" in script
    assert 'flock -w "${TOKENIZED_COPY_TIMEOUT}"' in script
    assert '"${SLURM_JOB_ID:?SLURM_JOB_ID is required for job-scoped paths}"' in script
    assert "verify_tokenized_copy" in script


def test_local_executor_stages_verifies_and_cleans_job_copy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.arrow").write_bytes(b"arrow-data")
    (source / "state.json").write_text('{"fingerprint":"abc"}', encoding="utf-8")
    destination = tmp_path / "copy-__PROOF_PIPELINE_SLURM_JOB_ID__"
    env = os.environ.copy()
    env.update(
        {
            "SLURM_JOB_ID": "1234",
            "PROOF_PIPELINE_TOKENIZED_LOCAL_COPY": "1",
            "PROOF_PIPELINE_TOKENIZED_SRC": str(source),
            "PROOF_PIPELINE_TOKENIZED_DST": str(destination),
            "PROOF_PIPELINE_TOKENIZED_COPY_TIMEOUT_SECONDS": "30",
            "PROOF_PIPELINE_TOKENIZED_VERIFY_FILES": "state.json",
            "PROOF_PIPELINE_TOKENIZED_CLEANUP": "1",
            "PROOF_PIPELINE_TOKENIZED_JOB_SCOPED": "1",
        }
    )
    resolved_destination = tmp_path / "copy-1234"
    command = [
        sys.executable,
        "-c",
        "import pathlib,sys; assert (pathlib.Path(sys.argv[1])/'train.arrow').is_file()",
        str(destination),
    ]
    LocalExecutor().run(command, env, False, tmp_path / "local.log")
    assert not resolved_destination.exists()
    assert "staging through LocalExecutor" in (tmp_path / "local.log").read_text(encoding="utf-8")


def test_strict_task_contract_validates_count_and_breakdown() -> None:
    tasks = (
        [f"belebele_lang{i}" for i in range(9)]
        + [f"flores_en_perplexity_lang{i}" for i in range(18)]
        + [f"sib200_lang{i}" for i in range(9)]
        + [f"taxi1500_lang{i}" for i in range(6)]
    )
    expected = {"Belebele": 9, "FLORES": 18, "SIB200": 9, "Taxi1500": 6}
    assert _validate_strict_task_selection(
        tasks,
        {"expected_task_count": 42, "expected_task_breakdown": expected},
    ) == expected

    with pytest.raises(SystemExit, match="selected 41 tasks, expected 42"):
        _validate_strict_task_selection(
            tasks[:-1],
            {"expected_task_count": 42, "expected_task_breakdown": expected},
        )


def test_strict_benchmark_check_rejects_partial_results(tmp_path: Path) -> None:
    summary = tmp_path / "benchmark_summary.json"
    summary.write_text(
        json.dumps(
            {
                "strict_completeness": True,
                "runs": [
                    {
                        "run_id": "partial",
                        "num_tasks_expected": 42,
                        "num_tasks_seen": 41,
                        "missing_tasks": ["taxi1500_missing"],
                        "strict_completeness": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="incomplete task results"):
        check_benchmark(tmp_path, summary, dry_run=False)


def test_per_run_summaries_aggregate_without_overwriting(tmp_path: Path) -> None:
    per_run = tmp_path / "run_summaries"
    aggregate = tmp_path / "summary.json"
    write_per_run_summary(
        per_run_path=per_run / "z.json",
        aggregate_path=aggregate,
        metadata={"strict_completeness": True},
        run={"run_id": "z", "value": 2},
    )
    write_per_run_summary(
        per_run_path=per_run / "a.json",
        aggregate_path=aggregate,
        metadata={"strict_completeness": True},
        run={"run_id": "a", "value": 1},
    )
    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    assert [row["run_id"] for row in payload["runs"]] == ["a", "z"]
    assert len(payload["per_run_summaries"]) == 2
