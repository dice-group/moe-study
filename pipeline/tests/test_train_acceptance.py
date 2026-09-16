from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import pipeline.run as run_module
from pipeline.executor import LocalExecutor, SlurmExecutor, build_executor
from pipeline.stages.train_acceptance import evaluate_training_run


def _write_valid_run(run_dir: Path, *, runtime: float = 100.0, step: int = 7) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "train_results.json").write_text(
        json.dumps({"train_loss": 1.25, "train_runtime": runtime}),
        encoding="utf-8",
    )
    (run_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "log_history": [{"loss": 1.25}]}),
        encoding="utf-8",
    )
    adapter = run_dir / f"checkpoint-{step}_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.bin").write_bytes(b"adapter")


def test_accepts_complete_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_valid_run(run_dir)
    log_path = tmp_path / "train.log"
    log_path.write_text("training completed cleanly\n", encoding="utf-8")

    decision = evaluate_training_run(
        run_id="run",
        run_dir=run_dir,
        config={
            "target_runtime_seconds": 100,
            "runtime_tolerance_seconds": 5,
            "require_direct_save_marker": False,
            "require_slurm_completed": True,
            "reject_log_patterns": ["CUDA out of memory", "NCCL.*error"],
            "required_log_patterns": ["training completed cleanly"],
        },
        log_paths=[log_path],
        train_job_id="123",
        slurm_query=lambda _job_id: {
            "available": True,
            "job_id": "123",
            "state": "COMPLETED",
            "exit_code": "0:0",
        },
    )

    assert decision["accepted"] is True
    assert decision["checks"]["runtime"]["pass"] is True
    assert decision["checks"]["checkpoint"]["nonempty"] is True
    assert decision["checks"]["slurm"]["pass"] is True
    assert decision["checks"]["logs"]["missing_required_patterns"] == []


def test_rejects_missing_required_signal_marker(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_valid_run(run_dir)
    log_path = tmp_path / "train.log"
    log_path.write_text("training completed without signal evidence\n", encoding="utf-8")

    decision = evaluate_training_run(
        run_id="run",
        run_dir=run_dir,
        config={
            "target_runtime_seconds": 100,
            "runtime_tolerance_seconds": 5,
            "require_direct_save_marker": False,
            "require_slurm_completed": False,
            "required_log_patterns": [r"\[SLURM_SIGNAL\].*signal=USR1"],
        },
        log_paths=[log_path],
        train_job_id="",
    )

    assert decision["accepted"] is False
    assert decision["checks"]["logs"]["pass"] is False
    assert decision["checks"]["logs"]["missing_required_patterns"]


def test_rejects_bad_runtime_log_and_slurm_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_valid_run(run_dir, runtime=50.0)
    log_path = tmp_path / "train.log"
    log_path.write_text("CUDA out of memory\n", encoding="utf-8")

    decision = evaluate_training_run(
        run_id="run",
        run_dir=run_dir,
        config={
            "target_runtime_seconds": 100,
            "runtime_tolerance_seconds": 5,
            "require_direct_save_marker": True,
            "require_slurm_completed": True,
            "reject_log_patterns": ["CUDA out of memory"],
        },
        log_paths=[log_path],
        train_job_id="123",
        slurm_query=lambda _job_id: {
            "available": True,
            "job_id": "123",
            "state": "NODE_FAIL",
            "exit_code": "1:0",
        },
    )

    assert decision["accepted"] is False
    assert decision["checks"]["runtime"]["pass"] is False
    assert decision["checks"]["logs"]["pass"] is False
    assert decision["checks"]["slurm"]["pass"] is False
    assert decision["checks"]["checkpoint"]["pass"] is False


@pytest.mark.parametrize("gate_enabled", [True, False])
def test_submit_chain_dry_run_uses_acceptance_gate(tmp_path: Path, monkeypatch, capsys, gate_enabled) -> None:
    runs = [{"id": "run_a", "kind": "lora"}, {"id": "run_b", "kind": "lora"}]
    profile = {
        "output_root": str(tmp_path / "out"),
        "logs_dir": str(tmp_path / "out" / "logs"),
        "executor": {"type": "slurm"},
        "sample": {"languages": ["eng_Latn"]},
        "training": {"runs": runs},
        "train_acceptance": {"enabled": gate_enabled},
        "final_eval_loss": {"enabled": True},
        "embedding_training": {"enabled": False},
        "benchmark": {"enabled": True},
    }

    class RecordingSlurm(SlurmExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, object]] = []

        def submit(self, cmd, env, dry_run, log_path, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(
                {
                    "cmd": cmd,
                    "env": dict(env),
                    "dry_run": dry_run,
                    "log_path": log_path,
                    "dependency": kwargs.get("dependency"),
                    "script_path": kwargs.get("script_path"),
                }
            )
            return None

    executor = RecordingSlurm()
    fake_ctx = SimpleNamespace(
        repo_root=tmp_path,
        env={},
        language_map_path=tmp_path / "language_map.json",
        tokenized_dir=tmp_path / "tokenized",
        embedding_training_cfg={},
        train_root=tmp_path / "out" / "runs",
        logs_root=tmp_path / "out" / "logs",
    )

    monkeypatch.setattr(run_module, "_load_pipeline_config", lambda _path: {"profiles": {"test": profile}})
    monkeypatch.setattr(run_module, "build_executor", lambda _profile, _stage_cfg=None: executor)
    monkeypatch.setattr(run_module, "_build_context", lambda _args: fake_ctx)
    monkeypatch.setattr(
        run_module,
        "resolve_training_stage_config",
        lambda _ctx: (profile["training"], "model", runs),
    )
    monkeypatch.setattr(
        run_module,
        "describe_training_run",
        lambda _ctx, run, train_cfg_for_stage: (
            run["id"],
            run["id"],
            tmp_path / "out" / "runs" / run["id"],
            tmp_path / "out" / "logs" / run["id"] / "shared.train.log",
        ),
    )
    monkeypatch.setattr(
        run_module,
        "build_training_cmd",
        lambda **kwargs: (["train", str(kwargs["run_cfg"]["id"])], {}),
    )
    monkeypatch.setattr(run_module, "_compute_pipeline_code_signature", lambda _root: "sig")

    args = argparse.Namespace(
        profile="test",
        config="unused.json",
        run_id=[],
        executor_type="",
        cuda_visible_devices="",
        dry_run=True,
        start_at="train",
        stop_after="benchmark",
        submit_chain=True,
        force_stage=[],
    )
    assert run_module._submit_chain(args) == 0

    dependencies = [call["dependency"] for call in executor.calls]
    if not gate_enabled:
        assert dependencies == [None, None] + ["afterany:dry_train_run_a", "afterany:dry_train_run_b"] * 2
        assert all("train_acceptance" not in call["cmd"] for call in executor.calls)
        return
    assert dependencies == [
        None,
        None,
        "afterany:dry_train_run_a",
        "afterany:dry_train_run_b",
        "afterok:dry_train_acceptance_run_a",
        "afterok:dry_train_acceptance_run_b",
        "afterok:dry_train_acceptance_run_a",
        "afterok:dry_train_acceptance_run_b",
    ]
    acceptance_envs = [call["env"] for call in executor.calls[2:4]]
    assert acceptance_envs[0]["PROOF_PIPELINE_TRAIN_JOB_ID"] == "dry_train_run_a"
    assert acceptance_envs[1]["PROOF_PIPELINE_TRAIN_JOB_ID"] == "dry_train_run_b"
    output = capsys.readouterr().out
    assert "stage=train_acceptance run_id=run_a" in output
    assert "stage=benchmark run_id=run_b" in output
    graph_paths = list((tmp_path / "out" / "logs" / "attempts").glob("*/chain/job_graph_*.json"))
    assert len(graph_paths) == 1
    graph = json.loads(graph_paths[0].read_text(encoding="utf-8"))
    assert graph["profile"] == "test"
    assert len(graph["jobs"]) == 8
    assert graph["jobs"][2]["dependency"] == "afterany:dry_train_run_a"
    assert graph["jobs"][4]["dependency"] == "afterok:dry_train_acceptance_run_a"
    script_paths = [Path(job["script_path"]) for job in graph["jobs"]]
    assert Path(executor.calls[0]["log_path"]).name == Path(executor.calls[1]["log_path"]).name
    assert len(script_paths) == len(set(script_paths)) == 8
    assert [Path(call["script_path"]).resolve() for call in executor.calls] == script_paths
    assert script_paths[0].name.startswith("chain_train.run_a.")
    assert script_paths[1].name.startswith("chain_train.run_b.")


def test_slurm_submit_honors_explicit_script_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "pipeline.executor.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="12345\n"),
    )
    executor = SlurmExecutor(conda_base="none", conda_env="none")
    log_path = tmp_path / "logs" / "shared.train.log"
    script_path = tmp_path / "sbatch" / "chain_train.run_a.20260718_120000.sbatch.sh"

    job_id = executor.submit(
        ["true"],
        {},
        False,
        log_path,
        script_path=script_path,
    )

    assert job_id == "12345"
    assert script_path.is_file()
    assert not (script_path.parent / "shared.train.sbatch.sh").exists()


def test_cpu_stage_overrides_clear_inherited_gpu_gres() -> None:
    profile = {
        "executor": {
            "type": "slurm",
            "partition": "gpu",
            "gres": "gpu:a100:4",
        },
        "stage_executors": {
            stage: {
                "type": "slurm",
                "partition": "normal",
                "gres": "none",
            }
            for stage in ("train_acceptance", "summary", "report")
        },
    }

    for stage in ("train_acceptance", "summary", "report"):
        stage_cfg = run_module._resolve_stage_cfg(profile, stage)
        executor = build_executor(profile, stage_cfg)
        assert isinstance(executor, SlurmExecutor)
        assert executor.partition == "normal"
        assert executor.gres == "none"
        script = executor._build_job_script(["true"], {}, f"test_{stage}", "/tmp/test.log")
        assert "#SBATCH --gres=" not in script


def test_cli_local_override_wins_over_stage_slurm_executor() -> None:
    profile = {
        "executor": {"type": "slurm"},
        "final_eval_loss": {"enabled": True},
        "stage_executors": {
            "final_eval_loss": {
                "type": "slurm",
                "partition": "gpu",
                "gres": "gpu:a100:1",
            }
        },
    }

    stage_cfg = run_module._resolve_stage_cfg(profile, "final_eval_loss")
    stage_cfg = run_module._apply_executor_cli_overrides(stage_cfg, executor_type="local")

    assert isinstance(build_executor(profile, stage_cfg), LocalExecutor)
