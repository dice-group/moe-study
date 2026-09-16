"""CPU fixtures for complete panels and concurrent evaluation artifacts."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from pipeline.checks import check_benchmark
from pipeline.runtime import _load_json, _save_stage_state, _select_tasks_for_languages
from pipeline.stages.benchmark import _validate_strict_task_selection
from pipeline.stages.per_run_summary import write_per_run_summary
from pipeline_tools.reporting.compact import parse_task_identity


def test_strict_panel_checks_count_breakdown_and_duplicates():
    tasks = ["belebele_eng_Latn", "sib200_eng_Latn", "taxi1500_v2_eng_Latn"]
    cfg = {"expected_task_count": 3, "expected_task_breakdown": {
        "Belebele": 1, "SIB200": 1, "Taxi1500": 1, "FLORES": 0,
    }}
    assert _validate_strict_task_selection(tasks, cfg) == cfg["expected_task_breakdown"]
    for invalid in ([], tasks[:-1], tasks + tasks, tasks[:2] + ["belebele_deu_Latn"]):
        with pytest.raises(SystemExit):
            _validate_strict_task_selection(invalid, cfg)


@pytest.mark.parametrize("seen,missing,requested", [
    (1, ["missing"], {"a"}), (2, [], {"a", "absent"}),
])
def test_benchmark_rejects_incomplete_requested_runs(tmp_path, seen, missing, requested):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"strict_completeness": True, "runs": [{
        "run_id": "a", "num_tasks_expected": 2, "num_tasks_seen": seen,
        "missing_tasks": missing,
    }]}))
    with pytest.raises(SystemExit, match="incomplete task results"):
        check_benchmark(tmp_path, summary, dry_run=False, run_ids=requested)


def test_legacy_partial_benchmark_remains_supported(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"runs": [{
        "run_id": "a", "num_tasks_expected": 2, "num_tasks_seen": 1,
        "missing_tasks": ["missing"],
    }]}))
    check_benchmark(tmp_path, summary, dry_run=False)


def test_parallel_summaries_and_state_preserve_all_runs(tmp_path):
    def write(index):
        run_id = f"run_{index}"
        write_per_run_summary(
            per_run_path=tmp_path / "runs" / f"{run_id}.json",
            aggregate_path=tmp_path / "summary.json",
            metadata={"strict_completeness": True},
            run={"run_id": run_id, "value": index},
        )
        _save_stage_state(tmp_path / "state.json", f"benchmark:{run_id}", "sig", [run_id])

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(12)))
    rows = _load_json(tmp_path / "summary.json")["runs"]
    assert {row["run_id"] for row in rows} == {f"run_{i}" for i in range(12)}
    assert len(rows) == 12
    assert len(_load_json(tmp_path / "state.json")["stages"]) == 12
    assert not list(tmp_path.rglob("*.tmp.*"))


@pytest.mark.parametrize("bad", ["{", "{}", '{"run":{"run_id":"wrong"}}'])
def test_bad_summary_cannot_silently_disappear(tmp_path, bad):
    summaries = tmp_path / "runs"
    summaries.mkdir()
    (summaries / "bad.json").write_text(bad)
    aggregate = tmp_path / "summary.json"
    aggregate.write_text('{"previous":true}')
    with pytest.raises(ValueError):
        write_per_run_summary(
            per_run_path=summaries / "good.json", aggregate_path=aggregate,
            metadata={}, run={"run_id": "good"},
        )
    assert _load_json(aggregate) == {"previous": True}


def test_taxi_v2_language_identity_matches_selection(tmp_path):
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("taxi1500_v2_eng_Latn\ntaxi1500_eng_Latn\ntaxi1500_v2_deu_Latn\n")
    selected = _select_tasks_for_languages(
        tasks_file=tasks, languages=["eng_Latn"], include_belebele=False,
        include_sib200=False, include_taxi1500=True, include_flores=False,
        flores_mode="both",
    )
    assert selected == ["taxi1500_v2_eng_Latn", "taxi1500_eng_Latn"]
    for task in selected:
        assert parse_task_identity(task, "acc,none") == ("Taxi1500", "eng_Latn", "acc")


@pytest.mark.parametrize("loss", [0.5, float("nan"), float("inf"), True, None])
def test_existing_final_loss_is_validated_before_caching(tmp_path, monkeypatch, loss):
    from pipeline.stages import final_eval_loss as stage

    result = tmp_path / "runs/a/final_eval_loss/eval_results.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"eval_loss": loss}))
    ctx = SimpleNamespace(
        active_stages=["final_eval_loss"], final_eval_loss_cfg={"enabled": True},
        train_cfg={"model_name_or_path": "fixture", "runs": [{"id": "a", "kind": "lora"}]},
        tokenized_dir=tmp_path / "tokens", final_eval_loss_root=tmp_path / "loss",
        train_root=tmp_path / "runs", state_path=tmp_path / "state.json",
        force_stages=set(), args=SimpleNamespace(run_id=["a"], dry_run=False),
        logger=logging.getLogger("test"), base_run_label="fixture",
    )
    monkeypatch.setattr(stage, "resolve_adapter_eval_path", lambda *a, **kw: tmp_path / "adapter")
    monkeypatch.setattr(stage, "adapter_checkpoint_provenance", lambda *a, **kw: {"valid": True})
    monkeypatch.setattr(stage, "_run_stage_cmd", lambda **kw: pytest.fail("Existing result must not rerun"))
    if loss == 0.5:
        stage.stage_final_eval_loss(ctx)
        assert _load_json(ctx.final_eval_loss_root / "summary.json")["runs"][0]["eval_loss"] == 0.5
        assert "final_eval_loss:a" in _load_json(ctx.state_path)["stages"]
    else:
        with pytest.raises(SystemExit, match="missing/non-finite"):
            stage.stage_final_eval_loss(ctx)
        assert not ctx.state_path.exists()
