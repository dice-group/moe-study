import copy

from pipeline_tools import cross_arch_launch as launch
from pipeline_tools.cross_arch_study import write_json, read_json


def test_submission_is_single_node_gated_and_idempotent(tmp_path, monkeypatch):
    from pipeline import training
    from pipeline_tools import cross_arch_eval, cross_arch_report, cross_arch_node
    root, out = tmp_path, tmp_path / "falcon_7b"
    run = {"id": "lora_r2", "kind": "lora", "train": {"logging_steps": 50}}
    write_json(out / "gates/lora_r2.json", {"pass": True})
    monkeypatch.setattr(launch, "campaign_root", lambda: root)
    monkeypatch.setattr(launch, "model_root", lambda _: out)
    monkeypatch.setattr(launch, "load_profile", lambda _: {"training": {}})
    monkeypatch.setattr(launch, "production_run", lambda *_: copy.deepcopy(run))
    monkeypatch.setattr(launch, "clean_environment", lambda *_: {})
    monkeypatch.setattr(training, "build_training_cmd", lambda **kwargs: (["accelerate", "launch", "test"], {}))
    calls = []
    class FakeSlurm:
        def __init__(self, **kwargs):
            assert kwargs["gres"] == "gpu:a100:4"
            assert kwargs["time_limit"] == "34:30:00"
            assert "--nodes=1" in kwargs["extra_sbatch_flags"]
            assert "--signal=B:USR1@1200" in kwargs["extra_sbatch_flags"]
        def submit(self, *args, **kwargs):
            calls.append(("train", kwargs))
            return "101"
    monkeypatch.setattr(cross_arch_node, "CrossArchSlurmExecutor", FakeSlurm)
    def acceptance(*args, **kwargs):
        assert kwargs["dependency"] == "afterany:101"
        calls.append(("acceptance", kwargs))
        return "102"
    monkeypatch.setattr(launch, "sbatch", acceptance)
    def evaluation(model, run_id, dependency, **kwargs):
        assert dependency == "afterok:102"
        calls.append(("evaluation", dependency))
        return {"collect": "103"}
    monkeypatch.setattr(cross_arch_eval, "submit_evaluation", evaluation)
    monkeypatch.setattr(cross_arch_report, "submit_ready_reports", lambda **kwargs: None)
    launch.advance("falcon_7b", "lora_r2", submit=True)
    launch.advance("falcon_7b", "lora_r2", submit=True)
    assert [x[0] for x in calls] == ["train", "acceptance", "evaluation"]
    assert read_json(root / "production_jobs.json")["jobs"][0]["train"] == "101"


def test_failed_pilot_does_not_submit_any_training(tmp_path, monkeypatch):
    root, out = tmp_path, tmp_path / "falcon_7b"
    write_json(out / "gates/lora_r2.json", {"pass": False, "reason": "capacity"})
    monkeypatch.setattr(launch, "campaign_root", lambda: root)
    monkeypatch.setattr(launch, "model_root", lambda _: out)
    monkeypatch.setattr(launch, "production_run", lambda *_: (_ for _ in ()).throw(AssertionError("must not launch")))
    launch.advance("falcon_7b", "lora_r2", submit=True)
    assert not (root / "production_jobs.json").exists()
