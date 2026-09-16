import math
import pytest

from pipeline_tools.cross_arch_study import capacity_check, contract, targets, benchmark_family


def test_frozen_scope_and_budget():
    cfg = contract()
    assert len(cfg["run_ids"]) == 7
    assert len(cfg["models"]) == 2
    assert cfg["nodes"] == 1 and cfg["gpus"] == 4
    assert cfg["trainer_seconds"] == 34 * 3600
    assert sum(cfg["expected_benchmarks"].values()) == 249
    assert targets("falcon_7b", "moelora") == ["query_key_value", "dense"]
    assert set(targets("falcon_7b", "lora")) == {"query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"}
    assert len(targets("gemma2_9b", "mola")) == 7


def test_capacity_is_fail_closed_and_uses_margin():
    assert capacity_check(10_000_000, 1.0)["pass"]
    assert not capacity_check(48 * 122400, 1.0)["pass"]
    assert not capacity_check(10_000_000, 1.0, steps_per_epoch=1000)["pass"]
    assert capacity_check(10_000_000, 1.0)["projected_steps_with_margin"] == 153000
    for value in (0, -1, math.inf, math.nan):
        with pytest.raises(ValueError):
            capacity_check(1000, value)


def test_panel_names():
    assert benchmark_family("taxi1500_v2_eng_Latn") == "Taxi1500"
    with pytest.raises(ValueError):
        benchmark_family("unknown")


def test_primary_values_rejects_partial_duplicate_and_nonfinite(tmp_path):
    import json
    from pipeline_tools.cross_arch_pilot import primary_values
    task = "belebele_eng_Latn"
    path = tmp_path / "one.json"
    path.write_text(json.dumps({"results": {task: {"acc_norm,none": 0.5}}}))
    assert primary_values(tmp_path, [task]) == {task: 0.5}
    with pytest.raises(ValueError, match="Missing"):
        primary_values(tmp_path, [task, "sib200_eng_Latn"])
    (tmp_path / "two.json").write_text(path.read_text())
    with pytest.raises(ValueError, match="duplicated"):
        primary_values(tmp_path, [task])


def test_selected_run_only_allows_systems_changes():
    import copy
    from pipeline_tools.cross_arch_launch import validate_selected_run
    original = {"kind": "lora", "lora": {"rank": 2}, "params": {},
                "train": {"learning_rate": 0.0002, "batch_size": 2, "grad_accum": 6, "flash_attn": "sdpa"}}
    selected = copy.deepcopy(original)
    selected["train"].update(batch_size=4, grad_accum=3, logging_steps=10)
    validate_selected_run(original, selected, "sdpa")
    selected["train"]["learning_rate"] = 0.0001
    with pytest.raises(ValueError, match="scientific"):
        validate_selected_run(original, selected, "sdpa")
    selected = copy.deepcopy(original)
    selected["lora"]["rank"] = 4
    with pytest.raises(ValueError, match="scientific"):
        validate_selected_run(original, selected, "sdpa")
    selected = copy.deepcopy(original)
    selected["train"]["grad_accum"] = 8
    with pytest.raises(ValueError, match="batch"):
        validate_selected_run(original, selected, "sdpa")


def test_benchmark_uses_native_tokenizer_and_original_routing_mode():
    from pipeline_tools.cross_arch_pilot import benchmark_command
    command = benchmark_command("falcon_7b", "moelora_tr32_e16_k8", "/ckpt", "/tasks.txt", "/output")
    assert command[command.index("--mode") + 1] == "no_ids"
    assert command[command.index("--torch-dtype") + 1] == "bf16"
    assert command[command.index("--tokenizer") + 1].endswith("falcon_7b/model")
    assert "Llama-3.1" not in " ".join(command)


def test_benchmark_sharding_is_complete_without_overlap():
    from pipeline_tools.cross_arch_eval import task_shards
    panel = [f"{prefix}_{i}" for prefix, count in [("belebele", 40), ("flores", 100), ("sib200", 51), ("taxi1500", 58)] for i in range(count)]
    shards = task_shards(panel)
    assert len(shards) == 15
    assert sum(map(len, shards)) == 249
    assert set(t for shard in shards for t in shard) == set(panel)
    with pytest.raises(ValueError, match="duplicate"):
        task_shards(panel + panel[:1])


def test_macro_ranking_equal_benchmark_and_direction_weights():
    import pandas as pd
    from pipeline_tools.cross_arch_report import macro_ranks
    rows = []
    # A wins Belebele/SIB/Taxi, loses FLORES en->x and wins x->en.
    # Even with nine en->x units and one reverse, the directions stay 50/50.
    for family, count, a in [("Belebele", 1, 1), ("SIB200", 1, 1), ("Taxi1500", 1, 1),
                             ("FLORES en->x", 9, 0), ("FLORES x->en", 1, 1)]:
        for i in range(count):
            for run, score in [("A", a), ("B", 1-a)]:
                rows.append({"score_family": family, "task": f"{family}_{i}", "language": str(i),
                             "metric": "score", "run_id": run, "score": score})
    ranking, benchmarks = macro_ranks(pd.DataFrame(rows))
    assert benchmarks.loc["A", "FLORES"] == 1.5
    assert ranking.loc["A", "macro_avg_rank"] == 1.125
    assert ranking.index[0] == "A"
    with pytest.raises(ValueError, match="paired"):
        macro_ranks(pd.DataFrame(rows[:-1]))


def test_production_refuses_pilot_environment(monkeypatch):
    from pipeline_tools.cross_arch_launch import clean_environment
    monkeypatch.setenv("PROOF_PIPELINE_CROSS_ARCH_STOP_STEPS", "120")
    with pytest.raises(ValueError, match="override"):
        clean_environment("falcon_7b", "lora_r2")


def test_checkpoint_accepts_original_bin_format_but_not_ambiguous_weights(tmp_path):
    from pipeline_tools.cross_arch_study import checkpoint_weights
    (tmp_path / "adapter_model.bin").write_bytes(b"test")
    assert checkpoint_weights(tmp_path).suffix == ".bin"
    (tmp_path / "adapter_model.safetensors").write_bytes(b"test")
    with pytest.raises(ValueError, match="one nonempty"):
        checkpoint_weights(tmp_path)


def test_slurm_preserves_pilot_and_production_thread_settings():
    from pipeline.executor import SlurmExecutor
    script = SlurmExecutor()._build_job_script(["python", "train.py"],
        {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}, "test", "/tmp/test.log")
    assert "export OMP_NUM_THREADS=2" in script
    assert "export MKL_NUM_THREADS=2" in script
    native = SlurmExecutor()._build_job_script(["python", "train.py"],
        {"PROOF_PIPELINE_PRESERVE_ENV": "1"}, "test", "/tmp/test.log")
    assert "setsid bash -c '" in native
    assert "setsid bash -lc '" not in native


def test_completed_slurm_dependencies_are_verified_before_removal(monkeypatch):
    from types import SimpleNamespace
    from pipeline_tools import cross_arch_study as study
    monkeypatch.setattr(study.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    monkeypatch.setattr(study.subprocess, "check_output", lambda *a, **k: "123|COMPLETED|0:0\n")
    assert study.live_dependency("afterok:123") is None
    monkeypatch.setattr(study.subprocess, "check_output", lambda *a, **k: "123|FAILED|1:0\n")
    with pytest.raises(ValueError, match="did not complete"):
        study.live_dependency("afterok:123")
    assert study.live_dependency("afterany:123") is None
    monkeypatch.setattr(study.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="PENDING\n"))
    assert study.live_dependency("afterok:123") == "afterok:123"


def test_runtime_profile_preserves_budget_and_recovery_settings():
    from pipeline_tools.cross_arch_study import runtime_profile, training_runtime_environment
    from pipeline_tools.cross_arch_node import excluded_nodes
    from pipeline_tools.cross_arch_launch import clean_environment
    cfg = runtime_profile()
    assert cfg["resources"]["train"] == {"cpus_per_task": 64, "mem": "384G"}
    assert excluded_nodes() == ["n2gpu1211", "n2gpu1213"]
    for model in contract()["models"]:
        for run in contract()["run_ids"]:
            expected = training_runtime_environment(model, run)
            env = clean_environment(model, run)
            assert all(env[key] == value for key, value in expected.items())
            if run in {"hydralora_r2", "mtllora_lr5e5_r8_nup16", "mola_tr16_e8_k6"}:
                assert expected["PROOF_PIPELINE_CUDA_MEMORY_FRACTION"] == "0.8"


def test_attempt_logs_do_not_change_artifact_root(tmp_path):
    from pipeline_tools.cross_arch_study import attempt_logs
    first, second = attempt_logs(tmp_path), attempt_logs(tmp_path)
    assert first.parent == second.parent == tmp_path / "logs" / "attempts"
    assert first != second


def test_evaluator_has_no_campaign_imports():
    import ast
    from pipeline_tools.cross_arch_study import ROOT
    tree = ast.parse((ROOT / "scripts/eval/lm_eval_language_ids.py").read_text())
    assert not any(isinstance(node, ast.ImportFrom) and "cross_arch" in (node.module or "")
                   for node in ast.walk(tree))
