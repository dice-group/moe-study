"""Exercise the actual workbook exporter on synthetic, isolated complete results."""
import copy
import pandas as pd
import pytest

from pipeline_tools.cross_arch_study import write_json, sha256
from pipeline_tools import cross_arch_report as report


def test_new_report_round_trip_and_missing_results_fail_closed(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    out = root / "falcon_7b"
    tasks = ["belebele_eng_Latn", "flores_en_perplexity_eng_Latn-deu_Latn",
             "flores_en_perplexity_deu_Latn-eng_Latn", "sib200_eng_Latn", "taxi1500_v2_eng_Latn"]
    (root / "inputs").mkdir(parents=True)
    (root / "inputs/tasks.txt").write_text("\n".join(tasks) + "\n")
    cfg = {"campaign_id": "test", "run_ids": ["lora_r2", "lora_r4"], "models": {"falcon_7b": {"repo_id": "tiiuae/falcon-7b", "revision": "test"}}}
    write_json(out / "backend_ready.json", {"selected_backend": "sdpa"})
    profile = {"training": {"runs": [{"id": rid, "kind": "lora", "train": {}, "lora": {}, "params": {}} for rid in cfg["run_ids"]]}}
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "campaign_root", lambda: root)
    monkeypatch.setattr(report, "model_root", lambda _: out)
    monkeypatch.setattr(report, "contract", lambda: cfg)
    monkeypatch.setattr(report, "load_profile", lambda _: copy.deepcopy(profile))
    from pipeline_tools.reporting import artifacts
    monkeypatch.setattr(artifacts, "git_value", lambda *_: "test")
    for i, run_id in enumerate([*cfg["run_ids"], "baseline"]):
        results = {task: {("bits_per_byte,none" if task.startswith("flores_") else "acc_norm,none" if task.startswith("belebele_") else "acc,none"): 0.3+i*0.01} for task in tasks}
        destination = out / ("baseline" if run_id == "baseline" else "benchmarks/" + run_id)
        write_json(destination / "no_language_ids.json", {"results": results})
        write_json(destination / "complete.json", {"pass": True, "adapter_sha256": "adapter-hash", "results_sha256": sha256(destination / "no_language_ids.json")})
        if run_id == "baseline": continue
        folder = out / "runs" / run_id
        write_json(folder / "campaign_training.json", {"trainable_parameters": 123})
        write_json(folder / "trainable_params.json", {"trainable_params": 123, "all_params": 1000, "trainable_pct": 12.3})
        write_json(folder / "train_results.json", {"train_loss": 1, "train_runtime": 122400})
        write_json(folder / "final_eval_loss/complete.json", {"pass": True, "adapter_sha256": "adapter-hash", "eval_loss": 2, "checkpoint": "test"})
        write_json(folder / "frozen_run.json", profile["training"]["runs"][i])
        write_json(out / "acceptance" / run_id / "acceptance.json", {"accepted": True})
    data = {"documents": {"train": {"english": 10}, "validation": {"english": 1}},
            "sequences": {"train": {"english": 20}, "validation": {"english": 2}},
            "effective_tokens": {"train": {"english": 100}, "validation": {"english": 10}}}
    write_json(out / "data_ready.json", data)
    write_json(out / "tokenized/_check.json", {"samples_per_language": {"english": 22},
               "samples_per_language_by_split": data["sequences"], "tokens_per_language_by_split": data["effective_tokens"]})
    report.build_report("falcon_7b")
    ready = report.read_json(out / "report_ready.json")
    raw = pd.read_excel(ready["path"], sheet_name="benchmark_raw")
    assert set(raw.run_id) == {"lora_r2", "lora_r4", "baseline::falcon-7b"}
    assert len(report.primary_data(raw)) == 15
    documents = pd.read_excel(ready["path"], sheet_name="training_documents")
    assert documents.documents.sum() == 11
    configs = pd.read_excel(ready["path"], sheet_name="run_configs")
    assert configs.trainable_params.tolist() == [123, 123]
    (out / "benchmarks/lora_r2/complete.json").unlink()
    with pytest.raises(FileNotFoundError):
        report.build_report("falcon_7b")


def test_finished_collectors_are_reused_only_with_matching_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "campaign_root", lambda: tmp_path)
    monkeypatch.setattr(report, "model_root", lambda model: tmp_path / model)
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/tasks.txt").write_text("belebele_eng_Latn\n")
    job = {"model": "falcon_7b", "run_id": "lora_r2", "collect": "123"}
    assert report.pending_collectors([job]) == ["123"]
    folder = tmp_path / "falcon_7b/benchmarks/lora_r2"
    results = folder / "no_language_ids.json"
    write_json(results, {"results": {"belebele_eng_Latn": {"acc_norm,none": 0.5}}})
    write_json(folder / "complete.json", {"pass": True, "model": "falcon_7b", "run_id": "lora_r2",
        "tasks": ["belebele_eng_Latn"], "results_sha256": sha256(results)})
    assert report.pending_collectors([job]) == []
    write_json(results, {"results": {}})
    with pytest.raises(ValueError, match="Invalid completed"):
        report.pending_collectors([job])
