from __future__ import annotations

import json

from pipeline_tools.reporting.sources import iter_eval_rows


def test_iter_eval_rows_follows_consolidated_run_symlink(tmp_path):
    source_run = tmp_path / "source" / "run_a"
    result_dir = source_run / "checkpoint-1_adapter_lm_eval.jsonl" / "model"
    result_dir.mkdir(parents=True)
    (result_dir / "results_1.json").write_text(
        json.dumps({"results": {"belebele_eng_Latn": {"acc_norm,none": 0.75}}}),
        encoding="utf-8",
    )

    benchmark_dir = tmp_path / "consolidated" / "benchmarks"
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "run_a").symlink_to(source_run, target_is_directory=True)

    assert iter_eval_rows(benchmark_dir) == [
        {
            "run_id": "run_a",
            "task": "belebele_eng_Latn",
            "metric": "acc_norm,none",
            "value": 0.75,
            "wandb_project": "",
            "wandb_prefix": "",
        }
    ]
