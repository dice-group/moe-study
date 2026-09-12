#!/usr/bin/env python3
"""Fail-closed validation for the final LoRA 96/200-language campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.config_loader import load_pipeline_config  # noqa: E402
from pipeline.runtime import _resolve_profile  # noqa: E402
from pipeline.stages.benchmark import _task_family_counts, _validate_strict_task_selection  # noqa: E402


RUNS = {
    "lora_r2": {
        "rank": 2,
        "alpha": 4,
        "trainable_params": 5_242_880,
        "all_params": 8_035_504_128,
    },
    "lora_r4": {
        "rank": 4,
        "alpha": 8,
        "trainable_params": 10_485_760,
        "all_params": 8_040_747_008,
    },
}
TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
KNOWN_BAD_NODES = {"n2gpu1204", "n2gpu1208", "n2gpu1228"}
PANEL_ROOT = ROOT / "outputs/pipeline_runs/moe_study_final_lora_r2_r4_shared/panels"
SHARED_ROOT = ROOT / "outputs/pipeline_runs/moe_study_final_lora_r2_r4_shared"
ENGLISH_TASKS = ("belebele_eng_Latn", "sib200_eng_Latn", "taxi1500_v2_eng_Latn")

PROFILES = {
    "moe_study_final_lora_r2_r4_noctua2_a100_4gpu_96_langs_34p5h_v1": {
        "size": 96,
        "base_tasks": ROOT / "configs/benchmark_panels/v2/lm_eval_tasks_96_langs_v2.txt",
        "base_tasks_sha256": "45f8ca8c965473e04defb3b61fd18178642102a8e6214ad3cbf6aecfe3c19a68",
        "tasks": PANEL_ROOT / "lm_eval_tasks_96_langs_v2_plus_english.txt",
        "tasks_sha256": "bc6fddb4999dae0ce3e555878b33cc337dbe452938066bcc7011ab03c95e1f62",
        "task_count": 249,
        "task_breakdown": {"Belebele": 40, "FLORES": 100, "SIB200": 51, "Taxi1500": 58},
        "panel": "v2_96_plus_english",
        "dataset": ROOT / "outputs/pipeline_runs/moe_study_final_winners_noctua2_a100_4gpu_96_langs_34p5h/tokenized",
        "fingerprint": "ef19026c024549328cae81859b383933",
        "arrow_shards": 50,
        "arrow_bytes": 20_764_331_512,
        "train_time": "34:30:00",
        "signal": "--signal=B:USR1@1200",
        "active_seconds": 122_400,
        "report": "moe_study_final_lora_96_languages.xlsx",
        "baseline_source": ROOT / "result_analysis/reports/moe_study/moe_study_final_96_languages_2026-07-25.xlsx",
        "baseline_source_sha256": "9a0dd6158393dd8284a8f6be8806eae82ef11eeeef66396561404038f3d350db",
        "baseline_cache": SHARED_ROOT / "baseline_cache_96",
        "baseline_rows": 559,
    },
    "moe_study_final_lora_r2_r4_noctua2_a100_4gpu_200_langs_72h_v1": {
        "size": 200,
        "base_tasks": ROOT / "configs/benchmark_panels/v2/lm_eval_tasks_200_langs_v2.txt",
        "base_tasks_sha256": "ce1e056942bee9e70db44fb76de188fc8428a45fcae758bd5f2a198917428e08",
        "tasks": PANEL_ROOT / "lm_eval_tasks_200_langs_v2_plus_english.txt",
        "tasks_sha256": "aab7529237893d14adcac1f28d3355f332090a6e9b2273fad0e86a2637fb48a1",
        "task_count": 408,
        "task_breakdown": {"Belebele": 65, "FLORES": 166, "SIB200": 84, "Taxi1500": 93},
        "panel": "v2_200_plus_english",
        "dataset": ROOT / "outputs/pipeline_runs/moe_study_final_winners_noctua2_a100_4gpu_200_langs_72h/tokenized",
        "fingerprint": "0c9ef91af72343aaa8332237393fb293",
        "arrow_shards": 150,
        "arrow_bytes": 53_172_609_952,
        "train_time": "72:30:00",
        "signal": "--signal=B:USR1@1800",
        "active_seconds": 259_200,
        "report": "moe_study_final_lora_200_languages.xlsx",
        "baseline_source": ROOT / "result_analysis/reports/moe_study/moe_study_final_200_languages_2026-07-25.xlsx",
        "baseline_source_sha256": "9a232d5106bd67e96bea23ce7a8dee5cc78bc8e953c179c9edfcbafc120ed6c2",
        "baseline_cache": SHARED_ROOT / "baseline_cache_200",
        "baseline_rows": 918,
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tasks(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def prepare_panels() -> None:
    PANEL_ROOT.mkdir(parents=True, exist_ok=True)
    for spec in PROFILES.values():
        base = Path(spec["base_tasks"])
        if digest(base) != spec["base_tasks_sha256"]:
            raise ValueError(f"Base v2 task panel changed: {base}")
        tasks = read_tasks(base)
        if len(tasks) != len(set(tasks)):
            raise ValueError(f"Duplicate task in base panel: {base}")
        for task in ENGLISH_TASKS:
            if task in tasks:
                raise ValueError(f"English extension already present in base panel: {task}")
            tasks.append(task)
        payload = "\n".join(tasks) + "\n"
        if hashlib.sha256(payload.encode()).hexdigest() != spec["tasks_sha256"]:
            raise ValueError(f"Augmented panel digest mismatch for {spec['size']} languages")
        target = Path(spec["tasks"])
        temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, target)


def prepare_baseline_caches() -> None:
    for spec in PROFILES.values():
        source = Path(spec["baseline_source"])
        if digest(source) != spec["baseline_source_sha256"]:
            raise ValueError(f"Canonical baseline workbook changed: {source}")

        raw = pd.read_excel(source, sheet_name="benchmark_raw")
        baseline = raw[raw["run_id"].astype(str).eq("baseline::l-3-1-8b")].copy()
        if len(baseline) != spec["baseline_rows"]:
            raise ValueError(f"Unexpected baseline row count in {source}: {len(baseline)}")

        expected_tasks = set(read_tasks(Path(spec["tasks"])))
        actual_tasks = set(baseline["task"].astype(str))
        if actual_tasks != expected_tasks:
            raise ValueError(
                f"Baseline task contract changed for {spec['size']} languages: "
                f"missing={sorted(expected_tasks - actual_tasks)}, "
                f"extra={sorted(actual_tasks - expected_tasks)}"
            )

        rows: list[str] = []
        for row in baseline.itertuples(index=False):
            value = float(row.value)
            if not math.isfinite(value):
                raise ValueError(f"Non-finite canonical baseline value: {row.task} {row.metric}")
            metric = re.sub(r" (?:en->x|x->en)$", "", str(row.metric))
            rows.append(
                json.dumps(
                    {
                        "run_id": "baseline::l-3-1-8b",
                        "task": str(row.task),
                        "metric": metric,
                        "value": value,
                        "wandb_project": "",
                        "wandb_prefix": "",
                        "model": "meta-llama/Llama-3.1-8B",
                        "model_slug": "l-3-1-8b",
                    },
                    sort_keys=True,
                )
            )

        cache_file = Path(spec["baseline_cache"]) / "l-3-1-8b/metrics_long.jsonl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_name(f".{cache_file.name}.tmp.{os.getpid()}")
        temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
        os.replace(temporary, cache_file)


def flags(executor: dict[str, Any]) -> list[str]:
    return [str(item) for item in executor.get("extra_sbatch_flags", [])]


def excluded_nodes(executor: dict[str, Any]) -> set[str]:
    excluded: set[str] = set()
    for flag in flags(executor):
        if flag.startswith("--exclude="):
            excluded.update(part for part in flag.split("=", 1)[1].split(",") if part)
    return excluded


def linked_arrow_files(dataset: Path) -> list[Path]:
    files: list[Path] = []
    for rank_path in sorted(dataset.glob("rank_*")):
        target = rank_path.resolve(strict=True)
        files.extend(sorted(target.rglob("*.arrow")))
    return files


def validate_static(profile_name: str, require_no_training_artifacts: bool) -> list[str]:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    spec = PROFILES[profile_name]
    config = load_pipeline_config(ROOT / "configs/pipeline")
    profile = _resolve_profile(config, profile_name)
    output_root = ROOT / str(profile.get("output_root", ""))

    contract = profile.get("study_contract", {})
    check(tuple(contract.get("selected_run_ids", [])) == tuple(RUNS), "run contract changed")
    check(contract.get("expected_trainable_params") == {
        run_id: run["trainable_params"] for run_id, run in RUNS.items()
    }, "trainable-parameter contract changed")
    check(contract.get("global_batch_size") == 48, "global-batch contract changed")
    check(contract.get("benchmark_task_panel_sha256") == spec["tasks_sha256"], "task digest contract changed")

    training = profile.get("training", {})
    runs = training.get("runs", [])
    check(tuple(run.get("id") for run in runs) == tuple(RUNS), "profile run IDs/order changed")
    check(training.get("model_name_or_path") == "meta-llama/Llama-3.1-8B", "base model changed")
    check(int(training.get("seed", -1)) == 42, "training seed changed")
    check(float(training.get("train", {}).get("learning_rate", 0)) == 2e-4, "learning rate changed")
    check(training.get("train", {}).get("max_steps") is None, "training must be fixed-time")
    check(float(training.get("train", {}).get("num_train_epochs", 0)) == 1.0, "epoch ceiling changed")
    world_size = int(training.get("distributed", {}).get("num_processes", 0))
    check(world_size == 4, "distributed world size must be four")
    check(
        str(training.get("distributed", {}).get("config_file", "")).endswith(
            "fsdp_4gpu_config_size_sharded.yaml"
        ),
        "FSDP configuration changed",
    )

    for run in runs:
        run_id = str(run.get("id", ""))
        expected = RUNS.get(run_id, {})
        lora = run.get("lora", {})
        train = {**training.get("train", {}), **run.get("train", {})}
        check(run.get("kind") == "lora", f"{run_id}: kind must be lora")
        check(int(lora.get("rank", -1)) == expected.get("rank"), f"{run_id}: rank changed")
        check(int(lora.get("alpha", -1)) == expected.get("alpha"), f"{run_id}: alpha changed")
        check(float(lora.get("dropout", -1)) == 0.05, f"{run_id}: dropout changed")
        check(set(str(lora.get("target", "")).split(",")) == TARGET_MODULES, f"{run_id}: targets changed")
        check(int(train.get("batch_size", 0)) == 4, f"{run_id}: micro-batch changed")
        check(int(train.get("grad_accum", 0)) == 3, f"{run_id}: accumulation changed")
        check(4 * 3 * world_size == 48, f"{run_id}: global batch changed")
        check(train.get("disable_gradient_checkpointing") is False, f"{run_id}: checkpointing setting changed")

    tokenization = profile.get("tokenization", {})
    dataset = Path(spec["dataset"])
    check(tokenization.get("enabled") is False, "tokenization must remain disabled")
    check((ROOT / str(tokenization.get("output_dir", ""))).resolve() == dataset.resolve(), "wrong frozen dataset")
    state = json.loads((dataset / "state.json").read_text(encoding="utf-8"))
    token_check = json.loads((dataset / "_check.json").read_text(encoding="utf-8"))
    check(state.get("_fingerprint") == spec["fingerprint"] == contract.get("dataset_fingerprint"), "dataset fingerprint changed")
    check(token_check.get("pass") is True, "frozen dataset check failed")
    arrows = linked_arrow_files(dataset)
    check(len(arrows) == spec["arrow_shards"], "Arrow shard count changed")
    check(sum(path.stat().st_size for path in arrows) == spec["arrow_bytes"], "Arrow byte total changed")

    tasks_file = Path(spec["tasks"])
    tasks = read_tasks(tasks_file)
    benchmark = profile.get("benchmark", {})
    check(digest(tasks_file) == spec["tasks_sha256"], "generated task panel digest changed")
    check((ROOT / str(benchmark.get("tasks_file", ""))).resolve() == tasks_file.resolve(), "profile task path changed")
    check(_task_family_counts(tasks) == spec["task_breakdown"], "task-family breakdown changed")
    try:
        _validate_strict_task_selection(tasks, benchmark)
    except SystemExit as exc:
        errors.append(str(exc))

    report = profile.get("report", {})
    baseline_cache = Path(spec["baseline_cache"])
    check(
        (ROOT / str(report.get("baseline_cache_root", ""))).resolve()
        == baseline_cache.resolve(),
        "wrong dedicated baseline cache",
    )
    baseline_cache_file = baseline_cache / "l-3-1-8b/metrics_long.jsonl"
    check(baseline_cache_file.is_file(), "dedicated baseline cache is missing")
    if baseline_cache_file.is_file():
        baseline_rows = [
            json.loads(line)
            for line in baseline_cache_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        check(len(baseline_rows) == spec["baseline_rows"], "baseline cache row count changed")
        check(
            {str(row.get("task")) for row in baseline_rows} == set(tasks),
            "baseline cache task panel changed",
        )

    train_executor = profile.get("stage_executors", {}).get("train", {})
    check(train_executor.get("gres") == "gpu:a100:4", "training must request four A100 GPUs")
    check(train_executor.get("time") == spec["train_time"], "training allocation time changed")
    check("--nodes=1" in flags(train_executor), "training must request one node")
    check(spec["signal"] in flags(train_executor), "signal schedule changed")
    check(
        excluded_nodes(train_executor) == KNOWN_BAD_NODES,
        "training must exclude exactly the three explicitly unhealthy nodes",
    )
    preflight = training.get("preflight", {})
    check(preflight.get("required_gpu_count") == 4, "four-GPU preflight changed")
    check(preflight.get("required_gpu_name") == "A100", "A100 preflight changed")
    check(all(preflight.get(key) is True for key in ("check_ecc", "check_nvlink", "check_torch_distributed")), "GPU preflight is incomplete")

    acceptance = profile.get("train_acceptance", {})
    check(acceptance.get("enabled") is True, "acceptance stage must be enabled")
    check(acceptance.get("target_runtime_seconds") == spec["active_seconds"], "active-time target changed")
    check(acceptance.get("runtime_tolerance_seconds") == 1200, "runtime tolerance changed")
    check(acceptance.get("require_direct_save_marker") is True, "direct-save requirement changed")
    check(acceptance.get("require_slurm_completed") is True, "Slurm-completion requirement changed")

    final_eval = profile.get("final_eval_loss", {})
    check(final_eval.get("enabled") is True, "final evaluation loss must be enabled")
    check(final_eval.get("strict_final_checkpoint") is True, "final loss must use strict checkpoint")
    check(final_eval.get("require_direct_save_marker") is True, "final loss direct-save check changed")
    check(
        "--oversubscribe"
        in flags(profile.get("stage_executors", {}).get("final_eval_loss", {})),
        "final loss must use --oversubscribe so one requested GPU remains one allocated GPU",
    )
    check(
        "--ntasks=1"
        in flags(profile.get("stage_executors", {}).get("final_eval_loss", {})),
        "final loss must request one task so Slurm does not allocate the whole node",
    )
    for stage in ("final_eval_loss", "benchmark"):
        check(
            excluded_nodes(profile.get("stage_executors", {}).get(stage, {}))
            == KNOWN_BAD_NODES,
            f"{stage} must exclude exactly the three explicitly unhealthy nodes",
        )

    check(
        subprocess.check_output(["git", "-C", "LLaMA-Factory", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        == contract.get("llama_factory_revision"),
        "LLaMA-Factory revision changed",
    )
    check(
        subprocess.check_output(
            ["git", "--git-dir=.git/modules/lm-evaluation-harness", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        == contract.get("lm_eval_revision"),
        "lm-evaluation-harness revision changed",
    )

    if require_no_training_artifacts and output_root.exists():
        artifact_dirs = [
            output_root / "runs",
            output_root / "acceptance",
            output_root / "final_eval_loss",
            output_root / "benchmarks",
        ]
        artifact_files = [output_root / "summary.csv", output_root / spec["report"]]
        has_artifacts = any(
            path.is_file()
            for directory in artifact_dirs
            if directory.is_dir()
            for path in directory.rglob("*")
        ) or any(path.exists() for path in artifact_files)
        check(not has_artifacts, f"campaign artifacts already exist: {output_root}")
    return errors


def validate_results(profile_name: str) -> list[str]:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    spec = PROFILES[profile_name]
    profile = _resolve_profile(load_pipeline_config(ROOT / "configs/pipeline"), profile_name)
    root = ROOT / str(profile["output_root"])

    for run_id, expected in RUNS.items():
        acceptance_path = root / "acceptance" / run_id / "acceptance.json"
        check(acceptance_path.is_file(), f"{run_id}: missing acceptance result")
        if acceptance_path.is_file():
            acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
            check(acceptance.get("accepted") is True, f"{run_id}: training was not accepted")

        checkpoints = sorted(
            path
            for path in (root / "runs" / run_id).glob("checkpoint-*_adapter")
            if (path / ".adapter_direct_save_complete").is_file()
        )
        check(len(checkpoints) == 1, f"{run_id}: expected one direct-save checkpoint")
        if len(checkpoints) == 1:
            adapter = json.loads((checkpoints[0] / "adapter_config.json").read_text(encoding="utf-8"))
            check(adapter.get("r") == expected["rank"], f"{run_id}: saved adapter rank changed")
            check(adapter.get("lora_alpha") == expected["alpha"], f"{run_id}: saved adapter alpha changed")
            check(set(adapter.get("target_modules", [])) == TARGET_MODULES, f"{run_id}: saved targets changed")

    final_summary_path = root / "final_eval_loss" / "summary.json"
    check(final_summary_path.is_file(), "missing final-evaluation-loss summary")
    if final_summary_path.is_file():
        final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
        final_runs = {str(row.get("run_id")): row for row in final_summary.get("runs", [])}
        check(set(final_runs) == set(RUNS), "final-loss run set changed")
        for run_id in RUNS:
            value = final_runs.get(run_id, {}).get("eval_loss")
            check(isinstance(value, (int, float)) and math.isfinite(float(value)), f"{run_id}: invalid final loss")

    benchmark_summary_path = root / "benchmarks" / "benchmark_summary.json"
    check(benchmark_summary_path.is_file(), "missing benchmark summary")
    if benchmark_summary_path.is_file():
        benchmark_summary = json.loads(benchmark_summary_path.read_text(encoding="utf-8"))
        benchmark_runs = {str(row.get("run_id")): row for row in benchmark_summary.get("runs", [])}
        check(set(benchmark_runs) == set(RUNS), "benchmark run set changed")
        for run_id in RUNS:
            row = benchmark_runs.get(run_id, {})
            check(row.get("num_tasks_seen") == spec["task_count"], f"{run_id}: benchmark task count changed")
            check(row.get("missing_tasks") == [], f"{run_id}: benchmark tasks are missing")
            check(row.get("extra_tasks_seen") == [], f"{run_id}: unexpected benchmark tasks")
            check(row.get("seen_task_breakdown") == spec["task_breakdown"], f"{run_id}: benchmark breakdown changed")

    summary_path = root / "summary.csv"
    check(summary_path.is_file(), "missing training summary CSV")
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        check(set(summary["run_id"].astype(str)) == set(RUNS), "summary run set changed")
        for run_id, expected in RUNS.items():
            row = summary[summary["run_id"].astype(str).eq(run_id)]
            check(len(row) == 1, f"{run_id}: summary row missing or duplicated")
            if len(row) == 1:
                check(int(row["trainable_params"].iat[0]) == expected["trainable_params"], f"{run_id}: trainable count changed")
                check(int(row["all_params"].iat[0]) == expected["all_params"], f"{run_id}: total count changed")

    report_path = root / spec["report"]
    check(report_path.is_file(), "missing LoRA Excel report")
    if report_path.is_file():
        with zipfile.ZipFile(report_path) as archive:
            check(archive.testzip() is None, "corrupt Excel report")
            check("xl/workbook.xml" in archive.namelist(), "invalid Excel report")
        raw = pd.read_excel(report_path, sheet_name="benchmark_raw")
        expected_ids = {*RUNS, "baseline::l-3-1-8b"}
        check(set(raw["run_id"].astype(str)) == expected_ids, "Excel report run set changed")
        for run_id in expected_ids:
            rows = raw[raw["run_id"].astype(str).eq(run_id)]
            check(rows["task"].nunique() == spec["task_count"], f"{run_id}: Excel task count changed")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), action="append")
    parser.add_argument("--require-no-training-artifacts", action="store_true")
    parser.add_argument("--validate-results", action="store_true")
    args = parser.parse_args()

    prepare_panels()
    prepare_baseline_caches()
    profiles = args.profile or list(PROFILES)
    failed = False
    for profile_name in profiles:
        errors = validate_static(profile_name, args.require_no_training_artifacts)
        if args.validate_results and not errors:
            errors.extend(validate_results(profile_name))
        if errors:
            failed = True
            print(f"FAIL {profile_name}")
            for error in errors:
                print(f"  - {error}")
        else:
            mode = "results" if args.validate_results else "static"
            print(f"PASS {profile_name} ({mode})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
