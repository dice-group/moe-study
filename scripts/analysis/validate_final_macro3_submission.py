#!/usr/bin/env python3
"""Fail-closed validation for the promoted 96/200-language macro-rank runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.config_loader import load_pipeline_config  # noqa: E402
from pipeline.runtime import _resolve_profile, _select_tasks_for_languages  # noqa: E402


RANKING = ROOT / (
    "outputs/pipeline_runs/moe_study_mixed_rank_expert_followup_clean22_consolidated_v13/"
    "analysis/macro_ranking/configuration_macro_ranks.csv"
)
SOURCE_PROFILE = "moe_study_mixed_rank_expert_followup_clean22_recovery18_active7p2h_one_node_v11"
SELECTED = ("mtllora_r8_nup16", "moelora_tr32_e16_k8", "hydralora_r2")
ALLOWED_NODES = {"n2gpu1201", "n2gpu1202", "n2gpu1217", "n2gpu1222", "n2gpu1227"}
A100_NODES = {f"n2gpu{number}" for number in range(1201, 1233)}

PROFILES = {
    "moe_study_final_macro3_noctua2_a100_4gpu_96_langs_34p5h_v1": {
        "panel": ROOT / "result_analysis/l2v_nested_samples/96.csv",
        "tasks_file": ROOT / "configs/lm_eval_tasks_96_langs.txt",
        "task_count": 183,
        "task_breakdown": {"Belebele": 34, "FLORES": 92, "SIB200": 34, "Taxi1500": 23},
        "dataset": ROOT / "outputs/pipeline_runs/moe_study_final_winners_noctua2_a100_4gpu_96_langs_34p5h/tokenized",
        "fingerprint": "ef19026c024549328cae81859b383933",
        "arrow_shards": 50,
        "arrow_bytes": 20764331512,
        "train_time": "34:30:00",
        "signal": "--signal=B:USR1@1200",
        "active_seconds": 122400,
    },
    "moe_study_final_macro3_noctua2_a100_4gpu_200_langs_72h_v1": {
        "panel": ROOT / "result_analysis/l2v_nested_samples/200.csv",
        "tasks_file": ROOT / "configs/lm_eval_tasks_200_langs.txt",
        "task_count": 306,
        "task_breakdown": {"Belebele": 57, "FLORES": 156, "SIB200": 57, "Taxi1500": 36},
        "dataset": ROOT / "outputs/pipeline_runs/moe_study_final_winners_noctua2_a100_4gpu_200_langs_72h/tokenized",
        "fingerprint": "0c9ef91af72343aaa8332237393fb293",
        "arrow_shards": 150,
        "arrow_bytes": 53172609952,
        "train_time": "72:30:00",
        "signal": "--signal=B:USR1@1800",
        "active_seconds": 259200,
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flags(executor: dict[str, Any]) -> list[str]:
    return [str(item) for item in executor.get("extra_sbatch_flags", [])]


def excluded_nodes(executor: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for flag in flags(executor):
        if flag.startswith("--exclude="):
            result.update(part for part in flag.split("=", 1)[1].split(",") if part)
    return result


def linked_arrow_files(dataset: Path) -> list[Path]:
    files: list[Path] = []
    for rank_path in sorted(dataset.glob("rank_*")):
        target = rank_path.resolve(strict=True)
        files.extend(sorted(target.rglob("*.arrow")))
    return files


def resolved_tasks(panel: Path, tasks_file: Path) -> list[str]:
    with panel.open(newline="", encoding="utf-8") as handle:
        languages = sorted(
            {
                str(row.get("subset", "")).strip()
                for row in csv.DictReader(handle)
                if str(row.get("split", "")).strip() == "train" and str(row.get("subset", "")).strip()
            }
        )
    return _select_tasks_for_languages(
        tasks_file=tasks_file,
        languages=languages,
        include_belebele=True,
        include_sib200=True,
        include_taxi1500=True,
        include_flores=True,
        flores_mode="both",
    )


def validate(profile_name: str, require_no_training_artifacts: bool) -> list[str]:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    spec = PROFILES[profile_name]
    config = load_pipeline_config(ROOT / "configs/pipeline")
    profile = _resolve_profile(config, profile_name)
    source = _resolve_profile(config, SOURCE_PROFILE)

    with RANKING.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    check(tuple(row["run_id"] for row in rows[:3]) == SELECTED, "ranking CSV top three changed")
    check([int(row["macro_order"]) for row in rows[:3]] == [1, 2, 3], "ranking CSV order is invalid")
    contract = profile.get("study_contract", {})
    check(digest(RANKING) == contract.get("ranking_csv_sha256"), "ranking CSV digest changed")
    report_path = ROOT / str(contract.get("selection_report", ""))
    check(report_path.is_file(), "audited selection report is missing")
    if report_path.is_file():
        check(digest(report_path) == contract.get("selection_report_sha256"), "selection report digest changed")
    check(tuple(contract.get("selected_run_ids", [])) == SELECTED, "profile selected-run contract changed")

    runs = profile.get("training", {}).get("runs", [])
    check(tuple(run.get("id") for run in runs) == SELECTED, "profile run IDs/order differ from promoted top three")
    source_runs = {run["id"]: run for run in source.get("training", {}).get("runs", [])}
    for run in runs:
        run_id = str(run.get("id", ""))
        check(run == source_runs.get(run_id), f"{run_id}: configuration differs from audited mixed follow-up")

    training = profile.get("training", {})
    world_size = int(training.get("distributed", {}).get("num_processes", 0))
    check(world_size == 4, "distributed world size must be four")
    check(training.get("train", {}).get("max_steps") is None, "training must remain fixed-time, not fixed-step")
    for run in runs:
        run_train = {**training.get("train", {}), **run.get("train", {})}
        global_batch = int(run_train.get("batch_size", 0)) * int(run_train.get("grad_accum", 0)) * world_size
        check(global_batch == 48, f"{run.get('id')}: global batch is {global_batch}, expected 48")

    tokenization = profile.get("tokenization", {})
    dataset = spec["dataset"]
    check(tokenization.get("enabled") is False, "tokenization must be frozen/disabled")
    check((ROOT / str(tokenization.get("output_dir", ""))).resolve() == dataset.resolve(), "wrong frozen dataset path")
    check(digest(spec["panel"]) == contract.get("language_panel_sha256"), "language panel digest changed")
    token_check = json.loads((dataset / "_check.json").read_text(encoding="utf-8"))
    state = json.loads((dataset / "state.json").read_text(encoding="utf-8"))
    check(token_check.get("pass") is True, "frozen tokenized dataset check does not pass")
    check(state.get("_fingerprint") == spec["fingerprint"] == contract.get("dataset_fingerprint"), "dataset fingerprint changed")
    arrow_files = linked_arrow_files(dataset)
    check(len(arrow_files) == spec["arrow_shards"] == contract.get("dataset_arrow_shards"), "dataset Arrow shard count changed")
    check(sum(path.stat().st_size for path in arrow_files) == spec["arrow_bytes"] == contract.get("dataset_arrow_bytes"), "dataset Arrow bytes changed")

    tasks = resolved_tasks(spec["panel"], spec["tasks_file"])
    task_digest = hashlib.sha256(("\n".join(tasks) + "\n").encode()).hexdigest()
    benchmark = profile.get("benchmark", {})
    check(len(tasks) == spec["task_count"] == benchmark.get("expected_task_count"), "resolved benchmark task count changed")
    check(task_digest == contract.get("benchmark_task_panel_sha256"), "resolved benchmark task digest changed")
    check(benchmark.get("expected_task_breakdown") == spec["task_breakdown"], "benchmark breakdown contract changed")
    check(benchmark.get("strict_completeness") is True, "strict benchmark completeness must be enabled")
    check(benchmark.get("allow_missing_lm_eval") is False and benchmark.get("fail_if_no_tasks") is True, "benchmark must fail closed")

    train_executor = profile.get("stage_executors", {}).get("train", {})
    check(train_executor.get("gres") == "gpu:a100:4", "training must request four A100 GPUs")
    check(train_executor.get("time") == spec["train_time"], "training allocation time changed")
    check(spec["signal"] in flags(train_executor), "training signal schedule changed")
    check("--nodes=1" in flags(train_executor), "training must request exactly one node")
    check(A100_NODES - excluded_nodes(train_executor) == ALLOWED_NODES, "training node allowlist changed")
    preflight = training.get("preflight", {})
    check(preflight.get("required_gpu_count") == 4 and preflight.get("required_gpu_name") == "A100", "four-A100 preflight changed")
    check(all(preflight.get(key) is True for key in ("check_ecc", "check_nvlink", "check_torch_distributed")), "training health checks are incomplete")

    acceptance = profile.get("train_acceptance", {})
    check(acceptance.get("enabled") is True, "training acceptance gate must be enabled")
    check(acceptance.get("target_runtime_seconds") == spec["active_seconds"], "active-time target changed")
    check(acceptance.get("runtime_tolerance_seconds") == 1200, "active-time tolerance changed")
    check(acceptance.get("require_direct_save_marker") is True and acceptance.get("require_slurm_completed") is True, "acceptance gate is not strict")

    for stage in ("final_eval_loss", "benchmark"):
        stage_cfg = profile.get(stage, {})
        executor = profile.get("stage_executors", {}).get(stage, {})
        check(stage_cfg.get("enabled") is True, f"{stage} must be enabled")
        check(stage_cfg.get("strict_final_checkpoint") is True, f"{stage} must use the accepted final checkpoint")
        check(executor.get("gres") == "gpu:a100:1", f"{stage} must request one A100")
        check("--nodes=1" in flags(executor), f"{stage} must request one node")
        check(A100_NODES - excluded_nodes(executor) == ALLOWED_NODES, f"{stage} node allowlist changed")
    check(profile.get("stage_executors", {}).get("benchmark", {}).get("time") == "24:00:00", "benchmark must retain the corrected 24-hour limit")

    check(
        subprocess.check_output(["git", "-C", "LLaMA-Factory", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        == contract.get("llama_factory_revision"),
        "LLaMA-Factory revision changed",
    )
    check(
        subprocess.check_output(["git", "--git-dir=.git/modules/lm-evaluation-harness", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        == contract.get("lm_eval_revision"),
        "lm-evaluation-harness revision changed",
    )

    output_root = ROOT / str(profile.get("output_root", ""))
    if require_no_training_artifacts and output_root.exists():
        forbidden = [output_root / name for name in ("runs", "acceptance", "final_eval_loss", "benchmarks", "summary.csv", "report.xlsx")]
        check(not any(path.exists() for path in forbidden), f"output root already contains campaign artifacts: {output_root}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), action="append")
    parser.add_argument("--require-no-training-artifacts", action="store_true")
    args = parser.parse_args()
    profiles = args.profile or list(PROFILES)
    failed = False
    for profile in profiles:
        errors = validate(profile, args.require_no_training_artifacts)
        if errors:
            failed = True
            print(f"FAIL: {profile}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS: {profile}: frozen top-3 selection, data, fixed-time budget, one-node GPU policy, and evaluation contract")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
