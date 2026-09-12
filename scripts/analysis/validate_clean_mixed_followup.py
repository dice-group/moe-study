#!/usr/bin/env python3
"""Fail-closed pre-submission validation for the clean 22-run mixed follow-up."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.config_loader import load_pipeline_config  # noqa: E402
from pipeline.runtime import _resolve_profile, _select_tasks_for_languages  # noqa: E402


FULL_PROFILE = "moe_study_mixed_rank_expert_followup_clean22_active7p2h_v7"
RECOVERY_PROFILE = "moe_study_mixed_rank_expert_followup_clean22_recovery18_active7p2h_v8"
RECOVERY_RETRY_PROFILE = "moe_study_mixed_rank_expert_followup_clean22_recovery18_active7p2h_v9"
SAFE_RECOVERY_PROFILE = "moe_study_mixed_rank_expert_followup_clean22_recovery18_active7p2h_one_node_v11"
RETRY_V6_PROFILE = "moe_study_mixed_rank_expert_followup_clean22_signal_smoke_retry_v6"
HYDRA_RETRY_V7_PROFILE = "moe_study_mixed_rank_expert_followup_clean22_signal_smoke_hydra_retry_v7"
MTL_RETRY_V7_PROFILE = "moe_study_mixed_rank_expert_followup_clean22_signal_smoke_mtl_retry_v7"
MOLA_RETRY_V7_PROFILE = "moe_study_mixed_rank_expert_followup_clean22_signal_smoke_mola_retry_v7"

SELECTED_RUNS = (
    "moelora_tr64_e32_k8",
    "moelora_tr256_e32_k8",
    "moelora_tr64_e32_k4",
    "moelora_tr256_e32_k4",
    "moelora_tr32_e16_k8",
    "moelora_tr128_e16_k8",
    "moelora_tr16_e8_k2",
    "moelora_tr64_e8_k2",
    "mola_tr64_e32_k4",
    "mola_tr256_e32_k4",
    "mola_tr32_e16_k1",
    "mola_tr128_e16_k1",
    "mola_tr32_e16_k8",
    "mola_tr128_e16_k8",
    "mola_tr16_e8_k6",
    "mola_tr64_e8_k6",
    "hydralora_r2_heads32",
    "hydralora_r2_heads16",
    "hydralora_r2",
    "mtllora_r8_nup32",
    "mtllora_r8_nup16",
    "mtllora_r8",
)
RETRY_V6_RUNS = (
    "moelora_tr256_e32_k8",
    "hydralora_r2_heads32",
    "mtllora_r8_nup32",
)
HYDRA_RETRY_V7_RUNS = ("hydralora_r2_heads32",)
MTL_RETRY_V7_RUNS = ("mtllora_r8_nup32",)
MOLA_RETRY_V7_RUNS = ("mola_tr256_e32_k4",)
RECOVERY_RUNS = (
    "moelora_tr64_e32_k8",
    "moelora_tr64_e32_k4",
    "moelora_tr256_e32_k4",
    "moelora_tr32_e16_k8",
    "moelora_tr128_e16_k8",
    "moelora_tr64_e8_k2",
    "mola_tr64_e32_k4",
    "mola_tr256_e32_k4",
    "mola_tr32_e16_k1",
    "mola_tr32_e16_k8",
    "mola_tr128_e16_k8",
    "mola_tr16_e8_k6",
    "mola_tr64_e8_k6",
    "hydralora_r2_heads32",
    "hydralora_r2_heads16",
    "hydralora_r2",
    "mtllora_r8_nup16",
    "mtllora_r8",
)

ATTENTION_TARGETS = "q_proj,k_proj,v_proj,o_proj"
ALL_TARGETS = ATTENTION_TARGETS + ",gate_proj,up_proj,down_proj"
V6_BAD_NODES = {"n2gpu1204", "n2gpu1208", "n2gpu1211", "n2gpu1213", "n2gpu1226", "n2gpu1228", "n2gpu1229", "n2gpu1231"}
BAD_NODES = V6_BAD_NODES | {"n2gpu1219"}
RECOVERY_BAD_NODES = BAD_NODES | {"n2gpu1205", "n2gpu1206", "n2gpu1220", "n2gpu1221", "n2gpu1225", "n2gpu1230"}
RECOVERY_ALLOWED_NODES = {"n2gpu1212", "n2gpu1217", "n2gpu1222", "n2gpu1227"}
A100_NODE_INVENTORY = {f"n2gpu{node}" for node in range(1201, 1233)}
RECOVERY_EXCLUDED_NODES = A100_NODE_INVENTORY - RECOVERY_ALLOWED_NODES
SMOKE_PROFILES = {RETRY_V6_PROFILE, HYDRA_RETRY_V7_PROFILE, MTL_RETRY_V7_PROFILE, MOLA_RETRY_V7_PROFILE}
PROFILE_RUNS = {
    FULL_PROFILE: SELECTED_RUNS,
    RECOVERY_PROFILE: RECOVERY_RUNS,
    RECOVERY_RETRY_PROFILE: RECOVERY_RUNS,
    SAFE_RECOVERY_PROFILE: RECOVERY_RUNS,
    RETRY_V6_PROFILE: RETRY_V6_RUNS,
    HYDRA_RETRY_V7_PROFILE: HYDRA_RETRY_V7_RUNS,
    MTL_RETRY_V7_PROFILE: MTL_RETRY_V7_RUNS,
    MOLA_RETRY_V7_PROFILE: MOLA_RETRY_V7_RUNS,
}
PROFILE_BAD_NODES = {
    FULL_PROFILE: BAD_NODES,
    RECOVERY_PROFILE: RECOVERY_BAD_NODES,
    RECOVERY_RETRY_PROFILE: RECOVERY_BAD_NODES,
    SAFE_RECOVERY_PROFILE: RECOVERY_EXCLUDED_NODES,
    RETRY_V6_PROFILE: V6_BAD_NODES,
    HYDRA_RETRY_V7_PROFILE: BAD_NODES,
    MTL_RETRY_V7_PROFILE: BAD_NODES,
    MOLA_RETRY_V7_PROFILE: BAD_NODES,
}
PROFILE_ALLOWED_NODES = {
    RECOVERY_PROFILE: RECOVERY_ALLOWED_NODES,
    RECOVERY_RETRY_PROFILE: RECOVERY_ALLOWED_NODES,
    SAFE_RECOVERY_PROFILE: RECOVERY_ALLOWED_NODES,
}
DATASET_REL = Path(
    "outputs/pipeline_runs/moe_study_mixed_followup_expanded_data_rerun_10_langs/"
    "tokenized_expanded_400k_train"
)


def _git_revision(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args, "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _manifest_from_notebook(path: Path, variable: str) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == variable for target in targets):
                value = ast.literal_eval(node.value)
                return [str(item) for item in value]
    raise ValueError(f"{variable} was not found as a literal assignment in {path}")


def _flags(executor: dict[str, Any]) -> list[str]:
    return [str(item) for item in executor.get("extra_sbatch_flags", [])]


def _excluded_nodes(executor: dict[str, Any]) -> set[str]:
    excluded: set[str] = set()
    for flag in _flags(executor):
        if flag.startswith("--exclude="):
            excluded.update(item for item in flag.split("=", 1)[1].split(",") if item)
    return excluded


def _requested_nodes(executor: dict[str, Any]) -> set[str]:
    requested: set[str] = set()
    for flag in _flags(executor):
        if flag.startswith("--nodelist="):
            requested.update(item for item in flag.split("=", 1)[1].split(",") if item)
    return requested


def _requested_node_count(executor: dict[str, Any]) -> int | None:
    values = [flag.split("=", 1)[1] for flag in _flags(executor) if flag.startswith("--nodes=")]
    if len(values) != 1:
        return None
    try:
        return int(values[0])
    except ValueError:
        return None


def _live_a100_nodes() -> set[str]:
    result = subprocess.run(
        ["sinfo", "-N", "-p", "gpu", "-h", "-o", "%N|%G"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        line.split("|", 1)[0].strip()
        for line in result.stdout.splitlines()
        if "gpu:a100:" in line.lower()
    }


def validate(profile_name: str, require_fresh_output: bool) -> list[str]:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    config = load_pipeline_config(REPO_ROOT / "configs/pipeline")
    profile = _resolve_profile(config, profile_name)
    training = profile.get("training", {})
    runs = training.get("runs", [])
    run_ids = [str(run.get("id", "")) for run in runs]
    is_smoke = profile_name in SMOKE_PROFILES
    expected_ids = list(PROFILE_RUNS[profile_name])
    expected_bad_nodes = PROFILE_BAD_NODES[profile_name]
    expected_allowed_nodes = PROFILE_ALLOWED_NODES.get(profile_name)
    is_safe_recovery = profile_name == SAFE_RECOVERY_PROFILE

    check(run_ids == expected_ids, f"run IDs/order differ from the frozen {len(expected_ids)}-run manifest")
    check(len(run_ids) == len(set(run_ids)), "run IDs are not unique")
    check(not any("adamoe" in run_id for run_id in run_ids), "legacy adamoe alias remains in run IDs")

    contract = profile.get("study_contract", {})
    notebook_path = REPO_ROOT / str(contract.get("selection_notebook", ""))
    manifest_var = str(contract.get("selection_manifest_variable", ""))
    try:
        notebook_runs = _manifest_from_notebook(notebook_path, manifest_var)
        check(notebook_runs == list(SELECTED_RUNS), "notebook manifest differs from canonical 22-run manifest")
    except Exception as exc:  # fail closed with a concise diagnostic
        errors.append(f"could not verify notebook manifest: {exc}")

    common_train = training.get("train", {})
    common_lora = training.get("lora", {})
    world_size = int(training.get("distributed", {}).get("num_processes", 0))
    expected_global_batch = int(contract.get("global_batch_size", 0))

    check(world_size == 4, f"distributed world size is {world_size}, expected 4")
    check(int(common_train.get("max_steps", -1)) == 8333, "max_steps must be 8333")
    check(float(common_train.get("learning_rate", 0)) == 2e-4, "learning rate must be 2e-4")
    check(training.get("seed") == 42, "seed must be 42")
    check(float(common_lora.get("dropout", -1)) == 0.05, "LoRA dropout must be 0.05")
    check(training.get("model_name_or_path") == "meta-llama/Llama-3.1-8B", "wrong base model")
    check(training.get("dataset_name") == "identity", "training dataset registration must be identity")
    check(common_train.get("bf16") is True and common_train.get("fp16") is False, "training precision must be bf16")

    for run in runs:
        run_id = str(run.get("id", ""))
        kind = str(run.get("kind", ""))
        lora = {**common_lora, **run.get("lora", {})}
        params = run.get("params", {})
        run_train = {**common_train, **run.get("train", {})}
        global_batch = int(run_train.get("batch_size", 0)) * int(run_train.get("grad_accum", 0)) * world_size
        check(global_batch == expected_global_batch, f"{run_id}: global batch is {global_batch}, expected {expected_global_batch}")
        check(int(lora.get("alpha", -1)) == 2 * int(lora.get("rank", 0)), f"{run_id}: alpha must equal 2 * passed rank")
        check(float(lora.get("dropout", -1)) == 0.05, f"{run_id}: dropout must be 0.05")

        moe_match = re.fullmatch(r"(moelora|mola)_tr(\d+)_e(\d+)_k(\d+)", run_id)
        hydra_match = re.fullmatch(r"hydralora_r(\d+)(?:_heads(\d+))?", run_id)
        mtl_match = re.fullmatch(r"mtllora_r(\d+)(?:_nup(\d+))?", run_id)
        if moe_match:
            id_kind, total_rank_raw, experts_raw, top_k_raw = moe_match.groups()
            total_rank, experts, top_k = map(int, (total_rank_raw, experts_raw, top_k_raw))
            check(kind == id_kind, f"{run_id}: kind is {kind}")
            check(total_rank % experts == 0, f"{run_id}: total rank is not divisible by expert count")
            expected_passed_rank = total_rank if kind == "moelora" else total_rank // experts
            check(int(lora.get("rank", -1)) == expected_passed_rank, f"{run_id}: passed LoRA rank does not match family semantics")
            prefix = "moelora" if kind == "moelora" else "mola"
            expert_key = f"{prefix}_num_experts"
            top_k_key = "moelora_sparse_top_k" if kind == "moelora" else "mola_top_k"
            check(int(params.get(expert_key, -1)) == experts, f"{run_id}: expert count does not match ID")
            check(int(params.get(top_k_key, -1)) == top_k, f"{run_id}: top-k does not match ID")
            check(0 < top_k <= experts, f"{run_id}: invalid top-k/expert combination")
            expected_target = ATTENTION_TARGETS if kind == "moelora" else ALL_TARGETS
            check(lora.get("target") == expected_target, f"{run_id}: wrong target-module policy")
            if kind == "mola":
                check(int(params.get("mola_num_null_experts", -1)) == 2 * experts, f"{run_id}: null experts must be 2 * experts")
        elif hydra_match:
            rank, heads = int(hydra_match.group(1)), int(hydra_match.group(2) or 8)
            check(kind == "hydralora", f"{run_id}: kind is {kind}")
            check(int(lora.get("rank", -1)) == rank, f"{run_id}: rank does not match ID")
            check(int(params.get("hydralora_lora_num", -1)) == heads, f"{run_id}: head count does not match ID/anchor")
            check(params.get("hydralora_use_experts") is False, f"{run_id}: HydraLoRA experts must remain disabled")
            check(lora.get("target") == ALL_TARGETS, f"{run_id}: wrong target-module policy")
        elif mtl_match:
            rank, projections = int(mtl_match.group(1)), int(mtl_match.group(2) or 8)
            check(kind == "mtllora", f"{run_id}: kind is {kind}")
            check(int(lora.get("rank", -1)) == rank, f"{run_id}: rank does not match ID")
            check(int(params.get("mtllora_num_up_projections", -1)) == projections, f"{run_id}: projection count does not match ID/anchor")
            check(lora.get("target") == ALL_TARGETS, f"{run_id}: wrong target-module policy")
        else:
            errors.append(f"{run_id}: ID does not match a supported canonical family")

    tokenization = profile.get("tokenization", {})
    dataset_path = REPO_ROOT / str(tokenization.get("output_dir", ""))
    language_panel_path = REPO_ROOT / str(tokenization.get("languages_csv", ""))
    check(dataset_path == REPO_ROOT / DATASET_REL, "wrong tokenized dataset path")
    check(tokenization.get("enabled") is False, "tokenization must be frozen/disabled")
    check(int(tokenization.get("max_length", 0)) == 1024, "tokenized sequence length must be 1024")
    check(language_panel_path == REPO_ROOT / "result_analysis/l2v_nested_samples/10.csv", "wrong language-panel path")
    try:
        panel_bytes = language_panel_path.read_bytes()
        check(hashlib.sha256(panel_bytes).hexdigest() == contract.get("language_panel_sha256"), "language panel changed")
    except Exception as exc:
        errors.append(f"could not verify language panel: {exc}")
    state_path = dataset_path / "state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        check(state.get("_fingerprint") == contract.get("dataset_fingerprint"), "dataset fingerprint changed")
        arrow_files = sorted(dataset_path.glob("*.arrow"))
        check(len(arrow_files) == int(contract.get("dataset_arrow_shards", -1)), "dataset shard count changed")
        check(sum(path.stat().st_size for path in arrow_files) == int(contract.get("dataset_arrow_bytes", -1)), "dataset Arrow bytes changed")
        check(all(path.stat().st_size > 0 for path in arrow_files), "dataset contains an empty Arrow shard")
    except Exception as exc:
        errors.append(f"could not verify tokenized dataset: {exc}")

    train_executor = profile.get("stage_executors", {}).get("train", {})
    expected_time = "00:20:00" if is_smoke else "07:36:00"
    expected_signal = "--signal=B:USR1@300" if is_smoke else "--signal=B:USR1@1200"
    check(train_executor.get("time") == expected_time, f"train allocation must be {expected_time}")
    check(expected_signal in _flags(train_executor), f"train executor is missing {expected_signal}")
    check(_excluded_nodes(train_executor) == expected_bad_nodes, "train denylist differs from the frozen bad-node list")
    if expected_allowed_nodes is not None:
        training_executor = training.get("executor", {})
        for name, executor in (("training", training_executor), ("train", train_executor)):
            check(_excluded_nodes(executor) == expected_bad_nodes, f"{name} executor denylist differs from the recovery policy")
            if is_safe_recovery:
                check(not _requested_nodes(executor), f"{name} executor must not use a multi-node nodelist")
                check(_requested_node_count(executor) == 1, f"{name} executor must request exactly one node")
                check(
                    A100_NODE_INVENTORY - _excluded_nodes(executor) == expected_allowed_nodes,
                    f"{name} executor does not leave exactly the verified-good node pool",
                )
            else:
                check(_requested_nodes(executor) == expected_allowed_nodes, f"{name} executor nodelist differs from the verified-node policy")

    if is_safe_recovery:
        try:
            live_a100_nodes = _live_a100_nodes()
            check(live_a100_nodes == A100_NODE_INVENTORY, "live A100 node inventory differs from the frozen recovery inventory")
        except Exception as exc:
            errors.append(f"could not verify live A100 node inventory: {exc}")

    preflight = training.get("preflight", {})
    check(preflight.get("enabled") is True, "training preflight must be enabled")
    check(int(preflight.get("required_gpu_count", 0)) == 4, "training preflight must require four GPUs")
    check(preflight.get("required_gpu_name") == "A100", "training preflight must require A100")
    check(all(preflight.get(key) is True for key in ("check_ecc", "check_nvlink", "check_torch_distributed")), "training health checks are incomplete")

    local_copy = training.get("tokenized_local_copy", {})
    check(local_copy.get("enabled") is True and local_copy.get("job_scoped") is True, "dataset copy must be job-scoped")
    check(local_copy.get("cleanup") is True and local_copy.get("dereference_symlinks") is True, "dataset copy cleanup/dereference policy changed")

    acceptance = profile.get("train_acceptance", {})
    acceptance_executor = profile.get("stage_executors", {}).get("train_acceptance", {})
    check(acceptance.get("enabled") is True, "train acceptance must be enabled")
    check(acceptance_executor.get("partition") == "normal", "train acceptance must use the CPU partition")
    check(str(acceptance_executor.get("gres", "")).lower() in {"none", "cpu"}, "train acceptance must not request GPUs")
    if is_safe_recovery:
        check(acceptance_executor.get("time") == "01:00:00", "recovery acceptance allocation must be one hour")
        check(_requested_node_count(acceptance_executor) == 1, "recovery acceptance must request exactly one CPU node")
        check("n2cn0167" in _excluded_nodes(acceptance_executor), "recovery acceptance must exclude slow node n2cn0167")
    expected_runtime = 600 if is_smoke else int(contract.get("active_trainer_seconds", 0))
    check(int(acceptance.get("target_runtime_seconds", -1)) == expected_runtime, "acceptance target runtime changed")
    check(acceptance.get("require_direct_save_marker") is True, "acceptance must require direct-save marker")
    check(acceptance.get("require_slurm_completed") is True, "acceptance must require Slurm completion")
    required_markers = [str(item) for item in acceptance.get("required_log_patterns", [])]
    check(any("SLURM_SIGNAL" in item and "USR1" in item for item in required_markers), "acceptance must prove Slurm USR1 delivery")
    check(any("Adapter checkpoint saved" in item for item in required_markers), "acceptance must prove adapter checkpoint save")
    check(any("signal-triggered checkpoint" in item for item in required_markers), "acceptance must prove signal-triggered final-save path")
    if is_smoke:
        check(int(acceptance.get("minimum_global_step", 0)) >= 1, "signal smoke must complete at least one optimizer step")
        check(profile.get("final_eval_loss", {}).get("enabled") is False, "smoke final evaluation must be disabled")
        check(profile.get("benchmark", {}).get("enabled") is False, "smoke benchmark must be disabled")
    else:
        check(int(acceptance.get("runtime_tolerance_seconds", -1)) == 600, "full runtime tolerance must be 600 seconds")
        for stage_name in ("summary", "report"):
            executor = profile.get("stage_executors", {}).get(stage_name, {})
            check(executor.get("partition") == "normal", f"{stage_name} must use the CPU partition")
            check(str(executor.get("gres", "")).lower() in {"none", "cpu"}, f"{stage_name} must not request GPUs")
        for stage_name in ("final_eval_loss", "benchmark"):
            stage = profile.get(stage_name, {})
            executor = profile.get("stage_executors", {}).get(stage_name, {})
            stage_preflight = stage.get("preflight", {})
            check(stage.get("enabled") is True, f"{stage_name} must be enabled")
            check(stage.get("strict_final_checkpoint") is True, f"{stage_name} must require the final checkpoint")
            check(stage.get("require_direct_save_marker") is True, f"{stage_name} must require direct-save provenance")
            check(_excluded_nodes(executor) == expected_bad_nodes, f"{stage_name} denylist differs from training")
            if expected_allowed_nodes is not None:
                if is_safe_recovery:
                    check(not _requested_nodes(executor), f"{stage_name} must not use a multi-node nodelist")
                    check(_requested_node_count(executor) == 1, f"{stage_name} must request exactly one node")
                    check(
                        A100_NODE_INVENTORY - _excluded_nodes(executor) == expected_allowed_nodes,
                        f"{stage_name} does not leave exactly the verified-good node pool",
                    )
                else:
                    check(_requested_nodes(executor) == expected_allowed_nodes, f"{stage_name} nodelist differs from the verified-node policy")
            check(stage_preflight.get("enabled") is True, f"{stage_name} preflight must be enabled")
            check(int(stage_preflight.get("required_gpu_count", 0)) == 1, f"{stage_name} preflight must require one GPU")
            check(stage_preflight.get("required_gpu_name") == "A100", f"{stage_name} preflight must require A100")
            check(stage_preflight.get("check_ecc") is True, f"{stage_name} must check ECC")
        benchmark = profile.get("benchmark", {})
        check(benchmark.get("allow_missing_lm_eval") is False, "benchmark must fail when lm-eval is missing")
        check(benchmark.get("fail_if_no_tasks") is True, "benchmark must fail when no tasks run")
        check(
            benchmark.get("expected_task_breakdown")
            == {"Belebele": 9, "FLORES": 18, "SIB200": 9, "Taxi1500": 6},
            "benchmark task-breakdown contract changed",
        )
        check(benchmark.get("strict_completeness") is True, "benchmark completeness gate must be enabled")
        check(int(benchmark.get("expected_task_count", 0)) == 42, "benchmark must require exactly 42 tasks")
        try:
            languages: list[str] = []
            with language_panel_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if str(row.get("split", "")).strip() == "train":
                        languages.append(str(row.get("subset", "")).strip())
            tasks = _select_tasks_for_languages(
                tasks_file=REPO_ROOT / str(benchmark.get("tasks_file", "")),
                languages=sorted(set(languages)),
                include_belebele=bool(benchmark.get("include_belebele", True)),
                include_sib200=bool(benchmark.get("include_sib200", True)),
                include_taxi1500=bool(benchmark.get("include_taxi1500", True)),
                include_flores=bool(benchmark.get("include_flores", True)),
                flores_mode=str(benchmark.get("flores_mode", "both")),
            )
            task_digest = hashlib.sha256(("\n".join(tasks) + "\n").encode()).hexdigest()
            check(len(tasks) == 42, f"resolved benchmark panel has {len(tasks)} tasks, expected 42")
            check(task_digest == contract.get("benchmark_task_panel_sha256"), "resolved benchmark task panel changed")
        except Exception as exc:
            errors.append(f"could not verify benchmark task panel: {exc}")

    try:
        check(_git_revision("-C", "LLaMA-Factory") == contract.get("llama_factory_revision"), "LLaMA-Factory revision changed")
        check(
            _git_revision("--git-dir=.git/modules/lm-evaluation-harness") == contract.get("lm_eval_revision"),
            "lm-evaluation-harness revision changed",
        )
    except Exception as exc:
        errors.append(f"could not verify source revisions: {exc}")
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    model_snapshot = hf_home / "hub/models--meta-llama--Llama-3.1-8B/snapshots" / str(contract.get("model_snapshot_revision", ""))
    check(model_snapshot.is_dir(), f"pinned model snapshot is unavailable: {model_snapshot}")

    output_root = REPO_ROOT / str(profile.get("output_root", ""))
    check("clean22" in output_root.name, "output root is not the clean22 namespace")
    check("corrected12" not in str(output_root) and "safe12" not in str(output_root), "output root reuses a previous campaign")
    if require_fresh_output and output_root.exists():
        files = [path for path in output_root.rglob("*") if path.is_file()]
        check(not files, f"fresh output root already contains {len(files)} file(s): {output_root}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=FULL_PROFILE, choices=tuple(PROFILE_RUNS))
    parser.add_argument("--require-fresh-output", action="store_true")
    args = parser.parse_args()
    errors = validate(args.profile, args.require_fresh_output)
    if errors:
        print(f"FAIL: {args.profile}")
        for error in errors:
            print(f"  - {error}")
        return 1
    expected_count = len(PROFILE_RUNS[args.profile])
    print(f"PASS: {args.profile}: {expected_count} runs, frozen data/code, ranks, batches, targets, health checks, and Slurm policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
