#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config_loader import deep_merge, load_pipeline_config  # noqa: E402
from pipeline.executor import SlurmExecutor, build_executor  # noqa: E402
from pipeline.runtime import _resolve_profile  # noqa: E402
from pipeline.run import _compute_pipeline_code_signature  # noqa: E402
from pipeline.training import build_training_cmd  # noqa: E402


DEFAULT_CLUSTER = "otus"
CLUSTER_PRESETS = {
    "otus": {
        "base_profile": "moe_study_profile_otus_cluster_10_langs",
        "source_profile": "moe_study_expert_head_sweep_r2_otus_cluster_10_langs",
        "calibrated_profile": "moe_study_expert_head_sweep_r2_otus_cluster_10_langs_calibrated",
        "experiment_tag": "moe-study-expert-head-sweep-r2-10langs",
        "profile_path": Path("configs/pipeline/profiles/moe_study_expert_head_sweep_r2.generated.json"),
        "calibrated_profile_path": Path("configs/pipeline/profiles/moe_study_expert_head_sweep_r2_calibrated.generated.json"),
        "output_root": Path("outputs/pipeline_runs/moe_study_expert_head_sweep_r2_otus_cluster_10_langs"),
        "calibrated_output_root": Path("outputs/pipeline_runs/moe_study_expert_head_sweep_r2_otus_cluster_10_langs_calibrated"),
        "probe_root": Path("outputs/pipeline_runs/moe_study_expert_head_sweep_r2_batch_search"),
    },
    "anonymous-cluster": {
        "base_profile": "moe_study_profile_noctua2_a100_10_langs",
        "source_profile": "moe_study_expert_head_sweep_r2_noctua2_a100_10_langs",
        "calibrated_profile": "moe_study_expert_head_sweep_r2_noctua2_a100_10_langs_calibrated",
        "experiment_tag": "moe-study-expert-head-sweep-r2-anonymous-cluster-a100-10langs",
        "profile_path": Path("configs/pipeline/profiles/moe_study_expert_head_sweep_r2_noctua2.generated.json"),
        "calibrated_profile_path": Path("configs/pipeline/profiles/moe_study_expert_head_sweep_r2_noctua2_calibrated.generated.json"),
        "output_root": Path("outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_10_langs"),
        "calibrated_output_root": Path("outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_10_langs_calibrated"),
        "probe_root": Path("outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_batch_search"),
    },
    "noctua2_4gpu": {
        "base_profile": "moe_study_profile_noctua2_a100_4gpu_10_langs",
        "source_profile": "moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs",
        "calibrated_profile": "moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_calibrated",
        "experiment_tag": "moe-study-expert-head-sweep-r2-anonymous-cluster-a100-4gpu-10langs",
        "profile_path": Path("configs/pipeline/profiles/moe_study_expert_head_sweep_r2_noctua2_4gpu.generated.json"),
        "calibrated_profile_path": Path("configs/pipeline/profiles/moe_study_expert_head_sweep_r2_noctua2_4gpu_calibrated.generated.json"),
        "output_root": Path("outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs"),
        "calibrated_output_root": Path("outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_calibrated"),
        "probe_root": Path("outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_4gpu_batch_search"),
    },
}
DEFAULT_PROFILE_PATH = CLUSTER_PRESETS[DEFAULT_CLUSTER]["profile_path"]
DEFAULT_OUTPUT_ROOT = CLUSTER_PRESETS[DEFAULT_CLUSTER]["output_root"]
DEFAULT_PROBE_ROOT = CLUSTER_PRESETS[DEFAULT_CLUSTER]["probe_root"]
DEFAULT_CALIBRATED_PROFILE_PATH = CLUSTER_PRESETS[DEFAULT_CLUSTER]["calibrated_profile_path"]
DEFAULT_CALIBRATED_OUTPUT_ROOT = CLUSTER_PRESETS[DEFAULT_CLUSTER]["calibrated_output_root"]
DEFAULT_MANIFEST = DEFAULT_PROBE_ROOT / "probe_manifest.csv"
DEFAULT_RESULTS = DEFAULT_PROBE_ROOT / "probe_results.csv"
DEFAULT_SETTINGS = DEFAULT_PROBE_ROOT / "approach_batch_settings.csv"
DEFAULT_BATCHES = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 48]
DEFAULT_TARGET_GLOBAL_BATCH = 48
DEFAULT_MAX_GRAD_ACCUM = 32
EXPERT_COUNTS = [2, 4, 8, 16, 32]
TOP_K_BY_EXPERTS = {
    2: [1],
    4: [1, 2],
    8: [1, 2, 4],
    16: [1, 2, 4, 8],
    32: [1, 2, 4, 8, 16],
}
APPROACH_BY_KIND = {
    "moelora": "moelora",
    "adamole": "adamole",
    "mola": "adamoe_1",
    "movlora": "mov",
    "hmora": "hmora",
    "cola": "cola",
    "hydralora": "hydralora",
    "mtllora": "mtllora",
}


def _load_profile(profile_name: str) -> dict[str, Any]:
    config = load_pipeline_config(REPO_ROOT / "configs/pipeline")
    return _resolve_profile(config, profile_name)


def _preset(cluster: str) -> dict[str, Any]:
    if cluster not in CLUSTER_PRESETS:
        raise SystemExit(f"unknown cluster preset '{cluster}'. Available: {sorted(CLUSTER_PRESETS)}")
    return CLUSTER_PRESETS[cluster]


def _arg_profile(args: argparse.Namespace, attr: str, preset_key: str) -> str:
    value = getattr(args, attr, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return str(_preset(args.cluster)[preset_key])


def _arg_path(args: argparse.Namespace, attr: str, preset_key: str) -> Path:
    value = getattr(args, attr, None)
    if isinstance(value, str) and value.strip():
        return Path(value)
    return Path(_preset(args.cluster)[preset_key])


def _run_templates(base_profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    templates: dict[str, dict[str, Any]] = {}
    for run in base_profile["training"]["runs"]:
        run_id = str(run.get("id", ""))
        kind = str(run.get("kind", ""))
        if run_id == "adamoe_1":
            templates["mola"] = run
        elif kind in APPROACH_BY_KIND:
            templates.setdefault(kind, run)
    missing = [kind for kind in APPROACH_BY_KIND if kind not in templates]
    if missing:
        raise SystemExit(f"base profile is missing templates for: {missing}")
    return templates


def _base_run(template: dict[str, Any], run_id: str) -> dict[str, Any]:
    run = {
        "id": run_id,
        "kind": template["kind"],
    }
    if "lora" in template:
        run["lora"] = deepcopy(template["lora"])
    if "params" in template:
        run["params"] = deepcopy(template["params"])
    train = deepcopy(template.get("train", {})) if isinstance(template.get("train"), dict) else {}
    if "disable_gradient_checkpointing" in train:
        run["train"] = {"disable_gradient_checkpointing": bool(train["disable_gradient_checkpointing"])}
    return run


def _round_cola_num_a(num_b: int) -> int:
    return max(1, int(round((2.0 * num_b) / 3.0)))


def build_sweep_runs(base_profile: dict[str, Any]) -> list[dict[str, Any]]:
    templates = _run_templates(base_profile)
    runs: list[dict[str, Any]] = []

    for num_experts in EXPERT_COUNTS:
        for top_k in TOP_K_BY_EXPERTS[num_experts]:
            run = _base_run(templates["moelora"], f"moelora_r2_e{num_experts}_k{top_k}")
            run["lora"] = {"rank": 2 * num_experts, "alpha": 4 * num_experts}
            run["params"].update(
                {
                    "moelora_gate_temperature": 1.0,
                    "moelora_num_experts": num_experts,
                    "moelora_sparse_top_k": top_k,
                    "moelora_task_embedding_dim": 64,
                }
            )
            runs.append(run)

    for num_experts in EXPERT_COUNTS:
        run = _base_run(templates["adamole"], f"adamole_r2_e{num_experts}")
        run["lora"] = {"rank": 2, "alpha": 4}
        run["params"].update(
            {
                "adamole_num_experts": num_experts,
                "adamole_aux_loss_coef": 0.001,
                "adamole_max_threshold": 0.5 / float(num_experts),
            }
        )
        runs.append(run)

    for num_experts in EXPERT_COUNTS:
        for top_k in TOP_K_BY_EXPERTS[num_experts]:
            run = _base_run(templates["mola"], f"adamoe_1_r2_e{num_experts}_k{top_k}")
            run["lora"] = {"rank": 2, "alpha": 4}
            run["params"].update(
                {
                    "mola_num_experts": num_experts,
                    "mola_num_null_experts": 2 * num_experts,
                    "mola_top_k": top_k,
                    "mola_router_aux_loss_coef": 0.01,
                }
            )
            runs.append(run)

    for num_experts in EXPERT_COUNTS:
        for top_k in TOP_K_BY_EXPERTS[num_experts]:
            run = _base_run(templates["movlora"], f"mov_e{num_experts}_k{top_k}")
            run["lora"] = {"target": "k_proj,v_proj,down_proj"}
            run["params"].update(
                {
                    "movlora_num_experts": num_experts,
                    "movlora_top_k": top_k,
                    "movlora_router_temperature": 1.0,
                    "movlora_router_jitter_noise": 0.0,
                    "movlora_router_bias": False,
                    "movlora_router_init_std": 0.02,
                    "movlora_router_ignore_padding_tokens": False,
                }
            )
            runs.append(run)

    for num_experts in EXPERT_COUNTS:
        for top_k in TOP_K_BY_EXPERTS[num_experts]:
            run = _base_run(templates["hmora"], f"hmora_r2_e{num_experts}_k{top_k}")
            run["lora"] = {"rank": 2, "alpha": 4}
            run["params"].update(
                {
                    "hmora_num_experts": num_experts,
                    "hmora_top_k": top_k,
                    "hmora_use_language_ids_as_task_ids": True,
                }
            )
            run.setdefault("train", {})["disable_gradient_checkpointing"] = True
            runs.append(run)

    for num_experts in EXPERT_COUNTS:
        num_a = _round_cola_num_a(num_experts)
        run = _base_run(templates["cola"], f"cola_r2_a{num_a}_b{num_experts}")
        run["lora"] = {"rank": 2, "alpha": 4}
        run["params"].update(
            {
                "cola_num_A": num_a,
                "cola_num_B": num_experts,
                "cola_strategy": "fully",
                "cola_use_experts": False,
                "use_cola_pissa_init": True,
            }
        )
        runs.append(run)

    for num_experts in EXPERT_COUNTS:
        run = _base_run(templates["hydralora"], f"hydralora_r2_heads{num_experts}")
        run["lora"] = {"rank": 2, "alpha": 4}
        run["params"].update(
            {
                "hydralora_lora_num": num_experts,
                "hydralora_use_experts": False,
                "language_guidance_scope": "none",
                "language_prior_weight": 0.0,
            }
        )
        runs.append(run)

    for num_experts in EXPERT_COUNTS:
        run = _base_run(templates["mtllora"], f"mtllora_r2_nup{num_experts}")
        run["lora"] = {
            "rank": 2,
            "alpha": 4,
            "target": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        }
        run["params"].update(
            {
                "mtllora_num_up_projections": num_experts,
                "mtllora_temperature": 0.1,
                "mtllora_lambda_format": "full",
                "mtllora_use_language_ids_as_task_ids": True,
            }
        )
        runs.append(run)

    validate_runs(runs)
    return runs


def _run_num_experts(run: dict[str, Any]) -> int | None:
    params = run.get("params", {})
    if not isinstance(params, dict):
        params = {}
    kind = str(run["kind"])
    if kind == "moelora":
        return int(params["moelora_num_experts"])
    if kind == "adamole":
        return int(params["adamole_num_experts"])
    if kind == "mola":
        return int(params["mola_num_experts"])
    if kind == "movlora":
        return int(params["movlora_num_experts"])
    if kind == "hmora":
        return int(params["hmora_num_experts"])
    if kind == "cola":
        return int(params["cola_num_B"])
    if kind == "hydralora":
        return int(params["hydralora_lora_num"])
    if kind == "mtllora":
        return int(params["mtllora_num_up_projections"])
    return None


def _run_top_k(run: dict[str, Any]) -> int | None:
    params = run.get("params", {})
    if not isinstance(params, dict):
        return None
    kind = str(run["kind"])
    if kind == "moelora":
        return int(params["moelora_sparse_top_k"])
    if kind == "mola":
        return int(params["mola_top_k"])
    if kind == "movlora":
        return int(params["movlora_top_k"])
    if kind == "hmora":
        return int(params["hmora_top_k"])
    return None


def _approach_key(run: dict[str, Any]) -> str:
    kind = str(run["kind"])
    return APPROACH_BY_KIND.get(kind, kind)


def validate_runs(runs: list[dict[str, Any]]) -> None:
    ids = [str(run["id"]) for run in runs]
    if len(ids) != len(set(ids)):
        duplicates = sorted({run_id for run_id in ids if ids.count(run_id) > 1})
        raise SystemExit(f"duplicate run ids: {duplicates}")
    if len(runs) != 80:
        raise SystemExit(f"expected 80 runs, got {len(runs)}")

    for run in runs:
        kind = str(run["kind"])
        params = run.get("params", {}) if isinstance(run.get("params"), dict) else {}
        lora = run.get("lora", {}) if isinstance(run.get("lora"), dict) else {}
        num_experts = _run_num_experts(run)
        top_k = _run_top_k(run)
        if num_experts not in EXPERT_COUNTS:
            raise SystemExit(f"{run['id']}: unexpected expert/head count {num_experts}")
        if top_k is not None and (top_k <= 0 or top_k > int(num_experts)):
            raise SystemExit(f"{run['id']}: invalid top-k {top_k} for N={num_experts}")
        if kind == "moelora":
            if int(lora.get("rank", 0)) != 2 * int(num_experts):
                raise SystemExit(f"{run['id']}: MoE-LoRA rank must be 2*N")
            if int(lora.get("alpha", 0)) != 4 * int(num_experts):
                raise SystemExit(f"{run['id']}: MoE-LoRA alpha must be 4*N")
        elif kind != "movlora":
            if int(lora.get("rank", 0)) != 2 or int(lora.get("alpha", 0)) != 4:
                raise SystemExit(f"{run['id']}: expected LoRA rank=2 alpha=4")
        if kind == "cola":
            if int(params["cola_num_A"]) != _round_cola_num_a(int(params["cola_num_B"])):
                raise SystemExit(f"{run['id']}: unexpected CoLA A/B mapping")
        if kind == "mola":
            if int(params["mola_num_null_experts"]) != 2 * int(num_experts):
                raise SystemExit(f"{run['id']}: MoLA null experts must be 2*N")
        if kind == "adamole":
            expected = 0.5 / float(num_experts)
            if abs(float(params["adamole_max_threshold"]) - expected) > 1e-12:
                raise SystemExit(f"{run['id']}: AdaMoLE threshold must be 0.5/N")


def _profile_payload(
    *,
    profile_name: str,
    base_profile: str,
    experiment_tag: str,
    output_root: Path,
    runs: list[dict[str, Any]],
    calibrated: bool,
) -> dict[str, Any]:
    output_root_str = str(output_root)
    profile: dict[str, Any] = {
        "extends": base_profile,
        "output_root": output_root_str,
        "logs_dir": f"{output_root_str}/logs",
        "run_naming": {
            "experiment_tag": experiment_tag + ("-calibrated" if calibrated else ""),
            "language_count_hint": 10,
        },
        "tokenization": {
            "enabled": False,
            "output_dir": "outputs/pipeline_runs/moe_study_profile_otus_cluster_10_langs/tokenized",
        },
        "training": {
            "output_dir": f"{output_root_str}/runs",
            "report_to": "none",
            "executor": {"time": "03:51:00" if calibrated else "00:45:00"},
            "eval": {"enabled": False, "strategy": "no"},
            "train": {
                "logging_steps": 50 if calibrated else 1,
                "save_strategy": "no" if calibrated else "steps",
                "save_steps": 1 if calibrated else 1,
                "save_total_limit": 1 if calibrated else 1,
                "cleanup_large_artifacts_after_train": False if calibrated else True,
            },
            "runs": runs,
        },
        "summary": {
            "baseline_id": runs[0]["id"],
            "summary_csv": f"{output_root_str}/summary.csv",
            "gate_json": f"{output_root_str}/gate.json",
        },
        "baseline_eval": {"enabled": False},
        "wandb": {"enabled": False, "mode": "disabled", "project": "moe-study-expert-head-sweep-r2"},
        "wandb_mode": "disabled",
        "stage_executors": {
            "train": {
                "extra_sbatch_flags": ["--signal=B:USR1@900"],
            }
        },
    }
    if calibrated:
        profile["training"]["train"]["num_train_epochs"] = 1.0
        profile["final_eval_loss"] = {
            "enabled": True,
            "output_subdir": "final_eval_loss",
            "batch_size": "train",
        }
        profile["benchmark"] = {
            "enabled": True,
            "lang_mode": "with_ids",
            "batch_size": "auto",
            "output_dir": f"{output_root_str}/benchmarks",
        }
        profile["report"] = {
            "enabled": True,
            "output_xlsx": f"{output_root_str}/{profile_name}.xlsx",
        }
    else:
        profile["training"]["train"].update(
            {
                "batch_size": 1,
                "grad_accum": 1,
                "max_steps": 1,
            }
        )
        profile["final_eval_loss"] = {"enabled": False}
        profile["benchmark"] = {"enabled": False, "output_dir": f"{output_root_str}/benchmarks"}
        profile["report"] = {"enabled": False, "output_xlsx": f"{output_root_str}/report.xlsx"}
    return {profile_name: profile}


def write_profile(args: argparse.Namespace) -> int:
    base_profile = _arg_profile(args, "base_profile", "base_profile")
    source_profile = _arg_profile(args, "source_profile", "source_profile")
    base = _load_profile(base_profile)
    runs = build_sweep_runs(base)
    payload = _profile_payload(
        profile_name=source_profile,
        base_profile=base_profile,
        experiment_tag=str(_preset(args.cluster)["experiment_tag"]),
        output_root=_arg_path(args, "output_root", "output_root"),
        runs=runs,
        calibrated=False,
    )
    path = _arg_path(args, "output_profile", "profile_path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[expert_head_sweep] wrote {len(runs)} runs to {path}")
    return 0


def validate_profile(args: argparse.Namespace) -> int:
    profile_name = _arg_profile(args, "profile", "source_profile")
    profile = _load_profile(profile_name)
    runs = profile["training"]["runs"]
    validate_runs(runs)
    anchors = _anchor_runs(runs)
    print(f"[expert_head_sweep] profile={profile_name} runs={len(runs)} anchors={len(anchors)}")
    for approach, run in anchors.items():
        print(f"{approach}\t{run['id']}")
    return 0


def _anchor_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    anchors: dict[str, dict[str, Any]] = {}
    for run in runs:
        approach = _approach_key(run)
        num_experts = int(_run_num_experts(run) or 0)
        top_k = _run_top_k(run)
        is_anchor = num_experts == 32 and (top_k is None or top_k == 16)
        if is_anchor:
            anchors[approach] = run
    missing = sorted(set(APPROACH_BY_KIND.values()) - set(anchors.keys()))
    if missing:
        raise SystemExit(f"missing calibration anchors for approaches: {missing}")
    return anchors


def _parse_batch_values(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise SystemExit("--batches must contain positive integers")
    return sorted(dict.fromkeys(values))


def _default_disable_gc(run: dict[str, Any]) -> bool:
    train = run.get("train", {}) if isinstance(run.get("train"), dict) else {}
    if "disable_gradient_checkpointing" in train:
        return bool(train["disable_gradient_checkpointing"])
    return str(run["kind"]) in {"moelora", "hmora"}


def _gc_modes(raw: str, run: dict[str, Any]) -> list[bool]:
    if raw == "default":
        return [_default_disable_gc(run)]
    if raw == "both":
        return [False, True]
    if raw == "on":
        return [False]
    if raw == "off":
        return [True]
    raise SystemExit(f"unsupported --gc-modes: {raw}")


def _probe_id(anchor_id: str, batch_size: int, disable_gc: bool) -> str:
    return f"{anchor_id}__bs{batch_size}__gc{'off' if disable_gc else 'on'}"


def _stage_train_cfg(profile: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(profile["training"])
    stage_exec = profile.get("stage_executors", {}).get("train", {})
    if isinstance(stage_exec, dict):
        executor_cfg = deepcopy(stage_exec)
        executor_cfg.update(cfg.get("executor", {}) if isinstance(cfg.get("executor"), dict) else {})
        cfg["executor"] = executor_cfg
    return cfg


def _run_env() -> dict[str, str]:
    env = os.environ.copy()
    python_exec = Path(sys.executable)
    env["PYTHON_EXECUTABLE"] = str(python_exec)
    env["PATH"] = f"{python_exec.parent}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'LLaMA-Factory' / 'src'}:{env.get('PYTHONPATH', '')}"
    env["PROOF_PIPELINE_REPO_ROOT"] = str(REPO_ROOT)
    env["PROOF_PIPELINE_RUN_TIMESTAMP"] = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    env["PROOF_PIPELINE_CODE_SIG"] = _compute_pipeline_code_signature(REPO_ROOT)
    env.setdefault("WANDB_MODE", "disabled")
    return env


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _append_csv_row(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields)
    existing_rows: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            existing_rows = list(reader)
        for field in fields:
            if field not in fieldnames:
                fieldnames.append(field)
        if reader.fieldnames != fieldnames:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows({field: old.get(field, "") for field in fieldnames} for old in existing_rows)

    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def _telemetry_path(row: dict[str, Any]) -> Path:
    return REPO_ROOT / Path(row["output_root"]) / "logs" / "gpu_telemetry" / f"{row['probe_id']}.gpu.csv"


def _wrap_cmd_with_gpu_telemetry(cmd: list[str], telemetry_path: Path, interval_seconds: int) -> list[str]:
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    interval = max(1, int(interval_seconds))
    wrapped = f"""set -u
TELEMETRY_PATH={shlex.quote(str(telemetry_path))}
mkdir -p "$(dirname "$TELEMETRY_PATH")"
echo "timestamp,index,uuid,gpu_util_pct,mem_util_pct,mem_used_mib,mem_total_mib" > "$TELEMETRY_PATH"
telemetry_loop() {{
  while true; do
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits \\
        | awk -v ts="$ts" 'BEGIN {{ FS=", *"; OFS="," }} {{ print ts,$1,$2,$3,$4,$5,$6 }}' >> "$TELEMETRY_PATH" || true
    fi
    sleep {interval}
  done
}}
telemetry_loop &
TELEMETRY_PID=$!
cleanup_telemetry() {{
  kill "$TELEMETRY_PID" 2>/dev/null || true
  wait "$TELEMETRY_PID" 2>/dev/null || true
}}
trap cleanup_telemetry EXIT
set +e
{shlex.join(cmd)}
RC=$?
set -e
exit "$RC"
"""
    return ["bash", "-lc", wrapped]


def _probe_rows(profile: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    batches = _parse_batch_values(args.batches)
    selected_approaches = set(getattr(args, "approach", []) or [])
    explicit_run_ids = set(getattr(args, "anchor_run_id", []) or [])
    if explicit_run_ids:
        by_id = {str(run.get("id", "")): run for run in profile["training"]["runs"]}
        missing = sorted(explicit_run_ids - set(by_id))
        if missing:
            raise SystemExit(f"unknown --anchor-run-id values: {missing}")
        anchors = {_approach_key(by_id[run_id]): by_id[run_id] for run_id in explicit_run_ids}
    else:
        anchors = _anchor_runs(profile["training"]["runs"])
    rows: list[dict[str, Any]] = []
    for approach, anchor_run in sorted(anchors.items()):
        if selected_approaches and approach not in selected_approaches:
            continue
        for disable_gc in _gc_modes(args.gc_modes, anchor_run):
            for batch_size in batches:
                probe_id = _probe_id(str(anchor_run["id"]), batch_size, disable_gc)
                output_root = Path(args.output_root) / approach / f"bs{batch_size}_gc{'off' if disable_gc else 'on'}"
                rows.append(
                    {
                        "approach": approach,
                        "anchor_id": anchor_run["id"],
                        "anchor_run": anchor_run,
                        "probe_id": probe_id,
                        "kind": anchor_run["kind"],
                        "num_experts": _run_num_experts(anchor_run),
                        "top_k": _run_top_k(anchor_run) or "",
                        "batch_size": batch_size,
                        "disable_gradient_checkpointing": disable_gc,
                        "output_root": output_root,
                    }
                )
    return rows


def _probe_group_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["approach"]), str(row["disable_gradient_checkpointing"]).lower()


def _is_hard_probe_failure(row: dict[str, str]) -> bool:
    if str(row.get("success", "")).lower() == "true":
        return False
    state = str(row.get("state", "")).upper()
    if state.startswith("CANCELLED"):
        return False
    if not state and str(row.get("returncode", "")).strip() == "0" and not str(row.get("train_runtime", "")).strip():
        return False
    return True


def _job_finished(job_id: str) -> tuple[bool, str, str]:
    sacct = subprocess.run(
        ["sacct", "-j", str(job_id), "--format=JobIDRaw,State,ExitCode", "--parsable2", "--noheader"],
        capture_output=True,
        text=True,
    )
    if sacct.returncode == 0 and (sacct.stdout or "").strip():
        for line in sacct.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 3 or parts[0] != str(job_id):
                continue
            state = parts[1]
            exit_code = parts[2].split(":", 1)[0]
            terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
            return any(state.startswith(item) for item in terminal), state, exit_code
    show = subprocess.run(["scontrol", "show", "job", "-o", str(job_id)], capture_output=True, text=True)
    text = show.stdout or ""
    state_match = re.search(r"\bJobState=(\w+)\b", text)
    exit_match = re.search(r"\bExitCode=([0-9]+):", text)
    state = state_match.group(1) if state_match else ""
    exit_code = exit_match.group(1) if exit_match else ""
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
    return any(state.startswith(item) for item in terminal), state, exit_code


def _append_manifest(path: Path, row: dict[str, Any]) -> None:
    fields = [
        "approach",
        "anchor_id",
        "probe_id",
        "kind",
        "num_experts",
        "top_k",
        "batch_size",
        "disable_gradient_checkpointing",
        "job_id",
        "output_root",
        "run_dir",
        "train_log",
        "slurm_log",
        "telemetry_path",
    ]
    _append_csv_row(path, fields, row)


def _submit_probe_row(
    *,
    source: dict[str, Any],
    train_cfg: dict[str, Any],
    executor: SlurmExecutor,
    env: dict[str, str],
    manifest_path: Path,
    row: dict[str, Any],
    max_steps: int,
    dry_run: bool,
    telemetry_interval_seconds: int,
    gpu_telemetry: bool,
) -> str:
    run = deepcopy(row["anchor_run"])
    run["id"] = row["probe_id"]
    train = deepcopy(run.get("train", {})) if isinstance(run.get("train"), dict) else {}
    train.update(
        {
            "batch_size": int(row["batch_size"]),
            "grad_accum": 1,
            "disable_gradient_checkpointing": bool(row["disable_gradient_checkpointing"]),
        }
    )
    run["train"] = train

    common = deepcopy(train_cfg)
    common["runs"] = [run]
    common["train"] = deep_merge(
        common.get("train", {}),
        {
            "max_steps": int(max_steps),
            "batch_size": int(row["batch_size"]),
            "grad_accum": 1,
            "logging_steps": 1,
            "save_strategy": "no",
            "save_total_limit": 0,
            "skip_final_save": True,
            "cleanup_large_artifacts_after_train": True,
        },
    )
    common["eval"] = {"enabled": False, "strategy": "no"}
    common["report_to"] = "none"

    output_root = Path(row["output_root"])
    run_dir = REPO_ROOT / output_root / "runs" / row["probe_id"]
    train_log = REPO_ROOT / output_root / "logs" / "train_runs" / row["probe_id"] / f"{row['probe_id']}.train.log"
    slurm_log = train_log.with_suffix(".train.slurm.log")
    cmd, run_env = build_training_cmd(
        repo_root=REPO_ROOT,
        env=env,
        output_dir=run_dir,
        run_cfg=run,
        common=common,
        language_map_path=REPO_ROOT / source["language_map_path"],
        tokenized_path=REPO_ROOT / source["tokenization"]["output_dir"],
        embedding_training_cfg=source.get("embedding_training", {}),
        run_name=row["probe_id"],
    )
    telemetry_path = _telemetry_path(row)
    if gpu_telemetry:
        cmd = _wrap_cmd_with_gpu_telemetry(cmd, telemetry_path, telemetry_interval_seconds)
    job_id = executor.submit(
        cmd,
        run_env,
        dry_run,
        train_log,
        job_name=f"ehbs_{row['approach']}_bs{row['batch_size']}",
        slurm_output=slurm_log,
    )
    if job_id:
        _append_manifest(
            manifest_path,
            {
                **row,
                "job_id": job_id,
                "run_dir": run_dir,
                "train_log": train_log,
                "slurm_log": slurm_log,
                "telemetry_path": telemetry_path if gpu_telemetry else "",
            },
        )
    return str(job_id or "")


def _async_submit_context(source: dict[str, Any]) -> tuple[dict[str, Any], SlurmExecutor, dict[str, str]]:
    train_cfg = _stage_train_cfg(source)
    executor = build_executor(source, train_cfg)
    if not isinstance(executor, SlurmExecutor):
        raise SystemExit("batch search submission requires a Slurm executor")
    return train_cfg, executor, _run_env()


def submit_wave(args: argparse.Namespace) -> int:
    source = _load_profile(_arg_profile(args, "profile", "source_profile"))
    validate_runs(source["training"]["runs"])
    train_cfg, executor, env = _async_submit_context(source)
    if getattr(args, "time_limit", ""):
        executor.time_limit = str(args.time_limit).strip()
        executor.extra_sbatch_flags = [
            flag for flag in executor.extra_sbatch_flags if not str(flag).strip().startswith("--signal=")
        ]
    probe_root = _arg_path(args, "output_root", "probe_root")
    args.output_root = str(probe_root)
    manifest_path = Path(args.manifest) if args.manifest else probe_root / "probe_manifest.csv"
    result_path = Path(args.results) if args.results else probe_root / "probe_results.csv"
    submitted_probe_ids = {row.get("probe_id", "") for row in _manifest_rows(manifest_path)}
    results_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    if result_path.exists():
        for row in _load_probe_results(result_path):
            results_by_key.setdefault(_probe_group_key(row), []).append(row)

    manifest_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in _manifest_rows(manifest_path):
        manifest_by_key.setdefault(_probe_group_key(row), []).append(row)

    candidates_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in _probe_rows(source, args):
        candidates_by_key.setdefault(_probe_group_key(row), []).append(row)

    submitted_count = 0
    for key, candidates in sorted(candidates_by_key.items()):
        approach, disable_gc_raw = key
        candidates.sort(key=lambda item: int(item["batch_size"]))
        results = results_by_key.get(key, [])
        finished_probe_ids = {row.get("probe_id", "") for row in results}
        waiting = False
        for row in manifest_by_key.get(key, []):
            if row.get("probe_id", "") in finished_probe_ids:
                continue
            finished, state, _exit_code = _job_finished(row["job_id"])
            if not finished:
                waiting = True
                print(
                    f"[expert_head_batch_wave] waiting {approach} gc={disable_gc_raw} "
                    f"probe={row['probe_id']} state={state or 'UNKNOWN'}"
                )
                break
        if waiting:
            continue
        if any(_is_hard_probe_failure(row) for row in results):
            print(f"[expert_head_batch_wave] stopped {approach} gc={disable_gc_raw} after first failed probe")
            continue
        max_success = max((int(row["batch_size"]) for row in results if str(row.get("success", "")).lower() == "true"), default=0)
        next_row = None
        for row in candidates:
            if int(row["batch_size"]) <= max_success:
                continue
            if row["probe_id"] in submitted_probe_ids:
                continue
            next_row = row
            break
        if next_row is None:
            print(f"[expert_head_batch_wave] complete {approach} gc={disable_gc_raw} within requested batches")
            continue
        job_id = _submit_probe_row(
            source=source,
            train_cfg=train_cfg,
            executor=executor,
            env=env,
            manifest_path=manifest_path,
            row=next_row,
            max_steps=int(args.max_steps),
            dry_run=bool(args.dry_run),
            telemetry_interval_seconds=int(args.telemetry_interval_seconds),
            gpu_telemetry=not bool(args.disable_gpu_telemetry),
        )
        if args.dry_run:
            submitted_count += 1
            print(f"[expert_head_batch_wave] dry-run built {next_row['probe_id']}")
        elif job_id:
            submitted_probe_ids.add(next_row["probe_id"])
            submitted_count += 1
            print(f"[expert_head_batch_wave] {next_row['probe_id']} job_id={job_id}")
        if args.max_submissions and submitted_count >= int(args.max_submissions):
            break
    print(f"[expert_head_batch_wave] submitted={submitted_count} manifest={manifest_path}")
    return 0


def _as_float(raw: Any) -> float | None:
    try:
        if raw is None or str(raw).strip() == "":
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def _trainable_from_log(path: Path) -> dict[str, str]:
    pattern = re.compile(r"trainable params:\s*([\d,]+)\s*\|\|\s*all params:\s*([\d,]+)\s*\|\|\s*trainable%:\s*([0-9.]+)")
    if not path.exists():
        return {}
    last = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            last = match
    if last is None:
        return {}
    return {
        "trainable_params": last.group(1).replace(",", ""),
        "all_params": last.group(2).replace(",", ""),
        "trainable_pct": last.group(3),
    }


def _gpu_telemetry_from_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    samples = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            gpu_util = _as_float(row.get("gpu_util_pct"))
            mem_util = _as_float(row.get("mem_util_pct"))
            mem_used = _as_float(row.get("mem_used_mib"))
            mem_total = _as_float(row.get("mem_total_mib"))
            if gpu_util is None or mem_util is None or mem_used is None or mem_total in {None, 0.0}:
                continue
            samples.append((gpu_util, mem_util, mem_used, float(mem_total)))
    if not samples:
        return {}
    gpu_utils = [sample[0] for sample in samples]
    mem_utils = [sample[1] for sample in samples]
    mem_used_values = [sample[2] for sample in samples]
    mem_total = max(sample[3] for sample in samples)
    peak_mem = max(mem_used_values)
    avg_mem = sum(mem_used_values) / len(mem_used_values)
    return {
        "gpu_samples": str(len(samples)),
        "gpu_util_avg_pct": f"{sum(gpu_utils) / len(gpu_utils):.3f}",
        "gpu_util_peak_pct": f"{max(gpu_utils):.3f}",
        "gpu_mem_util_avg_pct": f"{sum(mem_utils) / len(mem_utils):.3f}",
        "gpu_mem_util_peak_pct": f"{max(mem_utils):.3f}",
        "gpu_mem_avg_mib": f"{avg_mem:.3f}",
        "gpu_mem_peak_mib": f"{peak_mem:.3f}",
        "gpu_mem_total_mib": f"{mem_total:.3f}",
        "gpu_mem_avg_pct": f"{100.0 * avg_mem / mem_total:.3f}",
        "gpu_mem_peak_pct": f"{100.0 * peak_mem / mem_total:.3f}",
    }


def _append_result(path: Path, row: dict[str, Any]) -> None:
    fields = [
        "approach",
        "anchor_id",
        "probe_id",
        "kind",
        "num_experts",
        "top_k",
        "batch_size",
        "disable_gradient_checkpointing",
        "state",
        "success",
        "returncode",
        "trainable_params",
        "train_runtime",
        "effective_tokens_per_sec",
        "train_loss",
        "gpu_samples",
        "gpu_util_avg_pct",
        "gpu_util_peak_pct",
        "gpu_mem_util_avg_pct",
        "gpu_mem_util_peak_pct",
        "gpu_mem_avg_mib",
        "gpu_mem_peak_mib",
        "gpu_mem_total_mib",
        "gpu_mem_avg_pct",
        "gpu_mem_peak_pct",
        "output_root",
    ]
    _append_csv_row(path, fields, row)


def _load_probe_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cleanup_probe_artifacts(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    removable_files = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "all_results.json",
        "generation_config.json",
        "optimizer.pt",
        "pytorch_model.bin",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    }
    for child in run_dir.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            shutil.rmtree(child, ignore_errors=True)
            continue
        if child.is_file() and child.name in removable_files:
            child.unlink(missing_ok=True)


def collect_results(args: argparse.Namespace) -> int:
    probe_root = _arg_path(args, "output_root", "probe_root")
    manifest_path = Path(args.manifest) if args.manifest else probe_root / "probe_manifest.csv"
    manifest = _manifest_rows(manifest_path)
    results_path = Path(args.results) if args.results else probe_root / "probe_results.csv"
    existing_rows = {row.get("probe_id", ""): row for row in _load_probe_results(results_path)}
    collected = 0
    cleaned = 0
    pending = 0
    for row in manifest:
        probe_id = row["probe_id"]
        if probe_id in existing_rows:
            if bool(getattr(args, "cleanup_successful", False)) and str(existing_rows[probe_id].get("success", "")).lower() == "true":
                _cleanup_probe_artifacts(Path(row["run_dir"]))
                cleaned += 1
            continue
        finished, state, exit_code = _job_finished(row["job_id"])
        if not finished:
            pending += 1
            continue
        run_dir = Path(row["run_dir"])
        train_results = run_dir / "train_results.json"
        train_payload = {}
        if train_results.exists():
            train_payload = json.loads(train_results.read_text(encoding="utf-8"))
        effective_tokens_per_sec = train_payload.get("effective_tokens_per_sec") or train_payload.get("train_tokens_per_second")
        if not effective_tokens_per_sec:
            tokens_seen = _as_float(train_payload.get("num_input_tokens_seen"))
            train_runtime = _as_float(train_payload.get("train_runtime"))
            if tokens_seen is not None and train_runtime not in {None, 0.0}:
                effective_tokens_per_sec = f"{tokens_seen / float(train_runtime):.6f}"
        trainable = _trainable_from_log(Path(row["slurm_log"]))
        telemetry_path_raw = str(row.get("telemetry_path", "") or "").strip()
        telemetry = _gpu_telemetry_from_csv(Path(telemetry_path_raw)) if telemetry_path_raw else {}
        success = state == "COMPLETED" and train_results.exists()
        _append_result(
            results_path,
            {
                **row,
                "state": state,
                "success": success,
                "returncode": exit_code,
                "trainable_params": trainable.get("trainable_params", ""),
                "train_runtime": train_payload.get("train_runtime", ""),
                "effective_tokens_per_sec": effective_tokens_per_sec or "",
                "train_loss": train_payload.get("train_loss", ""),
                **telemetry,
            },
        )
        collected += 1
        print(f"[expert_head_batch_collect] {probe_id} state={state} success={success}")
        if success and bool(getattr(args, "cleanup_successful", False)):
            _cleanup_probe_artifacts(run_dir)
            cleaned += 1
    print(f"[expert_head_batch_collect] collected={collected} pending={pending} cleaned={cleaned} results={results_path}")
    return 0


def _parse_approach_batches(raw_batches: list[str]) -> dict[str, int]:
    batches: dict[str, int] = {}
    valid = set(APPROACH_BY_KIND.values())
    for raw in raw_batches:
        if "=" not in raw:
            raise SystemExit(f"--batch-override expects approach=N, got: {raw}")
        approach, value = raw.split("=", 1)
        approach = approach.strip()
        if approach not in valid:
            raise SystemExit(f"unknown batch override '{approach}'. Available: {sorted(valid)}")
        batch_size = int(value)
        if batch_size <= 0:
            raise SystemExit(f"batch override must be positive for {approach}")
        batches[approach] = batch_size
    return batches


def _best_settings_by_approach(
    rows: list[dict[str, str]],
    batch_overrides: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    batch_overrides = batch_overrides or {}
    by_approach: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if str(row.get("success", "")).lower() == "true":
            by_approach.setdefault(row["approach"], []).append(row)
    best: dict[str, dict[str, Any]] = {}
    for approach, candidates in by_approach.items():
        if approach in batch_overrides:
            requested = int(batch_overrides[approach])
            candidates = [row for row in candidates if int(row["batch_size"]) == requested]
            if not candidates:
                raise SystemExit(f"no successful calibration result for {approach} batch_size={requested}")
        selected = max(
            candidates,
            key=lambda row: (
                int(row["batch_size"]),
                float(row.get("effective_tokens_per_sec") or 0.0),
            ),
        )
        best[approach] = {
            "approach": approach,
            "anchor_id": selected["anchor_id"],
            "batch_size": int(selected["batch_size"]),
            "disable_gradient_checkpointing": str(selected["disable_gradient_checkpointing"]).lower() == "true",
            "effective_tokens_per_sec": selected.get("effective_tokens_per_sec", ""),
            "gpu_mem_peak_pct": selected.get("gpu_mem_peak_pct", ""),
            "gpu_util_avg_pct": selected.get("gpu_util_avg_pct", ""),
        }
    return best


def _load_calibration_rows(args: argparse.Namespace, probe_root: Path) -> list[dict[str, str]]:
    paths: list[Path] = []
    if args.results:
        paths.append(Path(args.results))
    else:
        paths.append(probe_root / "probe_results.csv")
    for raw in getattr(args, "extra_results", []) or []:
        paths.append(Path(raw))

    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(_load_probe_results(path))
    return rows


def _parse_approach_caps(raw_caps: list[str]) -> dict[str, int]:
    caps: dict[str, int] = {}
    valid = set(APPROACH_BY_KIND.values())
    for raw in raw_caps:
        if "=" not in raw:
            raise SystemExit(f"--cap-approach-max-experts expects approach=N, got: {raw}")
        approach, value = raw.split("=", 1)
        approach = approach.strip()
        if approach not in valid:
            raise SystemExit(f"unknown approach cap '{approach}'. Available: {sorted(valid)}")
        cap = int(value)
        if cap <= 0:
            raise SystemExit(f"approach cap must be positive for {approach}")
        caps[approach] = cap
    return caps


def _distributed_process_count(profile: dict[str, Any]) -> int:
    distributed = profile.get("training", {}).get("distributed", {})
    if not isinstance(distributed, dict) or not distributed.get("enabled", False):
        return 1
    return max(1, int(distributed.get("num_processes", 1)))


def _closest_grad_accum(target_global_batch: int, batch_size: int, process_count: int, max_grad_accum: int) -> int:
    unit = max(1, int(batch_size) * int(process_count))
    candidates = {1, max(1, math.floor(target_global_batch / unit)), max(1, math.ceil(target_global_batch / unit))}
    candidates = {min(max_grad_accum, value) for value in candidates if value >= 1}
    return min(
        candidates,
        key=lambda value: (
            abs((unit * value) - target_global_batch),
            unit * value > target_global_batch,
            -value,
        ),
    )


def write_calibrated_profile(args: argparse.Namespace) -> int:
    source = _load_profile(_arg_profile(args, "profile", "source_profile"))
    validate_runs(source["training"]["runs"])
    probe_root = _arg_path(args, "probe_root", "probe_root")
    results = _load_calibration_rows(args, probe_root)
    batch_overrides = _parse_approach_batches(getattr(args, "batch_override", []) or [])
    best = _best_settings_by_approach(results, batch_overrides=batch_overrides)
    missing = sorted(set(APPROACH_BY_KIND.values()) - set(best.keys()))
    if missing:
        raise SystemExit(f"missing successful batch-search results for approaches: {missing}")

    target_global_batch = int(args.target_global_batch)
    max_grad_accum = int(args.max_grad_accum)
    process_count = _distributed_process_count(source)
    approach_caps = _parse_approach_caps(getattr(args, "cap_approach_max_experts", []) or [])
    calibrated_runs: list[dict[str, Any]] = []
    settings_rows: list[dict[str, Any]] = []
    for run in source["training"]["runs"]:
        run = deepcopy(run)
        approach = _approach_key(run)
        cap = approach_caps.get(approach)
        if cap is not None and int(_run_num_experts(run) or 0) > cap:
            continue
        setting = best[approach]
        batch_size = int(setting["batch_size"])
        grad_accum = _closest_grad_accum(target_global_batch, batch_size, process_count, max_grad_accum)
        train = deepcopy(run.get("train", {})) if isinstance(run.get("train"), dict) else {}
        train.update(
            {
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "disable_gradient_checkpointing": bool(setting["disable_gradient_checkpointing"]),
            }
        )
        run["train"] = train
        calibrated_runs.append(run)
        settings_rows.append(
            {
                "run_id": run["id"],
                "approach": approach,
                "anchor_id": setting["anchor_id"],
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "num_processes": process_count,
                "actual_global_batch": batch_size * process_count * grad_accum,
                "target_global_batch": target_global_batch,
                "disable_gradient_checkpointing": bool(setting["disable_gradient_checkpointing"]),
                "effective_tokens_per_sec": setting.get("effective_tokens_per_sec", ""),
                "gpu_mem_peak_pct": setting.get("gpu_mem_peak_pct", ""),
                "gpu_util_avg_pct": setting.get("gpu_util_avg_pct", ""),
            }
        )

    payload = _profile_payload(
        profile_name=_arg_profile(args, "calibrated_profile", "calibrated_profile"),
        base_profile=str(_preset(args.cluster)["base_profile"]),
        experiment_tag=str(_preset(args.cluster)["experiment_tag"]),
        output_root=_arg_path(args, "calibrated_output_root", "calibrated_output_root"),
        runs=calibrated_runs,
        calibrated=True,
    )
    profile_path = _arg_path(args, "output_profile", "calibrated_profile_path")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    settings_path = Path(args.output_settings) if args.output_settings else probe_root / "approach_batch_settings.csv"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with settings_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(settings_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(settings_rows)
    print(f"[expert_head_sweep] wrote calibrated profile: {profile_path}")
    print(f"[expert_head_sweep] wrote settings: {settings_path}")
    return 0


def list_anchors(args: argparse.Namespace) -> int:
    profile = _load_profile(_arg_profile(args, "profile", "source_profile"))
    validate_runs(profile["training"]["runs"])
    for approach, run in sorted(_anchor_runs(profile["training"]["runs"]).items()):
        print(f"{approach}\t{run['id']}\tdefault_gc_off={_default_disable_gc(run)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and batch-calibrate the expert/head-count r=2 sweep.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_cluster_arg(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--cluster", choices=sorted(CLUSTER_PRESETS), default=DEFAULT_CLUSTER)

    write_parser = subparsers.add_parser("write-profile", help="Write the generated source/probe profile.")
    add_cluster_arg(write_parser)
    write_parser.add_argument("--base-profile", default="")
    write_parser.add_argument("--source-profile", default="")
    write_parser.add_argument("--output-profile", default="")
    write_parser.add_argument("--output-root", default="")
    write_parser.set_defaults(func=write_profile)

    validate_parser = subparsers.add_parser("validate-profile", help="Validate the generated profile grid.")
    add_cluster_arg(validate_parser)
    validate_parser.add_argument("--profile", default="")
    validate_parser.set_defaults(func=validate_profile)

    anchors_parser = subparsers.add_parser("list-anchors", help="List one worst-case batch-search anchor per approach.")
    add_cluster_arg(anchors_parser)
    anchors_parser.add_argument("--profile", default="")
    anchors_parser.set_defaults(func=list_anchors)

    wave_parser = subparsers.add_parser("submit-wave", help="Submit the next batch-size probe per approach.")
    add_cluster_arg(wave_parser)
    wave_parser.add_argument("--profile", default="")
    wave_parser.add_argument("--batches", default=",".join(str(value) for value in DEFAULT_BATCHES))
    wave_parser.add_argument("--gc-modes", default="default", choices=["default", "both", "on", "off"])
    wave_parser.add_argument("--max-steps", type=int, default=20)
    wave_parser.add_argument("--time-limit", default="", help="Optional SLURM walltime override for probe jobs, e.g. 00:10:00.")
    wave_parser.add_argument("--output-root", default="")
    wave_parser.add_argument("--manifest", default="")
    wave_parser.add_argument("--results", default="")
    wave_parser.add_argument("--approach", action="append", default=[], help="Probe only one approach key; can be repeated.")
    wave_parser.add_argument("--anchor-run-id", action="append", default=[], help="Use an explicit sweep run id as the calibration anchor; can be repeated.")
    wave_parser.add_argument("--max-submissions", type=int, default=0)
    wave_parser.add_argument("--telemetry-interval-seconds", type=int, default=5)
    wave_parser.add_argument("--disable-gpu-telemetry", action="store_true")
    wave_parser.add_argument("--dry-run", action="store_true")
    wave_parser.set_defaults(func=submit_wave)

    collect_parser = subparsers.add_parser("collect-results", help="Collect completed probe jobs.")
    add_cluster_arg(collect_parser)
    collect_parser.add_argument("--output-root", default="")
    collect_parser.add_argument("--manifest", default="")
    collect_parser.add_argument("--results", default="")
    collect_parser.add_argument("--cleanup-successful", action="store_true")
    collect_parser.set_defaults(func=collect_results)

    calibrated_parser = subparsers.add_parser("write-calibrated-profile", help="Write full training profile from stable per-approach batch sizes.")
    add_cluster_arg(calibrated_parser)
    calibrated_parser.add_argument("--profile", default="")
    calibrated_parser.add_argument("--results", default="")
    calibrated_parser.add_argument("--extra-results", action="append", default=[], help="Additional probe_results.csv files to merge into calibration; can be repeated.")
    calibrated_parser.add_argument("--target-global-batch", type=int, default=DEFAULT_TARGET_GLOBAL_BATCH)
    calibrated_parser.add_argument("--max-grad-accum", type=int, default=DEFAULT_MAX_GRAD_ACCUM)
    calibrated_parser.add_argument("--calibrated-profile", default="")
    calibrated_parser.add_argument("--output-profile", default="")
    calibrated_parser.add_argument("--output-root", dest="calibrated_output_root", default="")
    calibrated_parser.add_argument("--probe-root", default="")
    calibrated_parser.add_argument("--output-settings", default="")
    calibrated_parser.add_argument(
        "--batch-override",
        action="append",
        default=[],
        help="Use a specific successful calibration batch for an approach, e.g. mov=11. Can be repeated.",
    )
    calibrated_parser.add_argument(
        "--cap-approach-max-experts",
        action="append",
        default=[],
        help="Drop sweep runs above an approach-specific expert/head cap, e.g. moelora=4. Can be repeated.",
    )
    calibrated_parser.set_defaults(func=write_calibrated_profile)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
