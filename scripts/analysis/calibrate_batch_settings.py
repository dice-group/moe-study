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
import time
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
        "source_profile": "moe_study_rank_expert_sweep_otus_cluster_10_langs",
        "calibrated_profile": "moe_study_rank_expert_sweep_otus_cluster_10_langs_calibrated",
        "experiment_tag": "moe-study-rank-expert-sweep-10langs",
        "profile_path": Path("configs/pipeline/profiles/moe_study_rank_expert_sweep.json"),
        "calibrated_profile_path": Path("configs/pipeline/profiles/moe_study_rank_expert_sweep_calibrated.generated.json"),
        "output_root": Path("outputs/pipeline_runs/moe_study_rank_expert_sweep_otus_cluster_10_langs"),
        "calibrated_output_root": Path("outputs/pipeline_runs/moe_study_rank_expert_sweep_otus_cluster_10_langs_calibrated"),
        "probe_root": Path("outputs/pipeline_runs/moe_study_batch_calibration"),
    },
    "noctua2_4gpu": {
        "base_profile": "moe_study_profile_noctua2_a100_4gpu_10_langs",
        "source_profile": "moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs",
        "calibrated_profile": "moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_calibrated",
        "experiment_tag": "moe-study-rank-expert-sweep-noctua2-a100-4gpu-10langs",
        "profile_path": Path("configs/pipeline/profiles/moe_study_rank_expert_sweep_noctua2_4gpu.generated.json"),
        "calibrated_profile_path": Path("configs/pipeline/profiles/moe_study_rank_expert_sweep_noctua2_4gpu_calibrated.generated.json"),
        "output_root": Path("outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs"),
        "calibrated_output_root": Path("outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_calibrated"),
        "probe_root": Path("outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_4gpu_batch_calibration"),
    },
}
SOURCE_PROFILE = str(CLUSTER_PRESETS[DEFAULT_CLUSTER]["source_profile"])
CALIBRATED_PROFILE = str(CLUSTER_PRESETS[DEFAULT_CLUSTER]["calibrated_profile"])
PROBE_PROFILE = "moe_study_batch_probe_tmp"
PROBE_PROFILE_PATH = Path("configs/pipeline/profiles/moe_study_batch_probe.generated.json")
DEFAULT_RESULTS = Path(CLUSTER_PRESETS[DEFAULT_CLUSTER]["probe_root"]) / "probe_results.csv"
DEFAULT_MANIFEST = Path(CLUSTER_PRESETS[DEFAULT_CLUSTER]["probe_root"]) / "probe_manifest.csv"
DEFAULT_CALIBRATED_PROFILE_PATH = Path(CLUSTER_PRESETS[DEFAULT_CLUSTER]["calibrated_profile_path"])
DEFAULT_BATCHES = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32]
DEFAULT_TARGET_GLOBAL_BATCH = 48
DEFAULT_MAX_GRAD_ACCUM = 32


ANCHOR_IDS = [
    "lora_r2",
    "lora_r64",
    "lora_r1024",
    "cola_r2",
    "cola_r64",
    "cola_r128",
    "cola_r256",
    "cola_r512",
    "cola_r1024",
    "hydralora_r2",
    "hydralora_r64",
    "hydralora_r128",
    "hydralora_r256",
    "hydralora_r512",
    "hydralora_r1024",
    "mtllora_r2",
    "mtllora_r64",
    "mtllora_r128",
    "mtllora_r256",
    "mtllora_r512",
    "mtllora_r1024",
    "moelora_tr16_e8_k2",
    "moelora_tr128_e8_k2",
    "moelora_tr1024_e8_k2",
    "adamole_tr16_e8",
    "adamole_tr128_e8",
    "adamole_tr1024_e8",
    "mola_tr16_e8_k6",
    "mola_tr128_e8_k6",
    "mola_tr1024_e8_k6",
    "hmora_tr16_e8_k2",
    "hmora_tr128_e8_k2",
    "hmora_tr1024_e8_k2",
    "movlora_e8",
]


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


def _probe_root(args: argparse.Namespace) -> Path:
    return _arg_path(args, "output_root", "probe_root")


def _manifest_path(args: argparse.Namespace) -> Path:
    value = getattr(args, "manifest", None)
    if isinstance(value, str) and value.strip():
        return Path(value)
    return _probe_root(args) / "probe_manifest.csv"


def _results_path(args: argparse.Namespace) -> Path:
    value = getattr(args, "results", None)
    if isinstance(value, str) and value.strip():
        return Path(value)
    return _probe_root(args) / "probe_results.csv"


def _run_by_id(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(run["id"]): run for run in profile["training"]["runs"]}


def _parse_batch_values(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise SystemExit("--batches must contain positive integers")
    return sorted(dict.fromkeys(values))


def _rank_value(run_id: str) -> int:
    if "_tr" in run_id:
        return int(run_id.split("_tr", 1)[1].split("_", 1)[0])
    if "_r" in run_id:
        match = re.search(r"_r(\d+)(?:_|$)", run_id)
        if match:
            return int(match.group(1))
    return 0


def _group_key(run: dict[str, Any]) -> str:
    kind = str(run["kind"])
    return "movlora" if kind == "movlora" else kind


def _default_disable_gc(run: dict[str, Any]) -> bool:
    train = run.get("train", {}) if isinstance(run.get("train"), dict) else {}
    kind = str(run["kind"])
    if "disable_gradient_checkpointing" in train:
        return bool(train["disable_gradient_checkpointing"])
    return kind in {"moelora", "hmora"}


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


def _write_probe_profile(
    *,
    source_profile: str,
    anchor_run: dict[str, Any],
    probe_id: str,
    output_root: Path,
    batch_size: int,
    disable_gc: bool,
    max_steps: int,
) -> None:
    run = deepcopy(anchor_run)
    run["id"] = probe_id
    train = deepcopy(run.get("train", {})) if isinstance(run.get("train"), dict) else {}
    train.update(
        {
            "batch_size": batch_size,
            "grad_accum": 1,
            "disable_gradient_checkpointing": disable_gc,
        }
    )
    run["train"] = train
    payload = {
        PROBE_PROFILE: {
            "extends": source_profile,
            "output_root": str(output_root),
            "logs_dir": str(output_root / "logs"),
            "training": {
                "output_dir": str(output_root / "runs"),
                "runs": [run],
                "train": {
                    "max_steps": max_steps,
                    "batch_size": batch_size,
                    "grad_accum": 1,
                    "logging_steps": 1,
                    "save_strategy": "no",
                    "skip_final_save": True,
                    "save_total_limit": 0,
                    "cleanup_large_artifacts_after_train": True,
                },
                "eval": {
                    "enabled": False,
                    "strategy": "no",
                },
            },
            "summary": {
                "baseline_id": probe_id,
                "summary_csv": str(output_root / "summary.csv"),
                "gate_json": str(output_root / "gate.json"),
            },
            "final_eval_loss": {"enabled": False},
            "benchmark": {"enabled": False},
            "baseline_eval": {"enabled": False},
            "report": {"enabled": False},
            "wandb": {"enabled": False, "mode": "disabled"},
            "wandb_mode": "disabled",
        }
    }
    path = REPO_ROOT / PROBE_PROFILE_PATH
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


def _probe_rows(source: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    runs = _run_by_id(source)
    batches = _parse_batch_values(args.batches)
    selected_anchors = list(getattr(args, "anchor_id", []) or [])
    selected_anchor_set = set(selected_anchors)
    anchor_ids = selected_anchors if selected_anchors else ANCHOR_IDS
    rows = []
    for anchor_id in anchor_ids:
        if selected_anchor_set and anchor_id not in selected_anchor_set:
            continue
        anchor_run = runs.get(anchor_id)
        if anchor_run is None:
            raise SystemExit(f"anchor run missing from source profile: {anchor_id}")
        for disable_gc in _gc_modes(args.gc_modes, anchor_run):
            for batch_size in batches:
                probe_id = _probe_id(anchor_id, batch_size, disable_gc)
                output_root = _probe_root(args) / anchor_id / f"bs{batch_size}_gc{'off' if disable_gc else 'on'}"
                rows.append(
                    {
                        "anchor_id": anchor_id,
                        "anchor_run": anchor_run,
                        "probe_id": probe_id,
                        "kind": anchor_run["kind"],
                        "rank_value": _rank_value(anchor_id),
                        "batch_size": batch_size,
                        "disable_gradient_checkpointing": disable_gc,
                        "output_root": output_root,
                    }
                )
                limit = int(getattr(args, "limit", 0) or 0)
                if limit > 0 and len(rows) >= limit:
                    return rows
    return rows


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _append_manifest(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "anchor_id",
        "probe_id",
        "kind",
        "rank_value",
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
            "skip_final_save": True,
            "save_total_limit": 0,
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
        job_name=f"bprobe_{row['probe_id']}",
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
        raise SystemExit("asynchronous probe submission requires a Slurm executor")
    return train_cfg, executor, _run_env()


def _apply_distributed_config_override(train_cfg: dict[str, Any], args: argparse.Namespace) -> None:
    config_file = str(getattr(args, "accelerate_config_file", "") or "").strip()
    if not config_file:
        return
    distributed = train_cfg.setdefault("distributed", {})
    if not isinstance(distributed, dict):
        distributed = {}
        train_cfg["distributed"] = distributed
    distributed["enabled"] = True
    distributed["num_processes"] = int(distributed.get("num_processes", 4) or 4)
    distributed["config_file"] = config_file


def submit_probes(args: argparse.Namespace) -> int:
    source = _load_profile(_arg_profile(args, "profile", "source_profile"))
    train_cfg, executor, env = _async_submit_context(source)
    _apply_distributed_config_override(train_cfg, args)
    manifest_path = _manifest_path(args)
    submitted = {row.get("probe_id", "") for row in _manifest_rows(manifest_path)}
    submitted_count = 0

    for row in _probe_rows(source, args):
        if row["probe_id"] in submitted:
            continue
        job_id = _submit_probe_row(
            source=source,
            train_cfg=train_cfg,
            executor=executor,
            env=env,
            manifest_path=manifest_path,
            row=row,
            max_steps=int(args.max_steps),
            dry_run=bool(args.dry_run),
            telemetry_interval_seconds=int(args.telemetry_interval_seconds),
            gpu_telemetry=not bool(args.disable_gpu_telemetry),
        )
        if job_id:
            submitted_count += 1
            print(f"[batch_probe_submit] {row['probe_id']} job_id={job_id}")
    print(f"[batch_probe_submit] submitted={submitted_count} manifest={manifest_path}")
    return 0


def _probe_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["anchor_id"]), str(row["disable_gradient_checkpointing"]).lower()


def _truthy(raw: Any) -> bool:
    return str(raw).lower() == "true"


def _as_float(raw: Any) -> float | None:
    try:
        if raw is None or str(raw).strip() == "":
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def _submit_wave_impl(args: argparse.Namespace) -> int:
    source = _load_profile(_arg_profile(args, "profile", "source_profile"))
    train_cfg, executor, env = _async_submit_context(source)
    _apply_distributed_config_override(train_cfg, args)
    manifest_path = _manifest_path(args)
    result_path = _results_path(args)
    submitted_probe_ids = {row.get("probe_id", "") for row in _manifest_rows(manifest_path)}
    manifest_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in _manifest_rows(manifest_path):
        manifest_by_key.setdefault(_probe_key(row), []).append(row)
    result_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    if result_path.exists():
        for row in _load_probe_results(result_path):
            result_by_key.setdefault(_probe_key(row), []).append(row)

    candidates_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in _probe_rows(source, args):
        candidates_by_key.setdefault(_probe_key(row), []).append(row)

    submitted_count = 0
    for key, candidates in sorted(candidates_by_key.items()):
        candidates.sort(key=lambda item: int(item["batch_size"]))
        results = result_by_key.get(key, [])

        target_gpu_mem_pct = float(getattr(args, "target_gpu_mem_pct", 0.0) or 0.0)
        successful_with_mem: list[dict[str, str]] = []
        if target_gpu_mem_pct > 0.0:
            successful_with_mem = [
                row for row in results
                if _truthy(row.get("success", "")) and _as_float(row.get("gpu_mem_peak_pct")) is not None
            ]
            if any(float(row["gpu_mem_peak_pct"]) >= target_gpu_mem_pct for row in successful_with_mem):
                best = max(successful_with_mem, key=lambda row: float(row.get("effective_tokens_per_sec") or 0.0))
                print(
                    f"[batch_probe_wave] target satisfied {key[0]} gc={key[1]} "
                    f"best={best['probe_id']} peak_vram={float(best['gpu_mem_peak_pct']):.1f}%"
                )
                continue

        pending_or_running = False
        finished_probe_ids = {row.get("probe_id", "") for row in results}
        for row in manifest_by_key.get(key, []):
            if row.get("probe_id", "") not in finished_probe_ids:
                finished, state, _exit_code = _job_finished(row["job_id"])
                if not finished:
                    pending_or_running = True
                    print(f"[batch_probe_wave] waiting {key[0]} gc={key[1]} probe={row['probe_id']} state={state or 'UNKNOWN'}")
                    break
        if pending_or_running:
            continue

        successful_batches = [int(row["batch_size"]) for row in results if _truthy(row.get("success", ""))]
        min_next_batch = max(successful_batches, default=0)
        next_row = None
        has_recorded_failure = any(not _truthy(row.get("success", "")) for row in results)
        if has_recorded_failure:
            if target_gpu_mem_pct > 0.0 and successful_batches and not successful_with_mem:
                telemetry_batch = max(successful_batches)
                base_row = next((row for row in candidates if int(row["batch_size"]) == telemetry_batch), None)
                if base_row is not None:
                    next_row = dict(base_row)
                    next_row["probe_id"] = f"{base_row['probe_id']}__telemetry"
                    next_row["output_root"] = (
                        _probe_root(args)
                        / str(base_row["anchor_id"])
                        / f"bs{telemetry_batch}_gc{'off' if bool(base_row['disable_gradient_checkpointing']) else 'on'}_telemetry"
                    )
                    if next_row["probe_id"] in submitted_probe_ids:
                        print(f"[batch_probe_wave] stopped {key[0]} gc={key[1]} after recorded failure")
                        continue
                    print(
                        f"[batch_probe_wave] backfilling telemetry {key[0]} gc={key[1]} "
                        f"at max_successful_bs={telemetry_batch}"
                    )
            if next_row is None:
                print(f"[batch_probe_wave] stopped {key[0]} gc={key[1]} after recorded failure")
                continue

        for row in candidates:
            if next_row is not None:
                break
            if int(row["batch_size"]) <= min_next_batch:
                continue
            if row["probe_id"] in submitted_probe_ids:
                continue
            next_row = row
            break
        if next_row is None:
            print(f"[batch_probe_wave] complete {key[0]} gc={key[1]} within requested batches")
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
            print(f"[batch_probe_wave] dry-run built {next_row['probe_id']}")
        elif job_id:
            submitted_probe_ids.add(next_row["probe_id"])
            submitted_count += 1
            print(f"[batch_probe_wave] {next_row['probe_id']} job_id={job_id}")
        if args.max_submissions and submitted_count >= int(args.max_submissions):
            break
    print(f"[batch_probe_wave] submitted={submitted_count} manifest={manifest_path}")
    return submitted_count


def submit_wave(args: argparse.Namespace) -> int:
    _submit_wave_impl(args)
    return 0


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
            return state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}, state, exit_code
    show = subprocess.run(["scontrol", "show", "job", "-o", str(job_id)], capture_output=True, text=True)
    text = show.stdout or ""
    state_match = re.search(r"\bJobState=(\w+)\b", text)
    exit_match = re.search(r"\bExitCode=([0-9]+):", text)
    state = state_match.group(1) if state_match else ""
    exit_code = exit_match.group(1) if exit_match else ""
    return state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}, state, exit_code


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


def _cleanup_probe_artifacts(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    removable_files = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "all_results.json",
        "generation_config.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "trainer_state.json",
        "training_args.bin",
    }
    removable_prefixes = ("events.out.tfevents",)
    for child in run_dir.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            shutil.rmtree(child, ignore_errors=True)
            continue
        if child.is_file() and (child.name in removable_files or child.name.startswith(removable_prefixes)):
            child.unlink(missing_ok=True)


def collect_results(args: argparse.Namespace) -> int:
    manifest = _manifest_rows(_manifest_path(args))
    results_path = _results_path(args)
    existing_rows = {row.get("probe_id", ""): row for row in _load_probe_results(results_path)} if results_path.exists() else {}
    collected = 0
    cleaned = 0
    pending = 0
    for row in manifest:
        probe_id = row["probe_id"]
        if probe_id in existing_rows:
            if bool(getattr(args, "cleanup_successful", False)) and _truthy(existing_rows[probe_id].get("success", "")):
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
        trainable = _trainable_from_log(Path(row["slurm_log"]))
        telemetry_path_raw = str(row.get("telemetry_path", "") or "").strip()
        telemetry = _gpu_telemetry_from_csv(Path(telemetry_path_raw)) if telemetry_path_raw else {}
        success = state == "COMPLETED" and train_results.exists()
        _append_result(
            results_path,
            {
                **row,
                "success": success,
                "returncode": exit_code,
                "trainable_params": trainable.get("trainable_params", ""),
                "train_runtime": train_payload.get("train_runtime", ""),
                "effective_tokens_per_sec": train_payload.get("effective_tokens_per_sec", ""),
                "train_loss": train_payload.get("train_loss", ""),
                **telemetry,
            },
        )
        collected += 1
        print(f"[batch_probe_collect] {probe_id} state={state} success={success}")
        if success and bool(getattr(args, "cleanup_successful", False)):
            _cleanup_probe_artifacts(run_dir)
            cleaned += 1
            print(f"[batch_probe_collect] cleaned artifacts for {probe_id}")
    print(f"[batch_probe_collect] collected={collected} pending={pending} cleaned={cleaned} results={results_path}")
    return 0


def _calibration_status(args: argparse.Namespace) -> dict[str, Any]:
    source = _load_profile(_arg_profile(args, "profile", "source_profile"))
    manifest = _manifest_rows(_manifest_path(args))
    results_path = _results_path(args)
    results = _load_probe_results(results_path) if results_path.exists() else []
    result_probe_ids = {row.get("probe_id", "") for row in results}

    active_or_uncollected = 0
    active_running = 0
    state_counts: dict[str, int] = {}
    for row in manifest:
        if row.get("probe_id", "") in result_probe_ids:
            continue
        finished, state, _exit_code = _job_finished(row["job_id"])
        state_key = state or "UNKNOWN"
        state_counts[state_key] = state_counts.get(state_key, 0) + 1
        active_or_uncollected += 1
        if not finished:
            active_running += 1

    candidates_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in _probe_rows(source, args):
        candidates_by_key.setdefault(_probe_key(row), []).append(row)

    manifest_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in manifest:
        manifest_by_key.setdefault(_probe_key(row), []).append(row)

    result_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in results:
        result_by_key.setdefault(_probe_key(row), []).append(row)

    target_gpu_mem_pct = float(getattr(args, "target_gpu_mem_pct", 0.0) or 0.0)
    complete = 0
    infeasible = 0
    needs_more = 0
    waiting = 0
    for key, candidates in candidates_by_key.items():
        max_candidate_batch = max(int(row["batch_size"]) for row in candidates)
        has_uncollected = any(row.get("probe_id", "") not in result_probe_ids for row in manifest_by_key.get(key, []))
        if has_uncollected:
            waiting += 1
            continue

        key_results = result_by_key.get(key, [])
        key_successes = [row for row in key_results if _truthy(row.get("success", ""))]
        key_failures = [row for row in key_results if not _truthy(row.get("success", ""))]
        if key_failures:
            if not key_successes and any(int(row.get("batch_size", "0") or 0) == 1 for row in key_failures):
                infeasible += 1
            complete += 1
            continue
        if target_gpu_mem_pct > 0.0 and any((_as_float(row.get("gpu_mem_peak_pct")) or 0.0) >= target_gpu_mem_pct for row in key_successes):
            complete += 1
            continue
        if key_successes and max(int(row["batch_size"]) for row in key_successes) >= max_candidate_batch:
            complete += 1
            continue
        needs_more += 1

    return {
        "anchors": len(candidates_by_key),
        "complete": complete,
        "infeasible": infeasible,
        "needs_more": needs_more,
        "waiting": waiting,
        "active_or_uncollected": active_or_uncollected,
        "active_running": active_running,
        "state_counts": state_counts,
        "results": len(results),
        "manifest": len(manifest),
    }


def auto_calibrate(args: argparse.Namespace) -> int:
    max_rounds = int(getattr(args, "max_rounds", 0) or 0)
    poll_seconds = int(getattr(args, "poll_seconds", 60) or 60)
    round_idx = 0
    while True:
        round_idx += 1
        print(f"[batch_probe_auto] round={round_idx} collect")
        collect_results(args)
        status = _calibration_status(args)
        print(
            "[batch_probe_auto] status "
            f"complete={status['complete']}/{status['anchors']} "
            f"infeasible={status['infeasible']} needs_more={status['needs_more']} "
            f"waiting={status['waiting']} active_or_uncollected={status['active_or_uncollected']} "
            f"running={status['active_running']} states={status['state_counts']}"
        )

        if status["needs_more"] == 0 and status["active_or_uncollected"] == 0:
            break

        submitted = _submit_wave_impl(args)
        status = _calibration_status(args)
        print(
            "[batch_probe_auto] after_submit "
            f"submitted={submitted} needs_more={status['needs_more']} "
            f"waiting={status['waiting']} active_or_uncollected={status['active_or_uncollected']} "
            f"running={status['active_running']}"
        )
        if status["needs_more"] == 0 and status["active_or_uncollected"] == 0:
            break
        if submitted == 0 and status["active_or_uncollected"] == 0 and status["needs_more"] > 0:
            raise SystemExit("calibration made no progress but still has anchors needing probes")
        if max_rounds > 0 and round_idx >= max_rounds:
            raise SystemExit(f"stopping after --max-rounds={max_rounds}")
        print(f"[batch_probe_auto] sleeping {poll_seconds}s")
        time.sleep(poll_seconds)

    print("[batch_probe_auto] calibration complete")
    if bool(getattr(args, "write_profile", False)):
        return write_calibrated_profile(args)
    return 0


def _read_summary_row(summary_path: Path, run_id: str) -> dict[str, str]:
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("run_id") == run_id:
                return row
    return {}


def _append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "anchor_id",
        "probe_id",
        "kind",
        "rank_value",
        "batch_size",
        "disable_gradient_checkpointing",
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


def run_probes(args: argparse.Namespace) -> int:
    source = _load_profile(_arg_profile(args, "profile", "source_profile"))
    runs = _run_by_id(source)
    batches = _parse_batch_values(args.batches)
    result_path = _results_path(args)
    if result_path.exists() and not args.resume:
        raise SystemExit(f"results file already exists: {result_path}; pass --resume to append/skip existing probes")

    existing_probe_ids: set[str] = set()
    if args.resume and result_path.exists():
        with result_path.open("r", encoding="utf-8", newline="") as handle:
            existing_probe_ids = {row.get("probe_id", "") for row in csv.DictReader(handle)}

    for anchor_id in ANCHOR_IDS:
        anchor_run = runs.get(anchor_id)
        if anchor_run is None:
            raise SystemExit(f"anchor run missing from source profile: {anchor_id}")
        for disable_gc in _gc_modes(args.gc_modes, anchor_run):
            for batch_size in batches:
                probe_id = _probe_id(anchor_id, batch_size, disable_gc)
                if probe_id in existing_probe_ids:
                    continue
                output_root = _probe_root(args) / anchor_id / f"bs{batch_size}_gc{'off' if disable_gc else 'on'}"
                _write_probe_profile(
                    source_profile=_arg_profile(args, "profile", "source_profile"),
                    anchor_run=anchor_run,
                    probe_id=probe_id,
                    output_root=output_root,
                    batch_size=batch_size,
                    disable_gc=disable_gc,
                    max_steps=int(args.max_steps),
                )
                cmd = [
                    sys.executable,
                    "-m",
                    "pipeline.run",
                    "--config",
                    "configs/pipeline",
                    "--profile",
                    PROBE_PROFILE,
                    "--start-at",
                    "train",
                    "--stop-after",
                    "summary",
                    "--force-stage",
                    "train",
                    "--force-stage",
                    "summary",
                ]
                print(f"[batch_probe] running {probe_id}: {' '.join(cmd)}", flush=True)
                if args.dry_run:
                    returncode = 0
                else:
                    completed = subprocess.run(cmd, cwd=REPO_ROOT)
                    returncode = int(completed.returncode)
                summary_row = _read_summary_row(REPO_ROOT / output_root / "summary.csv", probe_id)
                success = returncode == 0 and bool(summary_row)
                _append_result(
                    result_path,
                    {
                        "anchor_id": anchor_id,
                        "probe_id": probe_id,
                        "kind": anchor_run["kind"],
                        "rank_value": _rank_value(anchor_id),
                        "batch_size": batch_size,
                        "disable_gradient_checkpointing": disable_gc,
                        "success": success,
                        "returncode": returncode,
                        "trainable_params": summary_row.get("trainable_params", ""),
                        "train_runtime": summary_row.get("train_runtime", ""),
                        "effective_tokens_per_sec": summary_row.get("effective_tokens_per_sec", ""),
                        "train_loss": summary_row.get("train_loss", ""),
                        "output_root": output_root,
                    },
                )
                if not success:
                    print(f"[batch_probe] stopping {anchor_id} gc={'off' if disable_gc else 'on'} at bs={batch_size}")
                    break
    return 0


def _load_probe_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"probe results file does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _best_anchor_settings(
    rows: list[dict[str, str]],
    min_gpu_mem_pct: float = 0.0,
    selection: str = "fastest",
) -> dict[str, dict[str, Any]]:
    if selection not in {"fastest", "highest-batch"}:
        raise SystemExit(f"unsupported selection mode: {selection}")
    by_anchor_gc: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        if str(row.get("success", "")).lower() != "true":
            continue
        key = (str(row["anchor_id"]), str(row["disable_gradient_checkpointing"]))
        by_anchor_gc.setdefault(key, []).append(row)

    best_by_anchor: dict[str, dict[str, Any]] = {}
    for (anchor_id, disable_gc_raw), candidates in by_anchor_gc.items():
        candidates.sort(key=lambda row: int(row["batch_size"]))
        eligible = candidates
        selection_priority = 0
        selection_basis = "max_successful_batch"
        if min_gpu_mem_pct > 0:
            mem_qualified = [
                row for row in candidates
                if (_as_float(row.get("gpu_mem_peak_pct")) or 0.0) >= min_gpu_mem_pct
            ]
            if mem_qualified:
                eligible = mem_qualified
                selection_priority = 2
                selection_basis = f"fastest_at_or_above_{min_gpu_mem_pct:g}_pct_peak_vram"
            else:
                telemetry_candidates = [
                    row for row in candidates
                    if _as_float(row.get("gpu_mem_peak_pct")) is not None
                ]
                if telemetry_candidates:
                    eligible = telemetry_candidates
                    selection_priority = 1
                selection_basis = f"fallback_max_batch_below_{min_gpu_mem_pct:g}_pct_peak_vram"
        if selection == "highest-batch":
            best = max(
                eligible,
                key=lambda row: (
                    int(row["batch_size"]),
                    float(row.get("effective_tokens_per_sec") or 0.0),
                ),
            )
            selection_basis = f"{selection_basis}__highest_batch"
        else:
            best = max(
                eligible,
                key=lambda row: (
                    float(row.get("effective_tokens_per_sec") or 0.0),
                    int(row["batch_size"]),
                ),
            )
        throughput = float(best["effective_tokens_per_sec"] or 0.0)
        candidate_payload = {
            "anchor_id": anchor_id,
            "rank_value": int(best["rank_value"] or 0),
            "batch_size": int(best["batch_size"]),
            "disable_gradient_checkpointing": disable_gc_raw.lower() == "true",
            "effective_tokens_per_sec": throughput,
            "gpu_mem_peak_pct": _as_float(best.get("gpu_mem_peak_pct")),
            "gpu_util_avg_pct": _as_float(best.get("gpu_util_avg_pct")),
            "selection_basis": selection_basis,
            "selection_priority": selection_priority,
        }
        current = best_by_anchor.get(anchor_id)
        if current is None:
            best_by_anchor[anchor_id] = candidate_payload
            continue
        if selection == "highest-batch":
            candidate_score = (
                candidate_payload["selection_priority"],
                candidate_payload["batch_size"],
                candidate_payload["effective_tokens_per_sec"],
            )
            current_score = (
                current["selection_priority"],
                current["batch_size"],
                current["effective_tokens_per_sec"],
            )
        else:
            candidate_score = (
                candidate_payload["selection_priority"],
                candidate_payload["effective_tokens_per_sec"],
                candidate_payload["batch_size"],
            )
            current_score = (
                current["selection_priority"],
                current["effective_tokens_per_sec"],
                current["batch_size"],
            )
        if candidate_score > current_score:
            best_by_anchor[anchor_id] = candidate_payload
    return best_by_anchor


def _infeasible_anchor_ids(rows: list[dict[str, str]]) -> set[str]:
    successful = {str(row["anchor_id"]) for row in rows if _truthy(row.get("success", ""))}
    infeasible = set()
    for row in rows:
        anchor_id = str(row["anchor_id"])
        if anchor_id in successful:
            continue
        if _truthy(row.get("success", "")):
            continue
        try:
            batch_size = int(row.get("batch_size", "0") or 0)
        except ValueError:
            batch_size = 0
        if batch_size == 1:
            infeasible.add(anchor_id)
    return infeasible


def _select_setting_for_run(run: dict[str, Any], anchors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    exact = anchors.get(str(run["id"]))
    if exact is not None:
        return exact
    group = _group_key(run)
    rank = _rank_value(str(run["id"]))
    group_anchors = [value for key, value in anchors.items() if key.startswith(f"{group}_") or (group == "movlora" and key == "movlora_e8")]
    if not group_anchors:
        raise SystemExit(f"no successful probe anchors for group {group}")
    if group == "movlora":
        return group_anchors[0]
    group_anchors.sort(key=lambda value: int(value["rank_value"]))

    # Conservative interpolation: use the nearest calibrated anchor at or above
    # the run rank. Larger ranks are at least as memory-hungry in these tuners.
    selected = group_anchors[-1]
    for anchor in group_anchors:
        if rank <= int(anchor["rank_value"]):
            selected = anchor
            break
    return selected


def _distributed_process_count(profile: dict[str, Any]) -> int:
    training = profile.get("training", {}) if isinstance(profile.get("training"), dict) else {}
    distributed = training.get("distributed", {}) if isinstance(training.get("distributed"), dict) else {}
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


def _source_profile_payload(
    *,
    profile_name: str,
    base_profile: str,
    experiment_tag: str,
    output_root: Path,
    template_profile: dict[str, Any],
) -> dict[str, Any]:
    output_root_str = str(output_root)
    tokenization = template_profile.get("tokenization", {}) if isinstance(template_profile.get("tokenization"), dict) else {}
    runs = deepcopy(template_profile["training"]["runs"])
    return {
        profile_name: {
            "extends": base_profile,
            "output_root": output_root_str,
            "logs_dir": f"{output_root_str}/logs",
            "run_naming": {
                "experiment_tag": experiment_tag,
                "language_count_hint": 10,
            },
            "tokenization": {
                "enabled": False,
                "output_dir": tokenization.get("output_dir", "outputs/pipeline_runs/moe_study_profile_otus_cluster_10_langs/tokenized"),
            },
            "training": {
                "output_dir": f"{output_root_str}/runs",
                "report_to": "none",
                "executor": {"time": "03:51:00"},
                "lora": {
                    "target": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                },
                "tokenized_local_copy": {
                    "enabled": True,
                    "root": "/dev/shm/moe-study-tokenized",
                    "dereference_symlinks": True,
                },
                "eval": {"enabled": False, "strategy": "no"},
                "train": {
                    "batch_size": 1,
                    "grad_accum": 1,
                    "max_steps": 1,
                    "logging_steps": 1,
                    "save_steps": 1,
                    "skip_final_save": True,
                    "save_total_limit": 1,
                    "cleanup_large_artifacts_after_train": True,
                },
                "runs": runs,
            },
            "final_eval_loss": {"enabled": False},
            "benchmark": {"enabled": False, "output_dir": f"{output_root_str}/benchmarks"},
            "baseline_eval": {"enabled": False},
            "summary": {
                "baseline_id": "lora_r2",
                "summary_csv": f"{output_root_str}/summary.csv",
                "gate_json": f"{output_root_str}/gate.json",
            },
            "report": {"enabled": False, "output_xlsx": f"{output_root_str}/report.xlsx"},
            "wandb": {"enabled": False, "mode": "disabled", "project": "moe-study-rank-expert-sweep"},
            "wandb_mode": "disabled",
            "stage_executors": {
                "train": {
                    "type": "slurm",
                    "partition": "gpu",
                    "gres": "gpu:a100:4",
                    "cpus_per_task": 64,
                    "mem": "384G",
                    "time": "03:51:00",
                    "extra_sbatch_flags": [
                        "--signal=B:USR1@900",
                        "--exclude=n2gpu1204,n2gpu1208,n2gpu1228",
                        "--export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                    ],
                }
            },
        }
    }


def write_profile(args: argparse.Namespace) -> int:
    template_profile = _load_profile(str(args.template_profile or SOURCE_PROFILE))
    runs = _run_by_id(template_profile)
    missing = [anchor_id for anchor_id in ANCHOR_IDS if anchor_id not in runs]
    if missing:
        raise SystemExit(f"template profile missing rank-sweep anchors: {missing}")
    payload = _source_profile_payload(
        profile_name=_arg_profile(args, "source_profile", "source_profile"),
        base_profile=str(_preset(args.cluster)["base_profile"]),
        experiment_tag=str(_preset(args.cluster)["experiment_tag"]),
        output_root=_arg_path(args, "output_root", "output_root"),
        template_profile=template_profile,
    )
    output_profile = _arg_path(args, "output_profile", "profile_path")
    output_profile.parent.mkdir(parents=True, exist_ok=True)
    output_profile.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[batch_calibration] wrote source profile: {output_profile}")
    print(f"[batch_calibration] runs={len(template_profile['training']['runs'])}")
    return 0


def write_calibrated_profile(args: argparse.Namespace) -> int:
    source_profile_name = _arg_profile(args, "profile", "source_profile")
    source = _load_profile(source_profile_name)
    probe_rows = _load_probe_results(_results_path(args))
    anchors = _best_anchor_settings(
        probe_rows,
        min_gpu_mem_pct=float(args.min_gpu_mem_pct),
        selection=str(getattr(args, "selection", "fastest")),
    )
    infeasible_anchor_ids = _infeasible_anchor_ids(probe_rows)
    selected_anchor_ids = list(getattr(args, "anchor_id", []) or [])
    expected_anchor_ids = selected_anchor_ids if selected_anchor_ids else ANCHOR_IDS
    expected_anchor_set = set(expected_anchor_ids)
    missing = [
        anchor_id
        for anchor_id in expected_anchor_ids
        if anchor_id not in anchors and anchor_id not in infeasible_anchor_ids
    ]
    if missing:
        raise SystemExit(f"missing successful probe results for anchors: {missing}")

    target_global_batch = int(args.target_global_batch)
    max_grad_accum = int(args.max_grad_accum)
    process_count = _distributed_process_count(source)
    source_runs = _run_by_id(source)
    groups_with_successful_anchor = {
        _group_key(source_runs[anchor_id])
        for anchor_id in anchors
        if anchor_id in source_runs
    }
    calibrated_runs = []
    settings_rows = []
    for run in source["training"]["runs"]:
        run = deepcopy(run)
        if expected_anchor_set and str(run["id"]) not in expected_anchor_set:
            settings_rows.append(
                {
                    "run_id": run["id"],
                    "kind": run["kind"],
                    "anchor_id": "",
                    "batch_size": "",
                    "grad_accum": "",
                    "num_processes": process_count,
                    "actual_global_batch": "",
                    "target_global_batch": target_global_batch,
                    "disable_gradient_checkpointing": "",
                    "constrained": "",
                    "effective_tokens_per_sec": "",
                    "gpu_mem_peak_pct": "",
                    "gpu_util_avg_pct": "",
                    "selection_basis": "",
                    "status": "excluded_not_requested",
                }
            )
            continue
        run_group = _group_key(run)
        if run_group not in groups_with_successful_anchor:
            settings_rows.append(
                {
                    "run_id": run["id"],
                    "kind": run["kind"],
                    "anchor_id": "",
                    "batch_size": "",
                    "grad_accum": "",
                    "num_processes": process_count,
                    "actual_global_batch": "",
                    "target_global_batch": target_global_batch,
                    "disable_gradient_checkpointing": "",
                    "constrained": "",
                    "effective_tokens_per_sec": "",
                    "gpu_mem_peak_pct": "",
                    "gpu_util_avg_pct": "",
                    "selection_basis": "",
                    "status": "infeasible_group_no_successful_anchor",
                }
            )
            continue
        if str(run["id"]) in infeasible_anchor_ids:
            settings_rows.append(
                {
                    "run_id": run["id"],
                    "kind": run["kind"],
                    "anchor_id": run["id"],
                    "batch_size": "",
                    "grad_accum": "",
                    "num_processes": process_count,
                    "actual_global_batch": "",
                    "target_global_batch": target_global_batch,
                    "disable_gradient_checkpointing": "",
                    "constrained": "",
                    "effective_tokens_per_sec": "",
                    "gpu_mem_peak_pct": "",
                    "gpu_util_avg_pct": "",
                    "selection_basis": "",
                    "status": "infeasible_bs1",
                }
            )
            continue
        setting = _select_setting_for_run(run, anchors)
        batch_size = int(setting["batch_size"])
        grad_accum_needed = max(1, math.ceil(target_global_batch / (batch_size * process_count)))
        constrained = grad_accum_needed > max_grad_accum
        grad_accum = _closest_grad_accum(target_global_batch, batch_size, process_count, max_grad_accum)
        actual_global_batch = batch_size * process_count * grad_accum
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
                "kind": run["kind"],
                "anchor_id": setting["anchor_id"],
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "num_processes": process_count,
                "actual_global_batch": actual_global_batch,
                "target_global_batch": target_global_batch,
                "disable_gradient_checkpointing": bool(setting["disable_gradient_checkpointing"]),
                "constrained": constrained,
                "effective_tokens_per_sec": setting.get("effective_tokens_per_sec", ""),
                "gpu_mem_peak_pct": setting.get("gpu_mem_peak_pct", ""),
                "gpu_util_avg_pct": setting.get("gpu_util_avg_pct", ""),
                "selection_basis": setting.get("selection_basis", ""),
                "status": "included",
            }
        )

    output_root = str(_arg_path(args, "calibrated_output_root", "calibrated_output_root"))
    source_tokenization = deepcopy(source.get("tokenization", {})) if isinstance(source.get("tokenization"), dict) else {}
    tokenized_output_dir = source_tokenization.get(
        "output_dir",
        "outputs/pipeline_runs/moe_study_profile_otus_cluster_10_langs/tokenized",
    )
    source_train_defaults = deepcopy(source.get("training", {}).get("train", {}))
    calibrated_train_defaults = {
        "logging_steps": 50,
        "max_steps": 10000,
        "num_train_epochs": 1.0,
        "save_steps": 999999,
        "save_strategy": "no",
        "save_total_limit": 1,
        "save_only_model": False,
        "skip_final_save": False,
        "cleanup_large_artifacts_after_train": False,
    }
    if isinstance(source_train_defaults, dict):
        for key in (
            "max_steps",
            "num_train_epochs",
            "logging_steps",
            "save_steps",
            "save_strategy",
            "save_total_limit",
            "save_only_model",
            "skip_final_save",
            "cleanup_large_artifacts_after_train",
        ):
            if key in source_train_defaults:
                calibrated_train_defaults[key] = source_train_defaults[key]
    baseline_id = str(calibrated_runs[0]["id"]) if calibrated_runs else "lora_r2"
    profile_payload = {
        _arg_profile(args, "calibrated_profile", "calibrated_profile"): {
            "extends": source_profile_name,
            "output_root": output_root,
            "logs_dir": f"{output_root}/logs",
            "run_naming": {
                "experiment_tag": f"{_preset(args.cluster)['experiment_tag']}-calibrated",
                "language_count_hint": 10,
            },
            "tokenization": {
                "enabled": False,
                "output_dir": tokenized_output_dir,
            },
            "training": {
                "output_dir": f"{output_root}/runs",
                "report_to": "none",
                "executor": {"time": "03:51:00"},
                "eval": {"enabled": False, "strategy": "no"},
                "train": calibrated_train_defaults,
                "runs": calibrated_runs,
            },
            "summary": {
                "baseline_id": baseline_id,
                "summary_csv": f"{output_root}/summary.csv",
                "gate_json": f"{output_root}/gate.json",
            },
            "final_eval_loss": {"enabled": False},
            "benchmark": {"enabled": False, "output_dir": f"{output_root}/benchmarks"},
            "baseline_eval": {"enabled": False},
            "report": {"enabled": False, "output_xlsx": f"{output_root}/report.xlsx"},
            "wandb": {"enabled": False, "mode": "disabled", "project": "moe-study-rank-expert-sweep"},
            "wandb_mode": "disabled",
            "stage_executors": {
                "train": {
                    "type": "slurm",
                    "partition": "gpu",
                    "gres": "gpu:a100:4",
                    "cpus_per_task": 64,
                    "mem": "384G",
                    "time": "03:51:00",
                    "extra_sbatch_flags": [
                        "--signal=B:USR1@900",
                        "--exclude=n2gpu1204,n2gpu1208,n2gpu1228",
                        "--export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                    ],
                }
            },
        }
    }

    source_final_eval_loss = (
        deepcopy(source.get("final_eval_loss", {})) if isinstance(source.get("final_eval_loss"), dict) else {}
    )
    source_benchmark = deepcopy(source.get("benchmark", {})) if isinstance(source.get("benchmark"), dict) else {}
    final_eval_loss = {
        "enabled": True,
        "output_subdir": "final_eval_loss",
        "batch_size": "train",
        "output_dir": f"{output_root}/final_eval_loss",
    }
    benchmark = {
        "enabled": True,
        "batch_size": "auto",
        "output_dir": f"{output_root}/benchmarks",
    }
    for key in ("strict_final_checkpoint", "require_direct_save_marker", "checkpoint_step_source"):
        if key in source_final_eval_loss:
            final_eval_loss[key] = source_final_eval_loss[key]
        if key in source_benchmark:
            benchmark[key] = source_benchmark[key]
    profile_payload[_arg_profile(args, "calibrated_profile", "calibrated_profile")]["final_eval_loss"] = final_eval_loss
    profile_payload[_arg_profile(args, "calibrated_profile", "calibrated_profile")]["benchmark"] = benchmark
    profile_payload[_arg_profile(args, "calibrated_profile", "calibrated_profile")]["report"] = {
        "enabled": True,
        "output_xlsx": f"{output_root}/report.xlsx",
    }

    profile_path = _arg_path(args, "output_profile", "calibrated_profile_path")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile_payload, indent=2) + "\n", encoding="utf-8")

    settings_path = Path(args.output_settings) if str(args.output_settings).strip() else _probe_root(args) / "calibrated_batch_settings.csv"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with settings_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(settings_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(settings_rows)

    print(f"[batch_calibration] wrote profile: {profile_path}")
    print(f"[batch_calibration] wrote settings: {settings_path}")
    return 0


def list_anchors(args: argparse.Namespace) -> int:
    source = _load_profile(_arg_profile(args, "profile", "source_profile"))
    runs = _run_by_id(source)
    for anchor_id in ANCHOR_IDS:
        run = runs.get(anchor_id)
        if run is None:
            raise SystemExit(f"anchor run missing from source profile: {anchor_id}")
        print(f"{anchor_id}\t{run['kind']}\t{_rank_value(anchor_id)}\tdefault_gc_off={_default_disable_gc(run)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate per-run batch size and grad accumulation for the rank/expert sweep.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_cluster_arg(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--cluster", choices=sorted(CLUSTER_PRESETS), default=DEFAULT_CLUSTER)

    write_parser = subparsers.add_parser("write-profile", help="Write a rank/expert sweep source profile for a cluster preset.")
    add_cluster_arg(write_parser)
    write_parser.add_argument("--template-profile", default=SOURCE_PROFILE)
    write_parser.add_argument("--source-profile", default="")
    write_parser.add_argument("--output-profile", default="")
    write_parser.add_argument("--output-root", default="")
    write_parser.set_defaults(func=write_profile)

    anchors_parser = subparsers.add_parser("list-anchors", help="List calibration anchors.")
    add_cluster_arg(anchors_parser)
    anchors_parser.add_argument("--profile", default="")
    anchors_parser.set_defaults(func=list_anchors)

    probes_parser = subparsers.add_parser("run-probes", help="Run batch-size probes one at a time with the normal pipeline.")
    add_cluster_arg(probes_parser)
    probes_parser.add_argument("--profile", default="")
    probes_parser.add_argument("--batches", default=",".join(str(value) for value in DEFAULT_BATCHES))
    probes_parser.add_argument("--gc-modes", default="both", choices=["default", "both", "on", "off"])
    probes_parser.add_argument("--max-steps", type=int, default=20)
    probes_parser.add_argument("--output-root", default="")
    probes_parser.add_argument("--results", default="")
    probes_parser.add_argument("--resume", action="store_true")
    probes_parser.add_argument("--dry-run", action="store_true")
    probes_parser.set_defaults(func=run_probes)

    submit_parser = subparsers.add_parser("submit-probes", help="Submit probe jobs to Slurm without waiting.")
    add_cluster_arg(submit_parser)
    submit_parser.add_argument("--profile", default="")
    submit_parser.add_argument("--batches", default=",".join(str(value) for value in DEFAULT_BATCHES))
    submit_parser.add_argument("--gc-modes", default="both", choices=["default", "both", "on", "off"])
    submit_parser.add_argument("--max-steps", type=int, default=20)
    submit_parser.add_argument("--output-root", default="")
    submit_parser.add_argument("--manifest", default="")
    submit_parser.add_argument("--anchor-id", action="append", default=[], help="Submit only selected anchor id(s).")
    submit_parser.add_argument("--limit", type=int, default=0, help="Submit at most N probe jobs.")
    submit_parser.add_argument("--accelerate-config-file", default="", help="Override training.distributed.config_file for submitted probes.")
    submit_parser.add_argument("--telemetry-interval-seconds", type=int, default=5)
    submit_parser.add_argument("--disable-gpu-telemetry", action="store_true")
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.set_defaults(func=submit_probes)

    wave_parser = subparsers.add_parser("submit-wave", help="Submit at most one next batch-size probe per anchor/GC mode.")
    add_cluster_arg(wave_parser)
    wave_parser.add_argument("--profile", default="")
    wave_parser.add_argument("--batches", default=",".join(str(value) for value in DEFAULT_BATCHES))
    wave_parser.add_argument("--gc-modes", default="default", choices=["default", "both", "on", "off"])
    wave_parser.add_argument("--max-steps", type=int, default=20)
    wave_parser.add_argument("--output-root", default="")
    wave_parser.add_argument("--manifest", default="")
    wave_parser.add_argument("--results", default="")
    wave_parser.add_argument("--anchor-id", action="append", default=[], help="Submit only selected anchor id(s).")
    wave_parser.add_argument("--limit", type=int, default=0, help="Consider at most N candidate probes before wave selection.")
    wave_parser.add_argument("--max-submissions", type=int, default=0, help="Submit at most N jobs in this wave.")
    wave_parser.add_argument("--accelerate-config-file", default="", help="Override training.distributed.config_file for submitted probes.")
    wave_parser.add_argument("--target-gpu-mem-pct", type=float, default=0.0, help="Stop expanding an anchor once a successful telemetry probe reaches this peak VRAM percentage.")
    wave_parser.add_argument("--telemetry-interval-seconds", type=int, default=5)
    wave_parser.add_argument("--disable-gpu-telemetry", action="store_true")
    wave_parser.add_argument("--dry-run", action="store_true")
    wave_parser.set_defaults(func=submit_wave)

    collect_parser = subparsers.add_parser("collect-results", help="Collect completed asynchronous probe jobs.")
    add_cluster_arg(collect_parser)
    collect_parser.add_argument("--output-root", default="")
    collect_parser.add_argument("--manifest", default="")
    collect_parser.add_argument("--results", default="")
    collect_parser.add_argument("--cleanup-successful", action="store_true", help="Delete saved adapter/checkpoint artifacts after successful collection.")
    collect_parser.set_defaults(func=collect_results)

    calibrated_parser = subparsers.add_parser("write-calibrated-profile", help="Write the final calibrated 69-run profile from probe results.")
    add_cluster_arg(calibrated_parser)
    calibrated_parser.add_argument("--profile", default="")
    calibrated_parser.add_argument("--results", default="")
    calibrated_parser.add_argument("--target-global-batch", type=int, default=DEFAULT_TARGET_GLOBAL_BATCH)
    calibrated_parser.add_argument("--max-grad-accum", type=int, default=DEFAULT_MAX_GRAD_ACCUM)
    calibrated_parser.add_argument("--min-gpu-mem-pct", type=float, default=0.0, help="Prefer the fastest setting whose telemetry reaches this peak VRAM percentage; fall back to max batch if none do.")
    calibrated_parser.add_argument("--selection", choices=["fastest", "highest-batch"], default="fastest")
    calibrated_parser.add_argument("--anchor-id", action="append", default=[], help="Include only selected run id(s) in the calibrated profile.")
    calibrated_parser.add_argument("--calibrated-profile", default="")
    calibrated_parser.add_argument("--output-profile", default="")
    calibrated_parser.add_argument("--output-root", dest="calibrated_output_root", default="")
    calibrated_parser.add_argument("--probe-root", dest="output_root", default="")
    calibrated_parser.add_argument("--output-settings", default="")
    calibrated_parser.set_defaults(func=write_calibrated_profile)

    auto_parser = subparsers.add_parser("auto-calibrate", help="Automatically collect, submit, and wait until batch calibration is complete.")
    add_cluster_arg(auto_parser)
    auto_parser.add_argument("--profile", default="")
    auto_parser.add_argument("--batches", default=",".join(str(value) for value in DEFAULT_BATCHES))
    auto_parser.add_argument("--gc-modes", default="default", choices=["default", "both", "on", "off"])
    auto_parser.add_argument("--max-steps", type=int, default=20)
    auto_parser.add_argument("--output-root", default="")
    auto_parser.add_argument("--manifest", default="")
    auto_parser.add_argument("--results", default="")
    auto_parser.add_argument("--anchor-id", action="append", default=[], help="Calibrate only selected anchor id(s).")
    auto_parser.add_argument("--limit", type=int, default=0, help="Consider at most N candidate probes before wave selection.")
    auto_parser.add_argument("--max-submissions", type=int, default=0, help="Submit at most N jobs per wave.")
    auto_parser.add_argument("--target-gpu-mem-pct", type=float, default=0.0, help="Stop expanding an anchor once a successful telemetry probe reaches this peak VRAM percentage. Leave at 0 to find highest stable batch among --batches.")
    auto_parser.add_argument("--telemetry-interval-seconds", type=int, default=5)
    auto_parser.add_argument("--disable-gpu-telemetry", action="store_true")
    auto_parser.add_argument("--cleanup-successful", action="store_true")
    auto_parser.add_argument("--poll-seconds", type=int, default=60)
    auto_parser.add_argument("--max-rounds", type=int, default=0)
    auto_parser.add_argument("--dry-run", action="store_true")
    auto_parser.add_argument("--write-profile", action="store_true", help="Write the calibrated full-sweep profile after calibration completes.")
    auto_parser.add_argument("--target-global-batch", type=int, default=DEFAULT_TARGET_GLOBAL_BATCH)
    auto_parser.add_argument("--max-grad-accum", type=int, default=DEFAULT_MAX_GRAD_ACCUM)
    auto_parser.add_argument("--min-gpu-mem-pct", type=float, default=0.0)
    auto_parser.add_argument("--selection", choices=["fastest", "highest-batch"], default="highest-batch")
    auto_parser.add_argument("--calibrated-profile", default="")
    auto_parser.add_argument("--output-profile", default="")
    auto_parser.add_argument("--output-root-calibrated", dest="calibrated_output_root", default="")
    auto_parser.add_argument("--probe-root", dest="output_root", default="")
    auto_parser.add_argument("--output-settings", default="")
    auto_parser.set_defaults(func=auto_calibrate)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
