from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
CALL_DIR = Path.cwd()
COMPARISON_DIR = SCRIPT_DIR


def load_config(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        sys.exit(f"[ERROR] Config file {path} not found.")
    except json.JSONDecodeError as exc:
        sys.exit(f"[ERROR] Failed to parse {path}: {exc}")


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path_str).resolve()
    return path


def ensure_dir(path: Path, label: str) -> None:
    if not path.exists():
        sys.exit(f"[ERROR] {label} path {path} does not exist.")


def discover_datasets(root: Path, specified: List[str]) -> List[Tuple[str, Path]]:
    datasets: List[Tuple[str, Path]] = []
    if specified:
        for name in specified:
            name = name.strip()
            if not name:
                continue
            candidate = root / name
            if candidate.exists():
                datasets.append((name, candidate))
            else:
                print(f"[WARN] Dataset {name} not found under {root}, skipping.")
    else:
        for child in sorted(root.iterdir()):
            if child.is_dir() and not child.name.endswith("_ranks"):
                datasets.append((child.name, child))
    if not datasets:
        print(f"[WARN] No datasets discovered in {root}.")
    return datasets


def to_env_value(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def prepare_env(assignments: Dict[str, str]) -> List[str]:
    return [f"{key}={value}" for key, value in assignments.items()]


def _short_model_label(model_name: str) -> str:
    lower = model_name.lower()
    if "llama" in lower:
        if "3.1" in lower and "8b" in lower:
            return "l3-8b"
        if "3.2" in lower and "1b" in lower:
            return "l3-1b"
        if "3.2" in lower and "3b" in lower:
            return "l3-3b"
    return model_name.split("/")[-1]


def _tokenizer_tag(tokenizer_name: str) -> str:
    return "ext" if "ext" in tokenizer_name.lower() else "base"


def accelerate_env_for_gpus(
    gpus_per_node: int,
    nodes: int,
    repo_root: Path,
    override: str | None = None,
) -> Dict[str, str]:
    if override:
        config_path = resolve_path(override)
    elif nodes > 1:
        config_path = repo_root / "LLaMA-Factory" / "examples" / "accelerate" / "fsdp_config_multiple_nodes.yaml"
    elif gpus_per_node >= 8:
        config_path = repo_root / "LLaMA-Factory" / "examples" / "accelerate" / "fsdp_8gpu_config.yaml"
    elif gpus_per_node == 4:
        config_path = repo_root / "LLaMA-Factory" / "examples" / "accelerate" / "fsdp_4gpu_config.yaml"
    elif gpus_per_node == 2:
        config_path = repo_root / "LLaMA-Factory" / "examples" / "accelerate" / "fsdp_2gpu_config.yaml"
    else:
        return {}
    if not config_path.exists():
        sys.exit(f"[ERROR] Accelerate config {config_path} not found.")
    return {
        "USE_ACCELERATE": "1",
        "ACCELERATE_CONFIG": str(config_path),
        "ACCELERATE_USE_FSDP": "1",
    }


def merge_resources(*configs: dict | None) -> dict:
    merged: dict = {}
    for cfg in configs:
        if cfg:
            merged.update(cfg)
    return merged


def _resolve_dataset_entry(entry) -> tuple[str, dict]:
    if isinstance(entry, dict):
        name = entry.get("name")
        if not name:
            sys.exit("[ERROR] Dataset entry missing 'name'.")
        return name, entry.get("resources", {}) or {}
    return str(entry), {}


def _resolve_gpus_per_node(resources: dict, variant: dict, cfg: dict, default_gpus: int) -> int:
    for key in ("gpus_per_node", "gpus_per_job"):
        if key in resources:
            return int(resources[key])
    for key in ("gpus_per_job", "gpus_per_node"):
        if key in variant:
            return int(variant[key])
    for key in ("gpus_per_job", "gpus_per_node"):
        if key in cfg:
            return int(cfg[key])
    return int(default_gpus)


def _resolve_gpu_type(resources: dict, variant: dict, cfg: dict, default_gpu_type: str) -> str:
    if "gpu_type" in resources:
        return str(resources["gpu_type"])
    if "gpu_type" in variant:
        return str(variant["gpu_type"])
    if "gpu_type" in cfg:
        return str(cfg["gpu_type"])
    return default_gpu_type


def _build_sbatch_args(
    resources: dict,
    gpus_per_node: int,
    gpu_type: str,
    gpu_hours: float | None,
    sbatch_args: str,
) -> str:
    args: List[str] = []
    partition = resources.get("partition")
    if partition:
        args.append(f"--partition={partition}")
    if gpus_per_node > 0:
        args.append(f"--gres=gpu:{gpu_type}:{gpus_per_node}")
    nodes = int(resources.get("nodes", 1))
    if nodes > 1:
        args.append(f"--nodes={nodes}")
        if "ntasks" in resources:
            args.append(f"--ntasks={resources['ntasks']}")
        else:
            args.append(f"--ntasks={nodes}")
        if "ntasks_per_node" in resources:
            args.append(f"--ntasks-per-node={resources['ntasks_per_node']}")
        else:
            args.append("--ntasks-per-node=1")
    elif "ntasks" in resources:
        args.append(f"--ntasks={resources['ntasks']}")
    if "cpus_per_task" in resources:
        args.append(f"--cpus-per-task={resources['cpus_per_task']}")
    if "mem" in resources:
        args.append(f"--mem={resources['mem']}")

    total_gpus = max(1, nodes * gpus_per_node)
    computed_time = _format_sbatch_time((gpu_hours / total_gpus) if gpu_hours else None)
    time_arg = resources.get("time") or computed_time
    if time_arg and "--time" not in sbatch_args:
        args.append(f"--time={time_arg}")
    signal_arg = resources.get("signal")
    if not signal_arg:
        signal_minutes = resources.get("signal_before_timeout_minutes")
        if signal_minutes is not None:
            try:
                seconds = int(float(signal_minutes) * 60)
                if seconds > 0:
                    signal_arg = f"USR1@{seconds}"
            except (TypeError, ValueError):
                signal_arg = None
    if signal_arg and "--signal" not in sbatch_args:
        args.append(f"--signal={signal_arg}")
    kill_wait = resources.get("kill_wait_seconds")
    if kill_wait is not None and "--kill-wait" not in sbatch_args:
        try:
            kill_wait_val = int(kill_wait)
            if kill_wait_val > 0:
                args.append(f"--kill-wait={kill_wait_val}")
        except (TypeError, ValueError):
            pass
    if resources.get("extra_sbatch_args"):
        args.append(str(resources["extra_sbatch_args"]))
    if sbatch_args:
        args.append(sbatch_args)
    return " ".join(args)


def _tier_gpu_hours(dataset_name: str) -> float | None:
    head = dataset_name.split("_", 1)[0]
    if not head.isdigit():
        return None
    lang_count = int(head)
    tier_hours = {
        5: 24.00,
        10: 27.69,
        22: 36.55,
        46: 54.28,
        96: 91.20,
        200: 168.00,
    }
    return tier_hours.get(lang_count)


def _format_sbatch_time(hours: float | None) -> str | None:
    if hours is None:
        return None
    total_minutes = int(math.ceil(hours * 60))
    days = total_minutes // (24 * 60)
    rem_minutes = total_minutes % (24 * 60)
    hours_part = rem_minutes // 60
    minutes_part = rem_minutes % 60
    if days > 0:
        return f"{days}-{hours_part:02d}:{minutes_part:02d}:00"
    return f"{hours_part:02d}:{minutes_part:02d}:00"


def submit_job(job_name: str, script: Path, log_path: Path, env_assignments: List[str], sbatch_args: str) -> str:
    script = script.resolve()
    if not script.exists():
        sys.exit(f"[ERROR] Job script {script} not found.")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["env", *env_assignments, "sbatch", f"--output={log_path}"]
    if sbatch_args:
        cmd.extend(shlex.split(sbatch_args))
    cmd.append(str(script))
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr)
        sys.exit(f"[ERROR] sbatch submission failed for {job_name}.")
    output = result.stdout.strip()
    if "Submitted batch job" not in output:
        sys.stderr.write(output + "\n")
        sys.exit(f"[ERROR] sbatch did not acknowledge submission for {job_name}.")
    job_id = output.split()[-1]
    print(f"[INFO] Submitted {job_name} job {job_id}")
    return job_id


def run_simple_variants(
    dataset_label: str,
    dataset_env: Dict[str, str],
    approach_name: str,
    cfg: dict,
    base_resources: dict,
    timestamp: str,
    output_root: Path,
    log_root: Path,
    sbatch_args: str,
    gpu_hours: float | None,
    registry: Dict[str, str],
    lm_eval_cfg: dict,
    default_gpus: int,
    default_gpu_type: str,
) -> None:
    script = COMPARISON_DIR / cfg["script"]
    batch_size = int(cfg.get("batch_size", 16))
    log_prefix = cfg.get("log_prefix", approach_name)
    variants = cfg.get("variants", [])
    for variant in variants:
        variant_name = variant["name"]
        variant_env = {k: to_env_value(v) for k, v in variant.get("env", {}).items()}
        resources = merge_resources(base_resources, cfg.get("resources"), variant.get("resources"))
        bs = int(variant.get("batch_size", batch_size))
        job_gpus = _resolve_gpus_per_node(resources, variant, cfg, default_gpus)
        job_gpu_type = _resolve_gpu_type(resources, variant, cfg, default_gpu_type)
        nodes = int(resources.get("nodes", 1))
        run_dir = output_root / approach_name / f"{variant_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"{log_prefix}_{variant_name}_{timestamp}.log"
        env = dict(dataset_env)
        env.update(variant_env)
        model_short = dataset_env.get("MODEL_SHORT", "model")
        tok_tag = dataset_env.get("TOKENIZER_TAG", "base")
        dataset_name = dataset_env.get("DATASET_NAME", dataset_label)
        env.update(
            {
                "OUTPUT_DIR": str(run_dir),
                "WANDB_NAME": f"{model_short}-{tok_tag}-{dataset_name}-{approach_name}-{variant_name}-{timestamp}",
                "PER_DEVICE_TRAIN_BATCH_SIZE": str(bs),
                "PER_DEVICE_EVAL_BATCH_SIZE": str(bs),
                "GPUS_PER_NODE": str(job_gpus),
                "GPU_TYPE": job_gpu_type,
            }
        )
        env.update(
            accelerate_env_for_gpus(
                job_gpus,
                nodes,
                REPO_ROOT,
                resources.get("accelerate_config"),
            )
        )
        job_key = f"{dataset_label}-{approach_name}-{variant_name}"
        print(f"[INFO] Queueing {job_key} (bs={bs})")
        env_assignments = prepare_env(env)
        job_specific_args = _build_sbatch_args(resources, job_gpus, job_gpu_type, gpu_hours, sbatch_args)
        job_id = submit_job(job_key, script, log_path, env_assignments, job_specific_args)
        registry[job_key] = job_id
        launch_eval_watcher(
            run_dir,
            dataset_label,
            approach_name,
            variant_name,
            dataset_env,
            log_root,
            lm_eval_cfg,
        )


def run_moelpr_variants(
    dataset_label: str,
    dataset_env: Dict[str, str],
    cfg: dict,
    base_resources: dict,
    timestamp: str,
    output_root: Path,
    log_root: Path,
    sbatch_args: str,
    gpu_hours: float | None,
    registry: Dict[str, str],
    lm_eval_cfg: dict,
    default_gpus: int,
    default_gpu_type: str,
) -> None:
    script = COMPARISON_DIR / cfg["script"]
    batch_size = int(cfg.get("batch_size", 16))
    log_prefix = cfg.get("log_prefix", "moelpr")
    for variant in cfg.get("variants", []):
        variant_name = variant["name"]
        variant_env = {k: to_env_value(v) for k, v in variant.get("env", {}).items()}
        resources = merge_resources(base_resources, cfg.get("resources"), variant.get("resources"))
        bs = int(variant.get("batch_size", batch_size))
        job_gpus = _resolve_gpus_per_node(resources, variant, cfg, default_gpus)
        job_gpu_type = _resolve_gpu_type(resources, variant, cfg, default_gpu_type)
        nodes = int(resources.get("nodes", 1))
        stage1_dir = output_root / "moelpr" / f"{variant_name}_stage1_{timestamp}"
        stage2_dir = output_root / "moelpr" / f"{variant_name}_stage2_{timestamp}"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        stage2_dir.mkdir(parents=True, exist_ok=True)
        env = dict(dataset_env)
        env.update(variant_env)
        model_short = dataset_env.get("MODEL_SHORT", "model")
        tok_tag = dataset_env.get("TOKENIZER_TAG", "base")
        dataset_name = dataset_env.get("DATASET_NAME", dataset_label)
        env.update(
            {
                "MOELPR_STAGE1_OUTPUT": str(stage1_dir),
                "MOELPR_STAGE2_OUTPUT": str(stage2_dir),
                "WANDB_STAGE1_NAME": f"{model_short}-{tok_tag}-{dataset_name}-moelpr-{variant_name}-{timestamp}-stage1",
                "WANDB_STAGE2_NAME": f"{model_short}-{tok_tag}-{dataset_name}-moelpr-{variant_name}-{timestamp}-stage2",
                "PER_DEVICE_TRAIN_BATCH_SIZE": str(bs),
                "PER_DEVICE_EVAL_BATCH_SIZE": str(bs),
                "GPUS_PER_NODE": str(job_gpus),
                "GPU_TYPE": job_gpu_type,
            }
        )
        env.update(
            accelerate_env_for_gpus(
                job_gpus,
                nodes,
                REPO_ROOT,
                resources.get("accelerate_config"),
            )
        )
        job_specific_args = _build_sbatch_args(resources, job_gpus, job_gpu_type, gpu_hours, sbatch_args)

        stage1_key = f"{dataset_label}-moelpr-{variant_name}-stage1"
        stage1_log = log_root / f"{log_prefix}_{variant_name}_{timestamp}_stage1.log"
        env_stage1 = dict(env)
        env_stage1["MOELPR_STAGE_ONLY"] = "1"
        print(f"[INFO] Queueing {stage1_key} (bs={bs})")
        stage1_id = submit_job(stage1_key, script, stage1_log, prepare_env(env_stage1), job_specific_args)
        registry[stage1_key] = stage1_id

        stage2_key = f"{dataset_label}-moelpr-{variant_name}-stage2"
        stage2_log = log_root / f"{log_prefix}_{variant_name}_{timestamp}_stage2.log"
        env_stage2 = dict(env)
        env_stage2["MOELPR_STAGE_ONLY"] = "2"
        dep_args = f"--dependency=afterok:{stage1_id}"
        stage2_args = f"{job_specific_args} {dep_args}".strip()
        print(f"[INFO] Queueing {stage2_key} (bs={bs})")
        stage2_id = submit_job(stage2_key, script, stage2_log, prepare_env(env_stage2), stage2_args)
        registry[stage2_key] = stage2_id

        launch_eval_watcher(
            stage2_dir,
            dataset_label,
            "moelpr",
            variant_name,
            dataset_env,
            log_root,
            lm_eval_cfg,
        )


def launch_eval_watcher(
    watch_dir: Path,
    dataset_label: str,
    approach_name: str,
    variant_name: str,
    dataset_env: Dict[str, str],
    log_root: Path,
    lm_eval_cfg: dict,
) -> None:
    if not lm_eval_cfg.get("enabled"):
        return
    eval_script = (COMPARISON_DIR / lm_eval_cfg.get("eval_script", "lm_eval_checkpoint.sh")).resolve()
    watcher_script = (COMPARISON_DIR / "watcher_job.sh").resolve()
    if not eval_script.exists() or not watcher_script.exists():
        return
    label = f"{dataset_label}-{approach_name}-{variant_name}"
    watcher_env = dict(dataset_env)
    eval_log_root = log_root / "eval"
    eval_log_root.mkdir(parents=True, exist_ok=True)
    watcher_env.update(
        {
            "WATCH_DIR": str(watch_dir),
            "WATCH_LABEL": label,
            "EVAL_SCRIPT": str(eval_script),
            "EVAL_LOG_DIR": str(eval_log_root),
            "WATCH_CHECKPOINTS_PATH": str(COMPARISON_DIR / "watch_checkpoints.py"),
            "LM_EVAL_TASKS": lm_eval_cfg.get("_tasks_resolved", lm_eval_cfg.get("tasks", "belebele")),
            "LM_EVAL_BATCH_SIZE": str(lm_eval_cfg.get("batch_size", "auto")),
            "LM_EVAL_LIMIT": str(lm_eval_cfg.get("limit", 0)),
            "LM_EVAL_POLL_INTERVAL": str(lm_eval_cfg.get("poll_interval_seconds", 120)),
            "LM_EVAL_IDLE_TIMEOUT": str(lm_eval_cfg.get("idle_timeout_minutes", 60)),
            "LM_EVAL_MAX_DURATION": str(lm_eval_cfg.get("max_duration_minutes", 1440)),
            "LM_EVAL_RESULTS_DIR": str(watch_dir / "lm_eval_results"),
            "TOKENIZER_PATH": dataset_env.get("MODEL_NAME_OR_PATH", ""),
            "WANDB_GROUP": dataset_env.get("WANDB_RUN_GROUP", label),
            "EVAL_PARTITION": lm_eval_cfg.get("eval_sbatch", {}).get("partition", ""),
            "EVAL_TIME": lm_eval_cfg.get("eval_sbatch", {}).get("time", ""),
            "EVAL_GPUS": str(lm_eval_cfg.get("eval_sbatch", {}).get("gpus", 1)),
            "EVAL_GPU_TYPE": dataset_env.get("GPU_TYPE", ""),
            "EVAL_EXTRA_SBATCH": lm_eval_cfg.get("eval_sbatch", {}).get("extra_args", ""),
            "EVAL_CONDA_ENV": lm_eval_cfg.get("conda_env", "lm_eval"),
        }
    )
    env_assignments = prepare_env(watcher_env)
    watcher_log_root = log_root / "watcher"
    watcher_log_root.mkdir(parents=True, exist_ok=True)
    log_path = watcher_log_root / f"watcher_{approach_name}_{variant_name}_{int(time.time())}.log"
    cmd = ["env", *env_assignments, "sbatch", f"--output={log_path}"]
    watcher_sbatch = lm_eval_cfg.get("watcher_sbatch", {})
    if watcher_sbatch.get("partition"):
        cmd.append(f"--partition={watcher_sbatch['partition']}")
    if watcher_sbatch.get("time"):
        cmd.append(f"--time={watcher_sbatch['time']}")
    if watcher_sbatch.get("cpus_per_task"):
        cmd.append(f"--cpus-per-task={watcher_sbatch['cpus_per_task']}")
    if watcher_sbatch.get("mem"):
        cmd.append(f"--mem={watcher_sbatch['mem']}")
    subprocess.run(cmd + [str(watcher_script)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit comparison runs from JSON config.")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "comparison_config.json", help="Path to comparison config JSON.")
    args = parser.parse_args()

    config = load_config(args.config)
    global_cfg = config.get("global", {})
    models = config.get("models", [])
    approaches = config.get("approaches", {})
    lm_eval_cfg = config.get("lm_eval", {})
    tasks_value = lm_eval_cfg.get("tasks", "belebele")
    tasks_file = lm_eval_cfg.get("tasks_file")
    if tasks_file:
        path = resolve_path(tasks_file)
        ensure_dir(path.parent, "LM_EVAL tasks file directory")
        if not path.exists():
            sys.exit(f"[ERROR] LM_EVAL tasks file {path} does not exist.")
        tasks_lines = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
        if tasks_lines:
            tasks_value = ",".join(tasks_lines)
    lm_eval_cfg["_tasks_resolved"] = tasks_value
    if not models or not approaches:
        sys.exit("[ERROR] Config must define both 'models' and 'approaches'.")

    output_root = resolve_path(global_cfg.get("output_root", "./comparison_runs"))
    log_root = resolve_path(global_cfg.get("log_root", "./scripts/slurm/logs/comparison"))
    dataset_dir = resolve_path(global_cfg.get("dataset_dir", "./LLaMA-Factory/data"))
    tokenized_base = resolve_path(global_cfg.get("tokenized_base", "./tokenized"))
    wandb_project = global_cfg.get("wandb_project", "")
    wandb_entity = global_cfg.get("wandb_entity", "")
    wandb_group_prefix = global_cfg.get("wandb_group_prefix", "refactor-benchmark")
    moelpr_language_column = global_cfg.get("moelpr_language_column", "language")
    moelpr_language_map = resolve_path(
        global_cfg.get("moelpr_language_map", "./configs/moelpr_lang_map.json")
    )
    moelpr_original_language = global_cfg.get("moelpr_original_language", "english")
    flash_attn = global_cfg.get("flash_attn", os.environ.get("FLASH_ATTN", "fa2"))
    auto_find_batch_size = global_cfg.get(
        "auto_find_batch_size", os.environ.get("AUTO_FIND_BATCH_SIZE", "false")
    )
    preprocessing_num_workers = global_cfg.get(
        "preprocessing_num_workers", os.environ.get("PREPROCESSING_NUM_WORKERS")
    )
    preprocessing_batch_size = global_cfg.get(
        "preprocessing_batch_size", os.environ.get("PREPROCESSING_BATCH_SIZE")
    )
    eval_steps = int(global_cfg.get("eval_steps", 200))
    save_steps = int(global_cfg.get("save_steps", eval_steps))
    global_resources = global_cfg.get("resources", {}) or {}
    dataset_resources_map = global_cfg.get("dataset_resources", {}) or {}
    default_gpu_type = global_resources.get("gpu_type", global_cfg.get("gpu_type", "h100"))
    default_gpus_per_job = int(
        global_resources.get(
            "gpus_per_node",
            global_resources.get("gpus_per_job", global_cfg.get("gpus_per_job", 4)),
        )
    )

    ensure_dir(dataset_dir, "DATASET_DIR")
    ensure_dir(moelpr_language_map, "MOELPR_LANGUAGE_MAP")
    ensure_dir(tokenized_base, "tokenized_base")
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    base_env = {
        "DATASET_DIR": str(dataset_dir),
        "WANDB_PROJECT": wandb_project,
        "WANDB_ENTITY": wandb_entity,
        "USE_ACCELERATE": "1",
        "MOELPR_LANGUAGE_COLUMN": moelpr_language_column,
        "MOELPR_LANGUAGE_MAP": str(moelpr_language_map),
        "MOELPR_ORIGINAL_LANGUAGE": moelpr_original_language,
        "EVAL_STEPS": str(eval_steps),
        "SAVE_STEPS": str(save_steps),
        "FLASH_ATTN": str(flash_attn),
        "AUTO_FIND_BATCH_SIZE": str(auto_find_batch_size),
    }
    if preprocessing_num_workers is not None:
        base_env["PREPROCESSING_NUM_WORKERS"] = str(preprocessing_num_workers)
    if preprocessing_batch_size is not None:
        base_env["PREPROCESSING_BATCH_SIZE"] = str(preprocessing_batch_size)
    for key in ("CONDA_BASE", "CONDA_ENV", "MODULE_INIT"):
        if os.environ.get(key):
            base_env[key] = os.environ[key]

    sbatch_args = os.environ.get("SBATCH_ARGS", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_registry: Dict[str, str] = {}

    for model_entry in models:
        model_name = model_entry["model_name_or_path"]
        model_label = model_name.replace("/", "-")
        model_short = _short_model_label(model_name)
        tokenizer_name = model_entry.get("tokenizer_name")
        if not tokenizer_name:
            sys.exit(f"[ERROR] tokenizer_name missing for model {model_name}.")
        tokenizer_tag = _tokenizer_tag(tokenizer_name)
        model_resources = model_entry.get("resources", {}) or {}
        model_gpus = int(model_entry.get("gpus_per_job", default_gpus_per_job))
        model_gpu_type = model_entry.get("gpu_type", default_gpu_type)
        tokenizer_root = (tokenized_base / tokenizer_name).resolve()
        ensure_dir(tokenizer_root, f"tokenizer_root ({model_name})")
        dataset_entries = model_entry.get("datasets", [])
        specified_names: List[str] = []
        dataset_entry_resources: Dict[str, dict] = {}
        for entry in dataset_entries:
            name, resources = _resolve_dataset_entry(entry)
            specified_names.append(name)
            merged_resources = merge_resources(dataset_resources_map.get(name, {}), resources)
            dataset_entry_resources[name] = merged_resources
        datasets = discover_datasets(tokenizer_root, specified_names)
        for dataset_name, dataset_path in datasets:
            dataset_label = f"{tokenizer_root.name}-{dataset_name}"
            dataset_resources = dataset_entry_resources.get(
                dataset_name,
                dataset_resources_map.get(dataset_name, {}),
            )
            base_resources = merge_resources(global_resources, model_resources, dataset_resources)
            gpu_hours = _tier_gpu_hours(dataset_name)
            dataset_output_root = output_root / dataset_label
            dataset_log_root = log_root / dataset_label
            dataset_output_root.mkdir(parents=True, exist_ok=True)
            dataset_log_root.mkdir(parents=True, exist_ok=True)
            dataset_env = dict(base_env)
            moelpr_root = (tokenized_base / "moelpr" / tokenizer_name / dataset_name).resolve()
            moelpr_stage1 = moelpr_root / "stage1_newlangs"
            moelpr_stage2 = moelpr_root / "stage2_english"
            if not moelpr_stage1.exists() or not moelpr_stage2.exists():
                print(
                    f"[WARN] MoE-LPR datasets missing for {tokenizer_name}/{dataset_name}: "
                    f"{moelpr_stage1} / {moelpr_stage2}"
                )
            dataset_env.update(
                {
                    "MODEL_NAME_OR_PATH": model_name,
                    "MODEL_LABEL": model_label,
                    "MODEL_SHORT": model_short,
                    "TOKENIZER_TAG": tokenizer_tag,
                    "DATASET_NAME": dataset_name,
                    "GENERIC_TOKENIZED_PATH": str(dataset_path),
                    "GENERIC_DATASET_NAME": dataset_name,
                    "MOELPR_TOKENIZED_PATH": str(moelpr_stage1),
                    "MOELPR_TOKENIZED_PATH_STAGE2": str(moelpr_stage2),
                    "MOELPR_DATASET_NAME": dataset_name,
                    "MOELPR_EVAL_DATASET_NAME": dataset_name,
                    "WANDB_RUN_GROUP": f"{wandb_group_prefix}-{dataset_label}-{timestamp}",
                }
            )
            for approach_name, approach_cfg in approaches.items():
                if approach_cfg.get("type") == "moelpr":
                    run_moelpr_variants(
                        dataset_label,
                        dataset_env,
                        approach_cfg,
                        base_resources,
                        timestamp,
                        dataset_output_root,
                        dataset_log_root,
                        sbatch_args,
                        gpu_hours,
                        job_registry,
                        lm_eval_cfg,
                        model_gpus,
                        model_gpu_type,
                    )
                else:
                    run_simple_variants(
                        dataset_label,
                        dataset_env,
                        approach_name,
                        approach_cfg,
                        base_resources,
                        timestamp,
                        dataset_output_root,
                        dataset_log_root,
                        sbatch_args,
                        gpu_hours,
                        job_registry,
                        lm_eval_cfg,
                        model_gpus,
                        model_gpu_type,
                    )

    if not job_registry:
        print("[WARN] No jobs submitted.")
    else:
        print("========================================")
        print(f"Submitted comparison sweeps at {timestamp}")
        for key in sorted(job_registry):
            print(f"  {key:<35} : {job_registry[key]}")
        print(f"Outputs stored under {output_root}")
        print(f"W&B project: {wandb_project or 'N/A'}")
        print("========================================")


if __name__ == "__main__":
    main()
