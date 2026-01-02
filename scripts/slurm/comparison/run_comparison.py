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


def accelerate_env_for_gpus(gpus_per_job: int, repo_root: Path) -> Dict[str, str]:
    if gpus_per_job == 2:
        config_path = repo_root / "LLaMA-Factory" / "examples" / "accelerate" / "fsdp_2gpu_config.yaml"
    elif gpus_per_job == 4:
        config_path = repo_root / "LLaMA-Factory" / "examples" / "accelerate" / "fsdp_4gpu_config.yaml"
    else:
        return {}
    if not config_path.exists():
        sys.exit(f"[ERROR] Accelerate config {config_path} not found.")
    return {
        "USE_ACCELERATE": "1",
        "ACCELERATE_CONFIG": str(config_path),
        "ACCELERATE_USE_FSDP": "1",
    }


def _tier_hours(dataset_name: str) -> float | None:
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
    timestamp: str,
    output_root: Path,
    log_root: Path,
    sbatch_args: str,
    time_limit: str | None,
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
        bs = int(variant.get("batch_size", batch_size))
        job_gpus = int(variant.get("gpus_per_job", cfg.get("gpus_per_job", default_gpus)))
        job_gpu_type = variant.get("gpu_type", cfg.get("gpu_type", default_gpu_type))
        run_dir = output_root / approach_name / f"{variant_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"{log_prefix}_{variant_name}_{timestamp}.log"
        env = dict(dataset_env)
        env.update(variant_env)
        env.update(
            {
                "OUTPUT_DIR": str(run_dir),
                "WANDB_NAME": f"{dataset_label}-{approach_name}-{variant_name}-{timestamp}",
                "PER_DEVICE_TRAIN_BATCH_SIZE": str(bs),
                "PER_DEVICE_EVAL_BATCH_SIZE": str(bs),
                "GPUS_PER_NODE": str(job_gpus),
                "GPU_TYPE": job_gpu_type,
            }
        )
        env.update(accelerate_env_for_gpus(job_gpus, REPO_ROOT))
        job_key = f"{dataset_label}-{approach_name}-{variant_name}"
        print(f"[INFO] Queueing {job_key} (bs={bs})")
        env_assignments = prepare_env(env)
        time_arg = ""
        if time_limit and "--time" not in sbatch_args:
            time_arg = f"--time={time_limit}"
        job_specific_args = " ".join(
            arg for arg in [sbatch_args, f"--gres=gpu:{job_gpu_type}:{job_gpus}", time_arg] if arg
        )
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
    timestamp: str,
    output_root: Path,
    log_root: Path,
    sbatch_args: str,
    time_limit: str | None,
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
        bs = int(variant.get("batch_size", batch_size))
        job_gpus = int(variant.get("gpus_per_job", cfg.get("gpus_per_job", default_gpus)))
        job_gpu_type = variant.get("gpu_type", cfg.get("gpu_type", default_gpu_type))
        stage1_dir = output_root / "moelpr" / f"{variant_name}_stage1_{timestamp}"
        stage2_dir = output_root / "moelpr" / f"{variant_name}_stage2_{timestamp}"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        stage2_dir.mkdir(parents=True, exist_ok=True)
        env = dict(dataset_env)
        env.update(variant_env)
        env.update(
            {
                "MOELPR_STAGE1_OUTPUT": str(stage1_dir),
                "MOELPR_STAGE2_OUTPUT": str(stage2_dir),
                "WANDB_STAGE1_NAME": f"moelpr-stage1-{dataset_label}-{variant_name}-{timestamp}",
                "WANDB_STAGE2_NAME": f"moelpr-stage2-{dataset_label}-{variant_name}-{timestamp}",
                "PER_DEVICE_TRAIN_BATCH_SIZE": str(bs),
                "PER_DEVICE_EVAL_BATCH_SIZE": str(bs),
                "GPUS_PER_NODE": str(job_gpus),
                "GPU_TYPE": job_gpu_type,
            }
        )
        env.update(accelerate_env_for_gpus(job_gpus, REPO_ROOT))
        log_path = log_root / f"{log_prefix}_{variant_name}_{timestamp}.log"
        job_key = f"{dataset_label}-moelpr-{variant_name}"
        print(f"[INFO] Queueing {job_key} (bs={bs})")
        env_assignments = prepare_env(env)
        time_arg = ""
        if time_limit and "--time" not in sbatch_args:
            time_arg = f"--time={time_limit}"
        job_specific_args = " ".join(
            arg for arg in [sbatch_args, f"--gres=gpu:{job_gpu_type}:{job_gpus}", time_arg] if arg
        )
        job_id = submit_job(job_key, script, log_path, env_assignments, job_specific_args)
        registry[job_key] = job_id
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
    eval_steps = int(global_cfg.get("eval_steps", 200))
    default_gpu_type = global_cfg.get("gpu_type", "h100")
    default_gpus_per_job = int(global_cfg.get("gpus_per_job", 4))

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
    }
    for key in ("CONDA_BASE", "CONDA_ENV", "MODULE_INIT"):
        if os.environ.get(key):
            base_env[key] = os.environ[key]

    sbatch_args = os.environ.get("SBATCH_ARGS", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_registry: Dict[str, str] = {}

    for model_entry in models:
        model_name = model_entry["model_name_or_path"]
        tokenizer_name = model_entry.get("tokenizer_name")
        if not tokenizer_name:
            sys.exit(f"[ERROR] tokenizer_name missing for model {model_name}.")
        model_gpus = int(model_entry.get("gpus_per_job", default_gpus_per_job))
        model_gpu_type = model_entry.get("gpu_type", default_gpu_type)
        tokenizer_root = (tokenized_base / tokenizer_name).resolve()
        ensure_dir(tokenizer_root, f"tokenizer_root ({model_name})")
        datasets = discover_datasets(tokenizer_root, model_entry.get("datasets", []))
        for dataset_name, dataset_path in datasets:
            dataset_label = f"{tokenizer_root.name}-{dataset_name}"
            time_limit = _format_sbatch_time(_tier_hours(dataset_name))
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
                        timestamp,
                        dataset_output_root,
                        dataset_log_root,
                        sbatch_args,
                        time_limit,
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
                        timestamp,
                        dataset_output_root,
                        dataset_log_root,
                        sbatch_args,
                        time_limit,
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
