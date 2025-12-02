from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
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


def run_simple_variants(dataset_label: str, dataset_env: Dict[str, str], approach_name: str, cfg: dict, timestamp: str, output_root: Path, log_root: Path, sbatch_args: str, registry: Dict[str, str]) -> None:
    script = COMPARISON_DIR / cfg["script"]
    batch_size = int(cfg.get("batch_size", 16))
    log_prefix = cfg.get("log_prefix", approach_name)
    variants = cfg.get("variants", [])
    for variant in variants:
        variant_name = variant["name"]
        variant_env = {k: to_env_value(v) for k, v in variant.get("env", {}).items()}
        bs = int(variant.get("batch_size", batch_size))
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
            }
        )
        job_key = f"{dataset_label}-{approach_name}-{variant_name}"
        print(f"[INFO] Queueing {job_key} (bs={bs})")
        env_assignments = prepare_env(env)
        job_id = submit_job(job_key, script, log_path, env_assignments, sbatch_args)
        registry[job_key] = job_id


def run_moelpr_variants(
    dataset_label: str,
    dataset_env: Dict[str, str],
    cfg: dict,
    timestamp: str,
    output_root: Path,
    log_root: Path,
    default_bs: int,
    sbatch_args: str,
    registry: Dict[str, str],
) -> None:
    script = COMPARISON_DIR / cfg["script"]
    log_prefix = cfg.get("log_prefix", "moelpr")
    for variant in cfg.get("variants", []):
        variant_name = variant["name"]
        variant_env = {k: to_env_value(v) for k, v in variant.get("env", {}).items()}
        bs = int(variant.get("batch_size", default_bs))
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
            }
        )
        log_path = log_root / f"{log_prefix}_{variant_name}_{timestamp}.log"
        job_key = f"{dataset_label}-moelpr-{variant_name}"
        print(f"[INFO] Queueing {job_key} (bs={bs})")
        env_assignments = prepare_env(env)
        job_id = submit_job(job_key, script, log_path, env_assignments, sbatch_args)
        registry[job_key] = job_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit comparison runs from JSON config.")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "comparison_config.json", help="Path to comparison config JSON.")
    args = parser.parse_args()

    config = load_config(args.config)
    global_cfg = config.get("global", {})
    models = config.get("models", [])
    approaches = config.get("approaches", {})
    if not models or not approaches:
        sys.exit("[ERROR] Config must define both 'models' and 'approaches'.")

    output_root = resolve_path(global_cfg.get("output_root", "./comparison_runs"))
    log_root = resolve_path(global_cfg.get("log_root", "./scripts/slurm/logs/comparison"))
    dataset_dir = resolve_path(global_cfg.get("dataset_dir", "./LLaMA-Factory/data"))
    tokenized_base = resolve_path(global_cfg.get("tokenized_base", "./tokenized"))
    wandb_project = global_cfg.get("wandb_project", "")
    wandb_entity = global_cfg.get("wandb_entity", "")
    wandb_group_prefix = global_cfg.get("wandb_group_prefix", "refactor-benchmark")
    moelpr_batch_size = int(global_cfg.get("moelpr_batch_size", 16))
    moelpr_language_column = global_cfg.get("moelpr_language_column", "language")
    moelpr_language_map = resolve_path(
        global_cfg.get("moelpr_language_map", "./configs/moelpr_lang_map.json")
    )
    moelpr_original_language = global_cfg.get("moelpr_original_language", "english")

    ensure_dir(dataset_dir, "DATASET_DIR")
    ensure_dir(moelpr_language_map, "MOELPR_LANGUAGE_MAP")
    ensure_dir(tokenized_base, "tokenized_base")
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    base_env = {
        "DATASET_DIR": str(dataset_dir),
        "WANDB_PROJECT": wandb_project,
        "WANDB_ENTITY": wandb_entity,
        "MOELPR_LANGUAGE_COLUMN": moelpr_language_column,
        "MOELPR_LANGUAGE_MAP": str(moelpr_language_map),
        "MOELPR_ORIGINAL_LANGUAGE": moelpr_original_language,
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
        tokenizer_root = (tokenized_base / tokenizer_name).resolve()
        ensure_dir(tokenizer_root, f"tokenizer_root ({model_name})")
        datasets = discover_datasets(tokenizer_root, model_entry.get("datasets", []))
        for dataset_name, dataset_path in datasets:
            dataset_label = f"{tokenizer_root.name}-{dataset_name}"
            dataset_output_root = output_root / dataset_label
            dataset_log_root = log_root / dataset_label
            dataset_output_root.mkdir(parents=True, exist_ok=True)
            dataset_log_root.mkdir(parents=True, exist_ok=True)
            dataset_env = dict(base_env)
            dataset_env.update(
                {
                    "MODEL_NAME_OR_PATH": model_name,
                    "GENERIC_TOKENIZED_PATH": str(dataset_path),
                    "GENERIC_DATASET_NAME": dataset_name,
                    "MOELPR_TOKENIZED_PATH": str(dataset_path),
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
                        moelpr_batch_size,
                        sbatch_args,
                        job_registry,
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
                        job_registry,
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
