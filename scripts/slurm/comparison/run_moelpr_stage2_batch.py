#!/usr/bin/env python3
"""Submit MoE-LPR Stage-2 jobs for a fixed list of Stage-1 outputs."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

CONFIG_PATHS = [
    "scripts/slurm/comparison/comparison_config_llama3_2_1b.json",
    "scripts/slurm/comparison/comparison_config_llama3_2_3b.json",
]
STAGE1_DIRS = [
    "/scratch/anonymous-project/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-1B_tokenizer-10_langs/moelpr/paper-best_stage1_20260126_204129",
    # "/scratch/anonymous-project/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-1B_tokenizer-96_langs/moelpr/paper-best_stage1_20260126_204008",
    # "/scratch/anonymous-project/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-1B_tokenizer-200_langs/moelpr/paper-best_stage1_20260126_203734",
    # "/scratch/anonymous-project/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-3B_tokenizer-10_langs/moelpr/paper-best_stage1_20260126_202526",
    # "/scratch/anonymous-project/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-3B_tokenizer-96_langs/moelpr/paper-best_stage1_20260126_201857",
    # "/scratch/anonymous-project/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-3B_tokenizer-200_langs/moelpr/paper-best_stage1_20260126_201628"
]
GPUS_PER_NODE = 4


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path_str).resolve()
    return path


def pick_model_by_tokenizer(models: list[dict], tokenizer_name: str) -> dict:
    for entry in models:
        if entry.get("tokenizer_name") == tokenizer_name:
            return entry
    raise SystemExit(f"[ERROR] No model matches tokenizer_name={tokenizer_name}.")


def infer_dataset_label(stage1_dir: Path) -> str:
    # .../<dataset_label>/moelpr/<variant>_stage1_*
    return stage1_dir.parent.parent.name


def infer_variant(stage1_dir: Path) -> str:
    name = stage1_dir.name
    if "_stage1_" in name:
        return name.split("_stage1_", 1)[0]
    return name


def infer_stage1_timestamp(stage1_dir: Path) -> str | None:
    name = stage1_dir.name
    if "_stage1_" in name:
        return name.split("_stage1_", 1)[1]
    return None


def infer_tokenizer_and_dataset(models: list[dict], dataset_label: str) -> tuple[str, str]:
    # dataset_label = f"{tokenizer_name}-{dataset}"
    for entry in models:
        tokenizer = entry.get("tokenizer_name")
        if tokenizer and dataset_label.startswith(f"{tokenizer}-"):
            dataset = dataset_label[len(tokenizer) + 1 :]
            return tokenizer, dataset
    raise SystemExit(f"[ERROR] Could not infer tokenizer/dataset from label {dataset_label}.")


def pick_accelerate_config(gpus: int) -> Path | None:
    accel_dir = REPO_ROOT / "LLaMA-Factory" / "examples" / "accelerate"
    mapping = {
        1: None,
        2: accel_dir / "fsdp_2gpu_config.yaml",
        4: accel_dir / "fsdp_4gpu_config.yaml",
        8: accel_dir / "fsdp_8gpu_config.yaml",
    }
    cfg = mapping.get(gpus)
    if cfg is None:
        raise SystemExit(f"[ERROR] Unsupported GPU count {gpus}.")
    if cfg and not cfg.exists():
        raise SystemExit(f"[ERROR] Accelerate config not found: {cfg}")
    return cfg


def main() -> None:
    configs = [load_config(resolve_path(p)) for p in CONFIG_PATHS]

    stage1_dirs = [Path(p).resolve() for p in STAGE1_DIRS]
    if not stage1_dirs:
        raise SystemExit("[ERROR] STAGE1_DIRS is empty.")

    for stage1_dir in stage1_dirs:
        if not stage1_dir.exists():
            print(f"[WARN] Stage-1 dir missing, skip: {stage1_dir}")
            continue

        dataset_label = infer_dataset_label(stage1_dir)
        cfg = None
        for candidate in configs:
            models = candidate.get("models", [])
            try:
                tokenizer_name, dataset = infer_tokenizer_and_dataset(models, dataset_label)
                model_entry = pick_model_by_tokenizer(models, tokenizer_name)
                cfg = candidate
                break
            except SystemExit:
                continue
        if cfg is None:
            print(f"[WARN] No config matched for {dataset_label}, skip.")
            continue

        global_cfg = cfg.get("global", {})
        approaches = cfg.get("approaches", {})
        moelpr_cfg = approaches.get("moelpr", {})
        variants = {v["name"]: v for v in moelpr_cfg.get("variants", [])}
        output_root = resolve_path(global_cfg.get("output_root", "./comparison_runs"))
        log_root = resolve_path(global_cfg.get("log_root", "./scripts/slurm/logs/comparison"))
        tokenized_base = resolve_path(global_cfg.get("tokenized_base", "./tokenized"))

        variant = infer_variant(stage1_dir)
        variant_cfg = variants.get(variant, {})
        if not variant_cfg:
            print(f"[WARN] Variant {variant} not found in config for {dataset_label}, skip.")
            continue
        variant_env = variant_cfg.get("env", {})
        batch_size = int(variant_cfg.get("batch_size", moelpr_cfg.get("batch_size", 16)))

        dataset_resources = (global_cfg.get("dataset_resources", {}) or {}).get(dataset, {})
        resources = global_cfg.get("resources", {}) or {}
        gpus = GPUS_PER_NODE

        stage1_ts = infer_stage1_timestamp(stage1_dir) or datetime.now().strftime("%Y%m%d_%H%M%S")
        stage2_dir = stage1_dir.parent / f"{variant}_stage2_{stage1_ts}"
        stage2_dir.mkdir(parents=True, exist_ok=True)

        moelpr_root = tokenized_base / "moelpr" / tokenizer_name / dataset
        tokenized_stage1 = moelpr_root / "stage1_newlangs"
        tokenized_stage2 = moelpr_root / "stage2_english"

        env = {
            "STAGE_ONLY": "2",
            "REPO_ROOT": str(REPO_ROOT),
            "MODEL_NAME_OR_PATH": model_entry.get("model_name_or_path"),
            "DATASET_DIR": str(resolve_path(global_cfg.get("dataset_dir", "./LLaMA-Factory/data"))),
            "MOELPR_TOKENIZED_PATH": str(tokenized_stage1),
            "MOELPR_TOKENIZED_PATH_STAGE2": str(tokenized_stage2),
            "MOELPR_STAGE1_OUTPUT": str(stage1_dir),
            "MOELPR_STAGE2_OUTPUT": str(stage2_dir),
            "MOELPR_LANGUAGE_COLUMN": global_cfg.get("moelpr_language_column", "language"),
            "MOELPR_LANGUAGE_MAP": str(resolve_path(global_cfg.get("moelpr_language_map", "./configs/moelpr_lang_map.json"))),
            "MOELPR_ORIGINAL_LANGUAGE": global_cfg.get("moelpr_original_language", "english"),
            "MOELPR_DATASET_NAME": dataset,
            "MOELPR_EVAL_DATASET_NAME": dataset,
            "WANDB_PROJECT": global_cfg.get("wandb_project", ""),
            "WANDB_ENTITY": global_cfg.get("wandb_entity", ""),
            "WANDB_RUN_GROUP": f"moelpr-restart-{dataset_label}-{variant}",
            "PER_DEVICE_TRAIN_BATCH_SIZE": str(batch_size),
            "PER_DEVICE_EVAL_BATCH_SIZE": str(batch_size),
            "GPUS_PER_NODE": str(gpus),
            "GPU_TYPE": dataset_resources.get("gpu_type") or resources.get("gpu_type") or global_cfg.get("gpu_type", ""),
            "USE_ACCELERATE": "1",
            "ACCELERATE_USE_FSDP": "1" if gpus > 1 else "0",
        }

        for k, v in variant_env.items():
            env[str(k)] = str(v)

        accel_cfg = pick_accelerate_config(gpus)
        if accel_cfg is not None:
            env["ACCELERATE_CONFIG"] = str(accel_cfg)

        log_root_dir = log_root / dataset_label
        log_root_dir.mkdir(parents=True, exist_ok=True)
        log_prefix = moelpr_cfg.get("log_prefix", "moelpr")
        log_path = log_root_dir / f"{log_prefix}_{variant}_{stage1_ts}_stage2.log"

        cmd = ["env", *[f"{k}={v}" for k, v in env.items()], "sbatch", f"--output={log_path}", str(SCRIPT_DIR / "moelpr_job.sh")]
        print(f"[INFO] Submit stage2 {dataset_label} | {variant}\nckpt={stage1_dir} -> {stage2_dir}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
