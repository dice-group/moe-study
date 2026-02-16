#!/usr/bin/env python3
"""Profile-driven local E2E validation orchestration."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


from pipeline.runtime import _abs, _deep_merge, _load_json, _missing_modules, _resolve_profile


def _run(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    rendered = " ".join(cmd)
    print(f"[CMD] {rendered}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, env=env, check=True)


def _manifest_data_files(manifest_path: Path) -> str:
    payload = _load_json(manifest_path)
    files = payload.get("data_files", [])
    if not files:
        raise SystemExit(f"No data_files in manifest: {manifest_path}")
    return ",".join(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run profile-based E2E validation.")
    parser.add_argument("--profile", default="cpu_sandbox", help="Profile key in configs/e2e_validation_profiles.json")
    parser.add_argument(
        "--config",
        default="configs/e2e_validation_profiles.json",
        help="Path to E2E profile JSON config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--skip-build", action="store_true", help="Skip synthetic data build stage.")
    parser.add_argument("--skip-training", action="store_true", help="Skip smoke training stage.")
    parser.add_argument("--skip-audit", action="store_true", help="Skip pad/label audit stage.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = _abs(repo_root, args.config)
    config = _load_json(config_path)
    profile = _resolve_profile(config, args.profile)

    language_map_path = _abs(repo_root, profile["language_map_path"])
    language_selection = profile.get("language_selection", {})
    synthetic = profile.get("synthetic", {})
    tokenization = profile.get("tokenization", {})
    training = profile.get("training", {})
    short_budget = training.get("short_run_budget", {})
    eval_policy = profile.get("eval_policy", {})
    audit = profile.get("audit", {})

    synthetic_output_dir = _abs(repo_root, synthetic["output_dir"])
    smoke_output_root = _abs(repo_root, training["smoke_output_root"])
    manifest_path = synthetic_output_dir / "manifest.json"
    tokenized_output_dir = _abs(repo_root, tokenization.get("output_dir", "outputs/e2e_validation/tokenized_from_merlin"))

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{repo_root / 'LLaMA-Factory' / 'src'}:{env.get('PYTHONPATH', '')}"
    env.setdefault("WANDB_MODE", "disabled")

    # Stage 0: repo sanity checks.
    validate_script = repo_root / "pipeline_tools" / "validate_repo_config.py"
    if validate_script.exists():
        _run(["python3", str(validate_script)], env, args.dry_run)

    if not args.skip_build:
        policy = language_selection.get("policy", "tier_subset")
        explicit_languages = ""
        max_languages = 0
        if policy == "explicit":
            explicit_languages = ",".join(language_selection.get("languages", []))
        else:
            max_languages = int(language_selection.get("max_languages", 0))

        sampling = synthetic.get("sampling_policy", {})
        cmd = [
            "python3",
            str(repo_root / "pipeline_tools" / "build_synthetic_multilingual_shards.py"),
            "--language-map",
            str(language_map_path),
            "--output-dir",
            str(synthetic_output_dir),
            "--total-samples",
            str(int(synthetic.get("total_samples", 96))),
            "--sampling-strategy",
            sampling.get("strategy", "uniform"),
            "--alpha",
            str(float(sampling.get("alpha", 0.3))),
            "--min-samples-per-language",
            str(int(sampling.get("min_samples_per_language", 4))),
            "--max-samples-per-language",
            str(int(sampling.get("max_samples_per_language", 64))),
            "--seed",
            str(int(synthetic.get("seed", 42))),
        ]
        if explicit_languages:
            cmd.extend(["--languages", explicit_languages])
        if max_languages > 0:
            cmd.extend(["--max-languages", str(max_languages)])
        _run(cmd, env, args.dry_run)

    if not manifest_path.exists() and not args.dry_run:
        raise SystemExit(f"Synthetic manifest not found: {manifest_path}")

    # Stage 1: tokenization correctness using merlin tokenizer.
    should_run_tokenization = tokenization.get("enabled", True)
    if should_run_tokenization:
        missing = _missing_modules(["datasets", "transformers"])
        if missing:
            print(
                f"[WARN] Skipping tokenization stage; missing Python modules: {', '.join(missing)}",
                flush=True,
            )
            should_run_tokenization = False
    if should_run_tokenization:
        tokenization_script = repo_root / "data_prep" / "merlin_data_prep" / "distributed_data_processor" / "src" / "tokenize_slurm.py"
        if not tokenization_script.exists():
            raise SystemExit(f"Missing tokenization script: {tokenization_script}")
        tokenized_output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python3",
            str(tokenization_script),
            "--shard_dir",
            str(synthetic_output_dir),
            "--save_tokenized_data_dir",
            str(tokenized_output_dir),
            "--model_name",
            str(tokenization.get("model_name", training.get("model_name_or_path"))),
            "--num_proc",
            str(int(tokenization.get("num_proc", 1))),
            "--eval_fraction",
            str(float(tokenization.get("eval_fraction", 0.1))),
            "--eval_seed",
            str(int(tokenization.get("eval_seed", 42))),
            "--sample_fraction",
            str(float(tokenization.get("sample_fraction", 1.0))),
            "--sample_seed",
            str(int(tokenization.get("sample_seed", 42))),
            "--max_length",
            str(int(tokenization.get("max_length", short_budget.get("dataset_max_length", 96)))),
            "--language-map",
            str(language_map_path),
            "--mask_pad_labels",
            "true" if tokenization.get("mask_pad_labels", True) else "false",
        ]
        max_samples = int(tokenization.get("max_samples", 0))
        if max_samples > 0:
            cmd.extend(["--max_samples", str(max_samples)])
        _run(cmd, env, args.dry_run)

    should_run_training = not args.skip_training and training.get("enabled", True)
    if should_run_training:
        missing = _missing_modules(["datasets", "transformers", "torch", "pytest"])
        if missing:
            print(
                f"[WARN] Skipping training stage; missing Python modules: {', '.join(missing)}",
                flush=True,
            )
            should_run_training = False

    if should_run_training:
        dataset_csv = "placeholder.csv" if args.dry_run else _manifest_data_files(manifest_path)
        script_path = _abs(repo_root, training["script"])
        smoke_output_root.mkdir(parents=True, exist_ok=True)

        env.update(
            {
                "RUN_TRAIN_SMOKE": str(training.get("run_train_smoke", 1)),
                "RUN_LM_EVAL_SMOKE": str(training.get("run_lm_eval_smoke", 0)),
                "ENABLE_LM_EVAL_LISTENER": str(training.get("enable_lm_eval_listener", 0)),
                "MODEL_NAME_OR_PATH": str(training.get("model_name_or_path")),
                "SMOKE_OUTPUT_ROOT": str(smoke_output_root),
                "SMOKE_DATASET_DATA_FILES": dataset_csv,
                "SMOKE_DATASET_DEFAULT_LANGUAGE": str(
                    (language_selection.get("languages") or ["aeb_Arab"])[0]
                ),
                "SMOKE_TRAIN_STEPS": str(short_budget.get("train_steps", 1)),
                "SMOKE_SAVE_STEPS": str(short_budget.get("save_steps", 1)),
                "SMOKE_BATCH_SIZE": str(short_budget.get("batch_size", 1)),
                "SMOKE_GRAD_ACCUM": str(short_budget.get("grad_accum", 1)),
                "SMOKE_LOGGING_STEPS": str(short_budget.get("logging_steps", 1)),
                "SMOKE_DATASET_MAX_SAMPLES": str(short_budget.get("dataset_max_samples", 24)),
                "SMOKE_DATASET_MAX_LENGTH": str(short_budget.get("dataset_max_length", 96)),
                "SMOKE_DATASET_MAX_SCAN": str(short_budget.get("dataset_max_scan", 500)),
                "LM_EVAL_TASKS": ",".join(eval_policy.get("tasks", [])),
                "LM_EVAL_BATCH_SIZE": str(eval_policy.get("batch_size", "auto")),
                "LM_EVAL_EXTRA_ARGS": str(eval_policy.get("extra_args", "--limit 5")),
                "LM_EVAL_WANDB_MODE": str(eval_policy.get("wandb_mode", "disabled")),
                # Keep E2E stable by targeting only required integration checks.
                "PYTEST_TARGET": str(
                    training.get(
                        "pytest_target",
                        "tests/integration/test_adapter_trainability_contract.py "
                        "tests/integration/test_smoke_training.py::test_smoke_training_all_variants",
                    )
                ),
            }
        )

        _run(["bash", str(script_path)], env, args.dry_run)

    should_run_audit = not args.skip_audit and audit.get("enabled", True)
    if should_run_audit:
        dataset_path = _abs(repo_root, audit.get("dataset_path", str(tokenized_output_dir)))
        missing = _missing_modules(["datasets"])
        if missing:
            print(
                f"[WARN] Skipping audit stage; missing Python modules: {', '.join(missing)}",
                flush=True,
            )
            should_run_audit = False
        elif not dataset_path.exists() and not args.dry_run:
            print(f"[WARN] Skipping audit stage; dataset path not found: {dataset_path}", flush=True)
            should_run_audit = False

    if should_run_audit:
        dataset_path = _abs(repo_root, audit.get("dataset_path", str(tokenized_output_dir)))
        cmd = [
            "python3",
            str(repo_root / "pipeline_tools" / "audit_pad_label_mask.py"),
            "--dataset-path",
            str(dataset_path),
            "--split",
            str(audit.get("split", "train")),
            "--pad-label-id",
            str(int(audit.get("pad_label_id", -100))),
            "--max-rows",
            str(int(audit.get("max_rows", 256))),
        ]
        if audit.get("strict", True):
            cmd.append("--strict")
        _run(cmd, env, args.dry_run)

    print(f"[DONE] profile={args.profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
