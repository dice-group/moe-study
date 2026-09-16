"""Bounded four-A100 calibration and acceptance for one model/configuration.

No production job is submitted here. A machine-readable gate is written only
after real training, reload, benchmark and dataset-capacity checks pass.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from pipeline_tools.cross_arch_study import (
    ROOT, PYTHON, campaign_root, model_root, contract, read_json, write_json, sha256,
    capacity_check, utc_now, sbatch,
)

BENCHMARK_CWD = ROOT / "scripts/eval/latest_checkpoint/lm-evaluation-harness"


def code_signature():
    paths = ["pipeline_tools/cross_arch_study.py", "pipeline_tools/cross_arch_pilot.py",
             "pipeline_tools/cross_arch_node.py", "configs/cross_arch_study/activate_node_cache.sh",
             "pipeline_tools/cross_arch_model_cache.py",
             "pipeline_tools/cross_arch_fsdp.py",
             "configs/cross_arch_study/python_startup/sitecustomize.py",
             "pipeline_tools/cross_arch_worker.py", "pipeline_tools/cross_arch_verify.py",
             "pipeline_tools/cross_arch_launch.py", "pipeline_tools/cross_arch_eval.py", "pipeline_tools/cross_arch_report.py",
             "pipeline_tools/cross_arch_runtime.py",
             "pipeline/training.py", "pipeline/executor.py", "pipeline/gpu_preflight.py",
             "configs/accelerate/fsdp_4gpu_config_size_sharded.yaml", "configs/cross_arch_study/96_languages.json",
             "scripts/eval/lm_eval_language_ids.py", "LLaMA-Factory/src/llamafactory/model/adapter.py"]
    paths.append(str((campaign_root() / "runtime/node_cache_ready.json").relative_to(ROOT)))
    paths.extend(str(p.relative_to(ROOT)) for folder in ("LLaMA-Factory/src/peft", "LLaMA-Factory/src/llamafactory")
                 for p in (ROOT / folder).rglob("*.py"))
    harness = ROOT / "scripts/eval/latest_checkpoint/lm-evaluation-harness/lm_eval"
    paths.append("configs/cross_arch_study/noctua2_runtime.json")
    paths.extend(str(p.relative_to(ROOT)) for folder in ("belebele", "flores_en_perplexity", "sib200", "taxi1500")
                 for p in (harness / "tasks" / folder).rglob("*") if p.suffix in {".py", ".yaml"})
    paths.extend(str((harness / p).relative_to(ROOT)) for p in ("models/huggingface.py", "evaluator.py"))
    # Every file is still hashed. Bound concurrent reads to avoid serial
    # shared-filesystem latency; no timestamp-only or stale-hash shortcut.
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(lambda p: (p, sha256(ROOT / p)), paths))


def load_profile(model):
    return read_json(campaign_root() / "pipeline_config.json")["profiles"][model]


def find_run(profile, run_id):
    matches = [r for r in profile["training"]["runs"] if r["id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one run {run_id}")
    return copy.deepcopy(matches[0])


def primary_values(folder, tasks):
    values = {}
    for path in Path(folder).glob("*.json"):
        payload = read_json(path)
        for task, metrics in payload.get("results", {}).items():
            if task not in tasks:
                continue
            prefix = "bits_per_byte," if task.startswith("flores_") else ("acc_norm," if task.startswith("belebele_") else "acc,")
            selected = [float(v) for k, v in metrics.items() if k.startswith(prefix) and isinstance(v, (int, float))]
            if len(selected) != 1 or not math.isfinite(selected[0]) or task in values:
                raise ValueError(f"Ambiguous, duplicated or nonfinite result: {task}")
            values[task] = selected[0]
    if set(values) != set(tasks):
        raise ValueError(f"Missing benchmark tasks: {set(tasks)-set(values)}")
    return values


def benchmark_command(model, run_id, checkpoint, tasks_file, output, *, limit=None):
    from pipeline_tools.cross_arch_model_cache import local_model_path
    cmd = [str(PYTHON), str(ROOT / "scripts/eval/lm_eval_language_ids.py"),
           "--checkpoint", str(checkpoint), "--tokenizer", str(model_root(model) / "model"),
           "--tasks", str(tasks_file), "--output-dir", str(output), "--batch-size", "1",
           "--torch-dtype", "bf16", "--device", "cuda", "--device-map", "auto",
           "--mode", "no_ids",  # Same final-benchmark protocol as the original seven runs.
           "--seed", "42", "--numpy-seed", "42", "--torch-seed", "42", "--fewshot-seed", "42"]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    cmd += ["--base-model-copy", str(local_model_path(model)),
            "--base-model-origin", str(model_root(model) / "model")]
    return cmd


def candidate(model, run_id, batch, disable_gc, label):
    from pipeline.training import build_training_cmd, adapter_checkpoint_provenance
    from pipeline.executor import LocalExecutor
    profile = load_profile(model)
    run = find_run(profile, run_id)
    out = model_root(model) / "pilots" / run_id / label
    receipt = out / "candidate.json"
    if receipt.exists():
        result = read_json(receipt)
        if result["code_signature"] != code_signature():
            raise ValueError("Pilot source changed; do not reuse stale pilot evidence")
        return result
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Interrupted attempt needs inspection before retry: {out}")
    out.mkdir(parents=True, exist_ok=True)
    backend = read_json(model_root(model) / "backend_ready.json")["selected_backend"]
    run["train"].update(batch_size=batch, grad_accum=12 // batch, disable_gradient_checkpointing=disable_gc,
                        logging_steps=10, flash_attn=backend)
    if not disable_gc and run["kind"] == "moelora":
        run["train"]["use_reentrant_gc"] = False
    env = dict(os.environ)
    from pipeline_tools.cross_arch_study import training_runtime_environment
    env.update(training_runtime_environment(model, run_id))
    env.update(PROOF_PIPELINE_CROSS_ARCH_STOP_STEPS="120", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false", PROOF_PIPELINE_PREFLIGHT_COMPLETE="1")
    cmd, env = build_training_cmd(repo_root=ROOT, env=env, output_dir=out, run_cfg=run,
                                  common=profile["training"], language_map_path=campaign_root() / "inputs/language_map.json",
                                  tokenized_path=model_root(model) / "tokenized", embedding_training_cfg={"enabled": False},
                                  run_name=f"{model}/{run_id}/pilot/{label}")
    # Avoid reliance on a login shell PATH for the distributed launcher.
    if cmd[0] == "accelerate":
        cmd[0] = str(PYTHON.parent / "accelerate")
    result = {"pass": False, "model": model, "run_id": run_id, "batch": batch,
              "disable_gradient_checkpointing": disable_gc, "label": label,
              "code_signature": code_signature(), "run_config": run, "command": cmd}
    log = out / "train.log"
    try:
        LocalExecutor().run(cmd, env, False, log)
        telemetry = read_json(out / "campaign_training.json")
        trained = read_json(out / "train_results.json")
        checkpoint = out / f"checkpoint-{telemetry['global_step']}_adapter"
        provenance = adapter_checkpoint_provenance(checkpoint, expected_step=telemetry["global_step"], require_direct_save_marker=True)
        if not provenance["valid"] or not telemetry["signal_checkpoint"] or telemetry["gradient_checks"] < 2:
            raise ValueError("Training, gradient or signal/checkpoint gate failed")
        if telemetry["global_step"] != 120 or len(telemetry["step_seconds_after_warmup"]) != 100:
            raise ValueError("Pilot did not complete the prescribed 20+100 steps")
        if not math.isfinite(trained["train_loss"]):
            raise ValueError("Nonfinite pilot loss")
        free_fraction = min(1 - telemetry["peak_reserved_bytes"] / telemetry["gpu_memory_bytes"],
                            telemetry["minimum_free_bytes"] / telemetry["gpu_memory_bytes"])
        result.update(pass_=free_fraction >= contract()["minimum_free_vram_fraction"],
                      median_step_seconds=statistics.median(telemetry["step_seconds_after_warmup"]),
                      free_vram_fraction=free_fraction, checkpoint=str(checkpoint), telemetry=telemetry)
        result["pass"] = result.pop("pass_")
        if not result["pass"]:
            result["reason"] = "Insufficient VRAM safety margin"
    except subprocess.CalledProcessError as exc:
        text = log.read_text(errors="replace") if log.exists() else ""
        if any(marker in text for marker in ("CUDA out of memory", "OutOfMemoryError", "CUBLAS_STATUS_ALLOC_FAILED", "CUDNN_STATUS_ALLOC_FAILED")):
            result["reason"] = "OOM"
        elif not disable_gc and "CheckpointError" in text:
            result["reason"] = "Unsupported checkpoint recomputation"
        else:
            result["reason"] = f"Training/infrastructure failure: {exc}"
            write_json(receipt, result)
            raise
    write_json(receipt, result)
    print(json.dumps({k: result.get(k) for k in ("model", "run_id", "label", "pass", "reason", "median_step_seconds")}), flush=True)
    return result


def run_pilot(model, run_id):
    cfg = contract()
    out = model_root(model)
    data = read_json(out / "data_ready.json")
    submitted = campaign_root() / "pilot_jobs.json"
    if submitted.exists() and read_json(submitted).get("code_signature") != code_signature():
        raise ValueError("Campaign code changed after pilot submission; inspect before retry")
    if not data["pass"] or not read_json(out / "backend_ready.json")["pass"]:
        raise ValueError("Data/backend readiness gate failed")
    subprocess.run([str(PYTHON), "-m", "pipeline.gpu_preflight", "--required-gpu-count", "4",
                    "--required-gpu-name", "A100", "--check-ecc", "--check-nvlink", "--check-torch-distributed"], check=True, cwd=ROOT)
    original = find_run(load_profile(model), run_id)
    original_gc = original["train"]["disable_gradient_checkpointing"]
    batches = list(dict.fromkeys([original["train"]["batch_size"], 1, 2, 3, 4, 6, 12]))
    results = []
    for disable_gc in (original_gc, not original_gc):
        oom_at = math.inf
        for batch in batches:
            if batch > oom_at:
                continue
            row = candidate(model, run_id, batch, disable_gc, f"b{batch}_gc{int(not disable_gc)}")
            results.append(row)
            if row.get("reason") == "OOM":
                oom_at = min(oom_at, batch)
            if row.get("reason") == "Unsupported checkpoint recomputation":
                break
    valid = [row for row in results if row["pass"]]
    if not valid:
        raise ValueError("No valid four-A100 configuration; production remains held")
    best = min(valid, key=lambda row: row["median_step_seconds"])
    confirmed = candidate(model, run_id, best["batch"], best["disable_gradient_checkpointing"], best["label"] + "_confirm")
    if not confirmed["pass"]:
        raise ValueError("Fastest candidate failed confirmation")
    rates = [1 / row["median_step_seconds"] for row in (best, confirmed)]
    if max(rates) / min(rates) > 1.10:
        raise ValueError("Throughput confirmations differ by over 10%; inspect node health")
    capacity = capacity_check(data["train_sequences"], max(rates), confirmed["telemetry"]["steps_per_epoch"])
    gate_path = out / "gates" / f"{run_id}.json"
    gate = {"pass": False, "model": model, "run_id": run_id, "code_signature": code_signature(),
            "pipeline_config_sha256": sha256(campaign_root() / "pipeline_config.json"),
            "data_sha256": sha256(out / "data_ready.json"), "backend_sha256": sha256(out / "backend_ready.json"),
            "capacity": capacity, "selected_run": confirmed["run_config"],
            "confirmed_steps_per_second": rates, "trainable_parameters": confirmed["telemetry"]["trainable_parameters"],
            "pilot_checkpoint": confirmed["checkpoint"], "checked_utc": utc_now()}
    if not capacity["pass"]:
        gate["reason"] = "Dataset exhaustion risk; no repetition, expansion or slower substitute authorized"
        write_json(gate_path, gate)
        print(json.dumps(gate, indent=2), flush=True)
        return
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    backend = read_json(out / "backend_ready.json")["selected_backend"]
    env["LM_EVAL_ATTN_IMPL"] = {"disabled": "eager", "fa2": "flash_attention_2", "sdpa": "sdpa"}[backend]
    checkpoint = Path(confirmed["checkpoint"])
    subprocess.run([str(PYTHON), "-m", "pipeline_tools.cross_arch_verify", "adapter", "--model", model,
                    "--checkpoint", str(checkpoint)], cwd=ROOT, check=True, env=env)
    panel = (campaign_root() / "inputs/tasks.txt").read_text().splitlines()
    smoke_tasks = smoke_panel(panel)
    smoke = checkpoint.parent / "benchmark_smoke"
    smoke.mkdir(exist_ok=True)
    tasks_file = smoke / "tasks.txt"
    tasks_file.write_text("\n".join(smoke_tasks) + "\n")
    subprocess.run(benchmark_command(model, run_id, checkpoint, tasks_file, smoke, limit=2), cwd=BENCHMARK_CWD, check=True, env=env)
    gate["benchmark_smoke"] = primary_values(smoke, smoke_tasks)
    gate["reload"] = read_json(checkpoint.parent / "reload_check.json")
    gate["pass"] = bool(gate["reload"]["pass"])
    write_json(gate_path, gate)
    print(json.dumps({"model": model, "run_id": run_id, "pass": gate["pass"], "batch": confirmed["batch"], "capacity": capacity}), flush=True)


def smoke_panel(panel):
    tasks = ["belebele_eng_Latn", "sib200_eng_Latn", "taxi1500_v2_eng_Latn",
             next(t for t in panel if t.startswith("flores_en_perplexity_eng_Latn-")),
             next(t for t in panel if t.startswith("flores_en_perplexity_") and t.endswith("-eng_Latn"))]
    if not set(tasks).issubset(panel) or len(set(tasks)) != 5:
        raise ValueError("Smoke panel must include English and both FLORES directions")
    return tasks


def submit_pilots(submit=False, resume=False):
    root = campaign_root()
    receipt = root / "pilot_jobs.json"
    prior = read_json(receipt) if receipt.exists() else {"jobs": []}
    if submit and receipt.exists() and not resume:
        raise ValueError("Pilots already submitted; use --resume only after inspecting the partial submission")
    if submit and prior["jobs"] and prior.get("code_signature") != code_signature():
        for job in prior["jobs"]:
            state = subprocess.check_output(["squeue", "-h", "-j", job["job_id"], "-o", "%T"], text=True).strip()
            if state != "PENDING":
                raise ValueError(f"Cannot refresh pilot code after execution began: {job['job_id']} {state}")
    data_jobs = {j["model"]: j["job_id"] for j in read_json(root / "preparation_jobs.json")["jobs"] if j["stage"] == "data_acceptance"}
    backend_jobs = {j["model"]: j["job_id"] for j in read_json(root / "backend_jobs.json")["jobs"]}
    jobs = prior["jobs"]
    if submit: write_json(receipt, {"status": "submitting", "jobs": jobs, "code_signature": code_signature()})
    for model in contract()["models"]:
        first_job = None
        for run_id in ["lora_r2", *[r for r in contract()["run_ids"] if r != "lora_r2"]]:
            existing = next((j for j in jobs if j["model"] == model and j["run_id"] == run_id), None)
            if existing:
                if run_id == "lora_r2": first_job = existing["job_id"]
                continue
            command = [str(PYTHON), "-m", "pipeline_tools.cross_arch_pilot", "run", "--model", model, "--run-id", run_id]
            job_id = sbatch(command, f"xa_pilot_{model}_{run_id}", gpu=True, hours="08:00:00", submit=submit,
                            dependency=f"afterok:{data_jobs[model]}:{backend_jobs[model]}" + (f":{first_job}" if first_job else ""))
            if run_id == "lora_r2": first_job = job_id
            jobs.append({"model": model, "run_id": run_id, "stage": "pilot", "job_id": job_id})
            if submit:
                write_json(receipt, {"status": "submitting", "jobs": jobs, "code_signature": code_signature()})
    if submit:
        write_json(receipt, {"status": "submitted", "jobs": jobs, "code_signature": code_signature()})
    print(json.dumps(jobs, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "submit"])
    parser.add_argument("--model", choices=list(contract()["models"]))
    parser.add_argument("--run-id", choices=contract()["run_ids"])
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "submit":
        submit_pilots(args.submit, args.resume)
    else:
        if not args.model or not args.run_id: parser.error("--model and --run-id are required")
        try:
            run_pilot(args.model, args.run_id)
        except Exception as exc:
            write_json(model_root(args.model) / "gates" / f"{args.run_id}.json",
                       {"pass": False, "model": args.model, "run_id": args.run_id,
                        "reason": f"{type(exc).__name__}: {exc}", "checked_utc": utc_now()})
            raise


if __name__ == "__main__":
    main()
