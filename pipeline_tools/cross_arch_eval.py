"""Complete-panel, checkpoint-verified benchmark shards and final validation loss."""
from __future__ import annotations

import argparse
import fcntl
import math
import os
from pathlib import Path
import subprocess

from pipeline_tools.cross_arch_study import (
    ROOT, PYTHON, campaign_root, model_root, contract, read_json, write_json,
    sha256, sbatch, benchmark_family, utc_now, checkpoint_weights,
)
from pipeline_tools.cross_arch_pilot import benchmark_command, primary_values, load_profile, find_run, BENCHMARK_CWD


def task_shards(panel):
    chunks = []
    sizes = {"Belebele": 10, "FLORES": 20, "SIB200": 17, "Taxi1500": 20}
    for family, size in sizes.items():
        tasks = [t for t in panel if benchmark_family(t) == family]
        chunks.extend(tasks[i:i+size] for i in range(0, len(tasks), size))
    flattened = [task for chunk in chunks for task in chunk]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(panel):
        raise ValueError("Benchmark shards duplicate or lose tasks")
    return chunks


def accepted_checkpoint(model, run_id):
    from pipeline.training import resolve_adapter_eval_path
    out = model_root(model)
    if not read_json(out / "acceptance" / run_id / "acceptance.json")["accepted"]:
        raise ValueError("Training was not accepted")
    state = read_json(out / "runs" / run_id / "trainer_state.json")
    return resolve_adapter_eval_path(out / "runs" / run_id, expected_step=state["global_step"],
                                     strict_final_checkpoint=True, require_direct_save_marker=True)


def eval_environment(model):
    out = model_root(model)
    backend = read_json(out / "backend_ready.json")
    if not backend["pass"] or backend["revision"] != contract()["models"][model]["revision"]:
        raise ValueError("Model/backend gate is absent or stale")
    env = dict(os.environ)
    env["LM_EVAL_ATTN_IMPL"] = {"disabled": "eager", "fa2": "flash_attention_2", "sdpa": "sdpa"}[backend["selected_backend"]]
    subprocess.run([str(PYTHON), "-m", "pipeline.gpu_preflight", "--required-gpu-count", "1",
                    "--required-gpu-name", "A100", "--check-ecc"], cwd=ROOT, check=True)
    return env


def baseline_benchmark(model, tasks, destination, env):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    # No PEFT adapter or other model's cached baseline is used here.
    from pipeline_tools.cross_arch_model_cache import local_model_path
    native = str(local_model_path(model))
    hf = HFLM(pretrained=native, tokenizer=native, dtype="bfloat16", device="cuda", batch_size=1,
              attn_implementation=env["LM_EVAL_ATTN_IMPL"], trust_remote_code=False)
    # FLORES task YAMLs resolve their checked-in JSON paths from the harness.
    previous_cwd = Path.cwd()
    try:
        os.chdir(BENCHMARK_CWD)
        results = evaluator.simple_evaluate(model=hf, tasks=tasks, random_seed=42, numpy_random_seed=42,
            torch_random_seed=42, fewshot_random_seed=42, log_samples=False)
    finally:
        os.chdir(previous_cwd)
    # Harness results contain serializable scalar metrics plus configuration objects.
    import json
    write_json(destination / "no_language_ids.json", json.loads(json.dumps(results, default=str)))


def benchmark_shard(model, run_id, index):
    panel = (campaign_root() / "inputs/tasks.txt").read_text().splitlines()
    shards = task_shards(panel)
    if not 0 <= index < len(shards):
        raise ValueError("Invalid shard index")
    tasks = shards[index]
    out = model_root(model)
    destination = out / "benchmark_shards" / run_id / f"shard_{index:02d}"
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / "complete.json"
    if receipt.exists():
        prior = read_json(receipt)
        if prior["pass"] and prior["tasks"] == tasks and prior["results_sha256"] == sha256(destination / "no_language_ids.json"):
            primary_values(destination, tasks)
            print(f"Already complete: {model}/{run_id}/{index}")
            return
        raise ValueError("Stale or invalid benchmark receipt")
    env = eval_environment(model)
    checkpoint = None if run_id == "baseline" else accepted_checkpoint(model, run_id)
    if checkpoint is None:
        baseline_benchmark(model, tasks, destination, env)
    else:
        tasks_file = destination / "tasks.txt"
        tasks_file.write_text("\n".join(tasks) + "\n")
        subprocess.run(benchmark_command(model, run_id, checkpoint, tasks_file, destination), env=env, cwd=BENCHMARK_CWD, check=True)
    values = primary_values(destination, tasks)
    write_json(receipt, {"pass": True, "model": model, "run_id": run_id, "tasks": tasks,
        "revision": contract()["models"][model]["revision"], "checkpoint": str(checkpoint) if checkpoint else None,
        "adapter_sha256": sha256(checkpoint_weights(checkpoint)) if checkpoint else None,
        "results_sha256": sha256(destination / "no_language_ids.json"), "n_primary": len(values), "utc": utc_now()})


def final_loss(model, run_id):
    from pipeline.training import build_final_eval_cmd
    from pipeline.executor import LocalExecutor
    from pipeline_tools.cross_arch_launch import production_run
    from pipeline_tools.cross_arch_model_cache import local_model_path
    checkpoint, out = accepted_checkpoint(model, run_id), model_root(model)
    env = eval_environment(model)
    destination = out / "runs" / run_id / "final_eval_loss"
    cmd, env = build_final_eval_cmd(repo_root=ROOT, env=env, output_dir=destination,
        run_cfg=production_run(model, run_id), common=load_profile(model)["training"],
        language_map_path=campaign_root() / "inputs/language_map.json", tokenized_path=out / "tokenized",
        embedding_training_cfg={"enabled": False}, run_name=f"{model}/{run_id}", model_name_or_path=str(local_model_path(model)),
        adapter_name_or_path=str(checkpoint), per_device_eval_batch_size=1)
    cmd[0] = str(PYTHON)
    from pipeline_tools.cross_arch_study import attempt_logs
    LocalExecutor().run(cmd, env, False, attempt_logs(out) / f"final_loss_{run_id}.log")
    value = read_json(destination / "eval_results.json")["eval_loss"]
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Missing or nonfinite final validation loss")
    write_json(destination / "complete.json", {"pass": True, "eval_loss": value, "checkpoint": str(checkpoint),
        "adapter_sha256": sha256(checkpoint_weights(checkpoint)),
        "results_sha256": sha256(destination / "eval_results.json")})


def collect(model, run_id):
    out = model_root(model)
    panel = (campaign_root() / "inputs/tasks.txt").read_text().splitlines()
    checkpoint = None if run_id == "baseline" else accepted_checkpoint(model, run_id)
    expected_hash = sha256(checkpoint_weights(checkpoint)) if checkpoint else None
    results, evidence = {}, []
    for index, tasks in enumerate(task_shards(panel)):
        folder = out / "benchmark_shards" / run_id / f"shard_{index:02d}"
        receipt = read_json(folder / "complete.json")
        if (not receipt["pass"] or receipt["model"] != model or receipt["run_id"] != run_id
            or receipt["revision"] != contract()["models"][model]["revision"] or receipt["tasks"] != tasks
            or receipt["adapter_sha256"] != expected_hash or receipt["results_sha256"] != sha256(folder / "no_language_ids.json")):
            raise ValueError(f"Invalid benchmark provenance: {folder}")
        primary_values(folder, tasks)
        payload = read_json(folder / "no_language_ids.json")
        if set(payload["results"]) != set(tasks) or set(results).intersection(payload["results"]):
            raise ValueError("Duplicate or unexpected benchmark results")
        results.update(payload["results"])
        evidence.append(receipt)
    target = out / ("baseline" if run_id == "baseline" else "benchmarks/" + run_id)
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "no_language_ids.json", {"results": results, "campaign_sources": evidence})
    primary_values(target, panel)
    if run_id != "baseline":
        loss_dir = out / "runs" / run_id / "final_eval_loss"
        loss = read_json(loss_dir / "complete.json")
        if not loss["pass"] or loss["adapter_sha256"] != expected_hash or loss["results_sha256"] != sha256(loss_dir / "eval_results.json"):
            raise ValueError("Missing or mismatched final validation loss")
    write_json(target / "complete.json", {"pass": True, "run_id": run_id, "model": model,
        "tasks": panel, "adapter_sha256": expected_hash, "results_sha256": sha256(target / "no_language_ids.json")})


def submit_evaluation(model, run_id, dependency, submit=False):
    root = campaign_root()
    with (root / ".evaluation_submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "evaluation_jobs.json"
        record = read_json(path) if path.exists() else {"jobs": []}
        if any(j["model"] == model and j["run_id"] == run_id for j in record["jobs"]):
            raise ValueError(f"Evaluation already recorded: {model}/{run_id}; no duplicate submission")
        cmd = [str(PYTHON), "-m", "pipeline_tools.cross_arch_eval"]
        suffix = ["--model", model, "--run-id", run_id]
        job = {"model": model, "run_id": run_id, "status": "submitting"}
        count = len(task_shards((root / "inputs/tasks.txt").read_text().splitlines()))
        job["benchmark"] = sbatch(cmd + ["benchmark", *suffix], f"xa_bench_{model}_{run_id}", gpu=True, gpu_count=1,
            array=f"0-{count-1}%4", dependency=dependency, hours="08:00:00", submit=submit)
        if submit:
            record["jobs"].append(job)
            write_json(path, record)
        dependencies = [job["benchmark"]]
        if run_id != "baseline":
            job["final_loss"] = sbatch(cmd + ["loss", *suffix], f"xa_loss_{model}_{run_id}", gpu=True, gpu_count=1,
                dependency=dependency, hours="12:00:00", submit=submit)
            dependencies.append(job["final_loss"])
            if submit: write_json(path, record)
        job["collect"] = sbatch(cmd + ["collect", *suffix], f"xa_collect_{model}_{run_id}", submit=submit,
            dependency="afterok:" + ":".join(dependencies) if submit else None, hours="00:30:00")
        job["status"] = "submitted"
        if submit: write_json(path, record)
        return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["benchmark", "loss", "collect", "submit"])
    parser.add_argument("--model", required=True, choices=list(contract()["models"]))
    parser.add_argument("--run-id", required=True, choices=["baseline", *contract()["run_ids"]])
    parser.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--dependency")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.action == "benchmark": benchmark_shard(args.model, args.run_id, args.index)
    elif args.action == "loss": final_loss(args.model, args.run_id)
    elif args.action == "collect": collect(args.model, args.run_id)
    else:
        if args.submit and not args.dependency: parser.error("A successful acceptance dependency is required")
        submit_evaluation(args.model, args.run_id, args.dependency, args.submit)


if __name__ == "__main__":
    main()
