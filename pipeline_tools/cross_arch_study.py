"""Gated, isolated 96-language Gemma/Falcon campaign; no implicit production launch.

python -m pipeline_tools.cross_arch_study prepare
python -m pipeline_tools.cross_arch_study submit-preparation --submit
python -m pipeline_tools.cross_arch_study status
"""
from __future__ import annotations

import argparse
import collections
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PROFILE = ROOT / "configs/cross_arch_study/noctua2_runtime.json"


def runtime_profile():
    return json.loads(RUNTIME_PROFILE.read_text())


PYTHON = Path(runtime_profile()["python"])
CONDA_BASE = runtime_profile()["conda_base"]
CONTRACT = ROOT / "configs/cross_arch_study/96_languages.json"
ARCHIVE_PROFILES = [
    "moe_study_final_macro3_96_200_v1.generated.json",
    "moe_study_final_macro45_96_200_v1.generated.json",
    "moe_study_final_mtllora_lr5e5_96_200_v1.generated.json",
    "moe_study_final_lora_r2_r4_96_200_v1.generated.json",
]
PROFILES = {
    "macro3": "moe_study_final_macro3_noctua2_a100_4gpu_96_langs_34p5h_v1",
    "macro45": "moe_study_final_macro45_noctua2_a100_4gpu_96_langs_34p5h_v1",
    "mtllora": "moe_study_final_mtllora_lr5e5_noctua2_a100_4gpu_96_langs_34p5h_v1",
    "lora": "moe_study_final_lora_r2_r4_noctua2_a100_4gpu_96_langs_34p5h_v1",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_weights(checkpoint):
    files = [Path(checkpoint) / name for name in ("adapter_model.safetensors", "adapter_model.bin")]
    existing = [p for p in files if p.is_file() and p.stat().st_size > 0]
    if len(existing) != 1:
        raise ValueError(f"Expected one nonempty adapter weight file: {checkpoint}")
    return existing[0]


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def contract():
    return read_json(CONTRACT)


def training_runtime_environment(model, run_id):
    settings = runtime_profile()
    default = "expandable_segments:True" if run_id.startswith(("moelora_", "mola_")) else ""
    return {
        "PYTORCH_CUDA_ALLOC_CONF": settings["allocator_overrides"].get(f"{model}/{run_id}", default),
        "PROOF_PIPELINE_CUDA_MEMORY_FRACTION": str(settings["memory_fraction_by_run"].get(run_id, "")),
    }


def attempt_logs(root):
    return Path(root) / "logs" / "attempts" / (dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{os.getpid()}")


def campaign_root():
    return ROOT / "outputs/pipeline_runs" / contract()["campaign_id"]


def model_root(model):
    if model not in contract()["models"]:
        raise ValueError(f"Unknown model: {model}")
    return campaign_root() / model


def runtime_pythonpath(model=None):
    paths = [str(ROOT / "configs/cross_arch_study/python_startup"), str(ROOT), str(ROOT / "LLaMA-Factory/src"), str(ROOT / "scripts/eval/latest_checkpoint/lm-evaluation-harness")]
    if model == "gemma2_9b":
        paths.insert(0, str(campaign_root() / "runtime/flash_attn_2_8_3_sm80/site-packages"))
    return os.pathsep.join(paths)


def archived_profiles():
    from pipeline.config_loader import load_pipeline_config, deep_merge
    from pipeline.runtime import _resolve_profile
    bundle = load_pipeline_config(ROOT / "configs/pipeline")
    evidence = {}
    for filename in ARCHIVE_PROFILES:
        path = "configs/pipeline/profiles/moe_study_rank_expert_sweep_configs/" + filename
        raw = subprocess.check_output(["git", "show", f"{contract()['archive_commit']}:{path}"], cwd=ROOT)
        evidence[path] = hashlib.sha256(raw).hexdigest()
        bundle["profiles"] = deep_merge(bundle["profiles"], json.loads(raw))
    return {key: _resolve_profile(bundle, name) for key, name in PROFILES.items()}, evidence


def targets(model, kind):
    attention = ["query_key_value", "dense"] if model == "falcon_7b" else ["q_proj", "k_proj", "v_proj", "o_proj"]
    ffn = ["dense_h_to_4h", "dense_4h_to_h"] if model == "falcon_7b" else ["gate_proj", "up_proj", "down_proj"]
    return attention if kind == "moelora" else attention + ffn


def benchmark_family(task):
    for prefix, name in [("belebele_", "Belebele"), ("flores_", "FLORES"), ("sib200_", "SIB200"), ("taxi1500_", "Taxi1500")]:
        if task.startswith(prefix):
            return name
    raise ValueError(f"Unexpected task: {task}")


def capacity_check(train_sequences, steps_per_second, steps_per_epoch=None):
    cfg = contract()
    if train_sequences <= 0 or not math.isfinite(steps_per_second) or steps_per_second <= 0:
        raise ValueError("Capacity check requires positive finite dataset size and throughput")
    # Replace conservative floor by the observed Trainer horizon once available.
    available = min(train_sequences // cfg["global_batch"], steps_per_epoch or math.inf)
    projected = math.ceil(cfg["trainer_seconds"] * steps_per_second * cfg["capacity_margin"])
    return {"pass": available > projected, "available_steps": available,
            "projected_steps_with_margin": projected, "margin": cfg["capacity_margin"],
            "estimated_epoch_hours": available / steps_per_second / 3600,
            "policy": cfg["exhaustion_policy"]}


def prepare():
    from pipeline.config_loader import deep_merge
    cfg = contract()
    root = campaign_root()
    root.mkdir(parents=True, exist_ok=True)
    old_manifest = root / "manifest.json"
    if old_manifest.exists() and read_json(old_manifest)["contract_sha256"] != sha256(CONTRACT):
        raise ValueError("Existing campaign contract differs; create a new campaign instead")
    original = ROOT / cfg["original_data_root"]
    panel = ROOT / cfg["original_panel"]
    if sha256(panel) != cfg["panel_sha256"]:
        raise ValueError("Original benchmark panel checksum differs")
    tasks = [t for t in panel.read_text().splitlines() if t and not t.startswith("#")]
    if len(tasks) != len(set(tasks)) or dict(collections.Counter(map(benchmark_family, tasks))) != cfg["expected_benchmarks"]:
        raise ValueError("Benchmark panel is incomplete or duplicated")
    # Snapshot only small manifests, never copy or mutate the published datasets.
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    snapshots = {"tasks.txt": panel, "language_map.json": original / "language_map_from_csv.json",
                 "language_rates.json": original / "tokenization_language_rates.json"}
    snapshots.update({"plan/" + p.name: p for p in (original / "tokenized_shard_plan").glob("*") if p.is_file()})
    hashes = {}
    for name, source in snapshots.items():
        dest = inputs / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha256(dest) != sha256(source):
            raise ValueError(f"Frozen input changed: {name}")
        if not dest.exists():
            dest.write_bytes(source.read_bytes())
        hashes[name] = sha256(dest)
    if len(list((inputs / "plan").glob("rank_*.jsonl"))) != cfg["tokenization_ranks"]:
        raise ValueError("Incomplete original tokenization plan")
    originals, profile_hashes = archived_profiles()
    selected = {}
    for profile in originals.values():
        common = profile["training"]
        for run in common["runs"]:
            if run["id"] in cfg["run_ids"]:
                effective = copy.deepcopy(run)
                for key in ("train", "lora", "params", "distributed", "eval"):
                    effective[key] = deep_merge(common.get(key, {}), run.get(key, {}))
                if run["id"] in selected and selected[run["id"]] != effective:
                    raise ValueError(f"Ambiguous original run: {run['id']}")
                selected[run["id"]] = effective
    if set(selected) != set(cfg["run_ids"]):
        raise ValueError(f"Missing original configurations: {set(cfg['run_ids']) - set(selected)}")
    profiles = {}
    for model, spec in cfg["models"].items():
        base = copy.deepcopy(originals["macro3"])
        out = model_root(model)
        out.mkdir(exist_ok=True)
        base.pop("extends", None)
        base.update(output_root=str(out), logs_dir=str(out / "logs"), language_map_path=str(inputs / "language_map.json"))
        base.pop("language_map_csv", None)
        base["run_naming"] = {"experiment_tag": cfg["campaign_id"] + "-" + model, "language_count_hint": 96}
        base["study_contract"] = {**cfg, "model": spec, "source_profile_sha256": profile_hashes}
        for key in ("sample", "tokenizer_extension", "embedding_init", "embedding_training", "embedding_verify", "audit", "wandb", "baseline_eval"):
            base[key] = {"enabled": False}
        base["wandb_mode"] = "disabled"
        base["tokenization"] = {"enabled": False, "output_dir": str(out / "tokenized"),
                                "model_name": str(out / "model"), "max_length": cfg["sequence_length"]}
        train = base["training"]
        # Local immutable snapshot is filled by assets(); no moving 'main' revision.
        train.update(model_name_or_path=str(out / "model"), output_dir=str(out / "runs"),
                     template="empty", resize_vocab=False, report_to="none", launcher_module="pipeline_tools.cross_arch_worker")
        train["tokenized_local_copy"].update(enabled=True, job_scoped=True, cleanup=True, verify_files=["state.json", "dataset_info.json", "_check.json"])
        runs = []
        for run_id in cfg["run_ids"]:
            run = copy.deepcopy(selected[run_id])
            run["lora"]["target"] = ",".join(targets(model, run["kind"]))
            run["train"].update(flash_attn=spec["attention_backend"], max_steps=None, num_train_epochs=1.0)
            run["train"].update(optim="adamw_torch", lr_scheduler_type="linear", warmup_ratio=0.0,
                                warmup_steps=0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, max_grad_norm=1.0)
            if 4 * run["train"]["batch_size"] * run["train"]["grad_accum"] != cfg["global_batch"]:
                raise ValueError(f"Unexpected original effective batch: {run_id}")
            runs.append(run)
        train["runs"] = runs
        runtime = runtime_profile()
        excluded = "--exclude=" + ",".join(runtime["exclude_gpu_nodes"])
        train_exec = {"type": "slurm", "partition": runtime["gpu_partition"], "gres": f"gpu:{runtime['gpu_type']}:4", **runtime["resources"]["train"], "time": cfg["slurm_time"],
                      "conda_base": CONDA_BASE, "conda_env": str(PYTHON.parent.parent),
                      "extra_sbatch_flags": ["--nodes=1", "--signal=B:USR1@1200", excluded,
                                             "--export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]}
        eval_exec = {"type": "slurm", "partition": runtime["gpu_partition"], "gres": f"gpu:{runtime['gpu_type']}:1", **runtime["resources"]["eval"], "time": "08:00:00", "extra_sbatch_flags": ["--nodes=1", excluded]}
        eval_exec.update(conda_base=CONDA_BASE, conda_env=str(PYTHON.parent.parent))
        cpu_exec = {"type": "slurm", "partition": runtime["cpu_partition"], "gres": "cpu", **runtime["resources"]["cpu"],
                    "time": "01:00:00", "conda_base": CONDA_BASE,
                    "conda_env": str(PYTHON.parent.parent), "extra_sbatch_flags": ["--nodes=1", "--kill-on-invalid-dep=yes"]}
        base["executor"] = cpu_exec
        base["stage_executors"] = {stage: copy.deepcopy(cpu_exec) for stage in ("train_acceptance", "summary", "report")}
        train["executor"] = train_exec
        base["stage_executors"]["train"] = train_exec
        for stage in ("final_eval_loss", "benchmark"):
            base["stage_executors"][stage] = copy.deepcopy(eval_exec)
            base[stage].pop("executor", None)
        base["train_acceptance"].update(output_dir=str(out / "acceptance"), target_runtime_seconds=cfg["trainer_seconds"], runtime_tolerance_seconds=cfg["runtime_tolerance_seconds"])
        base["final_eval_loss"].update(output_dir=str(out / "final_eval_loss"), batch_size=1)
        base["benchmark"].update(tasks_file=str(inputs / "tasks.txt"), output_dir=str(out / "benchmarks"),
                                 tokenizer=str(out / "model"), lang_mode="no_ids", batch_size=1,
                                 expected_task_count=len(tasks), expected_task_breakdown=cfg["expected_benchmarks"], strict_completeness=True)
        # Match the submitted final benchmark protocol, including no language IDs.
        # The new baseline must never fall back to the original Llama CSVs.
        base["report"].update(baseline_model_name_or_path=str(out / "model"), baseline_metrics_csvs=[str(out / "baseline_metrics.csv")])
        base["summary"].update(summary_csv=str(out / "summary.csv"), gate_json=str(out / "gate.json"))
        base["report"].update(output_xlsx=str(out / f"moe_study_final_{model}_96_languages.xlsx"), baseline_cache_root=str(out / "baseline_cache"))
        profiles[model] = base
    config_path = root / "pipeline_config.json"
    generated = {"schema_version": 1, "defaults": {}, "profiles": profiles}
    if not config_path.exists() or read_json(config_path) != generated:
        if (root / "production_jobs.json").exists() or (root / "pilot_jobs.json").exists():
            raise ValueError("Training jobs exist; do not overwrite their frozen configuration")
        write_json(config_path, generated)
    manifest = {"created_utc": utc_now(), "contract_sha256": sha256(CONTRACT), "inputs_sha256": hashes,
                "source_profile_sha256": profile_hashes, "original_runs": selected, "status": "prepared_not_launched"}
    if not old_manifest.exists():
        write_json(old_manifest, manifest)
    print(json.dumps({"campaign": str(root), "models": list(profiles), "trainings": 14, "panel_units": len(tasks), "production": "blocked_until_gates_pass"}, indent=2))


def assets(model):
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    spec = contract()["models"][model]
    out = model_root(model)
    path = Path(snapshot_download(spec["repo_id"], revision=spec["revision"], max_workers=4,
                                 allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "*.txt"]))
    config = AutoConfig.from_pretrained(path, trust_remote_code=False)
    if config.model_type != spec["model_type"]:
        raise ValueError("Wrong model architecture")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    if not list(path.glob("*.safetensors")):
        raise ValueError("No model weights downloaded")
    link = out / "model"
    if link.exists() and link.resolve() != path.resolve():
        raise ValueError("Model snapshot changed")
    if not link.exists():
        link.symlink_to(path, target_is_directory=True)
    record = {"pass": True, **spec, "snapshot": str(path), "tokenizer_size": len(tokenizer),
              "model_vocab_size": config.vocab_size, "bos_token_id": tokenizer.bos_token_id,
              "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
              "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in path.iterdir() if p.is_file()},
              "verified_utc": utc_now()}
    write_json(out / "assets.json", record)
    print(json.dumps({k: record[k] for k in ("pass", "repo_id", "revision", "snapshot", "tokenizer_size")}), flush=True)


def audit_rank(model, rank):
    import numpy as np
    from datasets import load_from_disk
    sys.path.insert(0, str(ROOT / "data_prep/merlin_data_prep/distributed_data_processor/src"))
    from tokenize_slurm import load_language_map, build_language_vocab
    cfg = contract()
    out = model_root(model)
    actual = load_from_disk(str(out / "tokenized_ranks" / f"rank_{rank:05d}"))
    original = load_from_disk(str(ROOT / cfg["original_data_root"] / "tokenized_ranks" / f"rank_{rank:05d}"))
    keys = ["doc_id", "chunk_idx", "language", "split", "language_ids", "family_ids"]
    def documents(dataset):
        docs = {}
        for batch in dataset.select_columns(keys).iter(batch_size=4096):
            for doc, chunk, language, split, language_id, family_id in zip(*(batch[k] for k in keys)):
                if chunk == 0:
                    if doc in docs:
                        raise ValueError(f"Duplicate document in rank {rank}: {doc}")
                    docs[doc] = (language, split, language_id, family_id)
        return docs
    expected, observed = documents(original), documents(actual)
    if expected != observed:
        raise ValueError(f"Document/split/routing mismatch in rank {rank}; original={len(expected)}, new={len(observed)}, missing={len(expected.keys()-observed.keys())}, extra={len(observed.keys()-expected.keys())}")
    language_map = load_language_map(str(campaign_root() / "inputs/language_map.json"))
    language_vocab, family_vocab = build_language_vocab(language_map)
    counts, token_counts = collections.Counter(), collections.Counter()
    vocab_size = read_json(out / "assets.json")["tokenizer_size"]
    for batch in actual.iter(batch_size=256):
        ids, labels, mask = (np.asarray(batch[k]) for k in ("input_ids", "labels", "attention_mask"))
        if ids.ndim != 2 or ids.shape[1] != cfg["sequence_length"]:
            raise ValueError("Incorrect sequence length")
        if np.any(ids < 0) or np.any(ids >= vocab_size) or not np.isin(mask, [0, 1]).all():
            raise ValueError("Invalid token IDs or attention mask")
        if not np.array_equal(labels, np.where(mask == 0, -100, ids)):
            raise ValueError("Padding label mask differs from the original objective")
        for i, (lang, split) in enumerate(zip(batch["language"], batch["split"])):
            if batch["language_ids"][i] != language_vocab[lang] or batch["family_ids"][i] != family_vocab[language_map[lang]]:
                raise ValueError("Routing IDs differ from original language map")
            counts[(split, lang)] += 1
            token_counts[(split, lang)] += int(mask[i].sum())
    docs = collections.Counter((row[1], row[0]) for row in observed.values())
    def nested(counter):
        return {split: {lang: n for (s, lang), n in counter.items() if s == split} for split in ("train", "validation")}
    result = {"pass": True, "rank": rank, "revision": cfg["models"][model]["revision"], "document_assignment_equal": True,
              "document_sha256": hashlib.sha256(json.dumps(sorted(observed.items())).encode()).hexdigest(),
              "sequences": nested(counts), "documents": nested(docs), "effective_tokens": nested(token_counts)}
    write_json(out / "data_checks" / f"rank_{rank:05d}.json", result)
    return result


def tokenize(model, rank):
    cfg = contract()
    out = model_root(model)
    if not read_json(out / "assets.json")["pass"]:
        raise ValueError("Model assets not validated")
    receipt = out / "data_checks" / f"rank_{rank:05d}.json"
    if receipt.exists() and read_json(receipt)["pass"]:
        print(f"Already verified {model} rank {rank}")
        return
    target = out / "tokenized_ranks" / f"rank_{rank:05d}"
    if not target.exists():
        command = [str(PYTHON), str(ROOT / "data_prep/merlin_data_prep/distributed_data_processor/src/tokenize_slurm.py"),
                   "--shard_dir", str(ROOT / cfg["raw_shards"]), "--save_tokenized_data_dir", str(out / "tokenized_ranks"),
                   "--model_name", str(out / "model"), "--num_proc", "4", "--rank", str(rank), "--world_size", str(cfg["tokenization_ranks"]),
                   "--eval_seed", str(cfg["validation_seed"]), "--sample_seed", str(cfg["sampling_seed"]),
                   "--max_length", str(cfg["sequence_length"]), "--language-map", str(campaign_root() / "inputs/language_map.json"),
                   "--mask_pad_labels", "true", "--shard-plan-dir", str(campaign_root() / "inputs/plan")]
        subprocess.run(command, check=True, cwd=ROOT)
    result = audit_rank(model, rank)
    print(json.dumps({"model": model, "rank": rank, "pass": result["pass"]}), flush=True)


def merge_data(model):
    from data_prep.merlin_data_prep.distributed_data_processor.src.merge_tokenized_ranks import fast_merge
    sys.path.insert(0, str(ROOT / "data_prep/merlin_data_prep/distributed_data_processor/src"))
    from tokenize_slurm import load_language_map
    cfg, out = contract(), model_root(model)
    parts = [read_json(out / "data_checks" / f"rank_{rank:05d}.json") for rank in range(cfg["tokenization_ranks"])]
    if not all(p["pass"] and p["document_assignment_equal"] and p["revision"] == cfg["models"][model]["revision"] for p in parts):
        raise ValueError("Incomplete or mismatched data checks")
    summary = {"pass": True, "stage": "tokenize", "model": model, "ranks": len(parts), "document_assignment_equal": True}
    for field in ("sequences", "documents", "effective_tokens"):
        summary[field] = {}
        for split in ("train", "validation"):
            totals = collections.Counter()
            for p in parts:
                totals.update(p[field][split])
            summary[field][split] = dict(totals)
    expected = set(load_language_map(str(campaign_root() / "inputs/language_map.json")))
    if set(summary["documents"]["train"]) != expected:
        raise ValueError("Missing training languages")
    summary["train_sequences"] = sum(summary["sequences"]["train"].values())
    summary["validation_sequences"] = sum(summary["sequences"]["validation"].values())
    summary["docs_per_language_by_split"] = summary["documents"]
    summary["samples_per_language_by_split"] = summary["sequences"]
    summary["effective_tokens_per_language_by_split"] = summary["effective_tokens"]
    summary["tokens_per_language_by_split"] = {
        split: {lang: count * cfg["sequence_length"] for lang, count in counts.items()}
        for split, counts in summary["sequences"].items()
    }
    for name, source in [("samples_per_language", "sequences"), ("tokens_per_language", "tokens_per_language_by_split")]:
        summary[name] = dict(collections.Counter(summary[source]["train"]) + collections.Counter(summary[source]["validation"]))
    summary.update(num_languages=len(expected), num_rows=summary["train_sequences"] + summary["validation_sequences"],
                   total_tokens=sum(summary["tokens_per_language"].values()),
                   splits={"train": summary["train_sequences"], "validation": summary["validation_sequences"]})
    if not (out / "tokenized").exists():
        fast_merge(out / "tokenized_ranks", out / "tokenized", overwrite=False)
    from datasets import load_from_disk
    if len(load_from_disk(str(out / "tokenized"))) != summary["num_rows"]:
        raise ValueError("Merged dataset length differs from audited shard counts")
    plan = out / "tokenized_shard_plan"
    if not plan.exists(): plan.symlink_to(campaign_root() / "inputs/plan", target_is_directory=True)
    write_json(out / "tokenized/_check.json", summary)
    write_json(out / "data_ready.json", summary)
    print(json.dumps({k: summary[k] for k in ("pass", "train_sequences", "validation_sequences")}), flush=True)


def live_dependency(dependency):
    """Slurm forgets short completed jobs; verify accounting before removing them."""
    if not dependency: return None
    mode, *job_ids = dependency.split(":")
    if mode not in {"afterok", "afterany"}: raise ValueError(f"Unsupported dependency: {dependency}")
    active = []
    for job_id in job_ids:
        query = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%T"], capture_output=True, text=True, timeout=30)
        if query.stdout.strip():
            active.append(job_id)
            continue
        accounting = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", job_id, "-o", "JobID,State,ExitCode"], text=True, timeout=30)
        rows = [line.split("|") for line in accounting.splitlines() if line.strip()]
        rows = [row for row in rows if row[0] == job_id or row[0].startswith(job_id + "_")]
        if not rows: raise ValueError(f"Cannot verify dependency job {job_id}")
        terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
        if any(row[1].split()[0] not in terminal for row in rows):
            active.append(job_id)
        elif mode == "afterok" and any(row[1] != "COMPLETED" or row[2] != "0:0" for row in rows):
            raise ValueError(f"Dependency {job_id} did not complete successfully; inspect before retry")
    return mode + ":" + ":".join(active) if active else None


def sbatch(command, label, *, gpu=False, gpu_count=4, dependency=None, array=None, submit=False, hours="02:00:00", extra_env=None):
    from pipeline_tools.cross_arch_node import CrossArchSlurmExecutor, excluded_nodes
    root = campaign_root()
    flags = ["--nodes=1", "--ntasks-per-node=1", "--kill-on-invalid-dep=yes"]
    if submit: dependency = live_dependency(dependency)
    if gpu:
        flags.append("--exclude=" + ",".join(excluded_nodes()))
    runtime = runtime_profile()
    resources = runtime["resources"]["train" if gpu and gpu_count == 4 else "eval" if gpu else "cpu"]
    executor = CrossArchSlurmExecutor(partition=runtime["gpu_partition" if gpu else "cpu_partition"], gres=f"gpu:{runtime['gpu_type']}:{gpu_count}" if gpu else "",
                             **resources, time_limit=hours, modules=runtime["modules"],
                             conda_base=CONDA_BASE, conda_env=str(PYTHON.parent.parent),
                             extra_sbatch_flags=flags)
    env = dict(os.environ)
    model = command[command.index("--model") + 1] if "--model" in command else None
    env.update(PYTHONPATH=runtime_pythonpath(model),
               PROOF_PIPELINE_REPO_ROOT=str(ROOT), PROOF_PIPELINE_PRESERVE_ENV="1", WANDB_MODE="disabled", HF_HOME=runtime["hf_home"],
               HF_DATASETS_CACHE=str(Path(runtime["hf_home"]) / "datasets"))
    env.update(extra_env or {})
    if model:
        env["PROOF_PIPELINE_CROSS_ARCH_MODEL"] = model
    # Preserve the exact interpreter in the receipt environment as well.
    env["PROOF_PIPELINE_PYTHON"] = str(PYTHON)
    log = attempt_logs(root) / f"{label}.log"
    if array:
        return executor.submit_array(command, env, not submit, log, array=array, dependency=dependency, job_name=label,
                                     slurm_output=str(log.parent / (label + ".%A_%a.log")))
    return executor.submit(command, env, not submit, log, dependency=dependency, job_name=label)


def submit_preparation(submit=False):
    root = campaign_root()
    receipt = root / "preparation_jobs.json"
    if submit and receipt.exists():
        raise ValueError("Preparation already submitted; inspect status before retrying individual failures")
    jobs = []
    for model in contract()["models"]:
        cmd = [str(PYTHON), "-m", "pipeline_tools.cross_arch_study"]
        download = sbatch(cmd + ["assets", "--model", model], "xa_assets_" + model, submit=submit)
        jobs.append({"model": model, "stage": "assets", "job_id": download})
        if submit:
            write_json(receipt, {"status": "submitting", "jobs": jobs})
        token = sbatch(cmd + ["tokenize", "--model", model], "xa_token_" + model, hours="08:00:00",
                       dependency=f"afterok:{download}" if submit else None, array="0-49%10", submit=submit)
        jobs.append({"model": model, "stage": "tokenize", "job_id": token})
        if submit:
            write_json(receipt, {"status": "submitting", "jobs": jobs})
        merge = sbatch(cmd + ["merge-data", "--model", model], "xa_data_" + model,
                       dependency=f"afterok:{token}" if submit else None, submit=submit)
        jobs.append({"model": model, "stage": "data_acceptance", "job_id": merge})
        if submit:
            write_json(receipt, {"status": "submitting", "jobs": jobs})
    if submit:
        write_json(receipt, {"status": "submitted", "created_utc": utc_now(), "jobs": jobs})
    print(json.dumps(jobs, indent=2))


def status():
    root = campaign_root()
    def passed(path, key="pass"):
        return bool(path.exists() and read_json(path).get(key))
    result = {"campaign": str(root), "models": {}, "job_manifests": [str(p) for p in root.glob("*_jobs.json")]}
    for model in contract()["models"]:
        out = model_root(model)
        gates = {p.stem: read_json(p) for p in (out / "gates").glob("*.json")}
        result["models"][model] = {
            "assets": passed(out / "assets.json"), "backend_gpu_gate": passed(out / "backend_ready.json"),
            "verified_tokenization_ranks": sum(passed(p) for p in (out / "data_checks").glob("rank_*.json")),
            "data_ready": passed(out / "data_ready.json"),
            "pilots_passed": [rid for rid, gate in gates.items() if gate.get("pass")],
            "held_pilots": {rid: gate.get("reason", "check gate file") for rid, gate in gates.items() if not gate.get("pass")},
            "training_accepted": [rid for rid in contract()["run_ids"] if passed(out / "acceptance" / rid / "acceptance.json", "accepted")],
            "benchmarks_complete": [rid for rid in contract()["run_ids"] if passed(out / "benchmarks" / rid / "complete.json")],
            "baseline_complete": passed(out / "baseline/complete.json"), "workbook_ready": passed(out / "report_ready.json"),
        }
    print(json.dumps(result, indent=2))


def repair_validation(submit=False):
    """Audit intact shards after the old import-error jobs; preserve their history."""
    path = campaign_root() / "preparation_jobs.json"
    record = read_json(path)
    if record.get("validation_repair"):
        raise ValueError("Validation repair already recorded; inspect it before resubmitting")
    record["validation_repair"] = {"reason": "Fix validation-hook import; reuse existing tokenized data", "utc": utc_now()}
    record.setdefault("history", [])
    for model in contract()["models"]:
        token = next(j["job_id"] for j in record["jobs"] if j["model"] == model and j["stage"] == "tokenize")
        merge = next(j for j in record["jobs"] if j["model"] == model and j["stage"] == "data_acceptance")
        if submit:
            state = subprocess.check_output(["squeue", "-h", "-j", str(merge["job_id"]), "-o", "%T"], text=True).strip()
            if state and state != "PENDING":
                raise ValueError(f"Do not replace active merge job {merge}: {state}")
            if state:
                subprocess.run(["scancel", str(merge["job_id"])], check=True)
        cmd = [str(PYTHON), "-m", "pipeline_tools.cross_arch_study"]
        retry = sbatch(cmd + ["tokenize", "--model", model], "xa_audit_retry_" + model,
                       dependency=f"afterany:{token}", array=f"0-{contract()['tokenization_ranks']-1}%10",
                       submit=submit, hours="08:00:00")
        record["history"].append(dict(merge, superseded=True))
        record["jobs"].append({"model": model, "stage": "validation_retry", "job_id": retry})
        if submit:
            write_json(path, record)
        merge["job_id"] = sbatch(cmd + ["merge-data", "--model", model], "xa_data_repaired_" + model,
                                  dependency=f"afterok:{retry}" if submit else None, submit=submit)
        if submit:
            write_json(path, record)
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "assets", "tokenize", "merge-data", "submit-preparation", "repair-validation", "status"])
    parser.add_argument("--model", choices=list(contract()["models"]))
    parser.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.action in {"assets", "tokenize", "merge-data"} and not args.model:
        parser.error("--model is required")
    if args.action == "prepare": prepare()
    elif args.action == "assets": assets(args.model)
    elif args.action == "tokenize": tokenize(args.model, args.rank)
    elif args.action == "merge-data": merge_data(args.model)
    elif args.action == "submit-preparation": submit_preparation(args.submit)
    elif args.action == "repair-validation": repair_validation(args.submit)
    else: status()


if __name__ == "__main__":
    main()
