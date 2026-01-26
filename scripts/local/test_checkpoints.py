import json
import os
import sys
import inspect
import torch
import torch.distributed.checkpoint as dist_cp
import torch.distributed as dist

WATCH_ROOTS = [
    "/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs/llama-3.2-3B_tokenizer-200_langs/mola/paper-best_20260124_223511"
]

APPROACHES = ("adamole", "mola", "cola", "hydralora")

# Trainable modules by approach:
# - CoLA: lora_A, lora_B (plus router if experts enabled)
# - HydraLoRA: lora_A, lora_B, lora_route (plus router if experts enabled)
# - AdaMoLE: lora_A, lora_B, lora_gating, lora_threshold
# - MoLA: lora_A, lora_B, lora_gating
APPROACH_PATTERNS = {
    "cola": ["lora_A", "lora_B", "router"],
    "hydralora": ["lora_A", "lora_B", "lora_route", "router"],
    "adamole": ["lora_A", "lora_B", "lora_gating", "lora_threshold"],
    "mola": ["lora_A", "lora_B", "lora_gating"],
}

CONFIG_KEYS = [
    "peft_type",
    "task_type",
    "r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "bias",
    "fan_in_fan_out",
    "modules_to_save",
    "use_rslora",
    "rank_pattern",
    "alpha_pattern",
    "lora_type",
    "lora_expert_num",
    "language_list",
    "language_router_mode",
    "language_guidance_scope",
    "language_prior_weight",
    "cola_num_experts",
    "cola_num_a",
    "cola_num_b",
    "cola_strategy",
    "hydralora_num_experts",
    "lora_num",
]


def load_config(config_path):
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_config(config):
    summary = {}
    for key in CONFIG_KEYS:
        if key in config:
            summary[key] = config[key]
    return summary


def detect_approach(root):
    lowered = root.lower()
    for approach in APPROACHES:
        if f"/{approach}/" in lowered:
            return approach
    for approach in APPROACHES:
        if approach in lowered:
            return approach
    return None


def build_runs():
    runs = {}
    for root in WATCH_ROOTS:
        approach = detect_approach(root)
        if approach is None:
            continue
        if approach not in APPROACHES:
            continue
        run_name = root
        runs[run_name] = {"root": root, "approach": approach}
    if not runs:
        print("[WARN] No runs detected from watch roots.")
    return runs


def ensure_dist():
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def load_full_state(ckpt_dir):
    ckpt_dir = str(ckpt_dir)
    bin_path = os.path.join(ckpt_dir, "adapter_model.bin")
    if os.path.isfile(os.path.join(ckpt_dir, "adapter_model.safetensors")):
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError("safetensors is required for adapter_model.safetensors") from exc
        return load_file(os.path.join(ckpt_dir, "adapter_model.safetensors"))
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu")

    ensure_dist()
    checkpoint_id = None
    meta_path = os.path.join(ckpt_dir, "adapter_sharded.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        checkpoint_id = meta.get("checkpoint_id")
    metadata_path = os.path.join(ckpt_dir, ".metadata")
    if not os.path.isfile(metadata_path) and checkpoint_id is None:
        raise RuntimeError(
            f"No adapter weights found in {ckpt_dir} (expected adapter_model.safetensors/.bin or DCP metadata)."
        )
    try:
        from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    except Exception as exc:  # pragma: no cover - depends on torch version
        raise RuntimeError(
            "torch.distributed.checkpoint._EmptyStateDictLoadPlanner is required "
            "to load a full state dict without a model."
        ) from exc
    planner = _EmptyStateDictLoadPlanner()
    state = {}
    if hasattr(dist_cp, "load_state_dict"):
        load_fn = dist_cp.load_state_dict
        params = inspect.signature(load_fn).parameters
        if not hasattr(dist_cp, "FileSystemReader"):
            raise RuntimeError("torch.distributed.checkpoint.FileSystemReader not available")
        reader = dist_cp.FileSystemReader(ckpt_dir)
        kwargs = {}
        if "planner" in params:
            kwargs["planner"] = planner
        load_fn(state, reader, **kwargs)
        return state
    if hasattr(dist_cp, "load"):
        load_fn = dist_cp.load
        params = inspect.signature(load_fn).parameters
        kwargs = {}
        if "planner" in params:
            kwargs["planner"] = planner
        if "storage_reader" in params and hasattr(dist_cp, "FileSystemReader"):
            reader = dist_cp.FileSystemReader(ckpt_dir)
            kwargs["storage_reader"] = reader
            if checkpoint_id is not None and "checkpoint_id" in params:
                kwargs["checkpoint_id"] = checkpoint_id
        else:
            if "checkpoint_id" in params:
                kwargs["checkpoint_id"] = checkpoint_id or ckpt_dir
        load_fn(state, **kwargs)
        return state
    raise RuntimeError("torch.distributed.checkpoint load API not available")


def flatten_tensors(obj, prefix=""):
    tensors = {}
    if torch.is_tensor(obj):
        tensors[prefix or "tensor"] = obj
        return tensors
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            next_prefix = f"{prefix}.{key_str}" if prefix else key_str
            tensors.update(flatten_tensors(value, next_prefix))
        return tensors
    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            tensors.update(flatten_tensors(value, next_prefix))
        return tensors
    return tensors


def load_sharded_state(ckpt_dir):
    state = load_full_state(ckpt_dir)
    return flatten_tensors(state)


def fingerprint_tensors(tensors):
    if not tensors:
        return {"tensor_keys": 0, "norm_sum": 0.0, "shape_sum": 0}
    norm_sum = 0.0
    shape_sum = 0
    for tensor in tensors.values():
        norm_sum += float(tensor.float().norm().item())
        shape_sum += tensor.numel()
    return {
        "tensor_keys": len(tensors),
        "norm_sum": norm_sum,
        "shape_sum": shape_sum,
    }


def compare_two(checkpoint_a, checkpoint_b):
    tensors_a = load_sharded_state(checkpoint_a)
    tensors_b = load_sharded_state(checkpoint_b)
    fp_a = fingerprint_tensors(tensors_a)
    fp_b = fingerprint_tensors(tensors_b)

    shared_keys = set(tensors_a.keys()) & set(tensors_b.keys())
    diff_norm = 0.0
    for key in sorted(shared_keys):
        diff_norm += float((tensors_a[key].float() - tensors_b[key].float()).norm().item())
    return tensors_a, tensors_b, fp_a, fp_b, diff_norm


def collect_keys(tensors, patterns):
    return {k for k in tensors.keys() if any(p in k for p in patterns)}


def summarize_group_diffs(tensors_a, tensors_b, label, patterns):
    keys_a = collect_keys(tensors_a, patterns)
    keys_b = collect_keys(tensors_b, patterns)
    if not keys_a and not keys_b:
        print(f"{label}: no matching keys for patterns={patterns}")
        return
    shared = keys_a & keys_b
    missing = keys_a - keys_b
    extra = keys_b - keys_a
    diff = 0.0
    for key in shared:
        diff += float((tensors_a[key].float() - tensors_b[key].float()).norm().item())
    sample_keys = ", ".join(sorted(shared)[:3])
    key_match = not missing and not extra
    print(
        f"{label}: key_match={key_match} shared={len(shared)} "
        f"A_only={len(missing)} B_only={len(extra)} sum_l2_diff={diff:.6f} sample={sample_keys}"
    )
    if not key_match:
        missing_sample = ", ".join(sorted(missing)[:3])
        extra_sample = ", ".join(sorted(extra)[:3])
        print(f"{label} key_mismatch sample missing={missing_sample} extra={extra_sample}")


def print_key_hints(tensors, label):
    substrings = ["lora", "cola", "router", "gate", "expert", "adapter"]
    hints = [k for k in tensors.keys() if any(s in k.lower() for s in substrings)]
    print(f"{label}: key_hints={len(hints)} sample={', '.join(sorted(hints)[:10])}")


def find_adapter_checkpoints(root, limit=2):
    if not os.path.isdir(root):
        return []
    candidates = []
    for dirpath, _, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if "_adapter" not in base:
            continue
        if "adapter_config.json" not in filenames:
            continue
        step = parse_checkpoint_step(dirpath)
        candidates.append((step, dirpath))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if limit is None or limit <= 0:
        return [path for _, path in candidates]
    return [path for _, path in candidates[-limit:]]


def parse_checkpoint_step(path):
    base = os.path.basename(path)
    if "checkpoint-" not in base:
        return sys.maxsize
    try:
        step_str = base.split("checkpoint-")[1].split("_", 1)[0]
        return int(step_str)
    except (IndexError, ValueError):
        return sys.maxsize


def report_expected_keys(tensors, approach, label):
    patterns = APPROACH_PATTERNS.get(approach, [])
    if not patterns:
        print(f"{label}: no expected patterns for approach={approach}")
        return
    matched = collect_keys(tensors, patterns)
    missing = [p for p in patterns if not any(p in key for key in matched)]
    print(f"{label}: expected_patterns={patterns} matched_keys={len(matched)}")
    if missing:
        print(f"{label}: missing_patterns={missing}")


def verify_required_keys(tensors, patterns, label):
    missing = [p for p in patterns if not any(p in key for key in tensors)]
    if missing:
        print(f"{label}: [ERROR] missing required patterns={missing}")
        return False
    print(f"{label}: required patterns present={patterns}")
    return True


def main():
    exit_code = 0
    runs = build_runs()
    for name, spec in runs.items():
        root = spec["root"]
        approach = spec["approach"]
        checkpoints = find_adapter_checkpoints(root, limit=2)
        if len(checkpoints) < 2:
            print(f"[SKIP] {name}: expected 2 adapter checkpoints under {root}")
            continue
        ckpt_a_path, ckpt_b_path = checkpoints

        config_a = summarize_config(load_config(os.path.join(ckpt_a_path, "adapter_config.json")))
        config_b = summarize_config(load_config(os.path.join(ckpt_b_path, "adapter_config.json")))
        config_match = config_a == config_b

        print(f"\n=== {name} ===")
        print(f"approach: {approach}")
        print(f"checkpoint A: {ckpt_a_path}")
        print(f"checkpoint B: {ckpt_b_path}")
        print(f"config_match: {config_match}")
        if not config_match:
            print("config_a:", config_a)
            print("config_b:", config_b)
        if "cola_num_a" in config_a and "cola_num_b" in config_a:
            cola_ratio_ok = config_a["cola_num_b"] == 3 * config_a["cola_num_a"]
            print(
                f"cola A/B ratio: A={config_a['cola_num_a']} B={config_a['cola_num_b']} "
                f"expected_B=3*A -> {cola_ratio_ok}"
            )

        tensors_a, tensors_b, fp_a, fp_b, diff_norm = compare_two(ckpt_a_path, ckpt_b_path)
        print(f"A tensors: {fp_a}")
        print(f"B tensors: {fp_b}")
        print(f"sum L2 diff across shared tensors: {diff_norm:.6f}")
        update_changed = diff_norm > 1e-6
        print(f"update_check: changed={update_changed}")
        keys_a = set(tensors_a.keys())
        keys_b = set(tensors_b.keys())
        key_set_match = keys_a == keys_b
        print(f"key_set_match: {key_set_match} total_A={len(keys_a)} total_B={len(keys_b)}")
        if not key_set_match:
            missing = ", ".join(sorted(keys_a - keys_b)[:3])
            extra = ", ".join(sorted(keys_b - keys_a)[:3])
            print(f"key_set_mismatch sample missing={missing} extra={extra}")

        print_key_hints(tensors_a, "A key hints")
        print_key_hints(tensors_b, "B key hints")
        report_expected_keys(tensors_a, approach, "A expected keys")
        report_expected_keys(tensors_b, approach, "B expected keys")

        if approach == "mola":
            required = ["lora_A", "lora_B", "lora_gating"]
            ok_a = verify_required_keys(tensors_a, required, "A required keys")
            ok_b = verify_required_keys(tensors_b, required, "B required keys")
            # Ensure weights changed between latest two checkpoints for MoLA
            diff_keys = collect_keys(tensors_a, required) & collect_keys(tensors_b, required)
            diff_norm = 0.0
            for key in diff_keys:
                diff_norm += float((tensors_a[key].float() - tensors_b[key].float()).norm().item())
            print(f"MoLA diff_norm on required keys: {diff_norm:.6f}")
            if not (ok_a and ok_b) or diff_norm <= 1e-6:
                exit_code = 1

        summarize_group_diffs(tensors_a, tensors_b, "LoRA A diff", ["lora_A", "lora_a"])
        summarize_group_diffs(tensors_a, tensors_b, "LoRA B diff", ["lora_B", "lora_b"])
        if approach in ("cola", "hydralora"):
            summarize_group_diffs(tensors_a, tensors_b, "Router diff", ["router"])
        if approach in ("adamole", "mola"):
            summarize_group_diffs(tensors_a, tensors_b, "Gating diff", ["lora_gating"])
        if approach == "adamole":
            summarize_group_diffs(tensors_a, tensors_b, "Threshold diff", ["lora_threshold"])
        if approach == "hydralora":
            summarize_group_diffs(tensors_a, tensors_b, "Head router diff", ["lora_route"])

        a_keys_a = collect_keys(tensors_a, ["lora_A", "lora_a", "cola_A", "cola_a"])
        b_keys_a = collect_keys(tensors_a, ["lora_B", "lora_b", "cola_B", "cola_b"])
        a_keys_b = collect_keys(tensors_b, ["lora_A", "lora_a", "cola_A", "cola_a"])
        b_keys_b = collect_keys(tensors_b, ["lora_B", "lora_b", "cola_B", "cola_b"])
        if a_keys_a and b_keys_a and a_keys_b and b_keys_b:
            ratio_a = len(b_keys_a) / max(len(a_keys_a), 1)
            ratio_b = len(b_keys_b) / max(len(a_keys_b), 1)
            ratio_match = abs(ratio_a - ratio_b) < 1e-6
            print(
                f"A/B key count: "
                f"A={len(a_keys_a)} B={len(b_keys_a)} "
                f"ratio_A={ratio_a:.2f} ratio_B={ratio_b:.2f} ratio_match={ratio_match}"
            )
        head_router_keys_a = collect_keys(tensors_a, ["lora_route"])
        head_router_keys_b = collect_keys(tensors_b, ["lora_route"])
        if approach == "hydralora" and (head_router_keys_a or head_router_keys_b):
            print(f"Head router key count: A={len(head_router_keys_a)} B={len(head_router_keys_b)}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
