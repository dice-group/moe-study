"""Lightweight post-stage verification checks.

Each function inspects stage outputs, writes a ``_check.json`` manifest
into the stage's output directory, and raises on critical problems.
Designed for minimal LOC — every check is a single callable.
"""
from __future__ import annotations

import collections
import gzip
import json
import logging
import math
import re
import statistics
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("pipeline.checks")


def _write_check(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    LOGGER.info("[CHECK] wrote %s", path)


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

def check_sample(sampled_dir: Path, shard_dir: Path, *, dry_run: bool) -> None:
    """Verify sampled shards: per-language file counts, doc counts, disk size."""

    if dry_run:
        return

    out: dict[str, Any] = {"stage": "sample", "pass": True, "languages": {}}
    total_docs = 0
    total_bytes = 0
    shard_files = sorted(shard_dir.rglob("*.jsonl")) + sorted(shard_dir.rglob("*.jsonl.gz"))

    # iterate over shard files and count documents and count
    for jsonl in shard_files:
        lang = jsonl.parent.name  # e.g. arb_Arab/part_000.jsonl(.gz)
        size = jsonl.stat().st_size

        if jsonl.name.endswith(".jsonl.gz"):
            with gzip.open(jsonl, "rt", encoding="utf-8") as fh:
                docs = sum(1 for _ in fh)
        else:
            docs = sum(1 for _ in jsonl.open("r", encoding="utf-8"))

        entry = out["languages"].setdefault(lang, {"files": 0, "docs": 0, "bytes": 0})
        entry["files"] += 1
        entry["docs"] += docs
        entry["bytes"] += size
        total_docs += docs
        total_bytes += size

    out["total_docs"] = total_docs
    out["total_bytes"] = total_bytes
    out["total_bytes_human"] = f"{total_bytes / 1024 / 1024:.2f} MB"
    out["num_languages"] = len(out["languages"])

    if total_docs == 0:
        out["pass"] = False
        _write_check(sampled_dir / "_check.json", out)
        raise SystemExit("[CHECK FAIL] sample: zero documents produced")

    empty = [lang for lang, v in out["languages"].items() if v["docs"] == 0]
    if empty:
        out["pass"] = False
        out["empty_languages"] = empty
        _write_check(sampled_dir / "_check.json", out)
        raise SystemExit(f"[CHECK FAIL] sample: languages with 0 docs: {empty}")
    _write_check(sampled_dir / "_check.json", out)


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

def check_tokenize(
    tokenized_dir: Path,
    *,
    dry_run: bool,
    expected_languages: list[str] | None = None,
    english_median_check: bool = False,
    english_key: str = "english",
    english_median_tolerance: float = 0.05,
) -> None:
    """Verify tokenized dataset: row counts, token counts, splits, language IDs, example."""

    if dry_run:
        return

    from datasets import load_from_disk
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    from pipeline_tools.check_tokenized_parallel import _iter_ipc_batches

    rank_dirs = sorted(
        p for p in tokenized_dir.iterdir() if p.is_dir() and p.name.startswith("rank_")
    )

    out: dict[str, Any] = {"stage": "tokenize", "pass": True}

    if rank_dirs:
        # Stream through rank_* shards to avoid loading the entire merged dataset at once.
        split_langs: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
        split_tokens: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
        langs = collections.Counter()
        tokens_by_lang: dict[str, int] = collections.Counter()
        total_tokens = 0
        total_rows = 0
        unique_lids: set[int] = set()
        unique_fids: set[int] = set()
        pad_count = 0
        has_lang_ids = False
        has_family_ids = False
        columns_seen: set[str] = set()
        example: dict[str, Any] | None = None

        for rank_dir in rank_dirs:
            arrow_files = sorted(rank_dir.glob("data-*.arrow"))
            if not arrow_files:
                continue

            required = [
                "language",
                "split",
                "input_ids",
                "language_ids",
                "family_ids",
                "doc_id",
                "chunk_idx",
                "doc_token_count",
            ]
            try:
                dataset = ds.dataset([str(p) for p in arrow_files], format="ipc")
                available = set(dataset.schema.names)
                columns_seen.update(available)
                columns = [col for col in required if col in available]
                if "language" not in available:
                    out["pass"] = False
                    _write_check(tokenized_dir / "_check.json", out)
                    raise SystemExit("[CHECK FAIL] tokenize: missing 'language' column in tokenized data")

                batches_iter = dataset.scanner(columns=columns, batch_size=8192).to_batches()
            except pa.ArrowInvalid:
                # HuggingFace Datasets writes IPC stream shards; pyarrow.dataset expects file IPC.
                def _stream_batches():
                    schema_checked = False
                    columns: list[str] = []
                    for path in arrow_files:
                        for batch, schema in _iter_ipc_batches(path):
                            if not schema_checked:
                                available = set(schema.names)
                                columns_seen.update(available)
                                columns = [col for col in required if col in available]
                                if "language" not in available:
                                    out["pass"] = False
                                    _write_check(tokenized_dir / "_check.json", out)
                                    raise SystemExit("[CHECK FAIL] tokenize: missing 'language' column in tokenized data")
                                schema_checked = True
                            yield batch.select(columns) if columns else batch

                batches_iter = _stream_batches()

            for batch in batches_iter:
                batch_rows = batch.num_rows
                total_rows += batch_rows
                lang_list = batch.column("language").to_pylist()
                split_list = batch.column("split").to_pylist() if "split" in batch.column_names else ["train"] * batch_rows

                if "input_ids" in batch.column_names:
                    lengths = pc.list_value_length(batch.column("input_ids")).to_pylist()
                else:
                    lengths = [0] * batch_rows

                for split_name, lang, length in zip(split_list, lang_list, lengths):
                    lang_key = str(lang)
                    split_key = str(split_name)
                    langs[lang_key] += 1
                    tokens_by_lang[lang_key] += int(length)
                    split_langs[split_key][lang_key] += 1
                    split_tokens[split_key][lang_key] += int(length)
                    total_tokens += int(length)

                if "language_ids" in batch.column_names:
                    has_lang_ids = True
                    lids = batch.column("language_ids").to_pylist()
                    for lid in lids:
                        lid_int = int(lid)
                        unique_lids.add(lid_int)
                        if lid_int < 0:
                            pad_count += 1

                if "family_ids" in batch.column_names:
                    has_family_ids = True
                    fids = batch.column("family_ids").to_pylist()
                    for fid in fids:
                        unique_fids.add(int(fid))

                if example is None and batch_rows > 0:
                    idx = 0
                    example = {
                        "language": lang_list[idx],
                        "split": split_list[idx],
                        "input_ids_len": int(lengths[idx]) if lengths else 0,
                        "input_ids_head": [],
                        "language_ids": int(batch.column("language_ids")[idx]) if "language_ids" in batch.column_names else None,
                        "family_ids": int(batch.column("family_ids")[idx]) if "family_ids" in batch.column_names else None,
                    }

        out["num_rows"] = total_rows
        out["columns"] = sorted(columns_seen)
        out["splits"] = {k: sum(v.values()) for k, v in split_langs.items()}
        out["samples_per_language_by_split"] = {split: dict(counts) for split, counts in split_langs.items()}
        out["tokens_per_language_by_split"] = {split: dict(counts) for split, counts in split_tokens.items()}
        out["samples_per_language"] = dict(langs)
        out["num_languages"] = len(langs)
        out["total_tokens"] = total_tokens
        out["tokens_per_language"] = dict(tokens_by_lang)
        out["has_language_ids"] = has_lang_ids
        if has_lang_ids:
            out["unique_language_ids"] = sorted(unique_lids)
            out["num_unique_language_ids"] = len(unique_lids)
            out["language_ids_pad_count"] = pad_count
            if pad_count > 0:
                out["language_ids_warning"] = (
                    f"{pad_count} samples have pad language_id (<0) — "
                    "these will be ignored by LPR guidance"
                )
        out["has_family_ids"] = has_family_ids
        if has_family_ids:
            out["unique_family_ids"] = sorted(unique_fids)
            out["num_unique_family_ids"] = len(unique_fids)
        if example is not None:
            out["example"] = example
    else:
        ds = load_from_disk(str(tokenized_dir))
        out["num_rows"] = len(ds)
        out["columns"] = ds.column_names

        # Splits
        splits = collections.Counter(ds["split"])
        out["splits"] = dict(splits)
        split_langs: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
        split_tokens: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
        for split_name, lang, ids in zip(ds["split"], ds["language"], ds["input_ids"]):
            split_langs[str(split_name)][str(lang)] += 1
            split_tokens[str(split_name)][str(lang)] += len(ids)
        out["samples_per_language_by_split"] = {split: dict(counts) for split, counts in split_langs.items()}
        out["tokens_per_language_by_split"] = {split: dict(counts) for split, counts in split_tokens.items()}

        # Language counts
        langs = collections.Counter(ds["language"])
        out["samples_per_language"] = dict(langs)
        out["num_languages"] = len(langs)

        # Token counts
        total_tokens = sum(len(ids) for ids in ds["input_ids"])
        out["total_tokens"] = total_tokens
        tokens_by_lang: dict[str, int] = collections.Counter()
        for lang, ids in zip(ds["language"], ds["input_ids"]):
            tokens_by_lang[lang] += len(ids)
        out["tokens_per_language"] = dict(tokens_by_lang)

        # Language IDs
        has_lang_ids = "language_ids" in ds.column_names
        out["has_language_ids"] = has_lang_ids
        if has_lang_ids:
            all_lids = ds["language_ids"]
            unique_lids = sorted(set(all_lids))
            out["unique_language_ids"] = unique_lids
            out["num_unique_language_ids"] = len(unique_lids)
            # Check for invalid pad IDs (-1)
            pad_count = sum(1 for lid in all_lids if lid < 0)
            out["language_ids_pad_count"] = pad_count
            if pad_count > 0:
                out["language_ids_warning"] = (
                    f"{pad_count} samples have pad language_id (<0) — "
                    "these will be ignored by LPR guidance"
                )

        # Family IDs (cluster/expert mapping for LPR)
        has_family_ids = "family_ids" in ds.column_names
        out["has_family_ids"] = has_family_ids
        if has_family_ids:
            unique_fids = sorted(set(ds["family_ids"]))
            out["unique_family_ids"] = unique_fids
            out["num_unique_family_ids"] = len(unique_fids)

        # Example (first row, truncated)
        ex = ds[0]
        out["example"] = {
            "language": ex.get("language"),
            "split": ex.get("split"),
            "input_ids_len": len(ex.get("input_ids", [])),
            "input_ids_head": ex.get("input_ids", [])[:10],
            "language_ids": ex.get("language_ids"),
            "family_ids": ex.get("family_ids"),
        }

    # Fail conditions
    if int(out.get("num_rows", 0)) == 0:
        out["pass"] = False
        _write_check(tokenized_dir / "_check.json", out)
        raise SystemExit("[CHECK FAIL] tokenize: dataset has 0 rows")
    if "train" not in out.get("splits", {}):
        out["pass"] = False
        _write_check(tokenized_dir / "_check.json", out)
        raise SystemExit("[CHECK FAIL] tokenize: no 'train' split found")
    if not has_lang_ids:
        out["pass"] = False
        _write_check(tokenized_dir / "_check.json", out)
        raise SystemExit("[CHECK FAIL] tokenize: missing 'language_ids' column")

    required_chunk_columns = {"doc_id", "chunk_idx", "doc_token_count"}
    missing_chunk_columns = sorted(required_chunk_columns - set(out.get("columns", [])))
    if missing_chunk_columns:
        out["pass"] = False
        out["missing_chunk_columns"] = missing_chunk_columns
        _write_check(tokenized_dir / "_check.json", out)
        raise SystemExit(
            "[CHECK FAIL] tokenize: missing chunk metadata columns: "
            f"{missing_chunk_columns}"
        )

    if expected_languages:
        expected = sorted({str(item) for item in expected_languages if str(item).strip()})
        missing = [lang for lang in expected if lang not in langs]
        if missing:
            out["pass"] = False
            out["missing_languages"] = missing
            _write_check(tokenized_dir / "_check.json", out)
            raise SystemExit(f"[CHECK FAIL] tokenize: missing languages in tokenized data: {missing}")

    if english_median_check and expected_languages:
        english_key = str(english_key or "english").strip()
        if english_key not in langs:
            out["pass"] = False
            out["english_missing"] = english_key
            _write_check(tokenized_dir / "_check.json", out)
            raise SystemExit(f"[CHECK FAIL] tokenize: english key '{english_key}' missing from tokenized data")
        expected_set = {str(item) for item in expected_languages if str(item).strip()}
        other_counts = [langs[lang] for lang in expected_set if lang != english_key and lang in langs]
        if other_counts:
            median_count = int(statistics.median(other_counts))
            english_count = int(langs.get(english_key, 0))
            tolerance = max(1, int(median_count * float(english_median_tolerance)))
            if abs(english_count - median_count) > tolerance:
                out["pass"] = False
                out["english_median_expected"] = median_count
                out["english_median_actual"] = english_count
                out["english_median_tolerance"] = tolerance
                _write_check(tokenized_dir / "_check.json", out)
                raise SystemExit(
                    "[CHECK FAIL] tokenize: english sample count is not near median of other languages "
                    f"(expected ~{median_count} ±{tolerance}, got {english_count})"
                )

    _write_check(tokenized_dir / "_check.json", out)


# ---------------------------------------------------------------------------
# tokenizer_extend  (already strong – add summary _check.json)
# ---------------------------------------------------------------------------

def check_tokenizer_extend(extension_manifest_path: Path, extended_tokenizer_dir: Path, *, dry_run: bool):
    """Write a summary _check.json from the already-validated extension manifest."""

    if dry_run:
        return

    manifest = json.loads(extension_manifest_path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {
        "stage": "tokenizer_extend",
        "pass": True,
        "original_vocab_size": manifest.get("original_vocab_size"),
        "new_vocab_size": manifest.get("new_vocab_size"),
        "vocab_growth": manifest.get("vocab_growth"),
        "extended_tokenizer_dir": str(extended_tokenizer_dir),
    }
    _write_check(extended_tokenizer_dir / "_check.json", out)


# ---------------------------------------------------------------------------
# embedding_init
# ---------------------------------------------------------------------------

def check_embedding_init(output_dir: Path, manifest_path: Path, expected_vocab_size: int | None, *, dry_run: bool):
    """Verify initialized model: config.json vocab_size matches extended tokenizer."""

    if dry_run:
        return

    config_path = output_dir / "config.json"
    out: dict[str, Any] = {"stage": "embedding_init", "pass": True}

    if not config_path.exists():
        out["pass"] = False
        _write_check(output_dir / "_check.json", out)
        raise SystemExit(f"[CHECK FAIL] embedding_init: {config_path} missing")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    out["model_vocab_size"] = config.get("vocab_size")
    out["expected_vocab_size"] = expected_vocab_size

    if expected_vocab_size is not None and config.get("vocab_size") != expected_vocab_size:
        out["pass"] = False
        _write_check(output_dir / "_check.json", out)
        raise SystemExit(f"[CHECK FAIL] embedding_init: model vocab_size={config.get('vocab_size')} != expected {expected_vocab_size}")

    _write_check(output_dir / "_check.json", out)


# ---------------------------------------------------------------------------
# train  — helpers
# ---------------------------------------------------------------------------

# each saved checkpoint includes different weight modules, depending how they are called in each approach
# e.g. Lora includes lora_A/B others include router, experts etc.
# these are listed in the saved weights and int the adapter_config.json
# For verification we check if the expected modules exist, also one check if they change between checkpoints to verify its trainings

_APPROACH_MODULES: dict[str, list[str]] = {
    "LORA":      ["lora_A", "lora_B"],
    "COLA":      ["lora_A", "lora_B", "router", "expert_"],
    "HYDRALORA": ["lora_A", "lora_B", "router", "expert_"],
    "ADAMOLE":   ["lora_A", "lora_B", "lora_gating", "lora_threshold"],
    "MIXLORA":   ["lora_A", "lora_B", "lora_router", "experts"],
    "MOLA":      ["lora_A", "lora_B", "lora_gating", "moe_layer"],
    "MOELPR":    ["moe_experts", "moe_router_embedding"],
    "MOELORA":   ["lora_A", "lora_B", "lora_gate", "lora_task_embedding"],
    "MOVLORA":   ["lora_router", "lora_mov_scaling"],
    "HMORA":     ["lora_A", "lora_B", "lora_router_pool"],
}


def _load_adapter_keys(adapter_bin: Path) -> list[str]:
    """Return sorted key list from an adapter_model.bin/.safetensors checkpoint."""
    if adapter_bin.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(adapter_bin), framework="pt") as f:
            return sorted(f.keys())
    else:
        import torch
        sd = torch.load(str(adapter_bin), map_location="cpu", weights_only=True)
        return sorted(sd.keys())


def _load_adapter_config(run_dir: Path) -> dict[str, Any]:
    """Read adapter_config.json, or return empty dict."""
    cfg_path = run_dir / "adapter_config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text("utf-8"))
    for ck in reversed(_find_checkpoints(run_dir)):
        ck_cfg = ck / "adapter_config.json"
        if ck_cfg.exists():
            return json.loads(ck_cfg.read_text("utf-8"))
    return {}


def _detect_approach(run_dir: Path) -> str:
    """Read peft_type from adapter_config.json, fallback to UNKNOWN."""
    return _load_adapter_config(run_dir).get("peft_type", "UNKNOWN").upper()


def _find_checkpoints(run_dir: Path) -> list[Path]:
    """Return checkpoint dirs sorted by step number (ascending)."""

    cks = []
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            cks.append(d)
    return sorted(cks, key=lambda p: int(p.name.split("-")[1].split("_")[0]))


def _adapter_file(d: Path) -> Path | None:
    """Return the adapter weight file inside a directory, or None."""
    for name in (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
    ):
        p = d / name
        if p.exists():
            return p
    return None


def _load_state_dict(adapter_path: Path) -> dict[str, Any]:
    """Load a full state dict from .bin or .safetensors."""
    if adapter_path.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(adapter_path), framework="pt") as f:
            return {k: f.get_tensor(k) for k in f.keys()}
    import torch
    return torch.load(str(adapter_path), map_location="cpu", weights_only=True)


def _parse_expert_structure(keys: list[str]) -> dict[str, Any]:
    """Parse MoE adapter keys to extract expert count, A/B heads per expert."""
    experts_a: dict[int, set[int]] = collections.defaultdict(set)
    experts_b: dict[int, set[int]] = collections.defaultdict(set)
    for k in keys:
        # CoLA:  lora_A.expert_0.0.weight  (expert_idx . sub_idx)
        # Hydra: lora_A.expert_0.weight    (expert_idx, no sub)
        m = re.search(r"lora_A\.expert_(\d+)(?:\.(\d+))?\.weight", k)
        if m:
            experts_a[int(m.group(1))].add(int(m.group(2) or 0))
        m = re.search(r"lora_B\.expert_(\d+)\.(\d+)\.weight", k)
        if m:
            experts_b[int(m.group(1))].add(int(m.group(2)))
    all_expert_ids = sorted(set(experts_a.keys()) | set(experts_b.keys()))
    result: dict[str, Any] = {
        "num_experts": len(all_expert_ids),
        "expert_ids": all_expert_ids,
    }
    if all_expert_ids:
        # Use first expert as representative (all should be identical)
        rep = all_expert_ids[0]
        result["A_heads_per_expert"] = len(experts_a.get(rep, set()))
        result["B_heads_per_expert"] = len(experts_b.get(rep, set()))
    return result


def _extract_lpr_config(acfg: dict[str, Any], approach: str) -> dict[str, Any]:
    """Extract LPR-relevant config fields from adapter_config.json."""
    lpr: dict[str, Any] = {}
    lpr["guidance_scope"] = acfg.get("language_guidance_scope", "unknown")
    lpr["router_mode"] = acfg.get("language_router_mode", "unknown")
    lpr["head_router_mode"] = acfg.get("language_head_router_mode", "unknown")
    lpr["prior_weight"] = acfg.get("language_prior_weight")
    lpr["bias_value"] = acfg.get("language_bias_value")
    lpr["num_languages"] = len(acfg.get("language_list", []))
    lpr["language_to_family_ids"] = acfg.get("language_to_family_ids")
    if approach == "COLA":
        lpr["num_A"] = acfg.get("num_A")
        lpr["num_B"] = acfg.get("num_B")
        lpr["strategy"] = acfg.get("cola_strategy")
    elif approach == "HYDRALORA":
        lpr["lora_num"] = acfg.get("lora_num")
    return lpr


def _component_diff(keys: list[str], sd_first: dict[str, Any], sd_last: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Count changed/unchanged weights grouped by component type."""
    import torch
    buckets: dict[str, list[bool]] = collections.defaultdict(list)
    for k in keys:
        changed = not torch.equal(sd_first[k], sd_last[k])
        if ".router." in k or ".lora_route." in k or ".lora_router." in k:
            buckets["router"].append(changed)
        elif "expert_" in k and "lora_A" in k:
            buckets["expert_lora_A"].append(changed)
        elif "expert_" in k and "lora_B" in k:
            buckets["expert_lora_B"].append(changed)
        elif "lora_A" in k:
            buckets["lora_A"].append(changed)
        elif "lora_B" in k:
            buckets["lora_B"].append(changed)
        else:
            buckets["other"].append(changed)
    return {
        comp: {"changed": sum(v), "unchanged": len(v) - sum(v), "total": len(v)}
        for comp, v in sorted(buckets.items())
    }


def _expert_diversity(sd: dict[str, Any], keys: list[str]) -> dict[str, Any] | None:
    """Check whether different experts have diverged (non-identical weights).

    Compares expert_0 vs expert_1 for each (layer, proj, matrix_type) group.
    Returns a summary with the fraction of groups that are diverse.
    """
    import torch
    # Group keys by (prefix_before_expert, A_or_B, sub_idx) → {expert_idx: key}
    groups: dict[str, dict[int, str]] = collections.defaultdict(dict)
    for k in keys:
        m = re.search(r"(.+)\.(lora_[AB])\.expert_(\d+)(\..*)", k)
        if m:
            group_key = f"{m.group(1)}.{m.group(2)}{m.group(4)}"
            expert_idx = int(m.group(3))
            groups[group_key][expert_idx] = k
    if not groups:
        return None
    diverse_count = 0
    identical_count = 0
    for _gk, emap in groups.items():
        idxs = sorted(emap.keys())
        if len(idxs) < 2:
            continue
        ref = sd[emap[idxs[0]]]
        for idx in idxs[1:]:
            if torch.equal(ref, sd[emap[idx]]):
                identical_count += 1
            else:
                diverse_count += 1
    total = diverse_count + identical_count
    return {
        "pairs_compared": total,
        "diverse_pairs": diverse_count,
        "identical_pairs": identical_count,
        "diversity_ratio": round(diverse_count / total, 3) if total > 0 else 0.0,
    }


def _check_run_adapters(run_dir: Path) -> dict[str, Any]:
    """Deep-check a single training run's adapter checkpoints.

    Returns a dict with:
      approach, checkpoints_found, modules_present, modules_expected,
      modules_missing, weight_diff (between first and last checkpoint).
    """
    entry: dict[str, Any] = {"pass": True}
    approach = _detect_approach(run_dir)
    entry["approach"] = approach

    # --- locate checkpoints ---
    checkpoints = _find_checkpoints(run_dir)
    final_adapter = _adapter_file(run_dir)
    ck_paths: list[Path] = []
    for ck in checkpoints:
        af = _adapter_file(ck)
        if af:
            ck_paths.append(af)
    if final_adapter:
        ck_paths.append(final_adapter)
    entry["checkpoints_found"] = len(ck_paths)

    if not ck_paths:
        entry["pass"] = False
        entry["error"] = "no adapter checkpoint files found"
        return entry

    # --- module presence ---
    first_keys = _load_adapter_keys(ck_paths[0])
    entry["num_adapter_keys"] = len(first_keys)
    expected_fragments = _APPROACH_MODULES.get(approach, [])
    entry["modules_expected"] = expected_fragments
    found_fragments = [f for f in expected_fragments if any(f in k for k in first_keys)]
    missing_fragments = [f for f in expected_fragments if f not in found_fragments]
    entry["modules_present"] = found_fragments
    entry["modules_missing"] = missing_fragments

    # Group keys by component type for a concise summary.
    component_counts: dict[str, int] = collections.Counter()
    for k in first_keys:
        for frag in (
            "lora_A", "lora_B", "router", "lora_route", "expert_",
            "lora_gating", "lora_threshold", "moe_layer", "moe_experts",
            "moe_router_embedding", "lora_gate", "lora_task_embedding",
            "lora_router", "lora_mov_scaling", "lora_router_pool", "lora_task_encoder", "experts",
        ):
            if frag in k:
                component_counts[frag] += 1
    entry["component_key_counts"] = dict(component_counts)

    if missing_fragments:
        entry["pass"] = False
        entry["error"] = f"missing expected modules: {missing_fragments}"

    # for MoE expert structure (CoLA / HydraLoRA / AdaMoLE / MoLA / MoE-LPR / MoE-LoRA / MoV-LoRA)
    if approach in ("COLA", "HYDRALORA"):
        entry["expert_structure"] = _parse_expert_structure(first_keys)
        acfg = _load_adapter_config(run_dir) # Read expected values from adapter_config.json
        entry["lpr_config"] = _extract_lpr_config(acfg, approach)
        # Cross-check: num_experts in config vs checkpoint
        cfg_num_experts = acfg.get("cola_num_experts" if approach == "COLA" else "num_experts")
        ck_num_experts = entry["expert_structure"]["num_experts"]
        if cfg_num_experts is not None and ck_num_experts != cfg_num_experts:
            entry["pass"] = False
            entry.setdefault("errors", []).append(
                f"expert count mismatch: config={cfg_num_experts} vs checkpoint={ck_num_experts}"
            )
    elif approach in ("ADAMOLE", "MIXLORA", "MOLA", "MOELPR", "MOELORA", "MOVLORA", "HMORA"):
        acfg = _load_adapter_config(run_dir)
        moe_info: dict[str, Any] = {"approach": approach}

        if approach == "ADAMOLE":
            moe_info["num_experts"] = acfg.get("adamole_num_experts")
        elif approach == "MIXLORA":
            moe_info["num_experts"] = acfg.get("num_experts", acfg.get("mixlora_num_experts"))
            moe_info["top_k"] = acfg.get("top_k", acfg.get("mixlora_top_k"))
            moe_info["router_aux_loss_coef"] = acfg.get(
                "router_aux_loss_coef", acfg.get("mixlora_router_aux_loss_coef")
            )
        elif approach == "MOLA":
            moe_info["num_experts"] = acfg.get("mola_num_experts")
            moe_info["top_k"] = acfg.get("mola_top_k")
        elif approach == "MOELPR":
            moe_info["num_experts"] = acfg.get("moelpr_num_experts")
            moe_info["top_k"] = acfg.get("moelpr_top_k")
            moe_info["stage"] = acfg.get("moelpr_stage")
        elif approach == "MOELORA":
            moe_info["num_experts"] = acfg.get("moelora_num_experts")
            moe_info["task_num"] = acfg.get("moelora_task_num")
        elif approach == "MOVLORA":
            moe_info["num_experts"] = acfg.get("movlora_num_experts")
            moe_info["top_k"] = acfg.get("movlora_top_k")
        elif approach == "HMORA":
            moe_info["num_experts"] = acfg.get("num_experts", acfg.get("hmora_num_experts"))
            moe_info["top_k"] = acfg.get("top_k", acfg.get("hmora_top_k"))
            moe_info["use_task_router"] = acfg.get("use_task_router", acfg.get("hmora_use_task_router"))
            moe_info["use_hydra_lora"] = acfg.get("use_hydra_lora", acfg.get("hmora_use_hydra_lora"))
        entry["moe_config"] = moe_info

    # here we do weight diff between first and last checkpoint, this verifies that parameters were updated
    if len(ck_paths) >= 2:
        first_keys_set = set(first_keys)
        last_keys = _load_adapter_keys(ck_paths[-1])
        last_keys_set = set(last_keys)
        if first_keys_set != last_keys_set:
            entry["keys_match"] = False
            entry["keys_only_first"] = len(first_keys_set - last_keys_set)
            entry["keys_only_last"] = len(last_keys_set - first_keys_set)
        else:
            entry["keys_match"] = True
            # Lazy-import torch only here (heaviest dep, only needed for diff).
            import torch
            sd_first = _load_state_dict(ck_paths[0])
            sd_last = _load_state_dict(ck_paths[-1])
            changed = sum(1 for k in first_keys if not torch.equal(sd_first[k], sd_last[k]))
            entry["weights_changed"] = changed
            entry["weights_unchanged"] = len(first_keys) - changed
            entry["weights_total"] = len(first_keys)
            if changed == 0 and len(ck_paths) > 2:
                entry["warning"] = "no weights changed between first and last checkpoint"

            # --- component-level diff (router vs experts vs lora) ---
            if approach in ("COLA", "HYDRALORA", "ADAMOLE", "MIXLORA", "MOLA", "MOELPR", "MOELORA", "MOVLORA", "HMORA"):
                diff_by_component = _component_diff(first_keys, sd_first, sd_last)
                entry["component_diff"] = diff_by_component
                # Router must have trained if guidance_scope != "none"
                if approach in ("COLA", "HYDRALORA"):
                    acfg = _load_adapter_config(run_dir)
                    scope = acfg.get("language_guidance_scope", "all")
                    router_changed = diff_by_component.get("router", {}).get("changed", 0)
                    if scope != "none" and router_changed == 0 and len(ck_paths) > 2:
                        entry.setdefault("warnings", []).append(
                            "router weights did not change despite LPR guidance_scope=" + scope
                        )
                # Expert diversity: check that different experts have different weights
                diversity = _expert_diversity(sd_last, first_keys)
                if diversity is not None:
                    entry["expert_diversity"] = diversity
    else:
        entry["weight_diff"] = "skipped (only 1 checkpoint)"

    return entry


# ---------------------------------------------------------------------------
# train  — public API
# ---------------------------------------------------------------------------

def check_train(train_root: Path, runs: list[dict[str, Any]], *, dry_run: bool):
    """Verify training results: loss, adapter modules, and weight updates."""

    if dry_run:
        return

    out: dict[str, Any] = {"stage": "train", "pass": True, "runs": {}}

    for run in runs:
        run_id = str(run["id"])
        run_dir = train_root / run_id
        entry: dict[str, Any] = {}

        # 1. Basic: train_results.json
        results_path = run_dir / "train_results.json"
        entry["results_exist"] = results_path.exists()
        if results_path.exists():
            results = json.loads(results_path.read_text(encoding="utf-8"))
            loss = results.get("train_loss")
            entry["train_loss"] = loss
            entry["train_runtime"] = results.get("train_runtime")
            entry["tokens_seen"] = results.get("num_input_tokens_seen")
            if loss is None or (isinstance(loss, float) and (math.isnan(loss) or math.isinf(loss))):
                entry["loss_ok"] = False
                out["pass"] = False
            else:
                entry["loss_ok"] = True
        else:
            entry["loss_ok"] = False
            out["pass"] = False

        # 2. Deep: adapter checkpoint analysis
        if run_dir.exists():
            adapter_info = _check_run_adapters(run_dir)
            entry["adapter_check"] = adapter_info
            if not adapter_info.get("pass", True):
                out["pass"] = False
        out["runs"][run_id] = entry

    _write_check(train_root / "_check.json", out)
    if not out["pass"]:
        failed = [rid for rid, v in out["runs"].items()
                  if not v.get("loss_ok") or not v.get("adapter_check", {}).get("pass", True)]
        raise SystemExit(f"[CHECK FAIL] train: failed runs: {failed}")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

def check_benchmark(benchmark_root: Path, summary_path: Path, *, dry_run: bool):
    """Verify benchmark: all runs have expected task count."""

    if dry_run:
        return

    if not summary_path.exists():
        return  # benchmark may be disabled

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    out: dict[str, Any] = {"stage": "benchmark", "pass": True, "runs": {}}
    for run_info in summary.get("runs", []):
        run_id = run_info.get("run_id", "?")
        expected = run_info.get("num_tasks_expected", 0)
        seen = run_info.get("num_tasks_seen", 0)
        missing = run_info.get("missing_tasks", [])
        entry = {"expected": expected, "seen": seen, "missing_count": len(missing)}
        if expected > 0 and seen == 0:
            entry["pass"] = False
            out["pass"] = False
        else:
            entry["pass"] = True
        out["runs"][run_id] = entry
    _write_check(benchmark_root / "_check.json", out)
    if not out["pass"]:
        failed = [rid for rid, v in out["runs"].items() if not v.get("pass")]
        raise SystemExit(f"[CHECK FAIL] benchmark: runs with 0 tasks evaluated: {failed}")
