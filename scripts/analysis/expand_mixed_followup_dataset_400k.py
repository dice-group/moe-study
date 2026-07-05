import gzip
import hashlib
import heapq
import itertools
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterator

from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer

from mixed_followup_dataset_utils import (
    dataset_distribution,
    extract_text,
    load_language_map,
    strip_zero_counts,
    write_json,
    write_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "outputs/pipeline_runs/moe_study_mixed_followup_expanded_data_rerun_10_langs"

BASE_TOKENIZED = RUN_ROOT / "tokenized"
BASE_SHARD_PLAN = RUN_ROOT / "tokenized_shard_plan"
SOURCE_SHARD_ROOT = REPO_ROOT / "outputs/sharded_samples"
ORIGINAL_SOURCE_ROOT = Path("/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/sharded_samples")
LANGUAGE_MAP = RUN_ROOT / "language_map_from_csv.json"
TOKENIZER_PATH = Path(
    "/scratch/hpc-prf-merlin/shared_cache/huggingface/hub/"
    "models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
)

OUTPUT_TOKENIZED = RUN_ROOT / "tokenized_expanded_400k_train"
AUDIT_DIR = RUN_ROOT / "tokenized_expanded_400k_train_audit"

TARGET_TRAIN_CHUNKS = 400_000
TARGET_VALIDATION_CHUNKS = 0  # preserve original validation/train ratio
SAMPLE_SEED = 1234
EVAL_SEED = 42
MAX_LENGTH = 1024
CANDIDATE_DOC_SURPLUS = 4.0
PLANNING_BATCH_SIZE = 128
GENERATION_BATCH_SIZE = 64
PROGRESS_EVERY_SHARDS = 25
LANGUAGE_PAD_ID = -1

def hash_score(seed: int, language: str, shard_rel: str, line_idx: int) -> float:
    key = f"{language}\0{shard_rel}\0{seed}\0{line_idx}".encode("utf-8", errors="ignore")  # replay original sampling
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def document_id(language: str, shard_rel: str, line_idx: int, text: str) -> str:
    shard_path = ORIGINAL_SOURCE_ROOT / shard_rel  # keep copied-dataset doc_ids reproducible
    key = f"{language}\0{shard_path}\0{line_idx}\0{text}".encode("utf-8", errors="ignore")
    return hashlib.sha256(key).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def load_shard_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(BASE_SHARD_PLAN.glob("rank_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rank_*.jsonl shard-plan rows found in {BASE_SHARD_PLAN}")
    return rows


def apportion_counts(current: dict[str, int], target_total: int, languages: list[str]) -> dict[str, int]:
    current_total = sum(int(current.get(language, 0)) for language in languages)
    if current_total <= 0:
        raise RuntimeError("Cannot apportion counts from an empty distribution.")

    raw = {language: int(current.get(language, 0)) / current_total * target_total for language in languages}  # preserve old language ratios
    apportioned = {language: math.floor(raw[language]) for language in languages}
    remaining = int(target_total) - sum(apportioned.values())
    order = sorted(
        languages,
        key=lambda language: (
            raw[language] - apportioned[language],
            int(current.get(language, 0)),
            language,
        ),
        reverse=True,
    )
    for language in order[:remaining]:
        apportioned[language] += 1
    return apportioned


def candidate_doc_limit(append_chunks: int, current_chunks: int, current_docs: int) -> int:
    if append_chunks <= 0:
        return 0
    avg_chunks_per_doc = current_chunks / max(1, current_docs)
    estimated_docs = math.ceil(append_chunks / max(avg_chunks_per_doc, 1e-9))
    if current_docs < 25:  # low-resource ratios are noisy in the old sample
        estimated_docs = max(estimated_docs, append_chunks)
    return max(8, math.ceil(estimated_docs * CANDIDATE_DOC_SURPLUS))


def shard_records(path: Path, language: str, shard_rel: str) -> list[tuple[float, float, int, str]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    records: list[tuple[float, float, int, str]] = []
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line_idx, line in enumerate(handle):
            text = extract_text(line)
            if text is None:
                continue
            records.append(
                (
                    hash_score(SAMPLE_SEED, language, shard_rel, line_idx),
                    hash_score(EVAL_SEED, language, shard_rel, line_idx),
                    line_idx,
                    text,
                )
            )
    return records


def scan_sources(
    languages: list[str],
    append_train_targets: dict[str, int],
    append_validation_targets: dict[str, int],
    current_samples: dict[str, dict[str, int]],
    current_docs: dict[str, dict[str, int]],
) -> dict[str, Any]:
    train_limits = {
        language: candidate_doc_limit(
            append_chunks=int(append_train_targets.get(language, 0)),
            current_chunks=int(current_samples.get("train", {}).get(language, 0)),
            current_docs=int(current_docs.get("train", {}).get(language, 0)),
        )
        for language in languages
    }
    validation_limits = {
        language: candidate_doc_limit(
            append_chunks=int(append_validation_targets.get(language, 0)),
            current_chunks=int(current_samples.get("validation", {}).get(language, 0)),
            current_docs=int(current_docs.get("validation", {}).get(language, 0)),
        )
        for language in languages
    }
    train_heaps = {language: [] for language in languages}  # bound candidate memory during source scan
    validation_heaps = {language: [] for language in languages}

    old_doc_ids: set[str] = set()
    old_source_keys: set[str] = set()
    old_text_hashes: set[str] = set()
    old_split_by_key: dict[str, str] = {}
    scanned_docs = 0
    plan_rows = load_shard_plan()
    sequence = itertools.count(1)

    def accepts(heap: list[tuple[tuple[Any, ...], int, dict[str, Any]]], limit: int, key: tuple[Any, ...]) -> bool:
        return limit > 0 and (len(heap) < limit or key < heap[0][0])

    def add(
        heap: list[tuple[tuple[Any, ...], int, dict[str, Any]]],
        limit: int,
        key: tuple[Any, ...],
        candidate: dict[str, Any],
    ) -> None:
        if limit <= 0:
            return
        entry = (key, next(sequence), candidate)
        if len(heap) < limit:
            heap.append(entry)
            heapq._siftdown_max(heap, 0, len(heap) - 1)
        elif key < heap[0][0]:
            heapq._heapreplace_max(heap, entry)

    print(f"[scan] shard plan rows={len(plan_rows):,} tokenizer={TOKENIZER_PATH}", flush=True)
    for scanned_shards, row in enumerate(plan_rows, start=1):
        language = str(row["language"])
        shard_rel = str(row["shard"])
        expected_doc_count = int(row["doc_count"])
        selected_quota = int(row["selected_quota"])
        validation_quota = int(row["validation_quota"])
        shard_path = SOURCE_SHARD_ROOT / shard_rel
        if not shard_path.is_file():
            raise RuntimeError(f"Shard plan references missing local shard: {shard_path}")

        records = shard_records(shard_path, language, shard_rel)
        if len(records) != expected_doc_count:
            raise RuntimeError(f"Shard doc count mismatch for {shard_rel}: planned={expected_doc_count} actual={len(records)}")

        selected_records = heapq.nsmallest(selected_quota, records, key=lambda item: (item[0], item[2]))  # replay old shard quota
        selected_line_ids = {line_idx for _sample, _eval, line_idx, _text in selected_records}
        validation_records = heapq.nsmallest(validation_quota, selected_records, key=lambda item: (item[1], item[2]))  # replay old validation split
        validation_line_ids = {line_idx for _sample, _eval, line_idx, _text in validation_records}

        for sample_score, eval_score, line_idx, text in records:
            scanned_docs += 1
            source_key = f"{language}\0{shard_rel}\0{line_idx}"
            if line_idx in selected_line_ids:  # reconstruct copied rows before appending
                doc_id = document_id(language, shard_rel, line_idx, text)
                old_doc_ids.add(doc_id)
                old_source_keys.add(source_key)
                old_text_hashes.add(text_hash(text))
                old_split_by_key[source_key] = "validation" if line_idx in validation_line_ids else "train"
                continue

            train_key = (sample_score, eval_score, shard_rel, line_idx)
            validation_key = (eval_score, sample_score, shard_rel, line_idx)  # keep validation selection independent from train order
            keep_train = accepts(train_heaps[language], train_limits[language], train_key)
            keep_validation = accepts(
                validation_heaps[language],
                validation_limits[language],
                validation_key,
            )
            if not keep_train and not keep_validation:
                continue

            candidate = {
                "language": language,
                "shard": shard_rel,
                "line_idx": line_idx,
                "text": text,
                "sample_score": sample_score,
                "eval_score": eval_score,
                "doc_id": document_id(language, shard_rel, line_idx, text),
                "text_hash": text_hash(text),
                "source_key": source_key,
            }
            if keep_train:
                add(train_heaps[language], train_limits[language], train_key, candidate)
            if keep_validation:
                add(validation_heaps[language], validation_limits[language], validation_key, candidate)

        if PROGRESS_EVERY_SHARDS > 0 and scanned_shards % PROGRESS_EVERY_SHARDS == 0:
            print(
                f"[scan] shards={scanned_shards:,}/{len(plan_rows):,} "
                f"docs={scanned_docs:,} old={len(old_doc_ids):,}",
                flush=True,
            )

    return {
        "train_candidates": {
            language: [entry[2] for entry in sorted(heap, key=lambda entry: entry[0])]
            for language, heap in train_heaps.items()
        },
        "validation_candidates": {
            language: [entry[2] for entry in sorted(heap, key=lambda entry: entry[0])]
            for language, heap in validation_heaps.items()
        },
        "old_doc_ids": old_doc_ids,
        "old_source_keys": old_source_keys,
        "old_text_hashes": old_text_hashes,
        "old_split_by_key": old_split_by_key,
        "candidate_heap_sizes": {
            "train": {language: len(heap) for language, heap in train_heaps.items()},
            "validation": {language: len(heap) for language, heap in validation_heaps.items()},
        },
        "candidate_heap_limits": {"train": train_limits, "validation": validation_limits},
        "scanned_docs": scanned_docs,
        "scanned_shards": len(plan_rows),
    }


def tokenize_chunk_counts(tokenizer: Any, texts: list[str]) -> list[int]:
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_overflowing_tokens=True,
    )
    mapping = encoded.pop("overflow_to_sample_mapping", list(range(len(texts))))  # track overflow chunks back to documents
    chunk_counts = [0 for _ in texts]
    for sample_idx in mapping:
        chunk_counts[int(sample_idx)] += 1
    return chunk_counts


def select_docs(
    tokenizer: Any,
    language: str,
    candidates: list[dict[str, Any]],
    target_chunks: int,
    split: str,
    excluded_source_keys: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    total_chunks = 0
    cursor = 0
    while cursor < len(candidates) and total_chunks < target_chunks:
        batch_candidates: list[dict[str, Any]] = []
        while cursor < len(candidates) and len(batch_candidates) < PLANNING_BATCH_SIZE:
            candidate = candidates[cursor]
            cursor += 1
            if candidate["source_key"] not in excluded_source_keys:
                batch_candidates.append(candidate)
        if not batch_candidates:
            continue

        chunk_counts = tokenize_chunk_counts(tokenizer, [candidate["text"] for candidate in batch_candidates])
        for candidate, full_chunks in zip(batch_candidates, chunk_counts):
            if total_chunks >= target_chunks:
                break
            if full_chunks <= 0:
                continue
            keep_chunks = min(int(full_chunks), target_chunks - total_chunks)  # truncate final doc to hit chunk target
            selected.append(
                {
                    **candidate,
                    "split": split,
                    "full_chunks": int(full_chunks),
                    "keep_chunks": int(keep_chunks),
                }
            )
            total_chunks += int(keep_chunks)

    if total_chunks != target_chunks:
        raise RuntimeError(
            f"Could not select enough {split} chunks for {language}: "
            f"selected={total_chunks} target={target_chunks} candidates={len(candidates)}. "
            "Increase CANDIDATE_DOC_SURPLUS or the low-resource candidate floor."
        )
    return selected


def generated_rows(
    selected_by_split_language: dict[str, dict[str, list[dict[str, Any]]]],
    languages: list[str],
    language_map: dict[str, str],
) -> Iterator[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    lang_vocab = {language: idx for idx, language in enumerate(sorted(language_map))}
    family_vocab = {family: idx for idx, family in enumerate(sorted(set(language_map.values())))}

    for split in ("validation", "train"):
        for language in languages:
            selected_docs = selected_by_split_language.get(split, {}).get(language, [])
            for start in range(0, len(selected_docs), GENERATION_BATCH_SIZE):
                batch_docs = selected_docs[start : start + GENERATION_BATCH_SIZE]
                texts = [item["text"] for item in batch_docs]
                encoded = tokenizer(
                    texts,
                    truncation=True,
                    padding="max_length",
                    max_length=MAX_LENGTH,
                    return_overflowing_tokens=True,
                )
                mapping = encoded.pop("overflow_to_sample_mapping", list(range(len(texts))))  # emit only selected chunks per document
                doc_token_totals = [0 for _ in batch_docs]
                running_chunks = [0 for _ in batch_docs]
                full_chunk_idx: list[int] = []
                for sample_idx, attention_mask in zip(mapping, encoded["attention_mask"]):
                    doc_idx = int(sample_idx)
                    full_chunk_idx.append(running_chunks[doc_idx])
                    running_chunks[doc_idx] += 1
                    doc_token_totals[doc_idx] += int(sum(attention_mask))

                emitted_by_doc = [0 for _ in batch_docs]
                for row_idx, sample_idx in enumerate(mapping):
                    doc_idx = int(sample_idx)
                    selected_doc = batch_docs[doc_idx]
                    if emitted_by_doc[doc_idx] >= selected_doc["keep_chunks"]:
                        continue

                    input_ids = [int(value) for value in encoded["input_ids"][row_idx]]
                    attention_mask = [int(value) for value in encoded["attention_mask"][row_idx]]
                    labels = [token if mask == 1 else -100 for token, mask in zip(input_ids, attention_mask)]
                    language = selected_doc["language"]
                    family = language_map.get(language)
                    lang_id = lang_vocab.get(language, LANGUAGE_PAD_ID)
                    family_id = family_vocab.get(family, LANGUAGE_PAD_ID) if family is not None else LANGUAGE_PAD_ID
                    emitted_by_doc[doc_idx] += 1
                    yield {
                        "language": language,
                        "doc_id": selected_doc["doc_id"],
                        "split": split,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                        "chunk_idx": int(full_chunk_idx[row_idx]),
                        "doc_token_count": int(doc_token_totals[doc_idx]),
                        "language_ids": int(lang_id),
                        "family_ids": int(family_id),
                    }

def main():
    for path in (OUTPUT_TOKENIZED, AUDIT_DIR):
        if path.exists():
            shutil.rmtree(path)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] base={BASE_TOKENIZED}", flush=True)
    base = load_from_disk(str(BASE_TOKENIZED))
    base_info = dataset_distribution(base, keep_doc_ids=True)  # keep doc ids so old selection can be reproduced exactly
    current_samples = base_info["samples_per_language_by_split"]
    current_docs = base_info["docs_per_language_by_split"]
    languages = sorted(set(current_samples.get("train", {})) | set(current_samples.get("validation", {})))
    current_train = int(base_info["splits"].get("train", 0))
    current_validation = int(base_info["splits"].get("validation", 0))

    target_train = TARGET_TRAIN_CHUNKS
    if target_train < current_train:
        raise RuntimeError(f"target train chunks {target_train} is below current train chunks {current_train}")

    target_validation = TARGET_VALIDATION_CHUNKS
    if target_validation <= 0:
        target_validation = round(target_train * current_validation / current_train)  # scale validation with train size
    if target_validation < current_validation:
        raise RuntimeError(f"target validation chunks {target_validation} is below current validation chunks {current_validation}")

    target_train_by_lang = apportion_counts(current_samples.get("train", {}), target_train, languages)
    target_validation_by_lang = apportion_counts(current_samples.get("validation", {}), target_validation, languages)
    append_train_targets = {
        language: target_train_by_lang.get(language, 0) - int(current_samples.get("train", {}).get(language, 0))
        for language in languages
    }
    append_validation_targets = {
        language: target_validation_by_lang.get(language, 0) - int(current_samples.get("validation", {}).get(language, 0))
        for language in languages
    }
    if any(value < 0 for value in append_train_targets.values()) or any(value < 0 for value in append_validation_targets.values()):
        raise RuntimeError("Computed negative append target; check target totals.")

    write_json(
        AUDIT_DIR / "expanded_distribution_target.json",
        {
            "target_train_chunks": target_train,
            "target_validation_chunks": target_validation,
            "current_train_chunks": current_train,
            "current_validation_chunks": current_validation,
            "append_train_chunks": sum(append_train_targets.values()),
            "append_validation_chunks": sum(append_validation_targets.values()),
            "target_train_by_language": target_train_by_lang,
            "target_validation_by_language": target_validation_by_lang,
            "append_train_by_language": append_train_targets,
            "append_validation_by_language": append_validation_targets,
        },
    )

    scan = scan_sources(languages, append_train_targets, append_validation_targets, current_samples, current_docs)  # scan local copy once
    missing_old = sorted(base_info["doc_ids"] - scan["old_doc_ids"])
    extra_old = sorted(scan["old_doc_ids"] - base_info["doc_ids"])
    if missing_old or extra_old:
        write_json(
            AUDIT_DIR / "old_doc_id_reproduction_failure.json",
            {
                "missing_from_reconstructed_old": missing_old[:100],
                "extra_in_reconstructed_old": extra_old[:100],
                "missing_count": len(missing_old),
                "extra_count": len(extra_old),
            },
        )
        raise RuntimeError(
            "Local source did not reproduce the copied dataset doc_ids. "
            f"missing={len(missing_old)} extra={len(extra_old)}"
        )
    print("[verify] old doc IDs reproduced exactly", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    selected_by_split_language: dict[str, dict[str, list[dict[str, Any]]]] = {"validation": {}, "train": {}}
    selected_validation_keys: set[str] = set()
    selected_train_keys: set[str] = set()
    selected_text_hash_overlaps = {"train": 0, "validation": 0}

    for language in languages:  # pick validation first so train cannot reuse validation documents
        selected = select_docs(
            tokenizer,
            language,
            scan["validation_candidates"].get(language, []),
            int(append_validation_targets.get(language, 0)),
            "validation",
            set(),
        )
        selected_by_split_language["validation"][language] = selected
        for item in selected:
            selected_validation_keys.add(item["source_key"])
            if item["text_hash"] in scan["old_text_hashes"]:
                selected_text_hash_overlaps["validation"] += 1

    for language in languages:
        selected = select_docs(
            tokenizer,
            language,
            scan["train_candidates"].get(language, []),
            int(append_train_targets.get(language, 0)),
            "train",
            selected_validation_keys,
        )
        selected_by_split_language["train"][language] = selected
        for item in selected:
            selected_train_keys.add(item["source_key"])
            if item["text_hash"] in scan["old_text_hashes"]:
                selected_text_hash_overlaps["train"] += 1

    old_overlap = (selected_train_keys | selected_validation_keys) & scan["old_source_keys"]  # ensure appended rows are new
    train_validation_overlap = selected_train_keys & selected_validation_keys
    if old_overlap or train_validation_overlap:
        raise RuntimeError(
            f"Newness violation: old_overlap={len(old_overlap)} "
            f"train_val_overlap={len(train_validation_overlap)}"
        )

    selected_docs = [
        item
        for split in ("validation", "train")
        for language in languages
        for item in selected_by_split_language[split].get(language, [])
    ]
    write_jsonl(
        AUDIT_DIR / "append_selected_manifest.jsonl",
        (
            {
                "language": item["language"],
                "split": item["split"],
                "shard": item["shard"],
                "line_idx": item["line_idx"],
                "doc_id": item["doc_id"],
                "text_hash": item["text_hash"],
                "sample_score": item["sample_score"],
                "eval_score": item["eval_score"],
                "full_chunks": item["full_chunks"],
                "kept_chunks": item["keep_chunks"],
                "partial_doc": item["keep_chunks"] != item["full_chunks"],
            }
            for item in selected_docs
        ),
    )
    write_jsonl(
        AUDIT_DIR / "old_selected_manifest.jsonl",
        (
            {"source_key": key, "split": scan["old_split_by_key"][key]}
            for key in sorted(scan["old_source_keys"])
        ),
    )
    write_json(
        AUDIT_DIR / "newness_validation.json",
        {
            "old_source_keys": len(scan["old_source_keys"]),
            "new_train_source_keys": len(selected_train_keys),
            "new_validation_source_keys": len(selected_validation_keys),
            "new_vs_old_source_key_overlap": len(old_overlap),
            "new_train_vs_validation_source_key_overlap": len(train_validation_overlap),
            "text_hash_overlaps_with_old": selected_text_hash_overlaps,
            "candidate_heap_sizes": scan["candidate_heap_sizes"],
            "candidate_heap_limits": scan["candidate_heap_limits"],
            "scanned_docs": scan["scanned_docs"],
            "scanned_shards": scan["scanned_shards"],
        },
    )

    print("[build] generating appended tokenized dataset", flush=True)
    append_dataset = Dataset.from_generator(
        generated_rows,
        gen_kwargs={
            "selected_by_split_language": selected_by_split_language,
            "languages": languages,
            "language_map": load_language_map(LANGUAGE_MAP),
        },
        features=base.features,
        cache_dir=str(AUDIT_DIR / "hf_generator_cache"),
    )
    print(f"[build] append rows={len(append_dataset):,}", flush=True)

    final_dataset = concatenate_datasets([base, append_dataset])
    final_dataset.save_to_disk(str(OUTPUT_TOKENIZED))
    print(f"[save] {OUTPUT_TOKENIZED}", flush=True)

    actual = dataset_distribution(final_dataset)  # validate saved chunk counts before accepting the output
    write_json(AUDIT_DIR / "expanded_distribution_actual.json", actual)
    if actual["splits"].get("train") != target_train or actual["splits"].get("validation") != target_validation:
        raise RuntimeError(f"Final split counts do not match target: {actual['splits']}")
    if strip_zero_counts(actual["samples_per_language_by_split"].get("train", {})) != strip_zero_counts(target_train_by_lang):
        raise RuntimeError("Final train language distribution does not match target.")
    if strip_zero_counts(actual["samples_per_language_by_split"].get("validation", {})) != strip_zero_counts(
        target_validation_by_lang
    ):
        raise RuntimeError("Final validation language distribution does not match target.")
    if actual["duplicate_doc_chunk_keys"] != 0:
        raise RuntimeError(f"Duplicate (doc_id, chunk_idx) pairs found: {actual['duplicate_doc_chunk_keys']}")

    write_json(
        AUDIT_DIR / "run_config.json",
        {
            "base_tokenized": str(BASE_TOKENIZED),
            "base_shard_plan": str(BASE_SHARD_PLAN),
            "source_shard_root": str(SOURCE_SHARD_ROOT),
            "original_source_root": str(ORIGINAL_SOURCE_ROOT),
            "language_map": str(LANGUAGE_MAP),
            "tokenizer": str(TOKENIZER_PATH),
            "resolved_tokenizer_path": str(TOKENIZER_PATH),
            "output_tokenized": str(OUTPUT_TOKENIZED),
            "audit_dir": str(AUDIT_DIR),
            "target_train_chunks": TARGET_TRAIN_CHUNKS,
            "target_validation_chunks": TARGET_VALIDATION_CHUNKS,
            "sample_seed": SAMPLE_SEED,
            "eval_seed": EVAL_SEED,
            "max_length": MAX_LENGTH,
            "candidate_doc_surplus": CANDIDATE_DOC_SURPLUS,
            "planning_batch_size": PLANNING_BATCH_SIZE,
            "generation_batch_size": GENERATION_BATCH_SIZE,
        },
    )
    print("[done] expanded dataset validated", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
