import collections
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def extract_text(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        row = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if isinstance(row, dict) and isinstance(row.get("text"), str) and row["text"].strip():
        return row["text"]
    return stripped


def load_language_map(path: Path) -> dict[str, str]:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Language map must be a JSON object: {path}")

    mapping: dict[str, str] = {}
    for group_id, entry in raw.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("languages"), list):
            raise RuntimeError(f"Unexpected language-map entry {group_id!r}: {entry!r}")
        group = str(entry.get("group") or group_id)
        for language in entry["languages"]:
            mapping[str(language)] = group
    if not mapping:
        raise RuntimeError(f"Language map is empty: {path}")
    return mapping


def dataset_distribution(
    dataset: Dataset,
    *,
    keep_doc_ids: bool = False,
    include_doc_counts: bool = True,
) -> dict[str, Any]:
    samples: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    docs: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    doc_ids: set[str] = set()
    doc_chunk_keys: set[tuple[str, int]] = set()

    columns = [name for name in ("language", "doc_id", "split", "chunk_idx") if name in dataset.column_names]
    metadata = dataset.select_columns(columns)
    for batch in metadata.iter(batch_size=10_000):
        size = len(batch["language"])
        splits = batch.get("split") or ["train"] * size
        chunk_indices = batch.get("chunk_idx") or [0] * size
        for idx in range(size):
            split = str(splits[idx])
            language = str(batch["language"][idx])
            doc_id = str(batch["doc_id"][idx])
            chunk_idx = int(chunk_indices[idx])
            samples[split][language] += 1
            if include_doc_counts and chunk_idx == 0:
                docs[split][language] += 1
            if keep_doc_ids:
                doc_ids.add(doc_id)
            doc_chunk_keys.add((doc_id, chunk_idx))

    summary: dict[str, Any] = {
        "rows": len(dataset),
        "splits": {split: sum(counts.values()) for split, counts in samples.items()},
        "samples_per_language_by_split": {split: dict(counts) for split, counts in samples.items()},
        "unique_doc_ids": len(doc_ids),
        "unique_doc_chunk_keys": len(doc_chunk_keys),
        "duplicate_doc_chunk_keys": len(dataset) - len(doc_chunk_keys),
    }
    if include_doc_counts:
        summary["docs_per_language_by_split"] = {split: dict(counts) for split, counts in docs.items()}
    if keep_doc_ids:
        summary["doc_ids"] = doc_ids
    return summary


def strip_zero_counts(counts: dict[str, int]) -> dict[str, int]:
    return {key: int(value) for key, value in counts.items() if int(value) != 0}
