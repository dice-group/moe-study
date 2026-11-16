from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import load_language_texts_with_provenance
from .io import atomic_write_json


def build_sample_manifest(
    *,
    sample_root: Path,
    languages: list[str],
    text_field: str,
    documents_per_language: int,
    output_path: Path,
) -> dict[str, Any]:
    entries: dict[str, dict[str, object]] = {}
    shortfalls: dict[str, int] = {}
    for index, language in enumerate(languages, start=1):
        _, provenance = load_language_texts_with_provenance(
            sample_root,
            language,
            text_field=text_field,
            limit=documents_per_language,
        )
        entries[language] = provenance
        observed = int(provenance["documents"])
        if observed != documents_per_language:
            shortfalls[language] = observed
        print(f"[{index}/{len(languages)}] {language}: {observed} documents")
    if shortfalls:
        raise ValueError(
            f"Cannot freeze {documents_per_language} documents for every routing id; shortfalls={shortfalls}"
        )

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "sorted_files_then_source_record_order",
        "sample_root": str(sample_root.resolve()),
        "text_field": text_field,
        "documents_per_language": int(documents_per_language),
        "num_routing_ids": len(languages),
        "routing_ids": languages,
        "languages": entries,
    }
    atomic_write_json(output_path, payload)
    return payload
