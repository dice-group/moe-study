from __future__ import annotations

import gzip
import json
from pathlib import Path


def iter_language_jsonl_files(sample_root: Path, language: str) -> list[Path]:
    language_dir = sample_root / language
    if not language_dir.exists() or not language_dir.is_dir():
        return []
    return sorted(language_dir.glob("*.jsonl")) + sorted(language_dir.glob("*.jsonl.gz"))


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_language_texts(sample_root: Path, language: str, *, text_field: str, limit: int) -> list[str]:
    texts: list[str] = []
    for path in iter_language_jsonl_files(sample_root, language):
        with _open_text(path) as handle:
            for line in handle:
                if len(texts) >= limit:
                    return texts
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                text = payload.get(text_field) or payload.get("text") or payload.get("content")
                if not text:
                    continue
                texts.append(str(text))
    return texts
