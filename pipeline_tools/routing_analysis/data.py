from __future__ import annotations

import gzip
import hashlib
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


def load_language_texts_with_provenance(
    sample_root: Path,
    language: str,
    *,
    text_field: str,
    limit: int,
) -> tuple[list[str], dict[str, object]]:
    texts: list[str] = []
    files: list[dict[str, object]] = []
    digest = hashlib.sha256()
    for path in iter_language_jsonl_files(sample_root, language):
        accepted = 0
        with _open_text(path) as handle:
            for line in handle:
                if len(texts) >= limit:
                    break
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
                value = str(text)
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
                texts.append(value)
                accepted += 1
        if accepted:
            files.append({"path": str(path.resolve()), "accepted_records": accepted})
        if len(texts) >= limit:
            break
    return texts, {
        "language": language,
        "documents": len(texts),
        "text_sha256": digest.hexdigest(),
        "files": files,
    }


def load_language_texts(sample_root: Path, language: str, *, text_field: str, limit: int) -> list[str]:
    texts, _ = load_language_texts_with_provenance(
        sample_root,
        language,
        text_field=text_field,
        limit=limit,
    )
    return texts


def load_sample_manifest(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("languages"), dict):
        raise ValueError(f"Invalid routing sample manifest: {path}")
    return payload


def validate_language_sample(
    *,
    provenance: dict[str, object],
    manifest: dict[str, object],
    expected_documents: int,
    strict: bool,
) -> None:
    language = str(provenance["language"])
    documents = int(provenance["documents"])
    if strict and documents != expected_documents:
        raise ValueError(f"{language}: expected {expected_documents} valid documents, found {documents}")
    if not manifest:
        return
    entries = manifest.get("languages", {})
    expected = entries.get(language) if isinstance(entries, dict) else None
    if not isinstance(expected, dict):
        raise ValueError(f"{language}: missing from routing sample manifest")
    for key in ("documents", "text_sha256"):
        if provenance.get(key) != expected.get(key):
            raise ValueError(
                f"{language}: sample manifest mismatch for {key}: "
                f"observed={provenance.get(key)!r}, expected={expected.get(key)!r}"
            )
