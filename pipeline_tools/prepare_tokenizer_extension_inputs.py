#!/usr/bin/env python3
"""Prepare per-language .jsonl.gz shards for tokenizer extension stages."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _resolve_language(record: dict[str, Any], path: Path, shard_dir: Path) -> str | None:
    for key in ("language", "lang", "subset"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    try:
        rel = path.relative_to(shard_dir)
    except ValueError:
        return None
    if rel.parts:
        return rel.parts[0]
    return None


def _resolve_text(record: dict[str, Any]) -> str | None:
    value = record.get("text")
    if isinstance(value, str) and value.strip():
        return value
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build tokenizer-extension jsonl.gz language shards.")
    parser.add_argument("--shard-dir", required=True, help="Input sampled shard directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for <lang>.jsonl.gz files.")
    parser.add_argument("--languages", default="", help="Optional comma-separated language allowlist.")
    parser.add_argument("--manifest-path", required=True, help="Output JSON manifest path.")
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    allow = {item.strip() for item in args.languages.split(",") if item.strip()}

    shard_paths = sorted([*shard_dir.rglob("*.jsonl"), *shard_dir.rglob("*.jsonl.gz")])
    if not shard_paths:
        raise SystemExit(f"No JSONL shards found under {shard_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    output_files: dict[str, str] = {}
    total_rows = 0

    with ExitStack() as stack:
        writers: dict[str, Any] = {}
        for shard_path in shard_paths:
            for record in _iter_jsonl(shard_path):
                language = _resolve_language(record, shard_path, shard_dir)
                if not language:
                    continue
                if allow and language not in allow:
                    continue
                text = _resolve_text(record)
                if text is None:
                    continue

                writer = writers.get(language)
                if writer is None:
                    out_path = output_dir / f"{language}.jsonl.gz"
                    writer = stack.enter_context(gzip.open(out_path, "wt", encoding="utf-8"))
                    writers[language] = writer
                    output_files[language] = str(out_path)
                    counts[language] = 0

                writer.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                counts[language] += 1
                total_rows += 1

    if not counts:
        raise SystemExit("No language rows found for tokenizer extension input.")

    payload = {
        "shard_dir": str(shard_dir),
        "output_dir": str(output_dir),
        "num_input_shards": len(shard_paths),
        "num_languages": len(counts),
        "total_rows": total_rows,
        "rows_per_language": counts,
        "output_files": output_files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[OK] prepared_languages={len(counts)} total_rows={total_rows}")
    print(f"[OK] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
