#!/usr/bin/env python3
"""Build tiny multilingual JSONL shards for local E2E validation."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path


def _load_languages_from_map(language_map_path: Path) -> list[str]:
    payload = json.loads(language_map_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {language_map_path}")
    languages: list[str] = []
    for value in payload.values():
        if isinstance(value, dict):
            items = value.get("languages")
            if isinstance(items, list):
                for lang in items:
                    if isinstance(lang, str):
                        languages.append(lang)
    unique = sorted(set(languages))
    if not unique:
        raise ValueError(f"No languages found in {language_map_path}")
    return unique


def _resolve_languages(
    language_map_path: Path,
    explicit: str,
    max_languages: int,
    seed: int,
) -> list[str]:
    all_languages = _load_languages_from_map(language_map_path)
    if explicit.strip():
        selected = [item.strip() for item in explicit.split(",") if item.strip()]
        missing = sorted(set(selected) - set(all_languages))
        if missing:
            raise ValueError(f"Unknown languages not present in map: {missing}")
        return selected
    if max_languages <= 0 or max_languages >= len(all_languages):
        return all_languages
    rng = random.Random(seed)
    picked = rng.sample(all_languages, max_languages)
    return sorted(picked)


def _build_counts(
    languages: list[str],
    total_samples: int,
    strategy: str,
    alpha: float,
    min_per_language: int,
    max_per_language: int,
) -> dict[str, int]:
    if total_samples < len(languages):
        raise ValueError("total_samples must be >= number of selected languages")

    if strategy == "uniform":
        weights = {lang: 1.0 for lang in languages}
    elif strategy == "alpha_smoothing":
        # Simulate long-tail source counts and apply alpha smoothing.
        base = {lang: float(len(languages) - idx) for idx, lang in enumerate(sorted(languages))}
        weights = {lang: math.pow(max(base[lang], 1.0), alpha) for lang in languages}
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    weight_sum = sum(weights.values())
    raw = {lang: (weights[lang] / weight_sum) * total_samples for lang in languages}
    counts = {lang: max(min_per_language, int(raw[lang])) for lang in languages}

    # Distribute remainder deterministically.
    remaining = total_samples - sum(counts.values())
    if remaining > 0:
        order = sorted(languages, key=lambda lang: raw[lang] - int(raw[lang]), reverse=True)
        idx = 0
        while remaining > 0:
            lang = order[idx % len(order)]
            if counts[lang] < max_per_language:
                counts[lang] += 1
                remaining -= 1
            idx += 1
            if idx > total_samples * 4 and remaining > 0:
                raise ValueError("Unable to allocate remaining samples under max_per_language")
    elif remaining < 0:
        # Trim extras while preserving minimum constraints.
        overflow = -remaining
        order = sorted(languages, key=lambda lang: raw[lang] - int(raw[lang]))
        idx = 0
        while overflow > 0:
            lang = order[idx % len(order)]
            if counts[lang] > min_per_language:
                counts[lang] -= 1
                overflow -= 1
            idx += 1
            if idx > total_samples * 4 and overflow > 0:
                raise ValueError("Unable to trim sample overflow under min_per_language")

    for lang in languages:
        counts[lang] = max(min_per_language, min(max_per_language, counts[lang]))

    diff = total_samples - sum(counts.values())
    if diff != 0:
        # Final fix-up in uniform round-robin.
        direction = 1 if diff > 0 else -1
        order = sorted(languages)
        idx = 0
        while diff != 0:
            lang = order[idx % len(order)]
            next_val = counts[lang] + direction
            if min_per_language <= next_val <= max_per_language:
                counts[lang] = next_val
                diff -= direction
            idx += 1
            if idx > total_samples * 8 and diff != 0:
                raise ValueError("Could not finalize sample counts")

    return counts


def _make_line(language: str, sample_idx: int) -> str:
    length_bucket = 4 + (sample_idx % 24)
    repeated = " ".join(["tok"] * length_bucket)
    sample = {
        "language": language,
        "text": (
            f"Synthetic training sample {sample_idx} for {language}. "
            f"This sentence is generated for E2E smoke validation. {repeated}"
        ),
    }
    return json.dumps(sample, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build synthetic multilingual JSONL shards.")
    parser.add_argument("--language-map", required=True, help="Path to language grouping JSON.")
    parser.add_argument("--output-dir", required=True, help="Output directory for shard files.")
    parser.add_argument("--languages", default="", help="Comma-separated explicit language list.")
    parser.add_argument("--max-languages", type=int, default=0, help="Sample this many languages from map.")
    parser.add_argument("--total-samples", type=int, default=96, help="Total synthetic records.")
    parser.add_argument("--sampling-strategy", default="uniform", choices=["uniform", "alpha_smoothing"])
    parser.add_argument("--alpha", type=float, default=0.3, help="Alpha used for alpha_smoothing policy.")
    parser.add_argument("--min-samples-per-language", type=int, default=4)
    parser.add_argument("--max-samples-per-language", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    languages = _resolve_languages(
        language_map_path=Path(args.language_map),
        explicit=args.languages,
        max_languages=args.max_languages,
        seed=args.seed,
    )

    counts = _build_counts(
        languages=languages,
        total_samples=args.total_samples,
        strategy=args.sampling_strategy,
        alpha=args.alpha,
        min_per_language=args.min_samples_per_language,
        max_per_language=args.max_samples_per_language,
    )

    files: list[str] = []
    for lang in sorted(languages):
        lang_dir = output_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        shard_path = lang_dir / "part_000.jsonl"
        files.append(str(shard_path.resolve()))
        with shard_path.open("w", encoding="utf-8") as handle:
            for idx in range(counts[lang]):
                handle.write(_make_line(lang, idx) + "\n")

    manifest = {
        "language_map": str(Path(args.language_map).resolve()),
        "languages": sorted(languages),
        "counts": counts,
        "total_samples": sum(counts.values()),
        "data_files": files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[OK] Wrote synthetic shards to {output_dir}")
    print(f"[OK] Languages: {len(languages)} | Samples: {manifest['total_samples']}")
    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] DATA_FILES_CSV={','.join(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
