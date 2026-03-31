from __future__ import annotations

import csv
import json
import statistics

from .runtime import (
    _load_json,
    _missing_modules,
    _run_stage_cmd,
    _save_stage_state,
    _should_skip_stage,
    _stage_signature,
)
from ..checks import check_tokenize
from ..context import PipelineContext


def _resolve_languages_for_tokenize(ctx: PipelineContext) -> list[str]:
    token_cfg = ctx.tokenize_cfg
    explicit = token_cfg.get("languages")
    if isinstance(explicit, list) and explicit:
        base = [str(item).strip() for item in explicit if str(item).strip()]
        extras = _resolve_language_extras(token_cfg)
        return sorted(set(base + extras))

    csv_path = str(token_cfg.get("languages_csv", "")).strip()
    if csv_path:
        column = str(token_cfg.get("languages_csv_column", "subset")).strip() or "subset"
        split_value = str(token_cfg.get("languages_csv_split", "train")).strip()
        split_column = str(token_cfg.get("languages_csv_split_column", "split")).strip() or "split"
        languages: set[str] = set()
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise SystemExit(f"Tokenize languages CSV is missing a header: {csv_path}")
            if column not in reader.fieldnames:
                raise SystemExit(f"Tokenize languages CSV missing column '{column}'. Found: {reader.fieldnames}")
            if split_value and split_column not in reader.fieldnames:
                raise SystemExit(f"Tokenize languages CSV missing split column '{split_column}'. Found: {reader.fieldnames}")
            for row in reader:
                if split_value:
                    if str(row.get(split_column, "")).strip() != split_value:
                        continue
                value = str(row.get(column, "")).strip()
                if value:
                    languages.add(value)
        if not languages:
            raise SystemExit(f"No languages resolved from {csv_path} (column={column}, split={split_value})")
        extras = _resolve_language_extras(token_cfg)
        return sorted(languages.union(extras))

    explicit = ctx.sample_cfg.get("languages")
    if isinstance(explicit, list) and explicit:
        base = [str(item).strip() for item in explicit if str(item).strip()]
        extras = _resolve_language_extras(token_cfg)
        return sorted(set(base + extras))

    return []


def _resolve_language_extras(token_cfg: dict) -> list[str]:
    extras = token_cfg.get("languages_extra")
    if extras is None:
        return []
    if isinstance(extras, str):
        return [item.strip() for item in extras.split(",") if item.strip()]
    if isinstance(extras, list):
        return [str(item).strip() for item in extras if str(item).strip()]
    return []


def _build_language_rates_for_tokenize(ctx: PipelineContext, languages: list[str]) -> Path | None:
    token_cfg = ctx.tokenize_cfg
    csv_path = str(token_cfg.get("language_rates_csv", "")).strip()
    if not csv_path:
        return None
    subset_column = str(token_cfg.get("language_rates_csv_column", "subset")).strip() or "subset"
    split_value = str(token_cfg.get("language_rates_csv_split", "train")).strip()
    split_column = str(token_cfg.get("language_rates_csv_split_column", "split")).strip() or "split"
    docs_column = str(token_cfg.get("language_rates_csv_documents_column", "documents")).strip() or "documents"
    english_key = str(token_cfg.get("language_rates_english_key", "english")).strip() or "english"
    english_override = token_cfg.get("language_rates_english_docs_override")
    english_docs_count = token_cfg.get("language_rates_english_docs_count")
    english_divisor = token_cfg.get("language_rates_english_undersample_divisor", 1)
    counts_tsv = str(token_cfg.get("language_rates_counts_tsv", "")).strip()
    counts_lang_col = str(token_cfg.get("language_rates_counts_tsv_language_column", "language")).strip() or "language"
    counts_val_col = str(token_cfg.get("language_rates_counts_tsv_count_column", "count")).strip() or "count"
    counts_has_header = bool(token_cfg.get("language_rates_counts_tsv_has_header", False))
    require_positive = bool(token_cfg.get("language_rates_require_positive_counts", True))

    if english_key not in languages:
        return None

    docs: dict[str, float] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"language_rates_csv is missing a header: {csv_path}")
        if subset_column not in reader.fieldnames:
            raise SystemExit(f"language_rates_csv missing column '{subset_column}'. Found: {reader.fieldnames}")
        if docs_column not in reader.fieldnames:
            raise SystemExit(f"language_rates_csv missing column '{docs_column}'. Found: {reader.fieldnames}")
        if split_value and split_column not in reader.fieldnames:
            raise SystemExit(f"language_rates_csv missing split column '{split_column}'. Found: {reader.fieldnames}")

        for row in reader:
            if split_value and str(row.get(split_column, "")).strip() != split_value:
                continue
            subset = str(row.get(subset_column, "")).strip()
            if not subset:
                continue
            raw = row.get(docs_column, 0)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = 0.0
            docs[subset] = max(0.0, value)

    counts: dict[str, float] = {}
    if counts_tsv:
        with open(counts_tsv, newline="", encoding="utf-8") as handle:
            if counts_has_header:
                reader = csv.DictReader(handle, delimiter="\t")
                if not reader.fieldnames:
                    raise SystemExit(f"language_rates_counts_tsv is missing a header: {counts_tsv}")
                if counts_lang_col not in reader.fieldnames or counts_val_col not in reader.fieldnames:
                    raise SystemExit(
                        f"language_rates_counts_tsv missing columns '{counts_lang_col}', '{counts_val_col}'. "
                        f"Found: {reader.fieldnames}"
                    )
                for row in reader:
                    lang = str(row.get(counts_lang_col, "")).strip()
                    if not lang:
                        continue
                    raw = row.get(counts_val_col, 0)
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        value = 0.0
                    counts[lang] = max(0.0, value)
            else:
                for line in handle:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    lang = parts[0].strip()
                    if not lang:
                        continue
                    try:
                        value = float(parts[1])
                    except (TypeError, ValueError):
                        value = 0.0
                    counts[lang] = max(0.0, value)

    if english_override is not None:
        try:
            target_map = counts if counts_tsv else docs
            target_map[english_key] = float(english_override)
        except (TypeError, ValueError):
            raise SystemExit(f"language_rates_english_docs_override must be numeric, got {english_override}")
    elif english_key not in docs:
        if english_docs_count is None and not counts_tsv:
            raise SystemExit(f"language_rates_csv missing '{english_key}' in column '{subset_column}': {csv_path}")
        try:
            base = float(english_docs_count)
        except (TypeError, ValueError):
            raise SystemExit(f"language_rates_english_docs_count must be numeric, got {english_docs_count}")
        try:
            divisor = float(english_divisor) if english_divisor is not None else 1.0
        except (TypeError, ValueError):
            divisor = 1.0
        if divisor <= 0:
            divisor = 1.0
        if not counts_tsv:
            docs[english_key] = base / divisor

    if counts_tsv:
        missing_counts = [lang for lang in languages if lang not in counts]
        if missing_counts:
            raise SystemExit(f"language_rates_counts_tsv missing languages: {missing_counts}")
        if require_positive:
            non_positive = [lang for lang in languages if counts.get(lang, 0.0) <= 0.0]
            if non_positive:
                raise SystemExit(f"language_rates_counts_tsv has non-positive counts for languages: {non_positive}")
        other_docs = [counts.get(lang, 0.0) for lang in languages if lang != english_key and counts.get(lang, 0.0) > 0.0]
        if not other_docs:
            raise SystemExit(f"language_rates_counts_tsv has no positive counts for non-english languages: {counts_tsv}")
        median_docs = float(statistics.median(other_docs))
        english_docs = float(counts.get(english_key, 0.0))
    else:
        other_docs = [docs.get(lang, 0.0) for lang in languages if lang != english_key and docs.get(lang, 0.0) > 0.0]
        if not other_docs:
            raise SystemExit(f"language_rates_csv has no positive docs for non-english languages: {csv_path}")
        median_docs = float(statistics.median(other_docs))
        english_docs = float(docs.get(english_key, 0.0))
    if english_docs <= 0:
        source = counts_tsv if counts_tsv else csv_path
        raise SystemExit(f"language_rates source has non-positive docs for '{english_key}': {source}")

    base_rate = float(token_cfg.get("sample_fraction", 1.0))
    english_rate = base_rate * median_docs / english_docs
    english_rate = max(0.0, min(1.0, english_rate))

    rates = {english_key: english_rate}
    output_path = ctx.output_root / "tokenization_language_rates.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rates, indent=2) + "\n", encoding="utf-8")
    return output_path


def stage_tokenize(ctx: PipelineContext) -> None:
    if "tokenize" in ctx.active_stages and bool(ctx.tokenize_cfg.get("enabled", True)):
        missing = _missing_modules(["datasets", "transformers"])
        if missing:
            raise SystemExit(f"Missing Python modules for tokenize stage: {missing}")
        tokenize_model_name = str(ctx.tokenize_cfg["model_name"])
        if bool(ctx.tokenizer_extension_cfg.get("enabled", False)) and ctx.extension_manifest_path.exists():
            extension_manifest = _load_json(ctx.extension_manifest_path)
            tokenize_model_name = str(extension_manifest.get("extended_tokenizer_dir", tokenize_model_name))
        signature = _stage_signature(
            {
                "stage": "tokenize",
                "cfg": ctx.tokenize_cfg,
                "sample_manifest": str(ctx.sampled_manifest),
                "tokenizer_model_name": tokenize_model_name,
            }
        )
        artifacts = [ctx.tokenized_dir / "dataset_info.json", ctx.tokenized_dir / "state.json"]
        if _should_skip_stage(
            state_path=ctx.state_path,
            stage="tokenize",
            signature=signature,
            expected_artifacts=artifacts,
            force_stages=ctx.force_stages,
            dry_run=ctx.args.dry_run,
        ):
            ctx.logger.info("[SKIP] tokenize stage (cache hit)")
        else:
            languages = _resolve_languages_for_tokenize(ctx)
            rates_path = _build_language_rates_for_tokenize(ctx, languages)
            cmd = [
                "python3",
                str(ctx.repo_root / "data_prep" / "merlin_data_prep" / "distributed_data_processor" / "src" / "tokenize_slurm.py"),
                "--shard_dir",
                str(ctx.sample_shard_dir),
                "--save_tokenized_data_dir",
                str(ctx.tokenized_dir),
                "--model_name",
                tokenize_model_name,
                "--num_proc",
                str(int(ctx.tokenize_cfg.get("num_proc", 1))),
                "--eval_fraction",
                str(float(ctx.tokenize_cfg.get("eval_fraction", 0.02))),
                "--eval_seed",
                str(int(ctx.tokenize_cfg.get("eval_seed", 42))),
                "--sample_fraction",
                str(float(ctx.tokenize_cfg.get("sample_fraction", 1.0))),
                "--sample_seed",
                str(int(ctx.tokenize_cfg.get("sample_seed", 42))),
                "--max_length",
                str(int(ctx.tokenize_cfg.get("max_length", 1024))),
                "--language-map",
                str(ctx.language_map_path),
                "--mask_pad_labels",
                "true" if bool(ctx.tokenize_cfg.get("mask_pad_labels", True)) else "false",
            ]
            if languages:
                cmd.extend(["--languages", *languages])
            if rates_path is not None:
                cmd.extend(["--language_rates_json", str(rates_path)])
            max_samples = int(ctx.tokenize_cfg.get("max_samples", 0))
            if max_samples > 0:
                cmd.extend(["--max_samples", str(max_samples)])
            _run_stage_cmd(cmd=cmd, env=ctx.env, dry_run=ctx.args.dry_run, logs_root=ctx.logs_root, stage="tokenize", executor=ctx.executor)
            if not ctx.args.dry_run:
                _save_stage_state(ctx.state_path, "tokenize", signature, [str(ctx.tokenized_dir)])
            check_tokenize(
                ctx.tokenized_dir,
                dry_run=ctx.args.dry_run,
                expected_languages=languages,
                english_median_check=bool(ctx.tokenize_cfg.get("english_median_check", False)),
                english_key=str(ctx.tokenize_cfg.get("language_rates_english_key", "english")).strip() or "english",
                english_median_tolerance=float(ctx.tokenize_cfg.get("english_median_tolerance", 0.05)),
            )
