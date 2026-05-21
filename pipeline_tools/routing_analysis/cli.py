from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .collectors import RoutingCollector, build_matrices
from .config import (
    AnalysisConfig,
    detect_method,
    flatten_language_map,
    language_ids_from_map,
    load_config,
    read_adapter_config,
    validate_config,
)
from .data import iter_language_jsonl_files, load_language_texts
from .plotting import aggregate_by_group, create_entropy_plot, create_heatmap, create_mola_null_plot


LOGGER = logging.getLogger("routing_analysis")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze routing behavior for supported PEFT adapter checkpoints.")
    parser.add_argument("--config", required=True, type=Path, help="Routing analysis JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and planned inputs without loading models.")
    parser.add_argument("--run-id", action="append", default=[], help="Limit to one or more configured run ids.")
    parser.add_argument("--skip-figures", action="store_true", help="Write NPZ/metadata only.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def ensure_local_imports(repo_root: Path) -> None:
    llama_src = repo_root / "LLaMA-Factory" / "src"
    for path in (repo_root, llama_src):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    for name in list(sys.modules):
        if name == "peft" or name.startswith("peft."):
            del sys.modules[name]


def torch_dtype(name: str):
    import torch

    value = name.lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {name}")


def load_model(config: AnalysisConfig, adapter_checkpoint: Path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device_map: str | dict[str, str] = config.device
    if config.device.startswith("cuda") and config.device not in {"auto", "balanced", "balanced_low_0", "sequential"}:
        device = 0 if config.device == "cuda" else config.device
        device_map = {"": device}
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch_dtype(config.torch_dtype),
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(model, str(adapter_checkpoint))
    model.eval()
    return model, tokenizer


def model_device(model: Any):
    return next(model.parameters()).device


def selected_adapters(config: AnalysisConfig, run_ids: list[str]):
    if not run_ids:
        return config.adapters
    wanted = set(run_ids)
    selected = [item for item in config.adapters if item.run_id in wanted]
    missing = sorted(wanted - {item.run_id for item in selected})
    if missing:
        raise SystemExit(f"Unknown --run-id values: {missing}")
    return selected


def dry_run(config: AnalysisConfig, run_ids: list[str]) -> int:
    summaries = validate_config(config)
    selected = {item.run_id for item in selected_adapters(config, run_ids)}
    language_dirs = {lang: bool(iter_language_jsonl_files(config.sample_root, lang)) for lang in config.languages}
    missing_languages = [lang for lang, present in language_dirs.items() if not present]
    print(f"[OK] config valid: {len(config.languages)} languages, {len(summaries)} adapters")
    print(f"[INFO] output_root={config.output_root}")
    if missing_languages:
        preview = ", ".join(missing_languages[:20])
        print(f"[WARN] {len(missing_languages)} languages have no sample files under {config.sample_root}: {preview}")
    for summary in summaries:
        if summary["run_id"] not in selected:
            continue
        adapter_cfg = read_adapter_config(Path(summary["checkpoint"]))
        method = detect_method(adapter_cfg)
        axis = RoutingCollector(method, adapter_cfg).axis_name
        print(f"[PLAN] {summary['run_id']}: peft_type={method} axis={axis} checkpoint={summary['checkpoint']}")
    return 0


def language_id_map(config: AnalysisConfig) -> dict[str, int]:
    language_map = flatten_language_map(config.language_map)
    if language_map:
        return language_ids_from_map(language_map)
    return {language: idx for idx, language in enumerate(sorted(config.languages))}


def batched(items: list[str], size: int):
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda" in text and "out of memory" in text


def run_language_batches(
    *,
    model: Any,
    tokenizer: Any,
    texts: list[str],
    config: AnalysisConfig,
    batch_size: int,
    language_id: int,
    run_id: str,
    language: str,
) -> int:
    import torch

    ordered_texts = sorted(texts, key=len) if config.length_sort_batches else texts
    device = model_device(model)
    effective_batch_size = max(1, int(batch_size))
    index = 0
    while index < len(ordered_texts):
        batch_texts = ordered_texts[index : index + effective_batch_size]
        try:
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=config.max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            encoded["logits_to_keep"] = 1
            if config.use_language_ids:
                encoded["language_ids"] = torch.full(
                    (len(batch_texts),),
                    int(language_id),
                    dtype=torch.long,
                    device=device,
                )
            with torch.no_grad():
                _ = model(**encoded)
            index += len(batch_texts)
        except RuntimeError as exc:
            if not config.auto_find_batch_size or effective_batch_size <= 1 or not is_cuda_oom(exc):
                raise
            new_batch_size = max(1, effective_batch_size // 2)
            LOGGER.warning(
                "CUDA OOM for %s/%s at batch_size=%d; retrying with batch_size=%d",
                run_id,
                language,
                effective_batch_size,
                new_batch_size,
            )
            effective_batch_size = new_batch_size
            torch.cuda.empty_cache()
    return effective_batch_size


def save_npz(output_path: Path, payload: dict[str, Any], *, languages: list[str], run_id: str, peft_type: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_payload = dict(payload)
    save_payload["languages"] = np.array(languages, dtype=object)
    save_payload["run_id"] = run_id
    save_payload["peft_type"] = peft_type
    np.savez_compressed(output_path, **save_payload)


def write_summary(output_root: Path, summaries: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps({"runs": summaries}, indent=2, sort_keys=True), encoding="utf-8")


def analyze_adapter(
    config: AnalysisConfig,
    *,
    spec,
    create_figures: bool,
) -> dict[str, Any]:
    import torch

    adapter_cfg = read_adapter_config(spec.checkpoint)
    method = detect_method(adapter_cfg)
    output_dir = config.output_root / spec.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading %s from %s", spec.run_id, spec.checkpoint)
    model, tokenizer = load_model(config, spec.checkpoint)
    collector = RoutingCollector(method, adapter_cfg)
    collector.attach(model)
    lang_to_id = language_id_map(config)
    adapter_batch_size = spec.batch_size or config.batch_size
    effective_batch_sizes: dict[str, int] = {}

    try:
        for language in config.languages:
            texts = load_language_texts(
                config.sample_root,
                language,
                text_field=config.text_field,
                limit=config.max_sequences,
            )
            if not texts:
                LOGGER.warning("No texts found for language %s", language)
                continue
            collector.set_language(language)
            language_id = lang_to_id.get(language, -1)
            LOGGER.info(
                "Analyzing %s for %s (%d sequences, batch_size=%d)",
                spec.run_id,
                language,
                len(texts),
                adapter_batch_size,
            )
            effective_batch_sizes[language] = run_language_batches(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                config=config,
                batch_size=adapter_batch_size,
                language_id=language_id,
                run_id=spec.run_id,
                language=language,
            )
    finally:
        collector.remove()

    payload = build_matrices(collector, config.languages)
    npz_path = output_dir / "routing_matrix.npz"
    save_npz(npz_path, payload, languages=config.languages, run_id=spec.run_id, peft_type=method)

    metadata = {
        "run_id": spec.run_id,
        "label": spec.label,
        "checkpoint": str(spec.checkpoint),
        "peft_type": method,
        "axis_name": str(payload["axis_name"]),
        "languages": config.languages,
        "captured_modules": len(payload["layer_labels"]),
        "captured_tokens": int(payload["tokens"].sum()),
        "configured_batch_size": int(adapter_batch_size),
        "effective_batch_sizes": effective_batch_sizes,
        "auto_find_batch_size": config.auto_find_batch_size,
        "length_sort_batches": config.length_sort_batches,
        "routing_matrix": str(npz_path),
        "adapter_config": adapter_cfg,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    figures: list[str] = []
    if create_figures:
        figures_dir = output_dir / "figures"
        create_heatmap(
            payload["routing_weight_normalized"],
            config.languages,
            f"{spec.run_id} weighted routing",
            figures_dir / "routing_weight_heatmap.png",
            axis_name=str(payload["axis_name"]),
        )
        figures.append("figures/routing_weight_heatmap.png")
        language_groups = flatten_language_map(config.language_map)
        if language_groups:
            grouped, group_labels = aggregate_by_group(
                payload["routing_weight_normalized"],
                config.languages,
                language_groups,
            )
            create_heatmap(
                grouped,
                group_labels,
                f"{spec.run_id} grouped weighted routing",
                figures_dir / "grouped_routing_weight_heatmap.png",
                axis_name=str(payload["axis_name"]),
            )
            figures.append("figures/grouped_routing_weight_heatmap.png")
        create_entropy_plot(payload["layer_entropy"], figures_dir / "routing_entropy.png")
        figures.append("figures/routing_entropy.png")
        if method == "MOLA":
            create_mola_null_plot(payload, figures_dir / "mola_null_usage.png")
            figures.append("figures/mola_null_usage.png")

    return {
        "run_id": spec.run_id,
        "peft_type": method,
        "axis_name": str(payload["axis_name"]),
        "captured_modules": len(payload["layer_labels"]),
        "captured_tokens": int(payload["tokens"].sum()),
        "configured_batch_size": int(adapter_batch_size),
        "min_effective_batch_size": int(min(effective_batch_sizes.values())) if effective_batch_sizes else int(adapter_batch_size),
        "output_dir": str(output_dir),
        "routing_matrix": str(npz_path),
        "figures": figures,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    if args.dry_run:
        return dry_run(config, args.run_id)

    ensure_local_imports(config.repo_root)
    validate_config(config)
    summaries = []
    for spec in selected_adapters(config, args.run_id):
        summaries.append(analyze_adapter(config, spec=spec, create_figures=config.create_figures and not args.skip_figures))
    write_summary(config.output_root, summaries)
    LOGGER.info("Routing analysis complete: %s", config.output_root)
    return 0
