from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .collectors import RoutingCollector, build_matrices, module_layer_label
from .config import (
    AnalysisConfig,
    detect_method,
    flatten_language_map,
    load_config,
    read_adapter_config,
    validate_config,
)
from .data import (
    iter_language_jsonl_files,
    load_language_texts_with_provenance,
    load_sample_manifest,
    validate_language_sample,
)
from .io import atomic_save_npz, atomic_write_json, sha256_file
from .plotting import aggregate_by_group, create_entropy_plot, create_heatmap, create_mola_null_plot
from .samples import build_sample_manifest
from .static import extract_static_routing, is_static_method, language_id_map as adapter_language_id_map


LOGGER = logging.getLogger("routing_analysis")
CANARY_LANGUAGES = ["english", "arb_Arab", "jpn_Jpan"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze routing behavior for supported PEFT adapter checkpoints.")
    parser.add_argument("--config", required=True, type=Path, help="Routing analysis JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and planned inputs without loading models.")
    parser.add_argument("--run-id", action="append", default=[], help="Limit to one or more configured run ids.")
    parser.add_argument("--skip-figures", action="store_true", help="Write NPZ/metadata only.")
    parser.add_argument("--force", action="store_true", help="Discard completion markers and recompute selected runs.")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Build the panel summary after every configured run has completed.",
    )
    parser.add_argument(
        "--prepare-sample-manifest",
        action="store_true",
        help="Freeze the configured deterministic input prefix into sample_manifest and exit.",
    )
    parser.add_argument(
        "--validate-static",
        action="store_true",
        help="For static routers, load the model and validate extraction against the runtime routing path.",
    )
    parser.add_argument(
        "--canary",
        action="store_true",
        help="Analyze 8 documents at length 128 for English, Arabic, and Japanese under a separate output root.",
    )
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
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_source = Path(config.base_model)
    if not model_source.exists():
        model_source = Path(snapshot_download(config.base_model, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device_map: str | dict[str, str] = config.device
    if config.device.startswith("cuda") and config.device not in {"auto", "balanced", "balanced_low_0", "sequential"}:
        device = 0 if config.device == "cuda" else config.device
        device_map = {"": device}
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch_dtype(config.torch_dtype),
        device_map=device_map,
        local_files_only=True,
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


def _axis_for(method: str, adapter_config: dict[str, Any]) -> str:
    if method == "MTLLORA":
        return "up_projection"
    return RoutingCollector(method, adapter_config).axis_name


def dry_run(config: AnalysisConfig, run_ids: list[str]) -> int:
    summaries = validate_config(config)
    selected = {item.run_id for item in selected_adapters(config, run_ids)}
    missing_languages = [
        language for language in config.languages if not iter_language_jsonl_files(config.sample_root, language)
    ]
    print(
        f"[OK] config valid: {len(config.languages)} routing ids "
        f"({len(config.analysis_languages)} active), {len(summaries)} adapters"
    )
    print(f"[INFO] output_root={config.output_root}")
    print(f"[INFO] sample_manifest={config.sample_manifest}")
    if missing_languages:
        preview = ", ".join(missing_languages[:20])
        print(f"[WARN] {len(missing_languages)} routing ids lack samples: {preview}")
    for summary in summaries:
        if summary["run_id"] not in selected:
            continue
        adapter_cfg = read_adapter_config(Path(summary["checkpoint"]))
        method = detect_method(adapter_cfg)
        adapter_language_id_map(adapter_cfg, config.languages)
        print(
            f"[PLAN] {summary['run_id']}: peft_type={method} axis={_axis_for(method, adapter_cfg)} "
            f"basis={'static' if is_static_method(method) else 'token'} checkpoint={summary['checkpoint']}"
        )
    return 0


def batched(items: list[str], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda" in text and "out of memory" in text


def run_language_batches(
    *,
    model: Any,
    tokenizer: Any,
    collector: RoutingCollector,
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
            collector.set_token_mask(encoded.get("attention_mask"))
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
        finally:
            collector.clear_token_mask()
    return effective_batch_size


def save_npz(
    output_path: Path,
    payload: dict[str, Any],
    *,
    languages: list[str],
    run_id: str,
    peft_type: str,
) -> None:
    save_payload = dict(payload)
    save_payload["languages"] = np.array(languages, dtype=object)
    save_payload["run_id"] = run_id
    save_payload["peft_type"] = peft_type
    atomic_save_npz(output_path, save_payload)


def _load_partial(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as source:
        return {key: source[key] for key in source.files}


def combine_partials(partial_paths: list[Path], languages: list[str]) -> dict[str, Any]:
    if len(partial_paths) != len(languages):
        raise ValueError(f"Expected {len(languages)} partials, got {len(partial_paths)}")
    partials = [_load_partial(path) for path in partial_paths]
    reference = partials[0]
    fixed_keys = {"layer_labels", "choice_labels", "axis_name", "routing_basis", "run_id", "peft_type"}
    for partial in partials[1:]:
        for key in ("layer_labels", "choice_labels", "axis_name", "routing_basis"):
            if not np.array_equal(partial[key], reference[key]):
                raise ValueError(f"Partial routing artifact mismatch for {key}")

    combined: dict[str, Any] = {}
    for key, value in reference.items():
        if key == "languages":
            continue
        if key in fixed_keys:
            combined[key] = value
        elif isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == 1:
            combined[key] = np.concatenate([partial[key] for partial in partials], axis=0)
        else:
            combined[key] = value
    return combined


def _checkpoint_provenance(config: AnalysisConfig, checkpoint: Path) -> dict[str, Any]:
    config_path = checkpoint / "adapter_config.json"
    weights_path = checkpoint / "adapter_model.safetensors"
    payload = {
        "adapter_config": {
            "path": str(config_path),
            "bytes": config_path.stat().st_size,
            "sha256": sha256_file(config_path),
        },
        "adapter_weights": {
            "path": str(weights_path),
            "bytes": weights_path.stat().st_size,
        },
    }
    if config.checkpoint_hashes:
        payload["adapter_weights"]["sha256"] = sha256_file(weights_path)
    return payload


def _create_figures(
    *,
    config: AnalysisConfig,
    spec,
    method: str,
    payload: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    figures_dir = output_dir / "figures"
    matrix = payload.get("routing_probability", payload["routing_weight_normalized"])
    create_heatmap(
        matrix,
        config.analysis_languages,
        f"{spec.run_id} routing probability",
        figures_dir / "routing_probability_heatmap.png",
        axis_name=str(payload["axis_name"]),
    )
    figures = ["figures/routing_probability_heatmap.png"]
    language_groups = flatten_language_map(config.language_map)
    if language_groups:
        grouped, group_labels = aggregate_by_group(matrix, config.analysis_languages, language_groups)
        create_heatmap(
            grouped,
            group_labels,
            f"{spec.run_id} grouped routing probability",
            figures_dir / "grouped_routing_probability_heatmap.png",
            axis_name=str(payload["axis_name"]),
        )
        figures.append("figures/grouped_routing_probability_heatmap.png")
    create_entropy_plot(payload["layer_entropy"], figures_dir / "routing_entropy.png")
    figures.append("figures/routing_entropy.png")
    if method == "MOLA" and bool(read_adapter_config(spec.checkpoint).get("mola_use_null_expert", False)):
        create_mola_null_plot(payload, figures_dir / "mola_null_usage.png")
        figures.append("figures/mola_null_usage.png")
    return figures


def _write_run_outputs(
    *,
    config: AnalysisConfig,
    spec,
    method: str,
    payload: dict[str, Any],
    output_dir: Path,
    adapter_config: dict[str, Any],
    input_provenance: dict[str, Any],
    effective_batch_sizes: dict[str, int],
    static_validation: dict[str, Any] | None,
    create_figures: bool,
) -> dict[str, Any]:
    npz_path = output_dir / "routing_matrix.npz"
    save_npz(npz_path, payload, languages=config.analysis_languages, run_id=spec.run_id, peft_type=method)
    figures = (
        _create_figures(config=config, spec=spec, method=method, payload=payload, output_dir=output_dir)
        if create_figures
        else []
    )
    checkpoint_provenance = _checkpoint_provenance(config, spec.checkpoint)
    routing_basis = str(payload.get("routing_basis", "token_conditioned"))
    metadata = {
        "schema_version": 2,
        "run_id": spec.run_id,
        "label": spec.label,
        "checkpoint": str(spec.checkpoint),
        "checkpoint_provenance": checkpoint_provenance,
        "peft_type": method,
        "axis_name": str(payload["axis_name"]),
        "routing_basis": routing_basis,
        "sample_dependent": routing_basis == "token_conditioned",
        "languages": config.analysis_languages,
        "captured_modules": len(payload["layer_labels"]),
        "captured_tokens": int(payload["tokens"].sum()),
        "configured_batch_size": int(spec.batch_size or config.batch_size),
        "effective_batch_sizes": effective_batch_sizes,
        "auto_find_batch_size": config.auto_find_batch_size,
        "length_sort_batches": config.length_sort_batches,
        "padding_tokens_excluded": routing_basis == "token_conditioned",
        "input_provenance": input_provenance,
        "static_validation": static_validation,
        "routing_matrix": str(npz_path),
        "adapter_config": adapter_config,
        "mola_null_router_enabled": (
            bool(adapter_config.get("mola_use_null_expert", False)) if method == "MOLA" else None
        ),
    }
    atomic_write_json(output_dir / "metadata.json", metadata)
    summary = {
        "run_id": spec.run_id,
        "peft_type": method,
        "axis_name": str(payload["axis_name"]),
        "routing_basis": routing_basis,
        "status": "complete",
        "captured_modules": len(payload["layer_labels"]),
        "captured_tokens": int(payload["tokens"].sum()),
        "routing_ids": len(config.analysis_languages),
        "configured_batch_size": int(spec.batch_size or config.batch_size),
        "min_effective_batch_size": (
            int(min(effective_batch_sizes.values()))
            if effective_batch_sizes
            else int(spec.batch_size or config.batch_size)
        ),
        "output_dir": str(output_dir),
        "routing_matrix": str(npz_path),
        "metadata": str(output_dir / "metadata.json"),
        "figures": figures,
    }
    atomic_write_json(output_dir / "run_summary.json", summary)
    return summary


def _find_moelora_parent(model: Any):
    for module in model.modules():
        if (
            callable(getattr(type(module), "get_routing_weights", None))
            and getattr(module, "lora_task_embedding", None) is not None
            and getattr(module, "lora_gate", None) is not None
        ):
            return module
    raise RuntimeError("Could not find loaded MoE-LoRA shared routing module")


def validate_static_forward(
    *,
    config: AnalysisConfig,
    spec,
    method: str,
    adapter_config: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    import torch

    model, tokenizer = load_model(config, spec.checkpoint)
    lang_to_id = adapter_language_id_map(adapter_config, config.languages)
    selected = [
        language for language in config.static_validation_languages if language in config.analysis_languages
    ]
    if not selected:
        raise ValueError("None of static_validation_languages are in the configured panel")
    max_error = 0.0
    forward_documents = 0
    try:
        if method == "MOELORA":
            from safetensors import safe_open

            parent = _find_moelora_parent(model)
            adapter_name = next(iter(parent.peft_config))
            weights_path = spec.checkpoint / "adapter_model.safetensors"
            with safe_open(weights_path, framework="pt", device="cpu") as handle:
                checkpoint_embedding = handle.get_tensor("base_model.lora_task_embedding.weight").float()
                checkpoint_gate = handle.get_tensor("base_model.lora_gate.weight").float()
            runtime_embedding = parent.lora_task_embedding[adapter_name].weight.detach().float().cpu()
            runtime_gate = parent.lora_gate[adapter_name].weight.detach().float().cpu()
            embedding_error = float(torch.max(torch.abs(runtime_embedding - checkpoint_embedding)))
            gate_error = float(torch.max(torch.abs(runtime_gate - checkpoint_gate)))
            LOGGER.info(
                "MoE static validation tensors: embedding_dtype=%s gate_dtype=%s "
                "embedding_max_abs_error=%g gate_max_abs_error=%g",
                parent.lora_task_embedding[adapter_name].weight.dtype,
                parent.lora_gate[adapter_name].weight.dtype,
                embedding_error,
                gate_error,
            )
            for language in selected:
                language_index = config.analysis_languages.index(language)
                routing_id = lang_to_id[language]
                parent._cached_routing_ids = torch.tensor([routing_id], device=model_device(model))
                parent._cached_routing_weights = {}
                runtime = parent.get_routing_weights(
                    adapter_name,
                    batch_size=1,
                    device=model_device(model),
                    dtype=torch.float32,
                )
                expected = payload["routing_probability"][language_index, 0]
                language_error = float(np.max(np.abs(runtime.detach().cpu().numpy()[0] - expected)))
                LOGGER.info(
                    "MoE static validation language=%s routing_id=%d max_abs_error=%g "
                    "runtime_topk=%s expected_topk=%s",
                    language,
                    routing_id,
                    language_error,
                    torch.nonzero(runtime[0] > 0, as_tuple=False).flatten().detach().cpu().tolist(),
                    np.flatnonzero(expected > 0).tolist(),
                )
                max_error = max(max_error, language_error)
                parent._cached_routing_ids = None
                parent._cached_routing_weights = None
        elif method == "MTLLORA":
            payload_layers = {
                str(label): index for index, label in enumerate(payload["layer_labels"].tolist())
            }
            observed_layers = set()
            for name, module in model.named_modules():
                weights_by_adapter = getattr(module, "lora_B_weights", None)
                if weights_by_adapter is None or not len(weights_by_adapter):
                    continue
                adapter_name = next(iter(weights_by_adapter))
                label = module_layer_label(name)
                if label not in payload_layers:
                    continue
                observed_layers.add(label)
                temperature = float(module.temperature[adapter_name])
                runtime = torch.softmax(
                    weights_by_adapter[adapter_name].detach().to(torch.float32) / temperature,
                    dim=-1,
                )
                expected = payload["routing_probability"][:, payload_layers[label], :]
                rows = torch.tensor(
                    [lang_to_id[language] for language in config.analysis_languages],
                    dtype=torch.long,
                    device=runtime.device,
                )
                observed = runtime[rows].cpu().numpy()
                max_error = max(max_error, float(np.max(np.abs(observed - expected))))
            missing_layers = sorted(set(payload_layers) - observed_layers)
            if missing_layers:
                raise ValueError(f"Loaded MTL-LoRA model is missing routing modules: {missing_layers}")

        for language in selected:
            texts, _ = load_language_texts_with_provenance(
                config.sample_root,
                language,
                text_field=config.text_field,
                limit=8,
            )
            if len(texts) != 8:
                raise ValueError(f"{language}: static validation requires 8 texts, found {len(texts)}")
            encoded = tokenizer(
                texts,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(model_device(model)) for key, value in encoded.items()}
            encoded["language_ids"] = torch.full(
                (len(texts),),
                lang_to_id[language],
                dtype=torch.long,
                device=model_device(model),
            )
            encoded["logits_to_keep"] = 1
            with torch.no_grad():
                output = model(**encoded)
            if not torch.isfinite(output.logits).all():
                raise ValueError(f"{language}: static validation forward produced non-finite logits")
            forward_documents += len(texts)
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if max_error > 1e-4:
        raise ValueError(f"{method} static extraction mismatch: max_abs_error={max_error}")
    return {
        "status": "passed",
        "languages": selected,
        "documents": forward_documents,
        "max_length": 128,
        "max_probability_error": max_error,
        "tolerance": 1e-4,
        "finite_forward_logits": True,
    }


def analyze_static_adapter(
    config: AnalysisConfig,
    *,
    spec,
    adapter_config: dict[str, Any],
    method: str,
    create_figures: bool,
    validate_static: bool,
) -> dict[str, Any]:
    output_dir = config.output_root / spec.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = extract_static_routing(
        method,
        spec.checkpoint,
        adapter_config,
        config.analysis_languages,
        config.languages,
    )
    validation = (
        validate_static_forward(
            config=config,
            spec=spec,
            method=method,
            adapter_config=adapter_config,
            payload=payload,
        )
        if validate_static
        else {"status": "not_run", "reason": "Run with --validate-static on a GPU node"}
    )
    return _write_run_outputs(
        config=config,
        spec=spec,
        method=method,
        payload=payload,
        output_dir=output_dir,
        adapter_config=adapter_config,
        input_provenance={
            "sample_dependent": False,
            "documents": 0,
            "note": "Routing is a learned language-conditioned distribution; text inputs are not routing observations.",
        },
        effective_batch_sizes={},
        static_validation=validation,
        create_figures=create_figures,
    )


def analyze_token_adapter(
    config: AnalysisConfig,
    *,
    spec,
    adapter_config: dict[str, Any],
    method: str,
    create_figures: bool,
) -> dict[str, Any]:
    output_dir = config.output_root / spec.run_id
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    sample_manifest = load_sample_manifest(config.sample_manifest)
    lang_to_id = adapter_language_id_map(adapter_config, config.languages)
    adapter_batch_size = spec.batch_size or config.batch_size
    effective_batch_sizes: dict[str, int] = {}
    input_provenance: dict[str, Any] = {}

    LOGGER.info("Loading %s from %s", spec.run_id, spec.checkpoint)
    model, tokenizer = load_model(config, spec.checkpoint)
    collector = RoutingCollector(method, adapter_config)
    collector.attach(model)
    try:
        for language in config.analysis_languages:
            partial_path = partial_dir / f"{language}.npz"
            partial_meta_path = partial_dir / f"{language}.json"
            texts, provenance = load_language_texts_with_provenance(
                config.sample_root,
                language,
                text_field=config.text_field,
                limit=config.max_sequences,
            )
            validate_language_sample(
                provenance=provenance,
                manifest=sample_manifest,
                expected_documents=config.max_sequences,
                strict=config.strict_samples,
            )
            input_provenance[language] = provenance
            expected_partial = {
                "language": language,
                "documents": len(texts),
                "text_sha256": provenance["text_sha256"],
                "max_length": config.max_length,
                "checkpoint": str(spec.checkpoint),
                "method": method,
            }
            if partial_path.exists() and partial_meta_path.exists():
                observed = json.loads(partial_meta_path.read_text(encoding="utf-8"))
                if all(observed.get(key) == value for key, value in expected_partial.items()):
                    effective_batch_sizes[language] = int(observed["effective_batch_size"])
                    LOGGER.info("Resume hit for %s/%s", spec.run_id, language)
                    continue

            collector.set_language(language)
            LOGGER.info(
                "Analyzing %s for %s (%d sequences, batch_size=%d)",
                spec.run_id,
                language,
                len(texts),
                adapter_batch_size,
            )
            effective = run_language_batches(
                model=model,
                tokenizer=tokenizer,
                collector=collector,
                texts=texts,
                config=config,
                batch_size=adapter_batch_size,
                language_id=lang_to_id[language],
                run_id=spec.run_id,
                language=language,
            )
            effective_batch_sizes[language] = effective
            partial_payload = build_matrices(collector, [language])
            save_npz(
                partial_path,
                partial_payload,
                languages=[language],
                run_id=spec.run_id,
                peft_type=method,
            )
            atomic_write_json(
                partial_meta_path,
                {**expected_partial, "effective_batch_size": effective, "status": "complete"},
            )
            collector.by_language.pop(language, None)
    finally:
        collector.remove()
        del model
        del tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    partial_paths = [partial_dir / f"{language}.npz" for language in config.analysis_languages]
    missing = [str(path) for path in partial_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing completed language partials for {spec.run_id}: {missing}")
    payload = combine_partials(partial_paths, config.analysis_languages)
    return _write_run_outputs(
        config=config,
        spec=spec,
        method=method,
        payload=payload,
        output_dir=output_dir,
        adapter_config=adapter_config,
        input_provenance={
            "sample_dependent": True,
            "documents_per_routing_id": config.max_sequences,
            "max_length": config.max_length,
            "sample_manifest": str(config.sample_manifest) if config.sample_manifest else None,
            "languages": input_provenance,
        },
        effective_batch_sizes=effective_batch_sizes,
        static_validation=None,
        create_figures=create_figures,
    )


def analyze_adapter(
    config: AnalysisConfig,
    *,
    spec,
    create_figures: bool,
    validate_static: bool,
    force: bool,
) -> dict[str, Any]:
    output_dir = config.output_root / spec.run_id
    completion = output_dir / "run_summary.json"
    matrix = output_dir / "routing_matrix.npz"
    if not force and completion.exists() and matrix.exists():
        summary = json.loads(completion.read_text(encoding="utf-8"))
        if summary.get("status") == "complete":
            LOGGER.info("Skipping completed run %s", spec.run_id)
            return summary
    if force and completion.exists():
        completion.unlink()

    adapter_config = read_adapter_config(spec.checkpoint)
    method = detect_method(adapter_config)
    if is_static_method(method):
        return analyze_static_adapter(
            config,
            spec=spec,
            adapter_config=adapter_config,
            method=method,
            create_figures=create_figures,
            validate_static=validate_static,
        )
    return analyze_token_adapter(
        config,
        spec=spec,
        adapter_config=adapter_config,
        method=method,
        create_figures=create_figures,
    )


def aggregate_summaries(config: AnalysisConfig, *, strict: bool) -> bool:
    summaries = []
    missing = []
    for spec in config.adapters:
        path = config.output_root / spec.run_id / "run_summary.json"
        if not path.exists():
            missing.append(spec.run_id)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            missing.append(spec.run_id)
            continue
        summaries.append(payload)
    if missing:
        if strict:
            raise RuntimeError(f"Cannot aggregate incomplete routing panel; missing={missing}")
        LOGGER.info("Panel aggregation deferred; incomplete runs: %s", ", ".join(missing))
        return False
    atomic_write_json(config.output_root / "summary.json", {"schema_version": 2, "runs": summaries})
    LOGGER.info("Routing panel complete: %s", config.output_root)
    return True


def _prepare_manifest(config: AnalysisConfig) -> int:
    if config.sample_manifest is None:
        raise SystemExit("Config must set sample_manifest for --prepare-sample-manifest")
    build_sample_manifest(
        sample_root=config.sample_root,
        languages=config.languages,
        text_field=config.text_field,
        documents_per_language=config.max_sequences,
        output_path=config.sample_manifest,
    )
    print(f"[OK] wrote routing sample manifest to {config.sample_manifest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_config(args.config)
    if args.prepare_sample_manifest:
        return _prepare_manifest(config)
    if args.canary:
        canary_languages = [language for language in CANARY_LANGUAGES if language in config.languages]
        config = replace(
            config,
            analysis_languages=canary_languages,
            output_root=config.output_root.parent / f"{config.output_root.name}_canary",
            max_sequences=8,
            max_length=128,
            strict_samples=False,
            sample_manifest=None,
            checkpoint_hashes=False,
        )
    if args.dry_run:
        return dry_run(config, args.run_id)

    ensure_local_imports(config.repo_root)
    validate_config(config)
    if args.aggregate_only:
        aggregate_summaries(config, strict=True)
        return 0

    for spec in selected_adapters(config, args.run_id):
        analyze_adapter(
            config,
            spec=spec,
            create_figures=config.create_figures and not args.skip_figures,
            validate_static=args.validate_static,
            force=args.force,
        )
    aggregate_summaries(config, strict=False)
    return 0
