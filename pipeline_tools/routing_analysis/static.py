from __future__ import annotations

from typing import Any

import numpy as np
from safetensors import safe_open

from .collectors import layer_sort_key, module_layer_label


STATIC_METHODS = {"MOELORA", "MTLLORA"}


def is_static_method(method: str) -> bool:
    return method in STATIC_METHODS


def language_id_map(adapter_config: dict[str, Any], languages: list[str]) -> dict[str, int]:
    task_num = int(adapter_config.get("task_num", 0) or 0)
    configured = adapter_config.get("language_list")
    if isinstance(configured, list) and configured:
        ordered = [str(item) for item in configured]
        if task_num > 0 and len(ordered) != task_num:
            raise ValueError(f"language_list has {len(ordered)} entries but task_num={task_num}")
        if set(ordered) != set(languages):
            missing = sorted(set(languages) - set(ordered))
            extra = sorted(set(ordered) - set(languages))
            raise ValueError(f"Checkpoint language_list mismatch: missing={missing}, extra={extra}")
    else:
        ordered = sorted(languages)
        if task_num > 0 and len(ordered) != task_num:
            raise ValueError(f"Resolved {len(ordered)} routing ids but checkpoint task_num={task_num}")
    return {language: index for index, language in enumerate(ordered)}


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values.astype(np.float64) - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def _payload(
    *,
    probabilities: np.ndarray,
    selections: np.ndarray,
    layer_labels: list[str],
    axis_name: str,
    routing_basis: str,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    selections = np.asarray(selections, dtype=np.float64)
    if probabilities.ndim != 3 or selections.shape != probabilities.shape:
        raise ValueError(
            f"Expected matching [language,module,choice] arrays, got {probabilities.shape} and {selections.shape}"
        )
    totals = selections.sum(axis=2, keepdims=True)
    selection_rate = np.divide(
        selections,
        totals,
        out=np.zeros_like(selections),
        where=totals > 0,
    )
    entropy = -(probabilities * np.log(probabilities + 1e-10)).sum(axis=2)
    languages, modules, choices = probabilities.shape
    return {
        "routing_counts": selections,
        "routing_count_normalized": selection_rate,
        "routing_weight_sums": probabilities,
        "routing_weight_normalized": probabilities,
        "routing_probability": probabilities,
        "selection_rate": selection_rate,
        "tokens": np.zeros((languages, modules), dtype=np.int64),
        "assignments": totals[:, :, 0].astype(np.int64),
        "observations": np.ones((languages, modules), dtype=np.int64),
        "null_assignments": np.zeros((languages, modules), dtype=np.int64),
        "bypass_tokens": np.zeros((languages, modules), dtype=np.int64),
        "layer_entropy": entropy,
        "layer_labels": np.array(layer_labels, dtype=object),
        "choice_labels": np.array([f"{axis_name}_{index}" for index in range(choices)], dtype=object),
        "axis_name": axis_name,
        "routing_basis": routing_basis,
    }


def extract_moelora(
    checkpoint,
    adapter_config: dict[str, Any],
    languages: list[str],
    language_universe: list[str] | None = None,
) -> dict[str, Any]:
    weights_path = checkpoint / "adapter_model.safetensors"
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        embedding_keys = [key for key in keys if key.endswith("lora_task_embedding.weight")]
        gate_keys = [key for key in keys if key.endswith("lora_gate.weight")]
        if len(embedding_keys) != 1 or len(gate_keys) != 1:
            raise ValueError(
                f"Expected one shared MoE-LoRA embedding and gate, got {embedding_keys} and {gate_keys}"
            )
        embedding = handle.get_tensor(embedding_keys[0]).float().numpy()
        gate = handle.get_tensor(gate_keys[0]).float().numpy()

    task_to_id = language_id_map(adapter_config, language_universe or languages)
    ordered_rows = np.array([task_to_id[language] for language in languages], dtype=np.int64)
    temperature = float(adapter_config.get("gate_temperature", 1.0) or 1.0)
    if temperature <= 0:
        raise ValueError(f"Invalid gate_temperature={temperature}")
    logits = (embedding @ gate.T) / temperature
    top_k = int(adapter_config.get("gate_top_k", 0) or 0)
    expert_num = int(adapter_config.get("expert_num", gate.shape[0]) or gate.shape[0])
    if logits.shape != (embedding.shape[0], expert_num):
        raise ValueError(f"Unexpected MoE-LoRA gate shape: logits={logits.shape}, expert_num={expert_num}")
    if 0 < top_k < expert_num:
        selected = np.argpartition(logits, -top_k, axis=1)[:, -top_k:]
        sparse = np.full_like(logits, -np.inf, dtype=np.float64)
        np.put_along_axis(sparse, selected, np.take_along_axis(logits, selected, axis=1), axis=1)
        probabilities = _softmax(sparse)
        selection = np.zeros_like(probabilities)
        np.put_along_axis(selection, selected, 1.0, axis=1)
    else:
        probabilities = _softmax(logits)
        selection = np.ones_like(probabilities)

    return _payload(
        probabilities=probabilities[ordered_rows, None, :],
        selections=selection[ordered_rows, None, :],
        layer_labels=["shared_gate"],
        axis_name="expert",
        routing_basis="static_language_conditioned",
    )


def extract_mtllora(
    checkpoint,
    adapter_config: dict[str, Any],
    languages: list[str],
    language_universe: list[str] | None = None,
) -> dict[str, Any]:
    weights_path = checkpoint / "adapter_model.safetensors"
    temperature = float(adapter_config.get("temperature", 0.1) or 0.1)
    if temperature <= 0:
        raise ValueError(f"Invalid MTL-LoRA temperature={temperature}")
    task_to_id = language_id_map(adapter_config, language_universe or languages)
    ordered_rows = np.array([task_to_id[language] for language in languages], dtype=np.int64)

    tensors: list[tuple[str, np.ndarray]] = []
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.endswith(".lora_B_weights"):
                continue
            label = module_layer_label(key[: -len(".lora_B_weights")])
            tensors.append((label, handle.get_tensor(key).float().numpy()))
    tensors.sort(key=lambda item: layer_sort_key(item[0]))
    if not tensors:
        raise ValueError(f"No MTL-LoRA lora_B_weights found in {weights_path}")

    expected_tasks = int(adapter_config.get("task_num", 0) or 0)
    expected_choices = int(adapter_config.get("num_up_projections", 0) or 0)
    probabilities = []
    for label, values in tensors:
        if values.shape != (expected_tasks, expected_choices):
            raise ValueError(f"Unexpected MTL-LoRA weight shape for {label}: {values.shape}")
        probabilities.append(_softmax(values / temperature)[ordered_rows])
    matrix = np.stack(probabilities, axis=1)
    argmax = np.argmax(matrix, axis=2, keepdims=True)
    selections = np.zeros_like(matrix)
    np.put_along_axis(selections, argmax, 1.0, axis=2)
    return _payload(
        probabilities=matrix,
        selections=selections,
        layer_labels=[label for label, _ in tensors],
        axis_name="up_projection",
        routing_basis="static_language_conditioned",
    )


def extract_static_routing(
    method: str,
    checkpoint,
    adapter_config: dict[str, Any],
    languages: list[str],
    language_universe: list[str] | None = None,
) -> dict[str, Any]:
    if method == "MOELORA":
        return extract_moelora(checkpoint, adapter_config, languages, language_universe)
    if method == "MTLLORA":
        return extract_mtllora(checkpoint, adapter_config, languages, language_universe)
    raise ValueError(f"No static routing extractor for {method}")
