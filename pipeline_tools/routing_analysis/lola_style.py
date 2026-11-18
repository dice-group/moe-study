from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LanguageMeta:
    language: str
    code: str
    script: str
    name: str
    family: str
    resource_category: str
    documents: int


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt

    return plt, LinearSegmentedColormap


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def _as_str_array(value: np.ndarray) -> list[str]:
    return [str(item) for item in value.tolist()]


def _load_language_metadata(csv_path: Path, languages: Iterable[str]) -> dict[str, LanguageMeta]:
    known = set(languages)
    metadata: dict[str, LanguageMeta] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("split", "")).strip() != "train":
                continue
            language = str(row.get("subset", "")).strip()
            if language not in known:
                continue
            documents_raw = str(row.get("documents", "0")).strip()
            try:
                documents = int(float(documents_raw))
            except ValueError:
                documents = 0
            metadata[language] = LanguageMeta(
                language=language,
                code=str(row.get("code", "")).strip() or language.split("_", 1)[0],
                script=str(row.get("script", "")).strip() or (language.split("_", 1)[1] if "_" in language else ""),
                name=str(row.get("name", "")).strip() or language,
                family=str(row.get("family", "")).strip() or "unknown",
                resource_category=str(row.get("resource_category", "")).strip() or "unknown",
                documents=documents,
            )

    if "english" in known and "english" not in metadata:
        metadata["english"] = LanguageMeta(
            language="english",
            code="eng",
            script="Latn",
            name="English",
            family="English",
            resource_category="high",
            documents=0,
        )

    for language in sorted(known - set(metadata)):
        metadata[language] = LanguageMeta(
            language=language,
            code=language.split("_", 1)[0],
            script=language.split("_", 1)[1] if "_" in language else "",
            name=language,
            family="unknown",
            resource_category="unknown",
            documents=0,
        )
    return metadata


def _group_label(meta: LanguageMeta, granularity: str) -> str:
    if granularity == "language":
        return meta.language
    if granularity == "family":
        return meta.family
    if granularity == "family_script":
        script = meta.script or "unknown-script"
        return f"{meta.family} / {script}"
    if granularity == "family_resource":
        resource = meta.resource_category or "unknown-resource"
        return f"{meta.family} / {resource}"
    raise ValueError(f"Unsupported granularity: {granularity}")


def _ordered_languages(languages: list[str], metadata: dict[str, LanguageMeta]) -> list[str]:
    return sorted(
        languages,
        key=lambda lang: (
            metadata[lang].family.lower(),
            metadata[lang].script.lower(),
            -metadata[lang].documents,
            metadata[lang].language,
        ),
    )


def _aggregate_by_group(
    matrix: np.ndarray,
    languages: list[str],
    metadata: dict[str, LanguageMeta],
    granularity: str,
) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    groups: dict[str, list[int]] = {}
    members: dict[str, list[str]] = {}
    ordered_languages = _ordered_languages(languages, metadata)
    language_index = {language: idx for idx, language in enumerate(languages)}
    for language in ordered_languages:
        label = _group_label(metadata[language], granularity)
        groups.setdefault(label, []).append(language_index[language])
        members.setdefault(label, []).append(language)

    labels = list(groups)
    aggregated = np.zeros((len(labels), matrix.shape[1], matrix.shape[2]), dtype=np.float64)
    for group_idx, label in enumerate(labels):
        aggregated[group_idx] = matrix[groups[label]].mean(axis=0)
    return aggregated, labels, members


def _normalize_per_module(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(matrix.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(matrix, axis=2, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(matrix.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _cosine_similarity(vectors: np.ndarray) -> np.ndarray:
    normalized = _normalize_rows(vectors)
    return normalized @ normalized.T


def _save_lola_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    output: Path,
    *,
    title: str,
    axis_name: str,
    module_labels: list[str],
    x_group_label: str = "captured module",
    paper_style: bool = False,
) -> None:
    plt, linear_segmented_colormap = _ensure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    data = np.nan_to_num(matrix.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0).reshape(matrix.shape[0], -1)
    data = np.clip(data, 0.0, 1.0)
    uniform = 1.0 / max(1, matrix.shape[2])
    # Match LOLA Figure 10: ratio scale with visual landmarks at uniform routing and its multiples.
    cmap = linear_segmented_colormap.from_list(
        "lola_ratio",
        [
            (0.0, "#ffffff"),
            (min(uniform, 1.0), "#b8f5a5"),
            (min(2 * uniform, 1.0), "#ffff00"),
            (min(4 * uniform, 1.0), "#ff1f1f"),
            (0.5, "#b00020"),
            (1.0, "#000000"),
        ],
    )

    fig_width = max(10, min(34 if paper_style else 42, data.shape[1] * (0.035 if paper_style else 0.045)))
    fig_height = max(5, min(24 if paper_style else 32, len(labels) * (0.18 if paper_style else 0.22)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    if not paper_style:
        ax.set_title(title)
    axis_title = axis_name.replace("_", " ").title()
    ax.set_xlabel(axis_title if paper_style else f"{axis_name}s grouped by {x_group_label}")
    ax.set_ylabel("Languages" if paper_style else "Language group")
    if len(labels) <= 140:
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=5 if paper_style and len(labels) > 70 else (6 if len(labels) > 70 else 8))
    else:
        ax.set_yticks([])

    choices = matrix.shape[2]
    for boundary in range(choices, data.shape[1], choices):
        ax.axvline(boundary - 0.5, color="black", alpha=0.35 if paper_style else 0.18, linewidth=0.45 if paper_style else 0.35)
    if len(module_labels) <= 48:
        tick_positions = np.arange(len(module_labels)) * choices + (choices - 1) / 2
        ax.set_xticks(tick_positions)
        if paper_style:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xticklabels(module_labels, rotation=90, fontsize=5)
    else:
        ax.set_xticks([])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.018, pad=0.01, label="" if paper_style else "routing ratio")
    tick_values = [0.0, uniform, 2 * uniform, 4 * uniform, 0.5, 1.0]
    tick_labels = ["0", f"1/{matrix.shape[2]}", f"2/{matrix.shape[2]}", f"4/{matrix.shape[2]}", "1/2", "1"]
    deduped = []
    seen = set()
    for value, label in zip(tick_values, tick_labels):
        value = min(float(value), 1.0)
        if value in seen:
            continue
        seen.add(value)
        deduped.append((value, label))
    colorbar.set_ticks([value for value, _ in deduped])
    colorbar.set_ticklabels([label for _, label in deduped])
    colorbar.ax.tick_params(labelsize=8 if paper_style else 9)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _parse_base_layer(label: str) -> int | None:
    match = re.match(r"L(\d+):", label)
    return int(match.group(1)) if match else None


def _aggregate_to_base_layers(npz: np.lib.npyio.NpzFile, module_labels: list[str]) -> tuple[np.ndarray, list[str]] | None:
    layer_ids = [_parse_base_layer(label) for label in module_labels]
    if not any(layer_id is not None for layer_id in layer_ids):
        return None
    unique_layers = sorted({layer_id for layer_id in layer_ids if layer_id is not None})
    if "routing_probability" in npz.files:
        matrix = npz["routing_probability"].astype(np.float64)
        layer_matrix = np.zeros((matrix.shape[0], len(unique_layers), matrix.shape[2]), dtype=np.float64)
        for layer_out_idx, layer_id in enumerate(unique_layers):
            module_indices = [idx for idx, current in enumerate(layer_ids) if current == layer_id]
            layer_matrix[:, layer_out_idx, :] = matrix[:, module_indices, :].mean(axis=1)
        return layer_matrix, [f"L{layer_id}" for layer_id in unique_layers]

    if "routing_counts" in npz.files and "assignments" in npz.files:
        counts = npz["routing_counts"].astype(np.float64)
        assignments = npz["assignments"].astype(np.float64)
        layer_matrix = np.zeros((counts.shape[0], len(unique_layers), counts.shape[2]), dtype=np.float64)
        for layer_out_idx, layer_id in enumerate(unique_layers):
            module_indices = [idx for idx, current in enumerate(layer_ids) if current == layer_id]
            layer_counts = counts[:, module_indices, :].sum(axis=1)
            layer_assignments = assignments[:, module_indices].sum(axis=1)
            layer_matrix[:, layer_out_idx, :] = np.divide(
                layer_counts,
                np.maximum(layer_assignments[:, None], 1),
                out=np.zeros_like(layer_counts),
                where=layer_assignments[:, None] > 0,
            )
        return layer_matrix, [f"L{layer_id}" for layer_id in unique_layers]

    matrix = npz["routing_count_normalized" if "routing_count_normalized" in npz.files else "routing_weight_normalized"].astype(np.float64)
    layer_matrix = np.zeros((matrix.shape[0], len(unique_layers), matrix.shape[2]), dtype=np.float64)
    for layer_out_idx, layer_id in enumerate(unique_layers):
        module_indices = [idx for idx, current in enumerate(layer_ids) if current == layer_id]
        layer_matrix[:, layer_out_idx, :] = matrix[:, module_indices, :].mean(axis=1)
    return layer_matrix, [f"L{layer_id}" for layer_id in unique_layers]


def _layer_zoom_ranges(layer_labels: list[str]) -> list[tuple[str, list[int]]]:
    layer_numbers = [int(label[1:]) for label in layer_labels if label.startswith("L") and label[1:].isdigit()]
    if not layer_numbers:
        return []
    max_layer = max(layer_numbers)
    label_to_idx = {label: idx for idx, label in enumerate(layer_labels)}

    candidates = [
        ("latest_layers", list(range(max(0, max_layer - 3), max_layer + 1))),
        ("previous_layers", list(range(max(0, max_layer - 7), max(0, max_layer - 3)))),
        ("middle_late_layers", list(range(max(0, max_layer - 15), max(0, max_layer - 11)))),
    ]
    ranges: list[tuple[str, list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for name, layer_range in candidates:
        indices = [label_to_idx[f"L{layer_id}"] for layer_id in layer_range if f"L{layer_id}" in label_to_idx]
        key = tuple(indices)
        if indices and key not in seen:
            seen.add(key)
            ranges.append((name, indices))
    return ranges


def _write_layer_expert_totals(
    matrix: np.ndarray,
    labels: list[str],
    layer_labels: list[str],
    choice_labels: list[str],
    output: Path,
    *,
    top_k: int,
) -> None:
    _write_top_experts(
        matrix,
        labels,
        layer_labels,
        choice_labels,
        output,
        top_k=top_k,
    )


def _save_similarity_heatmap(similarity: np.ndarray, labels: list[str], output: Path, *, title: str) -> None:
    plt, _ = _ensure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig_size = max(7, min(28, len(labels) * 0.22))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(similarity, aspect="equal", interpolation="nearest", cmap="viridis", vmin=-1, vmax=1)
    ax.set_title(title)
    if len(labels) <= 80:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _embedding_2d(vectors: np.ndarray) -> tuple[np.ndarray, str]:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    vectors = _normalize_rows(vectors)
    n_rows = vectors.shape[0]
    if n_rows < 4:
        return PCA(n_components=2, random_state=42).fit_transform(vectors), "PCA"
    perplexity = max(2, min(30, (n_rows - 1) // 3))
    try:
        embedding = TSNE(
            n_components=2,
            metric="cosine",
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=42,
        ).fit_transform(vectors)
        return embedding, "t-SNE"
    except Exception:
        return PCA(n_components=2, random_state=42).fit_transform(vectors), "PCA"


def _save_embedding(vectors: np.ndarray, labels: list[str], output: Path, *, title: str) -> str:
    plt, _ = _ensure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    embedding, method = _embedding_2d(vectors)
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=18, color="#2563eb", alpha=0.55)
    for idx, label in enumerate(labels):
        ax.text(embedding[idx, 0], embedding[idx, 1], label, fontsize=7, alpha=0.9)
    ax.set_title(f"{title} ({method})")
    ax.set_xlabel(f"{method} dimension 1")
    ax.set_ylabel(f"{method} dimension 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return method


def _write_pairwise_similarity(labels: list[str], similarity: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group_a", "group_b", "cosine_similarity"])
        for i, label_a in enumerate(labels):
            for j, label_b in enumerate(labels):
                if j <= i:
                    continue
                writer.writerow([label_a, label_b, f"{similarity[i, j]:.8f}"])


def _write_top_experts(
    matrix: np.ndarray,
    labels: list[str],
    module_labels: list[str],
    choice_labels: list[str],
    output: Path,
    *,
    top_k: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_per_module(matrix)
    per_choice = normalized.mean(axis=1)
    flattened = normalized.reshape(normalized.shape[0], -1)
    raw_per_choice = np.nan_to_num(matrix.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0).mean(axis=1)
    raw_flattened = np.nan_to_num(matrix.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0).reshape(matrix.shape[0], -1)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group",
                "rank",
                "summary_choice",
                "summary_lola_normalized_score",
                "summary_raw_activation_mean",
                "module_choice",
                "module_choice_lola_normalized_score",
                "module_choice_raw_activation",
            ]
        )
        for group_idx, label in enumerate(labels):
            choice_order = np.argsort(per_choice[group_idx])[::-1][:top_k]
            flat_order = np.argsort(flattened[group_idx])[::-1][:top_k]
            for rank in range(top_k):
                choice_idx = int(choice_order[rank]) if rank < len(choice_order) else -1
                flat_idx = int(flat_order[rank]) if rank < len(flat_order) else -1
                module_idx, expert_idx = divmod(flat_idx, matrix.shape[2])
                writer.writerow(
                    [
                        label,
                        rank + 1,
                        choice_labels[choice_idx] if choice_idx >= 0 else "",
                        f"{per_choice[group_idx, choice_idx]:.8f}" if choice_idx >= 0 else "",
                        f"{raw_per_choice[group_idx, choice_idx]:.8f}" if choice_idx >= 0 else "",
                        f"{module_labels[module_idx]}::{choice_labels[expert_idx]}" if flat_idx >= 0 else "",
                        f"{flattened[group_idx, flat_idx]:.8f}" if flat_idx >= 0 else "",
                        f"{raw_flattened[group_idx, flat_idx]:.8f}" if flat_idx >= 0 else "",
                    ]
                )


def _write_members(members: dict[str, list[str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "num_languages", "languages"])
        for label, languages in members.items():
            writer.writerow([label, len(languages), " ".join(languages)])


def _save_family_drilldowns(
    matrix: np.ndarray,
    languages: list[str],
    metadata: dict[str, LanguageMeta],
    output_root: Path,
    *,
    title_prefix: str,
    axis_name: str,
    module_labels: list[str],
    min_family_size: int,
) -> list[str]:
    families: dict[str, list[int]] = {}
    ordered = _ordered_languages(languages, metadata)
    language_index = {language: idx for idx, language in enumerate(languages)}
    for language in ordered:
        families.setdefault(metadata[language].family, []).append(language_index[language])

    written: list[str] = []
    for family, indices in families.items():
        if len(indices) < min_family_size:
            continue
        family_languages = [languages[idx] for idx in indices]
        path = output_root / f"{_slug(family)}.png"
        _save_lola_heatmap(
            matrix[indices],
            family_languages,
            path,
            title=f"{title_prefix}: {family} language drilldown",
            axis_name=axis_name,
            module_labels=module_labels,
        )
        written.append(str(path))
        paper_path = output_root / f"{_slug(family)}_lola_coling_style.png"
        _save_lola_heatmap(
            matrix[indices],
            family_languages,
            paper_path,
            title=f"{title_prefix}: {family} language drilldown",
            axis_name=axis_name,
            module_labels=module_labels,
            paper_style=True,
        )
        written.append(str(paper_path))
    return written


def analyze_run(
    run_dir: Path,
    output_root: Path,
    metadata: dict[str, LanguageMeta],
    *,
    granularities: list[str],
    top_k: int,
    min_family_size: int,
) -> dict[str, object]:
    npz = np.load(run_dir / "routing_matrix.npz", allow_pickle=True)
    languages = _as_str_array(npz["languages"])
    module_labels = _as_str_array(npz["layer_labels"])
    choice_labels = _as_str_array(npz["choice_labels"])
    axis_name = str(npz["axis_name"])
    run_id = str(npz["run_id"])
    peft_type = str(npz["peft_type"])

    if "routing_probability" in npz.files:
        matrix = npz["routing_probability"].astype(np.float64)
        matrix_source = "routing_probability"
    elif "routing_count_normalized" in npz.files:
        matrix = npz["routing_count_normalized"].astype(np.float64)
        matrix_source = "routing_count_normalized"
    else:
        matrix = npz["routing_weight_normalized"].astype(np.float64)
        matrix_source = "routing_weight_normalized"
    layer_aggregate = _aggregate_to_base_layers(npz, module_labels)

    run_output = output_root / run_id
    run_output.mkdir(parents=True, exist_ok=True)
    details: dict[str, object] = {
        "run_id": run_id,
        "peft_type": peft_type,
        "axis_name": axis_name,
        "matrix_source": matrix_source,
        "granularities": {},
        "family_drilldowns": [],
    }

    for granularity in granularities:
        group_matrix, labels, members = _aggregate_by_group(matrix, languages, metadata, granularity)
        vectors = _normalize_per_module(group_matrix).reshape(group_matrix.shape[0], -1)
        similarity = _cosine_similarity(vectors)
        granularity_dir = run_output / granularity
        _save_lola_heatmap(
            group_matrix,
            labels,
            granularity_dir / "lola_routing_ratio_heatmap.png",
            title=f"{run_id} LOLA-style routing ratio by {granularity}",
            axis_name=axis_name,
            module_labels=module_labels,
        )
        _save_lola_heatmap(
            group_matrix,
            labels,
            granularity_dir / "lola_coling_style_heatmap.png",
            title=f"{run_id} LOLA-style routing ratio by {granularity}",
            axis_name=axis_name,
            module_labels=module_labels,
            paper_style=True,
        )
        _save_similarity_heatmap(
            similarity,
            labels,
            granularity_dir / "routing_cosine_similarity.png",
            title=f"{run_id} routing-vector similarity by {granularity}",
        )
        embedding_method = _save_embedding(
            vectors,
            labels,
            granularity_dir / "routing_embedding_2d.png",
            title=f"{run_id} normalized routing vectors by {granularity}",
        )
        _write_pairwise_similarity(labels, similarity, granularity_dir / "pairwise_similarity.csv")
        _write_top_experts(
            group_matrix,
            labels,
            module_labels,
            choice_labels,
            granularity_dir / "top_experts_by_group.csv",
            top_k=top_k,
        )
        _write_members(members, granularity_dir / "group_members.csv")
        layer_files: list[str] = []
        if layer_aggregate is not None:
            layer_matrix_all, layer_labels = layer_aggregate
            group_layer_matrix, _, _ = _aggregate_by_group(layer_matrix_all, languages, metadata, granularity)
            layer_dir = granularity_dir / "by_base_layer"
            _save_lola_heatmap(
                group_layer_matrix,
                labels,
                layer_dir / "layer_expert_routing_ratio_heatmap.png",
                title=f"{run_id} routing by {granularity}: base layers x {axis_name}s",
                axis_name=axis_name,
                module_labels=layer_labels,
                x_group_label="base model layer",
            )
            _save_lola_heatmap(
                group_layer_matrix,
                labels,
                layer_dir / "layer_expert_lola_coling_style_heatmap.png",
                title=f"{run_id} routing by {granularity}: base layers x {axis_name}s",
                axis_name=axis_name,
                module_labels=layer_labels,
                x_group_label="base model layer",
                paper_style=True,
            )
            layer_files.append(str(layer_dir / "layer_expert_routing_ratio_heatmap.png"))
            layer_files.append(str(layer_dir / "layer_expert_lola_coling_style_heatmap.png"))
            _write_layer_expert_totals(
                group_layer_matrix,
                labels,
                layer_labels,
                choice_labels,
                layer_dir / "top_layer_experts_by_group.csv",
                top_k=top_k,
            )
            layer_files.append(str(layer_dir / "top_layer_experts_by_group.csv"))
            for zoom_name, layer_indices in _layer_zoom_ranges(layer_labels):
                zoom_labels = [layer_labels[idx] for idx in layer_indices]
                zoom_matrix = group_layer_matrix[:, layer_indices, :]
                zoom_path = layer_dir / f"zoom_{zoom_name}.png"
                _save_lola_heatmap(
                    zoom_matrix,
                    labels,
                    zoom_path,
                    title=f"{run_id} {granularity} zoom: {', '.join(zoom_labels)}",
                    axis_name=axis_name,
                    module_labels=zoom_labels,
                    x_group_label="base model layer",
                )
                layer_files.append(str(zoom_path))
                paper_zoom_path = layer_dir / f"zoom_{zoom_name}_lola_coling_style.png"
                _save_lola_heatmap(
                    zoom_matrix,
                    labels,
                    paper_zoom_path,
                    title=f"{run_id} {granularity} zoom: {', '.join(zoom_labels)}",
                    axis_name=axis_name,
                    module_labels=zoom_labels,
                    x_group_label="base model layer",
                    paper_style=True,
                )
                layer_files.append(str(paper_zoom_path))
        details["granularities"][granularity] = {
            "groups": len(labels),
            "embedding_method": embedding_method,
            "files": [
                str(granularity_dir / "lola_routing_ratio_heatmap.png"),
                str(granularity_dir / "lola_coling_style_heatmap.png"),
                str(granularity_dir / "routing_cosine_similarity.png"),
                str(granularity_dir / "routing_embedding_2d.png"),
                str(granularity_dir / "pairwise_similarity.csv"),
                str(granularity_dir / "top_experts_by_group.csv"),
                str(granularity_dir / "group_members.csv"),
            ]
            + layer_files,
        }

    drilldown_dir = run_output / "family_drilldowns"
    details["family_drilldowns"] = _save_family_drilldowns(
        matrix,
        languages,
        metadata,
        drilldown_dir,
        title_prefix=f"{run_id} LOLA-style routing",
        axis_name=axis_name,
        module_labels=module_labels,
        min_family_size=min_family_size,
    )
    return details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create LOLA-style grouped routing visualizations and metrics.")
    parser.add_argument("--routing-root", type=Path, required=True, help="Routing analysis output root with summary.json.")
    parser.add_argument("--languages-csv", type=Path, required=True, help="Language metadata CSV, e.g. result_analysis/l2v_nested_samples/96.csv.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to <routing-root>/lola_style.")
    parser.add_argument(
        "--granularity",
        action="append",
        choices=["language", "family", "family_script", "family_resource"],
        help="Granularity to render. Can be repeated.",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Number of top experts/module-experts to write per group.")
    parser.add_argument("--min-family-size", type=int, default=2, help="Minimum family size for per-family drilldown plots.")
    args = parser.parse_args(argv)

    routing_root = args.routing_root.resolve()
    output_root = (args.output_dir or (routing_root / "lola_style")).resolve()
    summary = json.loads((routing_root / "summary.json").read_text(encoding="utf-8"))
    run_ids = [str(item["run_id"]) for item in summary.get("runs", [])]
    if not run_ids:
        raise SystemExit(f"No runs found in {routing_root / 'summary.json'}")

    first_npz = np.load(routing_root / run_ids[0] / "routing_matrix.npz", allow_pickle=True)
    languages = _as_str_array(first_npz["languages"])
    metadata = _load_language_metadata(args.languages_csv.resolve(), languages)
    granularities = args.granularity or ["family", "family_script", "language"]

    output_root.mkdir(parents=True, exist_ok=True)
    details = {
        "routing_root": str(routing_root),
        "languages_csv": str(args.languages_csv.resolve()),
        "output_root": str(output_root),
        "num_languages": len(languages),
        "granularities": granularities,
        "runs": [],
        "lola_reference": {
            "paper": "LOLA -- An Open-Source Massively Multilingual Large Language Model",
            "local_pdf": "papers/local_library/LOLA/lola_coling_2025.pdf",
            "method": "Layer-normalized expert-routing vectors; grouped heatmap and normalized-vector similarity/embedding views.",
        },
    }
    for run_id in run_ids:
        details["runs"].append(
            analyze_run(
                routing_root / run_id,
                output_root,
                metadata,
                granularities=granularities,
                top_k=args.top_k,
                min_family_size=args.min_family_size,
            )
        )

    (output_root / "lola_style_summary.json").write_text(json.dumps(details, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] wrote LOLA-style routing outputs to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
