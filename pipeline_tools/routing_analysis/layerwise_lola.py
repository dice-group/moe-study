from __future__ import annotations

import argparse
import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .lola_style import (
    LanguageMeta,
    _aggregate_by_group,
    _as_str_array,
    _embedding_2d,
    _ensure_matplotlib,
    _load_language_metadata,
    _normalize_per_module,
    _write_members,
)


@dataclass(frozen=True)
class LayerViews:
    matrix: np.ndarray
    labels: list[str]
    scope: str
    note: str
    source_modules: int


def resolve_layer_views(matrix: np.ndarray, module_labels: list[str]) -> LayerViews:
    """Resolve honest layer views without duplicating a shared router."""
    if matrix.ndim != 3:
        raise ValueError(f"Expected [language, module, choice] routing matrix, got {matrix.shape}")
    if matrix.shape[1] != len(module_labels):
        raise ValueError(
            f"Routing module axis has {matrix.shape[1]} entries but {len(module_labels)} labels were provided"
        )

    parsed = []
    for label in module_labels:
        match = re.match(r"^L(\d+):", label)
        parsed.append(int(match.group(1)) if match else None)

    if all(layer is not None for layer in parsed):
        unique_layers = sorted(set(parsed))
        layer_matrix = np.stack(
            [
                matrix[:, [index for index, current in enumerate(parsed) if current == layer], :].mean(axis=1)
                for layer in unique_layers
            ],
            axis=1,
        )
        return LayerViews(
            matrix=layer_matrix,
            labels=[f"L{layer}" for layer in unique_layers],
            scope="base_layer_aggregate",
            note=(
                "Captured adapter modules are averaged within each base-model layer. "
                "Each displayed layer is a real checkpoint/runtime layer aggregate."
            ),
            source_modules=len(module_labels),
        )

    if any(layer is not None for layer in parsed):
        raise ValueError("Module labels mix base-layer-prefixed and unscoped routing modules")

    if matrix.shape[1] == 1:
        return LayerViews(
            matrix=matrix,
            labels=["global_shared"],
            scope="global_shared",
            note=(
                "This approach exposes one shared language-conditioned router. "
                "It is shown once and is not duplicated across artificial base layers."
            ),
            source_modules=1,
        )

    return LayerViews(
        matrix=matrix,
        labels=module_labels,
        scope="captured_module",
        note=(
            "No base-layer prefix is available, so each captured routing module is shown independently."
        ),
        source_modules=len(module_labels),
    )


def _routing_matrix(npz: np.lib.npyio.NpzFile) -> tuple[np.ndarray, str]:
    if "routing_probability" in npz.files:
        return npz["routing_probability"].astype(np.float64), "routing_probability"
    if "routing_count_normalized" in npz.files:
        return npz["routing_count_normalized"].astype(np.float64), "routing_count_normalized"
    return npz["routing_weight_normalized"].astype(np.float64), "routing_weight_normalized"


def _save_choice_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    choice_labels: list[str],
    output: Path,
    *,
    title: str,
    axis_name: str,
) -> None:
    plt, linear_segmented_colormap = _ensure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(
        np.nan_to_num(matrix.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0,
    )
    uniform = 1.0 / max(1, data.shape[1])
    cmap = linear_segmented_colormap.from_list(
        "lola_layer_ratio",
        [
            (0.0, "#ffffff"),
            (min(uniform, 1.0), "#b8f5a5"),
            (min(2 * uniform, 1.0), "#ffff00"),
            (min(4 * uniform, 1.0), "#ff1f1f"),
            (0.5, "#b00020"),
            (1.0, "#000000"),
        ],
    )
    fig_width = max(8, min(22, data.shape[1] * 0.7))
    fig_height = max(5, min(32, len(labels) * 0.22))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel(axis_name.replace("_", " ").title())
    ax.set_ylabel("Language group")
    ax.set_xticks(range(len(choice_labels)))
    ax.set_xticklabels(choice_labels, rotation=45, ha="right", fontsize=7)
    if len(labels) <= 140:
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6 if len(labels) > 70 else 8)
    else:
        ax.set_yticks([])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="routing probability")
    ticks = sorted({0.0, min(uniform, 1.0), min(2 * uniform, 1.0), min(4 * uniform, 1.0), 1.0})
    colorbar.set_ticks(ticks)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _categories(
    labels: list[str],
    metadata: dict[str, LanguageMeta],
    granularity: str,
) -> list[str]:
    if granularity == "language":
        return [metadata[label].family for label in labels]
    return [label.split(" / ", 1)[0] for label in labels]


def _save_tsne(
    vectors: np.ndarray,
    labels: list[str],
    categories: list[str],
    members: dict[str, list[str]],
    image_path: Path,
    coordinates_path: Path,
    *,
    title: str,
) -> str:
    plt, _ = _ensure_matplotlib()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    embedding, method = _embedding_2d(vectors)
    unique_categories = list(dict.fromkeys(categories))
    color_map = plt.get_cmap("tab20")
    colors = {
        category: color_map(index % color_map.N)
        for index, category in enumerate(unique_categories)
    }

    with coordinates_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "category", "x", "y", "embedding_method", "members"])
        for index, label in enumerate(labels):
            writer.writerow(
                [
                    label,
                    categories[index],
                    f"{embedding[index, 0]:.10f}",
                    f"{embedding[index, 1]:.10f}",
                    method,
                    " ".join(members[label]),
                ]
            )

    fig, ax = plt.subplots(figsize=(13, 9))
    for category in unique_categories:
        indices = [index for index, current in enumerate(categories) if current == category]
        ax.scatter(
            embedding[indices, 0],
            embedding[indices, 1],
            s=24,
            color=colors[category],
            alpha=0.72,
            label=category,
        )

    if len(labels) <= 120:
        for index, label in enumerate(labels):
            ax.text(embedding[index, 0], embedding[index, 1], label, fontsize=6, alpha=0.85)
    else:
        for category in unique_categories:
            indices = [index for index, current in enumerate(categories) if current == category]
            ax.text(
                float(embedding[indices, 0].mean()),
                float(embedding[indices, 1].mean()),
                category,
                fontsize=7,
                alpha=0.9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 1.5},
            )

    ax.set_title(f"{title} ({method})")
    ax.set_xlabel(f"{method} dimension 1")
    ax.set_ylabel(f"{method} dimension 2")
    ax.grid(True, alpha=0.2)
    if len(unique_categories) <= 24:
        ax.legend(
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=6,
            frameon=False,
            ncol=1 if len(unique_categories) <= 12 else 2,
        )
    fig.tight_layout()
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    return method


def _write_run_index(atlas_dir: Path, details: dict[str, object]) -> None:
    sections = []
    for layer in details["layers"]:
        views = []
        for granularity, view in layer["views"].items():
            views.append(
                "<div class='view'>"
                f"<h3>{html.escape(granularity)}</h3>"
                f"<img src='{html.escape(view['heatmap'])}' alt='{html.escape(granularity)} heatmap'>"
                f"<img src='{html.escape(view['tsne'])}' alt='{html.escape(granularity)} t-SNE'>"
                f"<p><a href='{html.escape(view['coordinates'])}'>t-SNE coordinates</a></p>"
                "</div>"
            )
        sections.append(
            f"<section><h2>{html.escape(layer['label'])}</h2>"
            "<div class='grid'>"
            + "".join(views)
            + "</div></section>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(details['run_id'])} layerwise routing atlas</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; color: #172033; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 1.5rem; }}
.view {{ border: 1px solid #d8deea; border-radius: 8px; padding: 1rem; }}
img {{ width: 100%; height: auto; }}
code {{ background: #f1f4f8; padding: 0.15rem 0.3rem; }}
</style>
</head>
<body>
<h1>{html.escape(details['run_id'])} layerwise routing atlas</h1>
<p><strong>Axis:</strong> {html.escape(details['axis_name'])};
<strong>scope:</strong> <code>{html.escape(details['routing_scope'])}</code>.</p>
<p>{html.escape(details['scope_note'])}</p>
<p>Every t-SNE is fitted independently with cosine distance and a fixed seed.
Coordinates support within-view neighborhoods only; axes and distances must not be compared across layers.</p>
{''.join(sections)}
</body>
</html>
"""
    (atlas_dir / "index.html").write_text(document, encoding="utf-8")


def analyze_run(
    run_dir: Path,
    output_root: Path,
    metadata: dict[str, LanguageMeta],
    *,
    granularities: list[str],
) -> dict[str, object]:
    with np.load(run_dir / "routing_matrix.npz", allow_pickle=True) as npz:
        languages = _as_str_array(npz["languages"])
        module_labels = _as_str_array(npz["layer_labels"])
        choice_labels = _as_str_array(npz["choice_labels"])
        axis_name = str(npz["axis_name"])
        run_id = str(npz["run_id"])
        peft_type = str(npz["peft_type"])
        matrix, matrix_source = _routing_matrix(npz)

    views = resolve_layer_views(matrix, module_labels)
    atlas_dir = output_root / run_id / "layerwise_atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    details: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "peft_type": peft_type,
        "axis_name": axis_name,
        "routing_basis": run_metadata["routing_basis"],
        "routing_scope": views.scope,
        "scope_note": views.note,
        "matrix_source": matrix_source,
        "source_modules": views.source_modules,
        "displayed_layers": len(views.labels),
        "choices": len(choice_labels),
        "num_languages": len(languages),
        "granularities": granularities,
        "coordinate_guardrail": (
            "Each t-SNE is fitted independently. Compare neighborhoods within a view, "
            "not absolute coordinates across layers or approaches."
        ),
        "layers": [],
    }

    grouped: dict[str, tuple[list[str], dict[str, list[str]]]] = {}
    for granularity in granularities:
        _, labels, members = _aggregate_by_group(
            views.matrix[:, :1, :],
            languages,
            metadata,
            granularity,
        )
        grouped[granularity] = (labels, members)
        _write_members(members, atlas_dir / granularity / "group_members.csv")

    for layer_index, layer_label in enumerate(views.labels):
        layer_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", layer_label).strip("_") or f"layer_{layer_index}"
        layer_slice = views.matrix[:, layer_index : layer_index + 1, :]
        layer_details: dict[str, object] = {
            "index": layer_index,
            "label": layer_label,
            "directory": f"layers/{layer_slug}",
            "views": {},
        }
        for granularity in granularities:
            group_matrix, labels, members = _aggregate_by_group(
                layer_slice,
                languages,
                metadata,
                granularity,
            )
            assert labels == grouped[granularity][0]
            view_dir = atlas_dir / "layers" / layer_slug / granularity
            heatmap_path = view_dir / "routing_heatmap.png"
            tsne_path = view_dir / "routing_tsne.png"
            coordinates_path = view_dir / "routing_tsne_coordinates.csv"
            _save_choice_heatmap(
                group_matrix[:, 0, :],
                labels,
                choice_labels,
                heatmap_path,
                title=f"{run_id} {layer_label}: routing by {granularity}",
                axis_name=axis_name,
            )
            vectors = _normalize_per_module(group_matrix).reshape(group_matrix.shape[0], -1)
            method = _save_tsne(
                vectors,
                labels,
                _categories(labels, metadata, granularity),
                members,
                tsne_path,
                coordinates_path,
                title=f"{run_id} {layer_label}: routing vectors by {granularity}",
            )
            layer_details["views"][granularity] = {
                "groups": len(labels),
                "embedding_method": method,
                "heatmap": str(heatmap_path.relative_to(atlas_dir)),
                "tsne": str(tsne_path.relative_to(atlas_dir)),
                "coordinates": str(coordinates_path.relative_to(atlas_dir)),
            }
        details["layers"].append(layer_details)

    (atlas_dir / "atlas_manifest.json").write_text(
        json.dumps(details, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_run_index(atlas_dir, details)
    return {
        "run_id": run_id,
        "axis_name": axis_name,
        "routing_scope": views.scope,
        "displayed_layers": len(views.labels),
        "manifest": str(atlas_dir / "atlas_manifest.json"),
        "index": str(atlas_dir / "index.html"),
    }


def _write_panel_index(output_root: Path, details: dict[str, object]) -> None:
    rows = []
    for run in details["runs"]:
        relative = Path(run["index"]).relative_to(output_root)
        rows.append(
            "<tr>"
            f"<td><a href='{html.escape(str(relative))}'>{html.escape(run['run_id'])}</a></td>"
            f"<td>{html.escape(run['axis_name'])}</td>"
            f"<td>{html.escape(run['routing_scope'])}</td>"
            f"<td>{run['displayed_layers']}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Layerwise routing atlases</title></head>
<body style="font-family:sans-serif;margin:2rem">
<h1>Layerwise routing atlases</h1>
<p>All approaches use the same output schema. Shared/global routers appear once rather than being
copied into fake base layers.</p>
<table border="1" cellspacing="0" cellpadding="8">
<thead><tr><th>Approach</th><th>Choice axis</th><th>Routing scope</th><th>Views</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body>
</html>
"""
    (output_root / "layerwise_atlas_index.html").write_text(document, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create honest per-layer LOLA-style routing heatmaps and t-SNE atlases."
    )
    parser.add_argument("--routing-root", type=Path, required=True)
    parser.add_argument("--languages-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <routing-root>/lola_style.",
    )
    parser.add_argument(
        "--granularity",
        action="append",
        choices=["language", "family", "family_script", "family_resource"],
        help="Can be repeated. Defaults to language and family.",
    )
    parser.add_argument("--run-id", action="append", default=[], help="Limit to selected runs.")
    args = parser.parse_args(argv)

    routing_root = args.routing_root.resolve()
    output_root = (args.output_dir or (routing_root / "lola_style")).resolve()
    panel_summary = json.loads((routing_root / "summary.json").read_text(encoding="utf-8"))
    configured_runs = [str(item["run_id"]) for item in panel_summary.get("runs", [])]
    run_ids = args.run_id or configured_runs
    unknown = sorted(set(run_ids) - set(configured_runs))
    if unknown:
        raise SystemExit(f"Unknown run ids: {unknown}")
    if not run_ids:
        raise SystemExit(f"No runs found in {routing_root / 'summary.json'}")

    with np.load(routing_root / run_ids[0] / "routing_matrix.npz", allow_pickle=True) as npz:
        languages = _as_str_array(npz["languages"])
    metadata = _load_language_metadata(args.languages_csv.resolve(), languages)
    granularities = args.granularity or ["language", "family"]

    output_root.mkdir(parents=True, exist_ok=True)
    details: dict[str, object] = {
        "schema_version": 1,
        "routing_root": str(routing_root),
        "languages_csv": str(args.languages_csv.resolve()),
        "output_root": str(output_root),
        "num_languages": len(languages),
        "granularities": granularities,
        "runs": [],
    }
    for run_id in run_ids:
        details["runs"].append(
            analyze_run(
                routing_root / run_id,
                output_root,
                metadata,
                granularities=granularities,
            )
        )

    (output_root / "layerwise_atlas_summary.json").write_text(
        json.dumps(details, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_panel_index(output_root, details)
    print(f"[OK] wrote layerwise LOLA/t-SNE atlases to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
