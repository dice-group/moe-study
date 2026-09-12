from __future__ import annotations

import argparse
import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .layerwise_lola import _routing_matrix
from .lola_style import (
    LanguageMeta,
    _aggregate_by_group,
    _as_str_array,
    _embedding_2d,
    _ensure_matplotlib,
    _load_language_metadata,
    _normalize_per_module,
    _ordered_languages,
)


TARGET_ORDER = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
EXPERT_COLORS = (
    "#2979ff",
    "#ff8a34",
    "#2fbf71",
    "#ef476f",
    "#8e63ce",
    "#8d6e63",
    "#ec77b4",
    "#26b6c9",
)
MODULE_PATTERN = re.compile(
    r"^L(?P<layer>\d+):.*\.(?P<target>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)(?:\.|$)"
)


@dataclass(frozen=True)
class StructuredRouting:
    matrix: np.ndarray
    layers: list[int]
    targets: list[str]


def structure_routing(
    matrix: np.ndarray,
    module_labels: list[str],
    *,
    target_order: tuple[str, ...] = TARGET_ORDER,
) -> StructuredRouting:
    """Map [group, module, choice] routing into [group, layer, target, choice]."""
    if matrix.ndim != 3:
        raise ValueError(f"Expected [group, module, choice], got {matrix.shape}")
    if matrix.shape[1] != len(module_labels):
        raise ValueError(
            f"Routing module axis has {matrix.shape[1]} entries but {len(module_labels)} labels were provided"
        )

    positions: dict[tuple[int, str], int] = {}
    for module_index, label in enumerate(module_labels):
        match = MODULE_PATTERN.match(label)
        if match is None:
            raise ValueError(f"Cannot resolve base layer and LoRA target from module label: {label}")
        key = (int(match.group("layer")), match.group("target"))
        if key in positions:
            raise ValueError(f"Duplicate routing module for L{key[0]}:{key[1]}")
        positions[key] = module_index

    layers = sorted({layer for layer, _ in positions})
    expected = {(layer, target) for layer in layers for target in target_order}
    missing = sorted(expected - set(positions))
    unexpected = sorted(set(positions) - expected)
    if missing or unexpected:
        raise ValueError(
            "Layer-target routing grid is incomplete or unexpected: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    structured = np.stack(
        [
            np.stack(
                [matrix[:, positions[(layer, target)], :] for target in target_order],
                axis=1,
            )
            for layer in layers
        ],
        axis=1,
    )
    return StructuredRouting(
        matrix=structured,
        layers=layers,
        targets=list(target_order),
    )


def _save_overall_heatmap(
    matrix: np.ndarray,
    layers: list[int],
    targets: list[str],
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
    ).reshape(len(layers), -1)
    choices = len(choice_labels)
    uniform = 1.0 / max(1, choices)
    cmap = linear_segmented_colormap.from_list(
        "target_expert_lola",
        [
            (0.0, "#ffffff"),
            (min(uniform, 1.0), "#b8f5a5"),
            (min(2 * uniform, 1.0), "#ffff00"),
            (min(4 * uniform, 1.0), "#ff1f1f"),
            (0.5, "#b00020"),
            (1.0, "#000000"),
        ],
    )

    fig, ax = plt.subplots(figsize=(18, 11))
    image = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, pad=38)
    ax.set_ylabel("Base-model layer")
    ax.set_xlabel(f"{axis_name.replace('_', ' ').title()} within each LoRA target")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{layer}" for layer in layers], fontsize=7)
    ax.set_xticks(range(len(targets) * choices))
    ax.set_xticklabels(
        [f"E{index}" for _ in targets for index in range(choices)],
        fontsize=5,
        rotation=90,
    )
    for boundary in range(1, len(targets)):
        ax.axvline(boundary * choices - 0.5, color="#172033", linewidth=1.4)
    for boundary in range(4, len(layers), 4):
        ax.axhline(boundary - 0.5, color="#ffffff", linewidth=0.8, alpha=0.8)

    top = ax.secondary_xaxis("top")
    top.set_xticks([index * choices + (choices - 1) / 2 for index in range(len(targets))])
    top.set_xticklabels(targets, fontsize=9)
    top.tick_params(length=0, pad=5)

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.028,
        pad=0.02,
        label="mean routing probability",
    )
    colorbar.set_ticks(
        sorted({0.0, min(uniform, 1.0), min(2 * uniform, 1.0), min(4 * uniform, 1.0), 1.0})
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _save_all_languages_heatmap(
    matrix: np.ndarray,
    languages: list[str],
    metadata: dict[str, LanguageMeta],
    layers: list[int],
    targets: list[str],
    choice_labels: list[str],
    output: Path,
    *,
    title: str,
) -> None:
    plt, linear_segmented_colormap = _ensure_matplotlib()
    ordered = _ordered_languages(languages, metadata)
    language_index = {language: index for index, language in enumerate(languages)}
    ordered_matrix = matrix[[language_index[language] for language in ordered]]
    data = ordered_matrix.transpose(0, 2, 1, 3).reshape(
        len(ordered) * len(targets),
        len(layers) * len(choice_labels),
    )
    choices = len(choice_labels)
    uniform = 1.0 / max(1, choices)
    cmap = linear_segmented_colormap.from_list(
        "all_language_target_expert_lola",
        [
            (0.0, "#ffffff"),
            (min(uniform, 1.0), "#b8f5a5"),
            (min(2 * uniform, 1.0), "#ffff00"),
            (min(4 * uniform, 1.0), "#ff1f1f"),
            (0.5, "#b00020"),
            (1.0, "#000000"),
        ],
    )
    fig, ax = plt.subplots(figsize=(18, 34))
    image = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, pad=30)
    ax.set_ylabel(
        "Language; seven target subrows each: q, k, v, o, gate, up, down"
    )
    ax.set_xlabel("Base-model layer; each layer contains E0–E7")
    ax.set_yticks(
        [index * len(targets) + (len(targets) - 1) / 2 for index in range(len(ordered))]
    )
    ax.set_yticklabels(ordered, fontsize=3.2)
    ax.set_xticks(
        [index * choices + (choices - 1) / 2 for index in range(len(layers))]
    )
    ax.set_xticklabels([f"L{layer}" for layer in layers], fontsize=6, rotation=90)
    for boundary in range(1, len(layers)):
        ax.axvline(boundary * choices - 0.5, color="#172033", linewidth=0.35)
    previous_family = None
    for index, language in enumerate(ordered):
        family = metadata[language].family
        if family != previous_family:
            ax.axhline(index * len(targets) - 0.5, color="#172033", linewidth=0.55)
            previous_family = family
    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.018,
        pad=0.012,
        label="routing probability",
    )
    colorbar.set_ticks(
        sorted({0.0, min(uniform, 1.0), min(2 * uniform, 1.0), min(4 * uniform, 1.0), 1.0})
    )
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def _hex_to_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.asarray(
        [int(value[index : index + 2], 16) / 255 for index in (0, 2, 4)],
        dtype=np.float64,
    )


def _usage_cmap():
    _, linear_segmented_colormap = _ensure_matplotlib()
    uniform = 1 / 8
    return linear_segmented_colormap.from_list(
        "lola_expert_usage",
        [
            (0.0, "#edf8e9"),
            (uniform, "#66bd63"),
            (2 * uniform, "#fee08b"),
            (4 * uniform, "#f46d43"),
            (1.0, "#a50026"),
        ],
    )


def _save_compact_usage_svg(
    matrix: np.ndarray,
    row_labels: list[str],
    layers: list[int],
    targets: list[str],
    output: Path,
    *,
    title: str,
    row_axis_label: str,
) -> None:
    """Write a compact browser-friendly vector SVG without a huge element tree."""
    from html import escape

    choices = matrix.shape[-1]
    if choices != 8:
        raise ValueError(f"The 2x4 expert glyph requires exactly 8 choices, got {choices}")

    width = 2200.0
    left, right, top, bottom, gap = 250.0, 35.0, 120.0, 145.0, 18.0
    row_height = 10.0
    plot_height = len(row_labels) * row_height
    height = top + plot_height + bottom
    panel_width = (width - left - right - gap * (len(targets) - 1)) / len(targets)
    cell_width = panel_width / len(layers)
    micro_width, micro_height = cell_width / 4, row_height / 2

    stops = [
        (0.0, "#edf8e9"),
        (0.125, "#66bd63"),
        (0.25, "#fee08b"),
        (0.5, "#f46d43"),
        (1.0, "#a50026"),
    ]

    def usage_hex(value: float) -> str:
        value = max(0.0, min(1.0, value))
        for index in range(1, len(stops)):
            if value <= stops[index][0]:
                start_value, start_hex = stops[index - 1]
                end_value, end_hex = stops[index]
                fraction = (value - start_value) / (end_value - start_value)
                start_rgb, end_rgb = _hex_to_rgb(start_hex), _hex_to_rgb(end_hex)
                rgb = np.rint((start_rgb + fraction * (end_rgb - start_rgb)) * 255).astype(int)
                return "#" + "".join(f"{channel:02x}" for channel in rgb)
        return stops[-1][1]

    palette = [usage_hex(index / 255) for index in range(256)]
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
            f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">'
        ),
        "<defs>",
        '<linearGradient id="usageScale" x1="0" x2="1" y1="0" y2="0">',
        '<stop offset="0" stop-color="#edf8e9"/>',
        '<stop offset="0.125" stop-color="#66bd63"/>',
        '<stop offset="0.25" stop-color="#fee08b"/>',
        '<stop offset="0.5" stop-color="#f46d43"/>',
        '<stop offset="1" stop-color="#a50026"/>',
        "</linearGradient>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        (
            f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" '
            f'font-family="sans-serif" font-size="20" font-weight="700">{escape(title)}</text>'
        ),
    ]

    for target_index, target in enumerate(targets):
        panel_x = left + target_index * (panel_width + gap)
        parts.append(
            f'<text x="{panel_x + panel_width / 2:.2f}" y="54" text-anchor="middle" '
            f'font-family="sans-serif" font-size="15" font-weight="700">'
            f'{escape(target.replace("_proj", ""))}</text>'
        )
        for layer_index, layer in enumerate(layers):
            layer_x = panel_x + layer_index * cell_width
            parts.append(
                f'<text x="{layer_x + cell_width / 2:.2f}" y="89" '
                f'transform="rotate(-90 {layer_x + cell_width / 2:.2f} 89)" '
                f'text-anchor="start" font-family="sans-serif" font-size="4.5" '
                f'fill="#344054">L{layer}</text>'
            )
            for expert in range(choices):
                expert_x = layer_x + (expert % 4 + 0.5) * micro_width
                expert_y = 108 + (expert // 4) * 5
                parts.append(
                    f'<text x="{expert_x:.2f}" y="{expert_y:.1f}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="2.2" font-weight="700" '
                    f'fill="#667085">{expert}</text>'
                )

        groups: list[list[str]] = [[] for _ in range(256)]
        values = np.clip(matrix[:, :, target_index, :], 0, 1)
        for row_index in range(len(row_labels)):
            row_y = top + row_index * row_height
            for layer_index in range(len(layers)):
                layer_x = panel_x + layer_index * cell_width
                for expert in range(choices):
                    probability = float(values[row_index, layer_index, expert])
                    color_index = max(0, min(255, round(probability * 255)))
                    tile_x = layer_x + (expert % 4) * micro_width
                    tile_y = row_y + (expert // 4) * micro_height
                    groups[color_index].append(
                        f"M{tile_x:.2f},{tile_y:.2f}h{micro_width:.2f}v{micro_height:.2f}"
                        f"h-{micro_width:.2f}z"
                    )
        for color_index, segments in enumerate(groups):
            for offset in range(0, len(segments), 1500):
                parts.append(
                    f'<path d="{"".join(segments[offset : offset + 1500])}" '
                    f'fill="{palette[color_index]}"/>'
                )

        for boundary in range(len(layers) + 1):
            boundary_x = panel_x + boundary * cell_width
            color = "#172033" if boundary in {10, 22} else "#667085"
            stroke_width = 1.4 if boundary in {10, 22} else 0.45
            parts.append(
                f'<line x1="{boundary_x:.2f}" x2="{boundary_x:.2f}" '
                f'y1="{top:.2f}" y2="{top + plot_height:.2f}" '
                f'stroke="{color}" stroke-width="{stroke_width}"/>'
            )
        for row_boundary in range(len(row_labels) + 1):
            boundary_y = top + row_boundary * row_height
            parts.append(
                f'<line x1="{panel_x:.2f}" x2="{panel_x + panel_width:.2f}" '
                f'y1="{boundary_y:.2f}" y2="{boundary_y:.2f}" '
                f'stroke="#ffffff" stroke-opacity="0.48" stroke-width="0.3"/>'
            )

    for row_index, label in enumerate(row_labels):
        parts.append(
            f'<text x="{left - 8:.2f}" y="{top + (row_index + 0.66) * row_height:.2f}" '
            f'text-anchor="end" font-family="sans-serif" font-size="4.6" '
            f'fill="#172033">{escape(label)}</text>'
        )
    parts.append(
        f'<text x="28" y="{top + plot_height / 2:.2f}" '
        f'transform="rotate(-90 28 {top + plot_height / 2:.2f})" text-anchor="middle" '
        f'font-family="sans-serif" font-size="13">{escape(row_axis_label)}</text>'
    )

    bar_x, bar_y, bar_width, bar_height = 690.0, height - 78.0, 820.0, 17.0
    parts.extend(
        [
            f'<rect x="{bar_x}" y="{bar_y}" width="{bar_width}" height="{bar_height}" '
            'fill="url(#usageScale)" stroke="#98a2b3" stroke-width="0.5"/>',
            f'<text x="{bar_x}" y="{bar_y + 34}" font-family="sans-serif" font-size="9">0</text>',
            f'<text x="{bar_x + bar_width * 0.125}" y="{bar_y + 34}" text-anchor="middle" font-family="sans-serif" font-size="9">uniform 1/8</text>',
            f'<text x="{bar_x + bar_width * 0.25}" y="{bar_y + 34}" text-anchor="middle" font-family="sans-serif" font-size="9">1/4</text>',
            f'<text x="{bar_x + bar_width * 0.5}" y="{bar_y + 34}" text-anchor="middle" font-family="sans-serif" font-size="9">1/2</text>',
            f'<text x="{bar_x + bar_width}" y="{bar_y + 34}" text-anchor="end" font-family="sans-serif" font-size="9">1</text>',
            f'<text x="{width / 2}" y="{height - 21}" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#344054">routing probability: green = low usage · red = high usage · within each layer: top E0–E3, bottom E4–E7</text>',
            "</svg>",
        ]
    )
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")



def _save_compact_usage_overview(
    matrix: np.ndarray,
    row_labels: list[str],
    layers: list[int],
    targets: list[str],
    output: Path,
    *,
    title: str,
    row_axis_label: str,
) -> None:
    if output.suffix.lower() == ".svg":
        _save_compact_usage_svg(
            matrix, row_labels, layers, targets, output,
            title=title, row_axis_label=row_axis_label,
        )
        return
    plt, _ = _ensure_matplotlib()
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    choices = matrix.shape[-1]
    if choices != 8:
        raise ValueError(f"The 2x4 expert glyph requires exactly 8 choices, got {choices}")
    cmap = _usage_cmap()
    figure_height = max(8.5, min(22, len(row_labels) * 0.095))
    figure, axes = plt.subplots(
        1,
        len(targets),
        figsize=(22, figure_height),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    for target_index, (target, axis) in enumerate(zip(targets, axes)):
        values = np.clip(matrix[:, :, target_index, :], 0, 1)
        usage_grid = np.zeros(
            (len(row_labels) * 2, len(layers) * 4),
            dtype=np.float64,
        )
        for expert in range(choices):
            usage_grid[expert // 4 :: 2, expert % 4 :: 4] = values[:, :, expert]
        if output.suffix.lower() in {".svg", ".pdf"}:
            x_edges = np.arange(usage_grid.shape[1] + 1, dtype=np.float64) - 0.5
            y_edges = np.arange(usage_grid.shape[0] + 1, dtype=np.float64) - 0.5
            axis.pcolormesh(
                x_edges,
                y_edges,
                usage_grid,
                cmap=cmap,
                vmin=0,
                vmax=1,
                shading="flat",
                rasterized=False,
            )
            axis.set_xlim(-0.5, usage_grid.shape[1] - 0.5)
            axis.set_ylim(usage_grid.shape[0] - 0.5, -0.5)
        else:
            axis.imshow(
                usage_grid,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=0,
                vmax=1,
            )
        axis.set_title(
            target.replace("_proj", ""),
            fontsize=13,
            fontweight="bold",
            pad=28,
        )
        axis.set_xticks([19.5, 63.5, 107.5])
        axis.set_xticklabels(["early\n0–9", "middle\n10–21", "late\n22–31"], fontsize=7)
        for boundary in range(1, len(layers)):
            axis.axvline(
                boundary * 4 - 0.5,
                color="#667085",
                linewidth=0.45,
                alpha=0.9,
            )
        axis.axvline(39.5, color="#172033", linewidth=1.2)
        axis.axvline(87.5, color="#172033", linewidth=1.2)
        layer_axis = axis.secondary_xaxis("top")
        layer_axis.set_xticks([index * 4 + 1.5 for index in range(len(layers))])
        layer_axis.set_xticklabels(
            [f"L{layer}" for layer in layers],
            rotation=90,
            fontsize=3.8,
        )
        layer_axis.tick_params(axis="x", length=0, pad=1)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks([index * 2 + 0.5 for index in range(len(row_labels))])
    axes[0].set_yticklabels(row_labels, fontsize=5 if len(row_labels) <= 60 else 3.2)
    axes[0].set_ylabel(f"{row_axis_label}; each layer cell is a 2x4 E0–E7 glyph")
    figure.suptitle(title, fontsize=16, y=0.97)
    colorbar = figure.colorbar(
        ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap),
        ax=list(axes),
        orientation="horizontal",
        fraction=0.025,
        pad=0.075,
        aspect=70,
        ticks=[0, 1 / 8, 1 / 4, 1 / 2, 1],
    )
    colorbar.solids.set_rasterized(False)
    colorbar.set_label("routing probability: green = low usage, red = high usage")
    colorbar.ax.set_xticklabels(["0", "uniform 1/8", "1/4", "1/2", "1"])
    figure.text(
        0.5,
        0.035,
        "Expert position inside every layer glyph: top E0 E1 E2 E3 · bottom E4 E5 E6 E7",
        ha="center",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.13, right=0.99, top=0.85, bottom=0.18, wspace=0.08)
    if output.suffix.lower() == ".svg":
        import matplotlib

        with matplotlib.rc_context({"svg.fonttype": "none"}):
            figure.savefig(output, format="svg")
    else:
        figure.savefig(output, dpi=160)
    plt.close(figure)


def _save_signature_tsne(
    embedding: np.ndarray,
    languages: list[str],
    metadata: dict[str, LanguageMeta],
    output: Path,
    *,
    title: str,
    method: str,
) -> None:
    plt, _ = _ensure_matplotlib()
    families = sorted({metadata[language].family for language in languages})
    family_index = {family: index for index, family in enumerate(families)}
    cmap = plt.get_cmap("gist_ncar")
    colors = {
        family: cmap(family_index[family] / max(1, len(families) - 1))
        for family in families
    }
    fig, ax = plt.subplots(figsize=(11, 8))
    for family in families:
        indices = [i for i, language in enumerate(languages) if metadata[language].family == family]
        ax.scatter(
            embedding[indices, 0],
            embedding[indices, 1],
            s=20,
            alpha=0.68,
            color=colors[family],
        )
    ax.set_title(f"{title} ({method})")
    ax.set_xlabel(f"{method} dimension 1")
    ax.set_ylabel(f"{method} dimension 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_coordinates(
    output: Path,
    embedding: np.ndarray,
    languages: list[str],
    metadata: dict[str, LanguageMeta],
    method: str,
) -> None:
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["language", "family", "x", "y", "embedding_method"])
        for index, language in enumerate(languages):
            writer.writerow(
                [
                    language,
                    metadata[language].family,
                    f"{embedding[index, 0]:.10f}",
                    f"{embedding[index, 1]:.10f}",
                    method,
                ]
            )


def _family_colors(families: list[str]) -> dict[str, str]:
    return {
        family: f"hsl({round(index * 360 / max(1, len(families)))},68%,48%)"
        for index, family in enumerate(families)
    }


def _selection_payload(
    structured: np.ndarray,
    languages: list[str],
    metadata: dict[str, LanguageMeta],
) -> tuple[dict[str, list[float]], list[str]]:
    flattened = structured.reshape(structured.shape[0], -1)
    selections: dict[str, list[float]] = {
        "overall": np.round(flattened.mean(axis=0), 6).tolist()
    }
    family_matrix, family_labels, _ = _aggregate_by_group(
        structured.reshape(structured.shape[0], -1, structured.shape[-1]),
        languages,
        metadata,
        "family",
    )
    for index, family in enumerate(family_labels):
        selections[f"family:{family}"] = np.round(family_matrix[index].reshape(-1), 6).tolist()
    for index, language in enumerate(languages):
        selections[f"language:{language}"] = np.round(flattened[index], 6).tolist()
    return selections, family_labels


COMPACT_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root { font-family:Inter,ui-sans-serif,system-ui,sans-serif; color:#172033; }
body { margin:0; background:#fff; }
main { max-width:1480px; margin:0 auto; padding:24px 28px; }
h1 { margin:0 0 5px; font-size:23px; }
.lede { margin:0 0 18px; color:#5d6879; font-size:13px; }
.controls { display:grid; grid-template-columns:repeat(4,minmax(220px,1fr)); gap:10px; max-width:1450px; }
.control label { display:block; font-size:12px; margin:0 0 5px; }
select { width:100%; padding:8px 10px; border:1px solid #cfd6df; border-radius:8px; background:white; color:#172033; }
.zoomline { display:flex; gap:8px; align-items:center; height:36px; }
.zoomline input { flex:1; }
.zoomline button { border:1px solid #cfd6df; border-radius:6px; background:#fff; padding:5px 9px; cursor:pointer; }
.matrix-wrap { max-height:820px; overflow:auto; margin-top:12px; border:1px solid #e5e9ef; background:#fff; }
canvas { display:block; }
#matrix { width:auto; max-width:none; height:auto; }
.legend { display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin:14px 0 10px; font-size:11px; }
.swatch { width:12px; height:12px; display:inline-block; vertical-align:-2px; margin-right:4px; }
.scale-gradient { width:240px; height:12px; display:inline-block; vertical-align:-2px; border:1px solid #cfd6df; background:linear-gradient(90deg,#edf8e9 0%,#66bd63 12.5%,#fee08b 25%,#f46d43 50%,#a50026 100%); }
.glyph { display:inline-grid; grid-template-columns:repeat(4,7px); grid-template-rows:repeat(2,7px); gap:1px; vertical-align:middle; margin-right:5px; }
.glyph i { display:block; width:7px; height:7px; }
.detail { border-top:1px solid #e5e9ef; padding-top:12px; display:grid; grid-template-columns:280px 1fr; gap:16px; align-items:start; }
.detail h2 { font-size:14px; margin:0 0 5px; }
.detail p { font-size:12px; color:#647087; margin:0; line-height:1.45; }
#bars { width:100%; height:auto; }
.footer { display:flex; justify-content:space-between; gap:18px; margin-top:12px; font-size:11px; color:#647087; }
.footer a { color:#2463c7; }
.tooltip { position:fixed; display:none; pointer-events:none; background:#172033; color:#fff; padding:6px 8px; border-radius:5px; font-size:11px; z-index:5; }
@media(max-width:950px){ .controls,.detail{grid-template-columns:1fr;} }
</style>
</head>
<body>
<main>
<h1>__TITLE__</h1>
<p class="lede">Each outlined projection × layer cell is a 2×4 E0–E7 glyph. Layer labels remain visible at every zoom; expert numbers appear when their tiles are large enough.</p>
<div class="controls">
  <div class="control"><label for="rows">Row resolution</label><select id="rows">
    <option value="family">Language families (44 rows)</option>
    <option value="language">Individual languages (215 rows)</option>
  </select></div>
  <div class="control"><label for="focus">Layer focus</label><select id="focus">
    <option value="all">All layers (32)</option>
    <option value="early">Early layers (0–9)</option>
    <option value="middle">Middle layers (10–21)</option>
    <option value="late">Late layers (22–31)</option>
  </select></div>
  <div class="control"><label for="colorMode">Color encoding</label><select id="colorMode">
    <option value="usage">Routing usage: green → red</option>
    <option value="identity">Expert identity hues</option>
  </select></div>
  <div class="control"><label>Zoom <strong id="zoomValue">100%</strong></label><div class="zoomline">
    <button id="zoomOut" type="button">−</button><input id="zoom" type="range" min="75" max="300" step="25" value="100"><button id="zoomIn" type="button">+</button><button id="zoomReset" type="button">Reset</button>
  </div></div>
</div>
<div class="matrix-wrap"><canvas id="matrix" width="1400" height="520"></canvas></div>
<div class="legend" id="legend"></div>
<section class="detail">
  <div><h2 id="detailTitle">Select an expert tile</h2><p id="detailText">Hover any microblock for its probability. Click it to reveal the full eight-expert distribution for that family/language, projection, and layer.</p></div>
  <canvas id="bars" width="960" height="190"></canvas>
</section>
<div class="footer">
  <span>Expert glyph: top row E0–E3 · bottom row E4–E7. Early 0–9 · middle 10–21 · late 22–31.</span>
  <span><a href="detailed_index.html">Full probability matrix and t-SNE</a> · <a href="compact_family_overview.png">Static families</a> · <a href="compact_language_overview.png">Static languages</a> · <a href="compact_language_overview.svg">Vector SVG</a></span>
</div>
</main>
<div class="tooltip" id="tooltip"></div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const D=JSON.parse(document.getElementById("payload").textContent);
const rows=document.getElementById("rows"),focus=document.getElementById("focus"),colorMode=document.getElementById("colorMode"),zoom=document.getElementById("zoom");
const canvas=document.getElementById("matrix"),ctx=canvas.getContext("2d");
const bars=document.getElementById("bars"),bctx=bars.getContext("2d");
const tooltip=document.getElementById("tooltip"),zoomValue=document.getElementById("zoomValue");
const base={width:1800,left:185,right:14,top:76,bottom:52,gap:14,panelWidth:210,rowHeight:9};
let geometry=null,selected=null;
function rgb(hex){const v=hex.slice(1);return [0,2,4].map(i=>parseInt(v.slice(i,i+2),16));}
function expertColor(expert,probability){const strength=.06+.94*Math.max(0,Math.min(1,probability)),baseColor=rgb(D.expertColors[expert]);return `rgb(${baseColor.map(v=>Math.round(255-strength*(255-v))).join(",")})`;}
function usageColor(probability){const stops=[[0,"#edf8e9"],[.125,"#66bd63"],[.25,"#fee08b"],[.5,"#f46d43"],[1,"#a50026"]],p=Math.max(0,Math.min(1,probability));for(let i=1;i<stops.length;i++){if(p<=stops[i][0]){const a=stops[i-1],b=stops[i],f=(p-a[0])/(b[0]-a[0]),ca=rgb(a[1]),cb=rgb(b[1]);return `rgb(${ca.map((v,j)=>Math.round(v+f*(cb[j]-v))).join(",")})`;}}return stops[stops.length-1][1];}
function tileColor(expert,probability){return colorMode.value==="usage"?usageColor(probability):expertColor(expert,probability);}
function currentLayers(){
  if(focus.value==="early")return D.layers.filter(x=>x<=9);
  if(focus.value==="middle")return D.layers.filter(x=>x>=10&&x<=21);
  if(focus.value==="late")return D.layers.filter(x=>x>=22);
  return D.layers;
}
function currentRows(){
  const labels=rows.value==="family"?D.familyOrder:D.languageOrder;
  const prefix=rows.value==="family"?"family:":"language:";
  return labels.map(label=>({label,key:prefix+label}));
}
function probabilities(key,layer,target){
  const values=D.selections[key],result=[];
  for(let expert=0;expert<D.choices.length;expert++)result.push(values[(layer*D.targets.length+target)*D.choices.length+expert]);
  return result;
}
function drawMatrix(){
  const scale=Number(zoom.value)/100,layers=currentLayers(),rowData=currentRows();
  const margin={left:base.left*scale,right:base.right*scale,top:base.top*scale,bottom:base.bottom*scale};
  const panelWidth=base.panelWidth*scale,gap=base.gap*scale,rowHeight=base.rowHeight*scale;
  canvas.width=Math.round(base.width*scale);canvas.height=Math.round(margin.top+margin.bottom+rowData.length*rowHeight);
  canvas.style.width=`${canvas.width}px`;canvas.style.height=`${canvas.height}px`;
  ctx.fillStyle="#fff";ctx.fillRect(0,0,canvas.width,canvas.height);
  const cellWidth=panelWidth/layers.length,microWidth=cellWidth/4,microHeight=rowHeight/2;
  D.targets.forEach((target,targetIndex)=>{
    const panelX=margin.left+targetIndex*(panelWidth+gap);
    ctx.fillStyle="#172033";ctx.font=`bold ${Math.max(11,15*scale)}px sans-serif`;ctx.textAlign="center";ctx.textBaseline="bottom";
    ctx.fillText(target.replace("_proj",""),panelX+panelWidth/2,margin.top-28*scale);
    ctx.fillStyle="#4b5565";ctx.font=`${Math.max(5,6*scale)}px sans-serif`;ctx.textAlign="center";ctx.textBaseline="middle";
    layers.forEach((layer,layerIndex)=>{
      const x=panelX+(layerIndex+.5)*cellWidth;
      ctx.save();ctx.translate(x,margin.top-7*scale);ctx.rotate(-Math.PI/2);ctx.fillText(`L${layer}`,0,0);ctx.restore();
    });
    rowData.forEach((row,rowIndex)=>layers.forEach((layer,layerIndex)=>{
      const probs=probabilities(row.key,layer,targetIndex),dominant=probs.indexOf(Math.max(...probs));
      const x=panelX+layerIndex*cellWidth,y=margin.top+rowIndex*rowHeight;
      probs.forEach((probability,expert)=>{
        const tileX=x+(expert%4)*microWidth,tileY=y+Math.floor(expert/4)*microHeight;
        ctx.fillStyle=tileColor(expert,probability);ctx.fillRect(tileX+.15*scale,tileY+.15*scale,Math.max(.5,microWidth-.3*scale),Math.max(.5,microHeight-.3*scale));
        if(expert===dominant&&microWidth>=1.8){ctx.strokeStyle="rgba(17,24,39,.75)";ctx.lineWidth=Math.max(.35,.45*scale);ctx.strokeRect(tileX+.3*scale,tileY+.3*scale,Math.max(.4,microWidth-.6*scale),Math.max(.4,microHeight-.6*scale));}
        if(microWidth>=4.5&&microHeight>=4.5){
          ctx.fillStyle=probability>=.42?"rgba(255,255,255,.92)":"rgba(17,24,39,.78)";
          ctx.font=`bold ${Math.max(4,Math.min(microWidth*.72,microHeight*.68))}px sans-serif`;
          ctx.textAlign="center";ctx.textBaseline="middle";
          ctx.fillText(String(expert),tileX+microWidth/2,tileY+microHeight/2);
        }
      });
      ctx.strokeStyle="rgba(71,85,105,.72)";ctx.lineWidth=Math.max(.45,.55*scale);ctx.strokeRect(x,y,cellWidth,rowHeight);
      if(selected&&selected.key===row.key&&selected.layer===layer&&selected.target===targetIndex){const sx=x+(selected.expert%4)*microWidth,sy=y+Math.floor(selected.expert/4)*microHeight;ctx.strokeStyle="#111827";ctx.lineWidth=Math.max(1,1.5*scale);ctx.strokeRect(sx,sy,microWidth,microHeight);}
    }));
    ctx.strokeStyle="#9aa5b4";ctx.lineWidth=Math.max(.7,scale);ctx.strokeRect(panelX,margin.top,panelWidth,rowData.length*rowHeight);
    if(focus.value==="all"){
      [10,22].forEach(boundary=>{const x=panelX+boundary*cellWidth;ctx.strokeStyle="#172033";ctx.lineWidth=Math.max(1.2,1.35*scale);ctx.beginPath();ctx.moveTo(x,margin.top);ctx.lineTo(x,canvas.height-margin.bottom);ctx.stroke();});
    }
    ctx.fillStyle="#4b5565";ctx.font=`${Math.max(7,9*scale)}px sans-serif`;ctx.textAlign="center";ctx.textBaseline="top";
    const layerLabel=focus.value==="all"?"32 layers":`${layers[0]}–${layers[layers.length-1]}`;
    ctx.fillText(layerLabel,panelX+panelWidth/2,canvas.height-margin.bottom+7*scale);
  });
  ctx.fillStyle="#374151";ctx.font=`${Math.max(6,7*scale)}px sans-serif`;ctx.textAlign="right";ctx.textBaseline="middle";
  rowData.forEach((row,index)=>ctx.fillText(row.label,margin.left-6*scale,margin.top+(index+.5)*rowHeight));
  ctx.save();ctx.translate(14*scale,margin.top+rowData.length*rowHeight/2);ctx.rotate(-Math.PI/2);ctx.font=`${Math.max(8,11*scale)}px sans-serif`;ctx.textAlign="center";ctx.fillText(rows.value==="family"?"Language families":"Languages",0,0);ctx.restore();
  geometry={layers,rowData,cellWidth,microWidth,microHeight,rowHeight,panelWidth,gap,margin};
  zoomValue.textContent=`${zoom.value}%`;
}
function drawBars(item){
  const probs=probabilities(item.key,item.layer,item.target),padding={left:35,right:12,top:14,bottom:28};
  bctx.clearRect(0,0,bars.width,bars.height);bctx.fillStyle="#fff";bctx.fillRect(0,0,bars.width,bars.height);
  const width=(bars.width-padding.left-padding.right)/probs.length;
  probs.forEach((p,i)=>{const h=p*(bars.height-padding.top-padding.bottom);const x=padding.left+i*width,y=bars.height-padding.bottom-h;
    bctx.fillStyle=tileColor(i,p);bctx.fillRect(x+width*.15,y,width*.7,h);if(i===item.expert){bctx.strokeStyle="#111827";bctx.lineWidth=2;bctx.strokeRect(x+width*.15,y,width*.7,h);}bctx.fillStyle="#172033";bctx.font="11px sans-serif";bctx.textAlign="center";bctx.fillText(`E${i}`,x+width/2,bars.height-9);bctx.fillText(p.toFixed(3),x+width/2,Math.max(11,y-4));});
  document.getElementById("detailTitle").textContent=`${item.label} · ${D.targets[item.target].replace("_proj","")} · L${item.layer} · E${item.expert}`;
  document.getElementById("detailText").textContent=`Selected expert probability: ${probs[item.expert].toFixed(6)}. Bars show the complete E0–E7 routing distribution.`;
}
function hit(event){
  if(!geometry)return null;const r=canvas.getBoundingClientRect(),x=(event.clientX-r.left)*canvas.width/r.width,y=(event.clientY-r.top)*canvas.height/r.height;
  const rowIndex=Math.floor((y-geometry.margin.top)/geometry.rowHeight);if(rowIndex<0||rowIndex>=geometry.rowData.length)return null;
  for(let target=0;target<D.targets.length;target++){const panelX=geometry.margin.left+target*(geometry.panelWidth+geometry.gap);if(x>=panelX&&x<panelX+geometry.panelWidth){const within=x-panelX,layerIndex=Math.floor(within/geometry.cellWidth);if(layerIndex>=0&&layerIndex<geometry.layers.length){const layerOffset=within-layerIndex*geometry.cellWidth,microColumn=Math.min(3,Math.floor(layerOffset/geometry.microWidth));const rowOffset=y-(geometry.margin.top+rowIndex*geometry.rowHeight),microRow=Math.min(1,Math.max(0,Math.floor(rowOffset/geometry.microHeight)));const expert=microRow*4+microColumn,row=geometry.rowData[rowIndex];return {...row,rowIndex,target,layer:geometry.layers[layerIndex],expert};}}}return null;
}
canvas.addEventListener("mousemove",event=>{const item=hit(event);if(!item){tooltip.style.display="none";return;}const probs=probabilities(item.key,item.layer,item.target),dominant=probs.indexOf(Math.max(...probs));tooltip.innerHTML=`${item.label}<br>${D.targets[item.target].replace("_proj","")} · L${item.layer} · E${item.expert}<br>p=${probs[item.expert].toFixed(6)}${item.expert===dominant?" · top expert":""}`;tooltip.style.display="block";tooltip.style.left=`${event.clientX+12}px`;tooltip.style.top=`${event.clientY+12}px`;});
canvas.addEventListener("mouseleave",()=>tooltip.style.display="none");
canvas.addEventListener("click",event=>{const item=hit(event);if(item){selected=item;drawMatrix();drawBars(item);}});
rows.addEventListener("change",()=>{selected=null;drawMatrix();});focus.addEventListener("change",()=>{selected=null;drawMatrix();});colorMode.addEventListener("change",()=>{drawMatrix();renderLegend();if(selected)drawBars(selected);});zoom.addEventListener("input",drawMatrix);
function setZoom(value){zoom.value=Math.max(Number(zoom.min),Math.min(Number(zoom.max),value));drawMatrix();}
document.getElementById("zoomOut").addEventListener("click",()=>setZoom(Number(zoom.value)-25));
document.getElementById("zoomIn").addEventListener("click",()=>setZoom(Number(zoom.value)+25));
document.getElementById("zoomReset").addEventListener("click",()=>setZoom(100));
const legend=document.getElementById("legend");
function renderLegend(){const layout=`<span><strong>Glyph position:</strong> top E0 E1 E2 E3 · bottom E4 E5 E6 E7</span>`;if(colorMode.value==="usage"){legend.innerHTML=layout+`<span><i class="scale-gradient"></i></span><span>0 low · 1/8 uniform · 1/4 · 1/2 · 1 high</span>`;}else{legend.innerHTML=layout+D.expertColors.map((color,i)=>`<span><i class="swatch" style="background:${color}"></i>E${i}</span>`).join("")+`<span>intensity = probability</span>`;}}
renderLegend();drawMatrix();
</script>
</body>
</html>
"""

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; color:#172033; }
body { margin:0; background:#f5f7fb; }
main { max-width:1720px; margin:0 auto; padding:24px; }
h1 { margin:0 0 6px; font-size:24px; }
.lede { margin:0 0 18px; color:#516078; }
.toolbar { display:flex; gap:14px; align-items:center; margin-bottom:14px; flex-wrap:wrap; }
select { min-width:330px; padding:8px 10px; border:1px solid #c8d1df; border-radius:6px; background:white; }
.layout { display:grid; grid-template-columns:minmax(720px,2fr) minmax(430px,1fr); gap:18px; }
.card { background:white; border:1px solid #d8deea; border-radius:10px; padding:14px; box-shadow:0 2px 7px #1720330d; }
.card h2 { margin:0 0 8px; font-size:16px; }
canvas { display:block; width:100%; height:auto; border:1px solid #eef1f6; }
.all-card { margin-top:18px; }
.all-scroll { max-height:900px; overflow:auto; border:1px solid #eef1f6; background:white; }
#allLanguages { width:1180px; max-width:none; border:0; }
.note { color:#637089; font-size:12px; line-height:1.45; margin:9px 0 0; }
.tooltip { position:fixed; display:none; pointer-events:none; background:#172033; color:white; padding:6px 8px; border-radius:5px; font-size:12px; z-index:4; }
@media (max-width:1100px) { .layout { grid-template-columns:1fr; } }
</style>
</head>
<body>
<main>
<h1>__TITLE__</h1>
<p class="lede">One view preserves all 32 base layers, seven LoRA targets, and eight true experts.</p>
<div class="toolbar">
  <label for="selection"><strong>Routing view</strong></label>
  <select id="selection">__OPTIONS__</select>
  <span id="selectionLabel"></span>
</div>
<div class="layout">
  <section class="card">
    <h2>Structured target–expert LoLA heatmap</h2>
    <canvas id="heatmap" width="1180" height="760"></canvas>
    <p class="note">Rows are real base-model layers. Columns are target blocks; each block contains E0–E7.
    Colour is the mean routing probability for the selected language group.</p>
  </section>
  <section class="card">
    <h2>Whole-signature language t-SNE</h2>
    <canvas id="tsne" width="620" height="660"></canvas>
    <p class="note">Each point embeds the complete 32×7×8 routing signature and is coloured by family.
    Click a point to load that language. t-SNE axes have no intrinsic meaning.</p>
  </section>
</div>
<section class="card all-card">
  <h2>All-language nested LoLA matrix</h2>
  <div class="all-scroll"><canvas id="allLanguages" width="1180" height="3100"></canvas></div>
  <p class="note">Every named language occupies seven target subrows in this order:
  q, k, v, o, gate, up, down. Columns are L0–L31, each subdivided into E0–E7.
  Languages are grouped by family. Hover for the exact coordinate and probability; click any
  language to load its detailed view above.</p>
</section>
</main>
<div class="tooltip" id="tooltip"></div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const D = JSON.parse(document.getElementById("payload").textContent);
const select = document.getElementById("selection");
const label = document.getElementById("selectionLabel");
const tooltip = document.getElementById("tooltip");
const heat = document.getElementById("heatmap");
const hctx = heat.getContext("2d");
const tsne = document.getElementById("tsne");
const tctx = tsne.getContext("2d");
const all = document.getElementById("allLanguages");
const actx = all.getContext("2d");
const hm = {left:58, right:18, top:72, bottom:46};
const tm = {left:44, right:18, top:20, bottom:42};
const am = {left:136, right:16, top:58, bottom:18};
let tsnePoints = [];
let allGeometry = null;

function hexRgb(hex) {
  const v = hex.replace("#","");
  return [parseInt(v.slice(0,2),16),parseInt(v.slice(2,4),16),parseInt(v.slice(4,6),16)];
}
function heatColor(value) {
  const u = 1 / D.choices.length;
  const raw = [[0,"#ffffff"],[u,"#b8f5a5"],[Math.min(2*u,1),"#ffff00"],
               [Math.min(4*u,1),"#ff1f1f"],[0.5,"#b00020"],[1,"#000000"]];
  const stops = raw.sort((a,b)=>a[0]-b[0]);
  const v = Math.max(0,Math.min(1,value));
  for (let i=1;i<stops.length;i++) {
    if (v <= stops[i][0]) {
      const a=stops[i-1], b=stops[i], span=Math.max(1e-9,b[0]-a[0]);
      const f=(v-a[0])/span, ca=hexRgb(a[1]), cb=hexRgb(b[1]);
      return `rgb(${ca.map((x,j)=>Math.round(x+f*(cb[j]-x))).join(",")})`;
    }
  }
  return "#000000";
}
function selectionText(key) {
  if (key === "overall") return "Overall mean across all languages";
  const [kind,...rest] = key.split(":");
  return `${kind[0].toUpperCase()+kind.slice(1)}: ${rest.join(":")}`;
}
function drawHeatmap() {
  const key=select.value, values=D.selections[key];
  label.textContent=selectionText(key);
  hctx.clearRect(0,0,heat.width,heat.height);
  hctx.fillStyle="#fff"; hctx.fillRect(0,0,heat.width,heat.height);
  const cols=D.targets.length*D.choices.length;
  const cw=(heat.width-hm.left-hm.right)/cols;
  const ch=(heat.height-hm.top-hm.bottom)/D.layers.length;
  for (let layer=0;layer<D.layers.length;layer++) {
    for (let col=0;col<cols;col++) {
      const index=layer*cols+col;
      hctx.fillStyle=heatColor(values[index]);
      hctx.fillRect(hm.left+col*cw,hm.top+layer*ch,Math.ceil(cw)+0.2,Math.ceil(ch)+0.2);
    }
  }
  hctx.strokeStyle="#172033"; hctx.lineWidth=1.5;
  for (let target=0;target<=D.targets.length;target++) {
    const x=hm.left+target*D.choices.length*cw;
    hctx.beginPath(); hctx.moveTo(x,hm.top); hctx.lineTo(x,heat.height-hm.bottom); hctx.stroke();
  }
  hctx.font="12px sans-serif"; hctx.textAlign="right"; hctx.textBaseline="middle"; hctx.fillStyle="#172033";
  D.layers.forEach((layer,i)=>hctx.fillText(`L${layer}`,hm.left-7,hm.top+(i+0.5)*ch));
  hctx.textAlign="center"; hctx.textBaseline="bottom"; hctx.font="bold 13px sans-serif";
  D.targets.forEach((target,i)=>hctx.fillText(target,hm.left+(i+0.5)*D.choices.length*cw,hm.top-26));
  hctx.font="9px sans-serif"; hctx.textBaseline="top";
  D.targets.forEach((_,target)=>D.choices.forEach((choice,expert)=>{
    hctx.fillText(`E${expert}`,hm.left+(target*D.choices.length+expert+0.5)*cw,heat.height-hm.bottom+7);
  }));
}
function drawTsne() {
  tctx.clearRect(0,0,tsne.width,tsne.height);
  tctx.fillStyle="#fff"; tctx.fillRect(0,0,tsne.width,tsne.height);
  const xs=D.points.map(p=>p.x), ys=D.points.map(p=>p.y);
  const xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys);
  const sx=x=>tm.left+(x-xmin)/Math.max(1e-9,xmax-xmin)*(tsne.width-tm.left-tm.right);
  const sy=y=>tsne.height-tm.bottom-(y-ymin)/Math.max(1e-9,ymax-ymin)*(tsne.height-tm.top-tm.bottom);
  tsnePoints=D.points.map(p=>({...p,px:sx(p.x),py:sy(p.y)}));
  tctx.strokeStyle="#dfe5ef"; tctx.lineWidth=1;
  tctx.strokeRect(tm.left,tm.top,tsne.width-tm.left-tm.right,tsne.height-tm.top-tm.bottom);
  tsnePoints.forEach(p=>{
    tctx.beginPath(); tctx.arc(p.px,p.py,4.2,0,2*Math.PI);
    tctx.fillStyle=D.familyColors[p.family]; tctx.globalAlpha=0.68; tctx.fill();
  });
  tctx.globalAlpha=1;
  const selected=select.value.startsWith("language:") ? select.value.slice(9) : null;
  if (selected) {
    const p=tsnePoints.find(point=>point.language===selected);
    if (p) {
      tctx.beginPath(); tctx.arc(p.px,p.py,8,0,2*Math.PI);
      tctx.strokeStyle="#111827"; tctx.lineWidth=2.5; tctx.stroke();
    }
  }
  tctx.fillStyle="#172033"; tctx.font="11px sans-serif"; tctx.textAlign="center";
  tctx.fillText(`${D.embeddingMethod} dimension 1`,(tm.left+tsne.width-tm.right)/2,tsne.height-10);
  tctx.save(); tctx.translate(13,(tm.top+tsne.height-tm.bottom)/2); tctx.rotate(-Math.PI/2);
  tctx.fillText(`${D.embeddingMethod} dimension 2`,0,0); tctx.restore();
}
function drawAllLanguages() {
  const expertWidth=(all.width-am.left-am.right)/(D.layers.length*D.choices.length);
  const targetHeight=2;
  const languageHeight=targetHeight*D.targets.length;
  const requiredHeight=Math.ceil(am.top+am.bottom+D.languageOrder.length*languageHeight);
  if(all.height!==requiredHeight) all.height=requiredHeight;
  actx.clearRect(0,0,all.width,all.height);
  actx.fillStyle="#fff"; actx.fillRect(0,0,all.width,all.height);
  D.languageOrder.forEach((language,languageIndex)=>{
    const values=D.selections[`language:${language}`];
    D.targets.forEach((_,target)=>{
      D.layers.forEach((__,layer)=>{
        D.choices.forEach((___,expert)=>{
          const source=((layer*D.targets.length+target)*D.choices.length+expert);
          const x=am.left+(layer*D.choices.length+expert)*expertWidth;
          const y=am.top+(languageIndex*D.targets.length+target)*targetHeight;
          actx.fillStyle=heatColor(values[source]);
          actx.fillRect(x,y,Math.ceil(expertWidth)+0.2,targetHeight+0.2);
        });
      });
    });
  });
  actx.strokeStyle="#172033"; actx.lineWidth=1;
  for(let layer=0;layer<=D.layers.length;layer++){
    const x=am.left+layer*D.choices.length*expertWidth;
    actx.beginPath();actx.moveTo(x,am.top);actx.lineTo(x,all.height-am.bottom);actx.stroke();
  }
  actx.font="8px sans-serif";actx.textAlign="right";actx.textBaseline="middle";actx.fillStyle="#172033";
  let previousFamily=null;
  D.languageOrder.forEach((language,index)=>{
    const family=D.languageFamilies[language];
    const y=am.top+(index+0.5)*languageHeight;
    actx.fillText(language,am.left-6,y);
    if(family!==previousFamily){
      const lineY=am.top+index*languageHeight;
      actx.strokeStyle="#172033";actx.lineWidth=1.4;
      actx.beginPath();actx.moveTo(0,lineY);actx.lineTo(all.width-am.right,lineY);actx.stroke();
      actx.save();actx.translate(7,lineY+2);actx.rotate(Math.PI/2);
      actx.font="bold 7px sans-serif";actx.textAlign="left";actx.fillText(family,0,0);actx.restore();
      previousFamily=family;
    }
  });
  actx.textAlign="center";actx.textBaseline="bottom";actx.font="bold 9px sans-serif";
  D.layers.forEach((layer,index)=>{
    const x=am.left+(index+0.5)*D.choices.length*expertWidth;
    actx.fillText(`L${layer}`,x,am.top-25);
  });
  actx.font="8px sans-serif";actx.textBaseline="top";
  D.layers.forEach((_,layer)=>D.choices.forEach((__,expert)=>{
    const x=am.left+(layer*D.choices.length+expert+0.5)*expertWidth;
    actx.save();actx.translate(x,am.top-21);actx.rotate(-Math.PI/2);actx.fillText(`E${expert}`,0,0);actx.restore();
  }));
  actx.textAlign="left";actx.font="10px sans-serif";actx.fillText(
    "Within each language: q · k · v · o · gate · up · down",
    am.left,12
  );
  allGeometry={expertWidth,targetHeight,languageHeight};
}
function nearestTsne(event) {
  const r=tsne.getBoundingClientRect(), x=(event.clientX-r.left)*tsne.width/r.width;
  const y=(event.clientY-r.top)*tsne.height/r.height;
  let best=null, distance=Infinity;
  tsnePoints.forEach(p=>{const d=(p.px-x)**2+(p.py-y)**2;if(d<distance){best=p;distance=d;}});
  return distance<180 ? best : null;
}
select.addEventListener("change",()=>{drawHeatmap();drawTsne();});
heat.addEventListener("mousemove",event=>{
  const r=heat.getBoundingClientRect(), x=(event.clientX-r.left)*heat.width/r.width;
  const y=(event.clientY-r.top)*heat.height/r.height;
  const cols=D.targets.length*D.choices.length;
  const cw=(heat.width-hm.left-hm.right)/cols, ch=(heat.height-hm.top-hm.bottom)/D.layers.length;
  const col=Math.floor((x-hm.left)/cw), layer=Math.floor((y-hm.top)/ch);
  if(col>=0&&col<cols&&layer>=0&&layer<D.layers.length){
    const target=Math.floor(col/D.choices.length), expert=col%D.choices.length;
    const value=D.selections[select.value][layer*cols+col];
    tooltip.innerHTML=`L${D.layers[layer]} · ${D.targets[target]} · E${expert}<br>p=${value.toFixed(4)}`;
    tooltip.style.display="block"; tooltip.style.left=`${event.clientX+12}px`;tooltip.style.top=`${event.clientY+12}px`;
  } else tooltip.style.display="none";
});
heat.addEventListener("mouseleave",()=>tooltip.style.display="none");
tsne.addEventListener("mousemove",event=>{
  const p=nearestTsne(event);
  if(p){tooltip.innerHTML=`${p.language}<br>${p.family}`;tooltip.style.display="block";
    tooltip.style.left=`${event.clientX+12}px`;tooltip.style.top=`${event.clientY+12}px`;}
  else tooltip.style.display="none";
});
tsne.addEventListener("mouseleave",()=>tooltip.style.display="none");
tsne.addEventListener("click",event=>{
  const p=nearestTsne(event);
  if(p){select.value=`language:${p.language}`;drawHeatmap();drawTsne();}
});
all.addEventListener("mousemove",event=>{
  const r=all.getBoundingClientRect(), x=(event.clientX-r.left)*all.width/r.width;
  const y=(event.clientY-r.top)*all.height/r.height;
  if(!allGeometry) return;
  const expertColumn=Math.floor((x-am.left)/allGeometry.expertWidth);
  const subrow=Math.floor((y-am.top)/allGeometry.targetHeight);
  const languageIndex=Math.floor(subrow/D.targets.length);
  const target=subrow%D.targets.length;
  if(expertColumn>=0&&expertColumn<D.layers.length*D.choices.length&&languageIndex>=0&&languageIndex<D.languageOrder.length){
    const layer=Math.floor(expertColumn/D.choices.length),expert=expertColumn%D.choices.length;
    const language=D.languageOrder[languageIndex];
    const source=((layer*D.targets.length+target)*D.choices.length+expert);
    const value=D.selections[`language:${language}`][source];
    tooltip.innerHTML=`${language} · ${D.languageFamilies[language]}<br>L${layer} · ${D.targets[target]} · E${expert} · p=${value.toFixed(4)}`;
    tooltip.style.display="block";tooltip.style.left=`${event.clientX+12}px`;tooltip.style.top=`${event.clientY+12}px`;
  } else tooltip.style.display="none";
});
all.addEventListener("mouseleave",()=>tooltip.style.display="none");
all.addEventListener("click",event=>{
  const r=all.getBoundingClientRect(), y=(event.clientY-r.top)*all.height/r.height;
  const subrow=Math.floor((y-am.top)/allGeometry.targetHeight);
  const languageIndex=Math.floor(subrow/D.targets.length);
  if(languageIndex>=0&&languageIndex<D.languageOrder.length){
    select.value=`language:${D.languageOrder[languageIndex]}`;drawHeatmap();drawTsne();
    window.scrollTo({top:0,behavior:"smooth"});
  }
});
drawHeatmap(); drawTsne(); drawAllLanguages();
</script>
</body>
</html>
"""


def _write_compact_html(
    output: Path,
    *,
    title: str,
    layers: list[int],
    targets: list[str],
    choices: list[str],
    selections: dict[str, list[float]],
    family_order: list[str],
    language_order: list[str],
) -> None:
    payload = {
        "layers": layers,
        "targets": targets,
        "choices": choices,
        "expertColors": list(EXPERT_COLORS[: len(choices)]),
        "selections": selections,
        "familyOrder": family_order,
        "languageOrder": language_order,
    }
    document = (
        COMPACT_HTML_TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"), allow_nan=False))
    )
    output.write_text(document, encoding="utf-8")


def _write_interactive_html(
    output: Path,
    *,
    title: str,
    layers: list[int],
    targets: list[str],
    choices: list[str],
    selections: dict[str, list[float]],
    family_labels: list[str],
    languages: list[str],
    metadata: dict[str, LanguageMeta],
    embedding: np.ndarray,
    embedding_method: str,
) -> None:
    options = ["<option value='overall'>Overall mean</option>"]
    options.append("<optgroup label='Families'>")
    options.extend(
        f"<option value='{html.escape('family:' + family)}'>{html.escape(family)}</option>"
        for family in family_labels
    )
    options.append("</optgroup><optgroup label='Languages'>")
    options.extend(
        f"<option value='{html.escape('language:' + language)}'>{html.escape(language)}</option>"
        for language in languages
    )
    options.append("</optgroup>")
    families = sorted(set(family_labels))
    language_order = _ordered_languages(languages, metadata)
    payload = {
        "layers": layers,
        "targets": targets,
        "choices": choices,
        "selections": selections,
        "embeddingMethod": embedding_method,
        "familyColors": _family_colors(families),
        "languageOrder": language_order,
        "languageFamilies": {
            language: metadata[language].family for language in language_order
        },
        "points": [
            {
                "language": language,
                "family": metadata[language].family,
                "x": round(float(embedding[index, 0]), 7),
                "y": round(float(embedding[index, 1]), 7),
            }
            for index, language in enumerate(languages)
        ],
    }
    document = (
        HTML_TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("__OPTIONS__", "".join(options))
        .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"), allow_nan=False))
    )
    output.write_text(document, encoding="utf-8")


def generate_prototype(
    routing_root: Path,
    run_id: str,
    languages_csv: Path,
    output_dir: Path | None = None,
) -> dict[str, object]:
    run_dir = routing_root / run_id
    with np.load(run_dir / "routing_matrix.npz", allow_pickle=True) as npz:
        languages = _as_str_array(npz["languages"])
        module_labels = _as_str_array(npz["layer_labels"])
        choice_labels = _as_str_array(npz["choice_labels"])
        axis_name = str(npz["axis_name"])
        peft_type = str(npz["peft_type"])
        matrix, matrix_source = _routing_matrix(npz)

    structured = structure_routing(matrix, module_labels)
    metadata = _load_language_metadata(languages_csv, languages)
    vectors = _normalize_per_module(
        structured.matrix.reshape(len(languages), -1, len(choice_labels))
    ).reshape(len(languages), -1)
    embedding, embedding_method = _embedding_2d(vectors)
    selections, family_labels = _selection_payload(structured.matrix, languages, metadata)
    family_flat, checked_family_labels, _ = _aggregate_by_group(
        structured.matrix.reshape(len(languages), -1, len(choice_labels)),
        languages,
        metadata,
        "family",
    )
    if checked_family_labels != family_labels:
        raise ValueError("Family aggregation order changed between prototype views")
    family_structured = family_flat.reshape(
        len(family_labels),
        len(structured.layers),
        len(structured.targets),
        len(choice_labels),
    )
    language_order = _ordered_languages(languages, metadata)
    language_index = {language: index for index, language in enumerate(languages)}
    language_structured = structured.matrix[
        [language_index[language] for language in language_order]
    ]
    output_dir = output_dir or (
        routing_root / "lola_style" / run_id / "target_expert_lola_prototype"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_path = output_dir / "overall_target_expert_lola.png"
    all_languages_path = output_dir / "all_languages_nested_lola.png"
    compact_path = output_dir / "compact_family_overview.png"
    compact_language_path = output_dir / "compact_language_overview.png"
    compact_language_svg_path = output_dir / "compact_language_overview.svg"
    tsne_path = output_dir / "whole_signature_tsne.png"
    coordinates_path = output_dir / "whole_signature_tsne_coordinates.csv"
    index_path = output_dir / "index.html"
    detailed_path = output_dir / "detailed_index.html"
    _save_overall_heatmap(
        structured.matrix.mean(axis=0),
        structured.layers,
        structured.targets,
        choice_labels,
        overall_path,
        title=f"{run_id}: complete layer–target–expert routing (overall mean)",
        axis_name=axis_name,
    )
    _save_all_languages_heatmap(
        structured.matrix,
        languages,
        metadata,
        structured.layers,
        structured.targets,
        choice_labels,
        all_languages_path,
        title=f"{run_id}: all-language layer–target–expert routing",
    )
    _save_compact_usage_overview(
        family_structured,
        family_labels,
        structured.layers,
        structured.targets,
        compact_path,
        title=f"{run_id}: LoLA expert-usage heatmap by family",
        row_axis_label="Language family",
    )
    _save_compact_usage_overview(
        language_structured,
        language_order,
        structured.layers,
        structured.targets,
        compact_language_path,
        title=f"{run_id}: LoLA expert-usage heatmap by language",
        row_axis_label="Language",
    )
    _save_compact_usage_overview(
        language_structured,
        language_order,
        structured.layers,
        structured.targets,
        compact_language_svg_path,
        title=f"{run_id}: LoLA expert-usage heatmap by language",
        row_axis_label="Language",
    )
    _save_signature_tsne(
        embedding,
        languages,
        metadata,
        tsne_path,
        title=f"{run_id}: whole-routing language signatures",
        method=embedding_method,
    )
    _write_coordinates(
        coordinates_path,
        embedding,
        languages,
        metadata,
        embedding_method,
    )
    _write_compact_html(
        index_path,
        title=f"{run_id} compact target–expert LoLA",
        layers=structured.layers,
        targets=structured.targets,
        choices=choice_labels,
        selections=selections,
        family_order=family_labels,
        language_order=language_order,
    )
    _write_interactive_html(
        detailed_path,
        title=f"{run_id} detailed target–expert LoLA",
        layers=structured.layers,
        targets=structured.targets,
        choices=choice_labels,
        selections=selections,
        family_labels=family_labels,
        languages=languages,
        metadata=metadata,
        embedding=embedding,
        embedding_method=embedding_method,
    )

    details: dict[str, object] = {
        "schema_version": 1,
        "prototype": True,
        "run_id": run_id,
        "peft_type": peft_type,
        "matrix_source": matrix_source,
        "axis_name": axis_name,
        "languages": len(languages),
        "families": len(family_labels),
        "layers": len(structured.layers),
        "targets": structured.targets,
        "choices": len(choice_labels),
        "source_modules": len(module_labels),
        "structured_shape": list(structured.matrix.shape),
        "embedding_method": embedding_method,
        "expert_glyph": "top row E0-E3; bottom row E4-E7",
        "usage_color_scale": {
            "0": "#edf8e9",
            "0.125_uniform": "#66bd63",
            "0.25": "#fee08b",
            "0.5": "#f46d43",
            "1": "#a50026",
        },
        "interpretation": (
            "Each heatmap row is a real base-model layer. Each target block contains the "
            "routing choices for one LoRA target. The t-SNE embeds each language's complete "
            "layer-target-choice signature and supports within-plot neighborhood interpretation only."
        ),
        "artifacts": {
            "interactive_html": index_path.name,
            "detailed_interactive_html": detailed_path.name,
            "compact_family_overview": compact_path.name,
            "compact_language_overview": compact_language_path.name,
            "compact_language_overview_svg": compact_language_svg_path.name,
            "overall_heatmap": overall_path.name,
            "all_languages_heatmap": all_languages_path.name,
            "signature_tsne": tsne_path.name,
            "signature_coordinates": coordinates_path.name,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(details, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate one structured layer-target-choice LoLA prototype."
    )
    parser.add_argument("--routing-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--languages-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    details = generate_prototype(
        args.routing_root.resolve(),
        args.run_id,
        args.languages_csv.resolve(),
        args.output_dir.resolve() if args.output_dir else None,
    )
    print(
        "[OK] wrote target-expert LoLA prototype for "
        f"{details['run_id']}: {details['layers']} layers x "
        f"{len(details['targets'])} targets x {details['choices']} choices"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
