from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def create_heatmap(matrix: np.ndarray, y_labels: list[str], title: str, output: Path, *, axis_name: str) -> None:
    plt = _ensure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    data = matrix.reshape(matrix.shape[0], -1)
    fig_width = max(12, min(40, data.shape[1] * 0.18))
    fig_height = max(6, min(42, len(y_labels) * 0.24))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(data, aspect="auto", interpolation="nearest", vmin=0, vmax=max(1.0, float(np.nanmax(data) or 1.0)))
    ax.set_title(title)
    ax.set_xlabel(f"{axis_name}s grouped by captured module")
    ax.set_ylabel("Language")
    if len(y_labels) <= 120:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=7)
    else:
        ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def create_entropy_plot(entropy: np.ndarray, output: Path) -> None:
    plt = _ensure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    mean_entropy = np.nanmean(entropy, axis=0) if entropy.size else np.array([])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(mean_entropy)), mean_entropy, marker="o", linewidth=1.5)
    ax.set_title("Mean Routing Entropy by Captured Module")
    ax.set_xlabel("Captured module index")
    ax.set_ylabel("Entropy")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def create_mola_null_plot(payload: dict[str, Any], output: Path) -> None:
    plt = _ensure_matplotlib()
    null_assignments = payload["null_assignments"].astype(float)
    assignments = payload["assignments"].astype(float)
    bypass = payload["bypass_tokens"].astype(float)
    tokens = payload["tokens"].astype(float)
    null_frac = np.divide(null_assignments.sum(axis=1), np.maximum(assignments.sum(axis=1), 1))
    bypass_frac = np.divide(bypass.sum(axis=1), np.maximum(tokens.sum(axis=1), 1))
    x = np.arange(len(null_frac))
    fig, ax = plt.subplots(figsize=(max(12, len(x) * 0.18), 4))
    ax.plot(x, null_frac, label="null assignment fraction")
    ax.plot(x, bypass_frac, label="all-selected-null token fraction")
    ax.set_ylim(0, 1)
    ax.set_title("MoLA Null Expert Usage")
    ax.set_xlabel("Language index")
    ax.set_ylabel("Fraction")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def aggregate_by_group(
    matrix: np.ndarray,
    languages: list[str],
    language_to_group: dict[str, str],
) -> tuple[np.ndarray, list[str]]:
    groups: dict[str, list[int]] = {}
    for idx, language in enumerate(languages):
        group = language_to_group.get(language, "unknown")
        groups.setdefault(group, []).append(idx)
    labels = sorted(groups)
    aggregated = np.zeros((len(labels), matrix.shape[1], matrix.shape[2]), dtype=np.float64)
    for group_idx, label in enumerate(labels):
        aggregated[group_idx] = matrix[groups[label]].mean(axis=0)
    return aggregated, labels
