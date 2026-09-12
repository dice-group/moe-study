from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon

from .io import atomic_write_json


def _load_run(root: Path, run_id: str) -> dict[str, Any]:
    with np.load(root / run_id / "routing_matrix.npz", allow_pickle=True) as source:
        payload = {key: source[key] for key in source.files}
    payload["languages"] = [str(item) for item in payload["languages"].tolist()]
    payload["layer_labels"] = [str(item) for item in payload["layer_labels"].tolist()]
    payload["choice_labels"] = [str(item) for item in payload["choice_labels"].tolist()]
    payload["axis_name"] = str(payload["axis_name"])
    payload["peft_type"] = str(payload["peft_type"])
    payload["routing_probability"] = np.asarray(payload["routing_probability"], dtype=np.float64)
    return payload


def _read_summary(root: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    return {str(item["run_id"]): item for item in payload.get("runs", [])}


def _language_metadata(path: Path, languages: set[str]) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split", "")).strip() != "train":
                continue
            language = str(row.get("subset", "")).strip()
            if language not in languages:
                continue
            metadata[language] = {
                "family": str(row.get("family", "")).strip() or "unknown",
                "script": str(row.get("script", "")).strip() or "unknown",
                "resource": str(row.get("resource_category", "")).strip() or "unknown",
            }
    if "english" in languages:
        metadata["english"] = {"family": "English", "script": "Latn", "resource": "high"}
    missing = languages - set(metadata)
    if missing:
        raise ValueError(f"Missing language metadata for {sorted(missing)}")
    return metadata


def _metric_record(
    *,
    panel: str,
    run_id: str,
    peft_type: str,
    axis_name: str,
    routing_basis: str,
    group_type: str,
    group: str,
    matrix: np.ndarray,
) -> dict[str, Any]:
    choices = matrix.shape[-1]
    entropy = -(matrix * np.log(matrix + 1e-12)).sum(axis=2)
    normalized_entropy = entropy / np.log(max(2, choices))
    mean_choice = matrix.mean(axis=(0, 1))
    mean_load = float(mean_choice.mean())
    load_cv = float(mean_choice.std() / mean_load) if mean_load > 0 else 0.0
    return {
        "panel": panel,
        "run_id": run_id,
        "peft_type": peft_type,
        "axis_name": axis_name,
        "routing_basis": routing_basis,
        "group_type": group_type,
        "group": group,
        "routing_ids": int(matrix.shape[0]),
        "modules": int(matrix.shape[1]),
        "choices": int(choices),
        "normalized_entropy_mean": float(normalized_entropy.mean()),
        "normalized_entropy_std": float(normalized_entropy.std()),
        "effective_choices_mean": float(np.exp(entropy).mean()),
        "max_probability_mean": float(matrix.max(axis=2).mean()),
        "load_cv": load_cv,
        "unused_choice_fraction": float((mean_choice < 1e-6).mean()),
    }


def build_metric_rows(
    *,
    panel: str,
    root: Path,
    metadata_csv: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    summaries = _read_summary(root)
    runs = {run_id: _load_run(root, run_id) for run_id in summaries}
    all_languages = {language for run in runs.values() for language in run["languages"]}
    metadata = _language_metadata(metadata_csv, all_languages)
    rows: list[dict[str, Any]] = []
    for run_id, run in runs.items():
        matrix = run["routing_probability"]
        common = {
            "panel": panel,
            "run_id": run_id,
            "peft_type": run["peft_type"],
            "axis_name": run["axis_name"],
            "routing_basis": str(summaries[run_id].get("routing_basis", "")),
        }
        rows.append(_metric_record(**common, group_type="overall", group="all", matrix=matrix))
        for group_type in ("family", "family_script", "family_resource"):
            groups: dict[str, list[int]] = defaultdict(list)
            for index, language in enumerate(run["languages"]):
                meta = metadata[language]
                if group_type == "family":
                    label = meta["family"]
                elif group_type == "family_script":
                    label = f"{meta['family']} / {meta['script']}"
                else:
                    label = f"{meta['family']} / {meta['resource']}"
                groups[label].append(index)
            for label, indices in sorted(groups.items()):
                rows.append(
                    _metric_record(
                        **common,
                        group_type=group_type,
                        group=label,
                        matrix=matrix[indices],
                    )
                )
    return rows, runs


def _column_cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=0, keepdims=True)
    right_norm = np.linalg.norm(right, axis=0, keepdims=True)
    left = np.divide(left, left_norm, out=np.zeros_like(left), where=left_norm > 0)
    right = np.divide(right, right_norm, out=np.zeros_like(right), where=right_norm > 0)
    return left.T @ right


def align_and_compare(
    run_id: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if left["axis_name"] != right["axis_name"]:
        raise ValueError(f"{run_id}: routing axes differ")
    if left["routing_probability"].shape[2] != right["routing_probability"].shape[2]:
        raise ValueError(f"{run_id}: choice counts differ")
    left_modules = {label: index for index, label in enumerate(left["layer_labels"])}
    right_modules = {label: index for index, label in enumerate(right["layer_labels"])}
    common_modules = [label for label in left["layer_labels"] if label in right_modules]
    if len(common_modules) != len(left_modules) or len(common_modules) != len(right_modules):
        raise ValueError(f"{run_id}: module labels differ across panels")

    common_languages = [language for language in left["languages"] if language in set(right["languages"])]
    left_language = {language: index for index, language in enumerate(left["languages"])}
    right_language = {language: index for index, language in enumerate(right["languages"])}
    left_rows = [left_language[language] for language in common_languages]
    right_rows = [right_language[language] for language in common_languages]
    per_language_cosine = np.zeros((len(common_languages), len(common_modules)), dtype=np.float64)
    per_language_js = np.zeros_like(per_language_cosine)
    alignments: list[dict[str, Any]] = []

    for module_out, label in enumerate(common_modules):
        left_matrix = left["routing_probability"][left_rows, left_modules[label], :]
        right_matrix = right["routing_probability"][right_rows, right_modules[label], :]
        similarity = _column_cosine_similarity(left_matrix, right_matrix)
        left_choices, right_choices = linear_sum_assignment(-similarity)
        permutation = np.empty(len(left_choices), dtype=np.int64)
        permutation[left_choices] = right_choices
        aligned = right_matrix[:, permutation]
        for choice, matched in enumerate(permutation):
            alignments.append(
                {
                    "run_id": run_id,
                    "module": label,
                    "left_choice": choice,
                    "right_choice": int(matched),
                    "choice_vector_cosine": float(similarity[choice, matched]),
                }
            )
        for language_index in range(len(common_languages)):
            a = left_matrix[language_index]
            b = aligned[language_index]
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            per_language_cosine[language_index, module_out] = (
                float(np.dot(a, b) / denominator) if denominator > 0 else 0.0
            )
            per_language_js[language_index, module_out] = float(jensenshannon(a, b, base=2.0))

    stability = [
        {
            "run_id": run_id,
            "language": language,
            "shared_modules": len(common_modules),
            "aligned_cosine_mean": float(per_language_cosine[index].mean()),
            "aligned_cosine_std": float(per_language_cosine[index].std()),
            "jensen_shannon_distance_mean": float(per_language_js[index].mean()),
            "jensen_shannon_distance_std": float(per_language_js[index].std()),
        }
        for index, language in enumerate(common_languages)
    ]
    return stability, alignments


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty comparison table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(metric_rows: list[dict[str, Any]], stability: list[dict[str, Any]], output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    overall = [row for row in metric_rows if row["group_type"] == "overall"]
    run_ids = sorted({str(row["run_id"]) for row in overall})
    panels = ["96", "200"]
    x = np.arange(len(run_ids))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5))
    for offset, panel in enumerate(panels):
        values = [
            next(float(row["normalized_entropy_mean"]) for row in overall if row["panel"] == panel and row["run_id"] == run_id)
            for run_id in run_ids
        ]
        ax.bar(x + (offset - 0.5) * width, values, width, label=f"{panel}-language panel")
    ax.set_xticks(x)
    ax.set_xticklabels(run_ids, rotation=25, ha="right")
    ax.set_ylabel("Mean normalized routing entropy")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    entropy_path = output / "normalized_entropy_96_vs_200.png"
    fig.savefig(entropy_path, dpi=180)
    plt.close(fig)

    stability_by_run: dict[str, list[float]] = defaultdict(list)
    for row in stability:
        stability_by_run[str(row["run_id"])].append(float(row["aligned_cosine_mean"]))
    fig, ax = plt.subplots(figsize=(10, 5))
    values = [float(np.mean(stability_by_run[run_id])) for run_id in run_ids]
    ax.bar(run_ids, values, color="#2563eb")
    ax.set_ylabel("Mean aligned routing cosine")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    stability_path = output / "shared_language_routing_stability.png"
    fig.savefig(stability_path, dpi=180)
    plt.close(fig)
    return [str(entropy_path), str(stability_path)]


def build_comparison(
    *,
    root_96: Path,
    root_200: Path,
    metadata_96: Path,
    metadata_200: Path,
    output: Path,
) -> dict[str, Any]:
    rows_96, runs_96 = build_metric_rows(panel="96", root=root_96, metadata_csv=metadata_96)
    rows_200, runs_200 = build_metric_rows(panel="200", root=root_200, metadata_csv=metadata_200)
    if set(runs_96) != set(runs_200):
        raise ValueError(f"Panel run ids differ: 96={sorted(runs_96)}, 200={sorted(runs_200)}")

    stability: list[dict[str, Any]] = []
    alignments: list[dict[str, Any]] = []
    for run_id in sorted(runs_96):
        run_stability, run_alignments = align_and_compare(run_id, runs_96[run_id], runs_200[run_id])
        stability.extend(run_stability)
        alignments.extend(run_alignments)

    output.mkdir(parents=True, exist_ok=True)
    metric_rows = rows_96 + rows_200
    _write_csv(output / "routing_metrics.csv", metric_rows)
    _write_csv(output / "shared_language_stability.csv", stability)
    _write_csv(output / "choice_alignments.csv", alignments)
    figures = _write_figures(metric_rows, stability, output / "figures")
    summary = {
        "schema_version": 1,
        "root_96": str(root_96.resolve()),
        "root_200": str(root_200.resolve()),
        "run_ids": sorted(runs_96),
        "shared_routing_ids": len({row["language"] for row in stability}),
        "metric_rows": len(metric_rows),
        "stability_rows": len(stability),
        "alignment_rows": len(alignments),
        "files": {
            "routing_metrics": str(output / "routing_metrics.csv"),
            "shared_language_stability": str(output / "shared_language_stability.csv"),
            "choice_alignments": str(output / "choice_alignments.csv"),
            "figures": figures,
        },
        "interpretation": {
            "cross_method": "Compare only normalized, permutation-invariant summary metrics.",
            "cross_panel": "Choice identities are matched per module with Hungarian assignment on shared-language vectors.",
        },
    }
    atomic_write_json(output / "comparison_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare final routing analyses for the 96- and 200-language panels.")
    parser.add_argument("--root-96", type=Path, required=True)
    parser.add_argument("--root-200", type=Path, required=True)
    parser.add_argument("--metadata-96", type=Path, required=True)
    parser.add_argument("--metadata-200", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = build_comparison(
        root_96=args.root_96,
        root_200=args.root_200,
        metadata_96=args.metadata_96,
        metadata_200=args.metadata_200,
        output=args.output,
    )
    print(f"[OK] wrote routing comparison for {len(summary['run_ids'])} approaches to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
