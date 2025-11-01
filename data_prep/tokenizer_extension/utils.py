from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import yaml

from .coverage import CoverageConfig, compute_coverage

def resolve_path(value: str | Path, base_dir: Path) -> Path:
    """Resolve relative paths relative to a base directory"""
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def load_or_compute_metrics(
    csv_path: Optional[Path],
    coverage_cfg: Optional[CoverageConfig],
    *,
    default_name: str,
    anchor: Path,
) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    """Load metrics from CSV or compute coverage if needed"""
    if csv_path and csv_path.exists():
        return pd.read_csv(csv_path), csv_path

    if not coverage_cfg:
        return None, None

    if not coverage_cfg.output_csv:
        coverage_cfg.output_csv = anchor / default_name

    df = compute_coverage(coverage_cfg)
    return df, coverage_cfg.output_csv


def load_yaml(path: Path) -> dict:
    """Load a YAML file safely"""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level.")
    return data

def load_pipeline_paths(config_path: Path) -> dict[str, Path]:
    """Load a YAML config and resolve all relevant paths."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {config_path} must contain a mapping at the top level.")

    root = config_path.parent
    work_dir = resolve_path(cfg.get("work_dir", "."), root)
    metrics_dir = work_dir / "metrics"
    work_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    def get_path(key: str, default: Optional[Path] = None) -> Optional[Path]:
        val = cfg.get(key)
        return resolve_path(val, root) if val else default

    return {
        "work_dir": work_dir,
        "metrics_dir": metrics_dir,
        "allocation_csv": get_path("allocation_csv", work_dir / "allocation.csv"),
        "comparison_csv": get_path("comparison_output_csv", metrics_dir / "comparison.csv"),
        "train_output_dir": get_path("train_output_dir", work_dir / "trained_tokenizer"),
        "extension_output_dir": get_path("extension_output_dir", work_dir / "extended_tokenizer"),
        "extended_cov_output": get_path("extended_coverage_output", metrics_dir / "extended_tokenizer.csv"),
        "extended_metrics_csv": get_path("extended_metrics_csv"),
        "base_metrics_csv": get_path("base_metrics_csv"),
    }