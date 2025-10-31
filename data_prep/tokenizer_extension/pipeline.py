from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import yaml

from .allocation import AllocationConfig, compute_allocation
from .coverage import CoverageConfig, compute_coverage
from .extension import ExtensionConfig, ExtensionResult, extend_tokenizer
from .training import TrainingConfig, TrainingResult, train_tokenizer


@dataclass
class ComparisonResult:
    dataframe: pd.DataFrame
    output_csv: Optional[Path]


@dataclass
class PipelineConfig:
    base_metrics_csv: Optional[Path]
    allocation_csv: Path
    allocation: AllocationConfig
    training: Optional[TrainingConfig] = None
    base_coverage: Optional[CoverageConfig] = None
    extension: Optional[ExtensionConfig] = None
    extended_metrics_csv: Optional[Path] = None
    extended_coverage: Optional[CoverageConfig] = None
    comparison_output_csv: Optional[Path] = None


@dataclass
class PipelineResult:
    base_metrics: pd.DataFrame
    allocation: pd.DataFrame
    allocation_csv: Path
    training: Optional[TrainingResult]
    extension: Optional[ExtensionResult]
    extended_metrics: Optional[pd.DataFrame]
    comparison: Optional[ComparisonResult]


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    training_result = None
    if config.training:
        training_result = train_tokenizer(config.training)

    base_metrics_df, _ = _load_or_compute_metrics(
        config.base_metrics_csv,
        config.base_coverage,
        default_name="tokenizer_metrics_base.csv",
        anchor=config.allocation_csv.parent,
    )
    if base_metrics_df is None:
        raise ValueError("Either provide base_metrics_csv or enable compute_base_coverage in the config.")

    allocation_df = compute_allocation(base_metrics_df, config.allocation)
    config.allocation_csv.parent.mkdir(parents=True, exist_ok=True)
    allocation_df.to_csv(config.allocation_csv, index=False)
    print(f"[tokenize_extension] Allocation saved to {config.allocation_csv}")

    extension_result = None
    if config.extension:
        extension_result = extend_tokenizer(config.extension, allocation_df)

    extended_metrics_df, _ = _load_or_compute_metrics(
        config.extended_metrics_csv,
        config.extended_coverage,
        default_name="tokenizer_metrics_extended.csv",
        anchor=config.allocation_csv.parent,
    )

    comparison_result = None
    if extended_metrics_df is not None:
        comparison_df = compare_metrics(base_metrics_df, extended_metrics_df)
        if config.comparison_output_csv:
            config.comparison_output_csv.parent.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(config.comparison_output_csv, index=False)
            comparison_result = ComparisonResult(comparison_df, config.comparison_output_csv)
        else:
            comparison_result = ComparisonResult(comparison_df, None)

    return PipelineResult(
        base_metrics=base_metrics_df,
        allocation=allocation_df,
        allocation_csv=config.allocation_csv,
        training=training_result,
        extension=extension_result,
        extended_metrics=extended_metrics_df,
        comparison=comparison_result,
    )


def compare_metrics(base_df: pd.DataFrame, extended_df: pd.DataFrame) -> pd.DataFrame:
    base = base_df.rename(
        columns={
            "pieces_per_word": "ppw_base",
            "chars_per_token": "cpt_base",
            "unknown_rate": "unk_base",
        }
    )
    extended = extended_df.rename(
        columns={
            "pieces_per_word": "ppw_new",
            "chars_per_token": "cpt_new",
            "unknown_rate": "unk_new",
        }
    )
    df = pd.merge(base, extended, on="language", how="inner")
    df["ppw_diff"] = df["ppw_base"] - df["ppw_new"]
    df["cpt_diff"] = df["cpt_new"] - df["cpt_base"]
    df["unk_diff"] = df["unk_base"] - df["unk_new"]
    return df


def _load_or_compute_metrics(
    csv_path: Optional[Path],
    coverage_config: Optional[CoverageConfig],
    *,
    default_name: str,
    anchor: Path,
) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    if csv_path:
        if not csv_path.exists():
            raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")
        return pd.read_csv(csv_path), csv_path

    if coverage_config is None:
        return None, None

    if coverage_config.output_csv is None:
        coverage_output = anchor / default_name
        coverage_config = CoverageConfig(
            data_dir=coverage_config.data_dir,
            tokenizer=coverage_config.tokenizer,
            max_lines=coverage_config.max_lines,
            num_workers=coverage_config.num_workers,
            output_csv=coverage_output,
        )

    df = compute_coverage(coverage_config)
    return df, coverage_config.output_csv


def _resolve_path(value, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _load_pipeline_config(config_path: Path) -> PipelineConfig:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {config_path} must contain a mapping at the top level.")

    root = config_path.parent

    work_dir = _resolve_path(cfg["work_dir"], root)
    work_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = work_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    def cfg_path(key: str, default: Optional[Path] = None) -> Optional[Path]:
        if key not in cfg or cfg[key] is None:
            return default
        return _resolve_path(cfg[key], root)

    allocation_csv = cfg_path("allocation_csv", work_dir / "allocation.csv")
    comparison_csv = cfg_path("comparison_output_csv", metrics_dir / "comparison.csv")
    train_output_dir = cfg_path("train_output_dir", work_dir / "trained_tokenizer")
    extension_output_dir = cfg_path("extension_output_dir", work_dir / "extended_tokenizer")
    extended_cov_output = cfg_path("extended_coverage_output", metrics_dir / "extended_tokenizer.csv")
    extended_metrics_csv = cfg_path("extended_metrics_csv")
    base_metrics_csv = cfg_path("base_metrics_csv")

    base_coverage = None
    if cfg.get("compute_base_coverage", False):
        base_coverage = CoverageConfig(
            data_dir=_resolve_path(cfg["base_data_dir"], root),
            tokenizer=cfg["base_tokenizer"],
            max_lines=cfg.get("base_max_lines", 10_000),
            num_workers=cfg.get("base_num_workers"),
            output_csv=cfg_path("base_coverage_output", metrics_dir / "base_tokenizer.csv"),
        )

    training = None
    if cfg.get("train_multilingual", False):
        training = TrainingConfig(
            data_dir=_resolve_path(cfg["train_data_dir"], root),
            output_dir=train_output_dir,
            vocab_size=cfg.get("train_vocab_size", 250_000),
            base_model=cfg.get("train_base_model", "meta-llama/Llama-3.2-1B"),
            text_key=cfg.get("train_text_key", "text"),
            max_samples=cfg.get("train_max_samples"),
        )

    extension = None
    if cfg.get("extend", False):
        multilingual_path = cfg.get("extension_multilingual_path")
        if multilingual_path is None:
            multilingual_resolved = train_output_dir
        else:
            multilingual_resolved = _resolve_path(multilingual_path, root)
        extension = ExtensionConfig(
            base_tokenizer_path=_resolve_path(cfg["extension_base_path"], root),
            multilingual_tokenizer_path=multilingual_resolved,
            data_dir=_resolve_path(cfg["extension_data_dir"], root),
            output_dir=extension_output_dir,
            sample_docs=cfg.get("extension_sample_docs", 50),
            text_key=cfg.get("extension_text_key", "text"),
            vocab_cap=cfg.get("extension_vocab_cap", 256_000),
        )

    extended_coverage = None
    if cfg.get("compute_extended_coverage", False):
        extended_data_dir = cfg.get("extended_data_dir", cfg.get("extension_data_dir"))
        extended_coverage = CoverageConfig(
            data_dir=_resolve_path(extended_data_dir, root),
            tokenizer=cfg.get("extended_tokenizer", str(extension_output_dir)),
            max_lines=cfg.get("extended_max_lines", 10_000),
            num_workers=cfg.get("extended_num_workers"),
            output_csv=extended_cov_output,
        )
        extended_metrics_csv = extended_cov_output

    allocation = AllocationConfig(
        total_new_tokens=cfg.get("total_new_tokens", 128_000),
        weight_ppw=cfg.get("weight_ppw", 0.5),
        weight_cpt=cfg.get("weight_cpt", 0.5),
        weight_unk=cfg.get("weight_unk", 0.0),
        gamma=cfg.get("gamma", 2.0),
    )

    return PipelineConfig(
        base_metrics_csv=base_metrics_csv,
        allocation_csv=allocation_csv,
        allocation=allocation,
        training=training,
        base_coverage=base_coverage,
        extension=extension,
        extended_metrics_csv=extended_metrics_csv,
        extended_coverage=extended_coverage,
        comparison_output_csv=comparison_csv,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> list[Path]:
    parser = argparse.ArgumentParser(description="Vocabulary extension pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        nargs="+",
        required=True,
        help="One or more YAML config files.",
    )
    args = parser.parse_args(argv)
    return args.config


def main(argv: Optional[Sequence[str]] = None) -> Optional[PipelineResult]:
    config_paths = _parse_args(argv)
    last_result: Optional[PipelineResult] = None
    for path in config_paths:
        pipeline_config = _load_pipeline_config(path)
        print(f"[tokenize_extension] Running config {path}")
        last_result = run_pipeline(pipeline_config)
    return last_result


if __name__ == "__main__":
    main()

