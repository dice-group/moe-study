from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .allocation import AllocationConfig, compute_allocation
from .coverage import CoverageConfig, compute_coverage
from .extension import ExtensionConfig, ExtensionResult, extend_tokenizer


@dataclass
class ComparisonResult:
    dataframe: pd.DataFrame
    output_csv: Optional[Path]


@dataclass
class PipelineConfig:
    base_metrics_csv: Optional[Path]
    allocation_csv: Path
    allocation: AllocationConfig
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
    extension: Optional[ExtensionResult]
    extended_metrics: Optional[pd.DataFrame]
    comparison: Optional[ComparisonResult]


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    base_metrics_df, _ = _load_or_compute_metrics(
        config.base_metrics_csv, config.base_coverage, default_name="tokenizer_metrics_base.csv", anchor=config.allocation_csv.parent
    )
    if base_metrics_df is None:
        raise ValueError("Currently base metrics need to be provided")

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
            output_path = config.comparison_output_csv
        else:
            output_path = None
        comparison_result = ComparisonResult(dataframe=comparison_df, output_csv=output_path)

    return PipelineResult(
        base_metrics=base_metrics_df,
        allocation=allocation_df,
        allocation_csv=config.allocation_csv,
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
) -> tuple[pd.DataFrame, Optional[Path]]:
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


def _parse_args(argv: Optional[Sequence[str]] = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Vocabulary extension pipeline."
    )
    parser.add_argument("--base-metrics-csv", type=Path)
    parser.add_argument("--allocation-csv", type=Path, required=True)
    parser.add_argument("--total-new-tokens", type=int, default=128_000)
    parser.add_argument("--weight-ppw", type=float, default=0.5)
    parser.add_argument("--weight-cpt", type=float, default=0.5)
    parser.add_argument("--weight-unk", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--comparison-output-csv", type=Path)

    parser.add_argument("--compute-base-coverage", action="store_true")
    parser.add_argument("--base-data-dir", type=Path)
    parser.add_argument("--base-tokenizer", type=str)
    parser.add_argument("--base-max-lines", type=int, default=10_000)
    parser.add_argument("--base-num-workers", type=int)
    parser.add_argument("--base-coverage-output", type=Path)

    parser.add_argument("--extend", action="store_true")
    parser.add_argument("--lang-dir", type=Path)
    parser.add_argument("--output-tokenizer-dir", type=Path)
    parser.add_argument("--extension-tokenizer", type=str)

    parser.add_argument("--compute-extended-coverage", action="store_true")
    parser.add_argument("--extended-data-dir", type=Path)
    parser.add_argument("--extended-tokenizer", type=str)
    parser.add_argument("--extended-max-lines", type=int, default=10_000)
    parser.add_argument("--extended-num-workers", type=int)
    parser.add_argument("--extended-coverage-output", type=Path)
    parser.add_argument("--extended-metrics-csv", type=Path)

    args = parser.parse_args(argv)

    base_coverage = None
    if args.compute_base_coverage:
        for name in ("base_data_dir", "base_tokenizer"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required when computing base coverage.")
        base_coverage = CoverageConfig(
            data_dir=args.base_data_dir,
            tokenizer=args.base_tokenizer,
            max_lines=args.base_max_lines,
            num_workers=args.base_num_workers,
            output_csv=args.base_coverage_output,
        )

    extension = None
    if args.extend:
        for name in ("extension_tokenizer", "lang_dir", "output_tokenizer_dir"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required when using --extend.")
        extension = ExtensionConfig(
            base_tokenizer=args.extension_tokenizer,
            lang_dir=args.lang_dir,
            output_dir=args.output_tokenizer_dir,
        )

    extended_coverage = None
    extended_metrics_csv = args.extended_metrics_csv
    if args.compute_extended_coverage:
        for name in ("extended_data_dir", "extended_tokenizer"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required when computing extended coverage.")
        extended_coverage = CoverageConfig(
            data_dir=args.extended_data_dir,
            tokenizer=args.extended_tokenizer,
            max_lines=args.extended_max_lines,
            num_workers=args.extended_num_workers,
            output_csv=args.extended_coverage_output,
        )
        extended_metrics_csv = None  # prefer freshly computed metrics #TODO:

    return PipelineConfig(
        base_metrics_csv=args.base_metrics_csv,
        allocation_csv=args.allocation_csv,
        allocation=AllocationConfig(
            total_new_tokens=args.total_new_tokens,
            weight_ppw=args.weight_ppw,
            weight_cpt=args.weight_cpt,
            weight_unk=args.weight_unk,
            gamma=args.gamma,
        ),
        base_coverage=base_coverage,
        extension=extension,
        extended_metrics_csv=extended_metrics_csv,
        extended_coverage=extended_coverage,
        comparison_output_csv=args.comparison_output_csv,
    )


def main(argv: Optional[Sequence[str]] = None) -> PipelineResult:
    config = _parse_args(argv)
    return run_pipeline(config)


if __name__ == "__main__":
    main()
