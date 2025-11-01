from __future__ import annotations
from pathlib import Path

import pandas as pd
import yaml

from .allocation import AllocationConfig
from .coverage import CoverageConfig
from .extension import ExtensionConfig
from .training import TrainingConfig

def resolve_path(value: str | Path, base_dir: Path) -> Path:
    """Resolve relative paths relative to a base directory"""
    path = Path(value)
    return path if path.is_absolute() else base_dir / path

def load_pipeline_config(config_path: Path):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config: {config_path}")

    root = config_path.parent
    output_dir = (root / cfg.get("output_dir", "results")).resolve()

    base_coverage = CoverageConfig(
        data_dir=resolve_path(cfg["base_data_dir"], root),
        tokenizer=cfg["base_tokenizer"],
        max_lines=cfg.get("base_max_lines", 10_000),
        num_workers=cfg.get("base_num_workers"),
        output_csv=None,  # is set later in pipeline
    )

    training = (
        TrainingConfig(
            data_dir=resolve_path(cfg["train_data_dir"], root),
            output_dir=output_dir / "trained_tokenizer",
            vocab_size=cfg.get("train_vocab_size", 250_000),
            max_samples=cfg.get("train_max_samples"),
        )
        if cfg.get("train_multilingual")
        else None
    )

    extension = (
        ExtensionConfig(
            base_tokenizer_path=resolve_path(cfg["extension_base_path"], root),
            multilingual_tokenizer_path=output_dir / "trained_tokenizer",
            data_dir=resolve_path(cfg["extension_data_dir"], root),
            output_dir=output_dir / "extended_tokenizer",
        )
        if cfg.get("extend")
        else None
    )

    extended_coverage = (
        CoverageConfig(
            data_dir=resolve_path(cfg.get("extended_data_dir", cfg["extension_data_dir"]), root),
            tokenizer=str(output_dir / "extended_tokenizer"),
            max_lines=cfg.get("extended_max_lines", 10_000),
        )
        if cfg.get("compute_extended_coverage")
        else None
    )

    allocation = AllocationConfig(
        total_new_tokens=cfg.get("total_new_tokens", 128_000),
        weight_ppw=cfg.get("weight_ppw", 0.5),
        weight_cpt=cfg.get("weight_cpt", 0.5),
    )

    return {
        "output_dir": output_dir,
        "allocation": allocation,
        "training": training,
        "base_coverage": base_coverage,
        "extension": extension,
        "extended_coverage": extended_coverage,
    }

def compare_metrics(base_df: pd.DataFrame, extended_df: pd.DataFrame) -> pd.DataFrame:
    """ Compares metrics of base tokenizer and extended tokenizer"""
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