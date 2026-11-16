#!/usr/bin/env python3
"""Context objects for proof pipeline stage execution."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .executor import Executor


@dataclass(slots=True)
class PipelineContext:
    args: argparse.Namespace
    repo_root: Path
    profile: dict[str, Any]
    force_stages: set[str]
    active_stages: list[str]

    output_root: Path
    state_path: Path

    language_map_path: Path
    sample_cfg: dict[str, Any]
    tokenize_cfg: dict[str, Any]
    audit_cfg: dict[str, Any]
    train_cfg: dict[str, Any]
    tokenizer_extension_cfg: dict[str, Any]
    embedding_init_cfg: dict[str, Any]
    embedding_training_cfg: dict[str, Any]
    summary_cfg: dict[str, Any]
    train_acceptance_cfg: dict[str, Any]
    final_eval_loss_cfg: dict[str, Any]
    benchmark_cfg: dict[str, Any]
    baseline_eval_cfg: dict[str, Any]
    report_cfg: dict[str, Any]
    wandb_cfg: dict[str, Any]

    env: dict[str, str]
    logger: logging.Logger
    executor: Executor

    sampled_dir: Path
    sampled_manifest: Path
    shard_subdir: str
    sample_shard_dir: Path
    tokenized_dir: Path
    train_root: Path
    summary_path: Path
    gate_path: Path
    train_acceptance_root: Path
    final_eval_loss_root: Path
    benchmark_root: Path
    baseline_cache_root: Path
    report_xlsx: Path
    lpr_contract_path: Path
    extension_root: Path
    extension_pipeline_root: Path
    extension_prepared_dir: Path
    extension_prepared_manifest: Path
    extension_manifest_path: Path
    extended_tokenizer_dir: Path
    embedding_init_output_dir: Path
    embedding_init_manifest_path: Path
    embedding_verify_root: Path
    embedding_verify_summary_path: Path
    lineage_manifest_path: Path
    logs_root: Path

    run_naming_cfg: dict[str, Any]
    run_timestamp: str
    naming_languages: list[str]
    lang_count: int
    model_slug: str
    exp_tag: str
    base_run_label: str
