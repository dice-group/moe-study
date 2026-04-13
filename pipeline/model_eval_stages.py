#!/usr/bin/env python3
"""Training/eval/report stage handler registry for multilingual proof pipeline."""

from __future__ import annotations

from .stages.baseline_eval import stage_baseline_eval
from .stages.benchmark import stage_benchmark
from .stages.embedding_verify import stage_embedding_verify
from .stages.final_eval_loss import stage_final_eval_loss
from .stages.report import stage_report
from .stages.summary import stage_summary
from .stages.train import stage_train


MODEL_STAGE_HANDLERS = {
    "train": stage_train,
    "final_eval_loss": stage_final_eval_loss,
    "embedding_verify": stage_embedding_verify,
    "benchmark": stage_benchmark,
    "summary": stage_summary,
    "baseline_eval": stage_baseline_eval,
    "report": stage_report,
}
