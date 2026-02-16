#!/usr/bin/env python3
"""Data-oriented stage handler registry for multilingual proof pipeline."""

from __future__ import annotations

from .stages.audit import stage_audit
from .stages.embedding_init import stage_embedding_init
from .stages.sample import stage_sample
from .stages.tokenize import stage_tokenize
from .stages.tokenizer_extend import stage_tokenizer_extend


DATA_STAGE_HANDLERS = {
    "sample": stage_sample,
    "tokenizer_extend": stage_tokenizer_extend,
    "embedding_init": stage_embedding_init,
    "tokenize": stage_tokenize,
    "audit": stage_audit,
}
