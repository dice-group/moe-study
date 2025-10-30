from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from transformers import AutoTokenizer

#TODO: list extraction of candidate should also be added instead of assuming it will be given as input

@dataclass
class ExtensionConfig:
    base_tokenizer: str
    lang_dir: Path
    output_dir: Path


@dataclass
class ExtensionResult:
    added_tokens: int
    total_vocab_size: int
    per_language: Dict[str, int]


def extend_tokenizer(
    config: ExtensionConfig,
    allocation: pd.DataFrame,
) -> ExtensionResult:
    tokenizer = AutoTokenizer.from_pretrained(config.base_tokenizer, use_fast=True)
    base_vocab = set(tokenizer.get_vocab().keys())

    per_language: Dict[str, int] = {}
    new_tokens: List[str] = []

    for _, row in allocation.iterrows():
        language = row["language"]
        n_tokens = int(row["token_alloc"])
        if n_tokens <= 0:
            continue

        freq_file = config.lang_dir / language / f"{language}_tokens_freq.csv"
        if not freq_file.exists():
            print(f"[tokenize_extension] Missing freq file for {language}: {freq_file}")
            continue

        freq_df = pd.read_csv(freq_file)
        if "token" not in freq_df.columns:
            raise ValueError(f"{freq_file} missing required 'token' column.")

        candidates = [
            tok for tok in freq_df["token"].head(n_tokens).tolist() if tok not in base_vocab
        ]
        new_tokens.extend(candidates)
        per_language[language] = len(candidates)
        print(
            f"[tokenize_extension] {language}: requested {n_tokens}, "
            f"adding {len(candidates)} tokens."
        )

    unique_tokens = list(dict.fromkeys(new_tokens))
    added = tokenizer.add_tokens(unique_tokens)
    print(f"[tokenize_extension] Added {added} new tokens.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(config.output_dir)
    print(f"[tokenize_extension] Extended tokenizer saved to {config.output_dir}")

    return ExtensionResult(
        added_tokens=added,
        total_vocab_size=len(tokenizer),
        per_language=per_language,
    )

