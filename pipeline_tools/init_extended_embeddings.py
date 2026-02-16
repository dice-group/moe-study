#!/usr/bin/env python3
"""Initialize model embeddings for extended tokenizer tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from data_prep.merlin_data_prep.tokenizer_extension.src.extension import (
    ExtensionConfig,
    _maybe_initialize_model_embeddings,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize model embeddings for an extended tokenizer.")
    parser.add_argument("--base-tokenizer", required=True, help="Base tokenizer path or HF id.")
    parser.add_argument("--extended-tokenizer", required=True, help="Extended tokenizer path.")
    parser.add_argument("--model-path", required=True, help="Base model path or HF id.")
    parser.add_argument("--output-model-dir", required=True, help="Output directory for initialized model.")
    parser.add_argument("--manifest-path", required=True, help="Output JSON manifest path.")
    args = parser.parse_args()

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    extended_tokenizer = AutoTokenizer.from_pretrained(args.extended_tokenizer, use_fast=True)
    base_vocab = base_tokenizer.get_vocab()
    ext_vocab = extended_tokenizer.get_vocab()
    new_tokens = sorted([tok for tok in ext_vocab if tok not in base_vocab])

    output_model_dir = Path(args.output_model_dir).resolve()
    cfg = ExtensionConfig(
        base_tokenizer_path=Path(args.base_tokenizer),
        multilingual_tokenizer_path=Path(args.extended_tokenizer),
        data_dir=Path(args.extended_tokenizer),
        output_dir=Path(args.extended_tokenizer),
        init_embeddings=True,
        model_path=args.model_path,
        initialized_model_dir=output_model_dir,
    )
    initialized_dir = _maybe_initialize_model_embeddings(cfg)
    if initialized_dir is None:
        raise SystemExit("Embedding initialization was skipped unexpectedly.")

    # Keep tokenizer colocated with the initialized model root for downstream consumers.
    extended_tokenizer.save_pretrained(initialized_dir)

    payload = {
        "base_tokenizer": str(args.base_tokenizer),
        "extended_tokenizer": str(Path(args.extended_tokenizer).resolve()),
        "model_path": str(args.model_path),
        "initialized_model_dir": str(initialized_dir.resolve()),
        "original_vocab_size": len(base_vocab),
        "new_vocab_size": len(ext_vocab),
        "added_tokens": len(new_tokens),
        "new_tokens_preview": new_tokens[:64],
    }
    manifest_path = Path(args.manifest_path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[OK] initialized_model_dir={initialized_dir}")
    print(f"[OK] vocab_growth={payload['new_vocab_size'] - payload['original_vocab_size']}")
    print(f"[OK] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
