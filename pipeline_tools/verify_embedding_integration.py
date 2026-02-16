#!/usr/bin/env python3
"""Verify embedding/lm_head integration contracts after short training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def _find_weight_key(keys: list[str], needle: str) -> str | None:
    matches = [key for key in keys if needle in key and key.endswith(".weight")]
    if not matches:
        return None
    matches.sort(key=len)
    return matches[0]


def _load_adapter_tensors(path: Path) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    if path.suffix == ".safetensors":
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            emb_key = _find_weight_key(keys, "embed_tokens")
            lm_key = _find_weight_key(keys, "lm_head")
            if emb_key is None or lm_key is None:
                raise RuntimeError(f"Missing embed/lm_head weights in {path}")
            emb = handle.get_tensor(emb_key).float().cpu()
            lm = handle.get_tensor(lm_key).float().cpu()
        return keys, emb, lm

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected adapter state format: {path}")
    keys = list(state.keys())
    emb_key = _find_weight_key(keys, "embed_tokens")
    lm_key = _find_weight_key(keys, "lm_head")
    if emb_key is None or lm_key is None:
        raise RuntimeError(f"Missing embed/lm_head weights in {path}")
    emb = state[emb_key].detach().float().cpu()
    lm = state[lm_key].detach().float().cpu()
    return keys, emb, lm


def _resolve_adapter_dir(run_dir: Path) -> Path:
    direct_cfg = run_dir / "adapter_config.json"
    if direct_cfg.exists():
        return run_dir

    candidates = sorted(run_dir.glob("checkpoint-*_adapter"))
    if candidates:
        return candidates[-1]
    raise SystemExit(f"Missing adapter checkpoint under: {run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify tokenizer extension + embedding training contracts.")
    parser.add_argument("--run-dir", required=True, help="Training run output directory.")
    parser.add_argument("--initialized-model-dir", required=True, help="Initialized model directory.")
    parser.add_argument("--base-tokenizer", required=True, help="Base tokenizer path or HF id.")
    parser.add_argument("--extended-tokenizer", required=True, help="Extended tokenizer path.")
    parser.add_argument("--manifest-path", required=True, help="Output JSON verification manifest path.")
    parser.add_argument("--require-weight-change", action="store_true", help="Fail if embeddings did not change.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    initialized_model_dir = Path(args.initialized_model_dir).resolve()
    adapter_dir = _resolve_adapter_dir(run_dir)
    adapter_cfg_path = adapter_dir / "adapter_config.json"
    adapter_weights_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_weights_path.exists():
        adapter_weights_path = adapter_dir / "adapter_model.bin"
    if not adapter_cfg_path.exists():
        raise SystemExit(f"Missing adapter config: {adapter_cfg_path}")
    if not adapter_weights_path.exists():
        raise SystemExit(f"Missing adapter weights: {adapter_weights_path}")

    adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
    modules = adapter_cfg.get("modules_to_save") or []
    modules_set = {str(item) for item in modules}
    has_trainable_targets = {"embed_tokens", "lm_head"}.issubset(modules_set)

    base_tok = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    ext_tok = AutoTokenizer.from_pretrained(args.extended_tokenizer, use_fast=True)
    base_vocab = base_tok.get_vocab()
    ext_vocab = ext_tok.get_vocab()
    new_token_ids = sorted([idx for token, idx in ext_vocab.items() if token not in base_vocab])

    model = AutoModelForCausalLM.from_pretrained(str(initialized_model_dir))
    init_emb = model.get_input_embeddings().weight.detach().float().cpu()
    out_layer = model.get_output_embeddings()
    if out_layer is None:
        raise SystemExit("Initialized model has no output embeddings.")
    init_lm = out_layer.weight.detach().float().cpu()

    adapter_keys, trained_emb, trained_lm = _load_adapter_tensors(adapter_weights_path)
    if trained_emb.shape != init_emb.shape:
        raise SystemExit(f"embed_tokens shape mismatch: {trained_emb.shape} vs {init_emb.shape}")
    if trained_lm.shape != init_lm.shape:
        raise SystemExit(f"lm_head shape mismatch: {trained_lm.shape} vs {init_lm.shape}")

    checked_ids = [idx for idx in new_token_ids if 0 <= idx < trained_emb.shape[0]]
    if not checked_ids:
        raise SystemExit("No valid new token ids found for embedding verification.")
    idx_tensor = torch.tensor(checked_ids, dtype=torch.long)

    emb_delta = (trained_emb.index_select(0, idx_tensor) - init_emb.index_select(0, idx_tensor)).abs().sum().item()
    lm_delta = (trained_lm.index_select(0, idx_tensor) - init_lm.index_select(0, idx_tensor)).abs().sum().item()
    weight_changed = bool((emb_delta > 0.0) or (lm_delta > 0.0))

    if args.require_weight_change and not weight_changed:
        raise SystemExit("Embedding/lm_head weights for new tokens did not change after training.")
    if not has_trainable_targets:
        raise SystemExit("adapter_config.modules_to_save is missing embed_tokens/lm_head.")

    payload = {
        "run_dir": str(run_dir),
        "adapter_dir": str(adapter_dir),
        "initialized_model_dir": str(initialized_model_dir),
        "base_tokenizer": str(args.base_tokenizer),
        "extended_tokenizer": str(Path(args.extended_tokenizer).resolve()),
        "modules_to_save": sorted(modules_set),
        "adapter_weight_keys": adapter_keys,
        "num_new_tokens": len(new_token_ids),
        "num_checked_new_token_ids": len(checked_ids),
        "embed_delta_l1_sum": emb_delta,
        "lm_head_delta_l1_sum": lm_delta,
        "weights_changed": weight_changed,
        "pass": True,
    }
    manifest_path = Path(args.manifest_path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] embedding_verify_manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
