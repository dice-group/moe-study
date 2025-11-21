import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from datasets import Dataset
from transformers import AutoTokenizer

from language_subsets import LANGUAGE_SUBSET_MAP


def _resolve_rank_world(rank: Optional[int], world_size: Optional[int]) -> tuple[int, int]:
    if rank is not None:
        resolved_rank = rank
    else:
        env_rank = (
            os.getenv("SLURM_ARRAY_TASK_ID")
            or os.getenv("SLURM_PROCID")
            or os.getenv("RANK")
            or os.getenv("LOCAL_RANK")
            or 0
        )
        resolved_rank = int(env_rank)

    if world_size is not None:
        resolved_world = world_size
    else:
        env_world = (
            os.getenv("SLURM_ARRAY_TASK_COUNT")
            or os.getenv("SLURM_NTASKS")
            or os.getenv("WORLD_SIZE")
            or 1
        )
        resolved_world = int(env_world)

    return resolved_rank, max(1, resolved_world)


def _list_shards(
    shard_dir: Path,
    languages: Optional[List[str]] = None,
    language_rates: Optional[Dict[str, float]] = None,
    default_rate: float = 1.0,
) -> List[Tuple[str, Path, float]]:
    valid_suffixes = (".jsonl", ".jsonl.gz")
    search_roots = [shard_dir] if not languages else [shard_dir / lang for lang in languages]
    shards: List[Tuple[str, Path, float]] = []
    for root in search_roots:
        if not root.exists():
            raise RuntimeError(f"Requested language directory is missing: {root}")
        for path in root.rglob("*"):
            if not path.is_file() or not any(path.name.endswith(sfx) for sfx in valid_suffixes):
                continue
            rel = path.relative_to(shard_dir)
            lang = rel.parts[0] if rel.parts else path.parent.name
            rate = language_rates.get(lang, default_rate) if language_rates else default_rate
            rate = max(0.0, min(1.0, rate))
            shards.append((lang, path, rate))
    shards.sort(key=lambda item: item[1])
    if not shards:
        raise RuntimeError(f"No .jsonl or .jsonl.gz files found under {shard_dir}")
    return shards


def _assign_shards(
    shards: List[Tuple[str, Path, float]],
    rank: int,
    world_size: int,
) -> List[Tuple[str, Path, float]]:
    if world_size <= 1:
        return shards
    assigned = [item for idx, item in enumerate(shards) if idx % world_size == rank]
    print(f"Rank {rank}: assigned {len(assigned)} part files out of {len(shards)}")
    return assigned


def _iter_shard_lines(
    shards: List[Tuple[str, Path, float]],
    eval_fraction: float,
    eval_seed: int,
    sample_seed: int,
    max_samples: Optional[int],
) -> Iterator[dict]:
    emitted = 0
    for language, shard_path, keep_rate in shards:
        opener = gzip.open if shard_path.suffix == ".gz" else open
        with opener(shard_path, "rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if max_samples is not None and emitted >= max_samples:
                    return
                text = _extract_text(line)
                if text is not None:
                    if keep_rate < 1.0 and not _should_keep_sample(language, text, keep_rate, sample_seed):
                        continue
                    record = {"text": text, "language": language}
                    if eval_fraction > 0:
                        record["split"] = (
                            "validation"
                            if _assign_to_eval(language, text, eval_fraction, eval_seed)
                            else "train"
                        )
                    yield record
                    emitted += 1


def _extract_text(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            text = obj.get("text")
            if isinstance(text, str) and text.strip():
                return text
    except json.JSONDecodeError:
        pass
    return stripped


def tokenize_fn(batch, tokenizer):
    tokenized = tokenizer(batch["text"], truncation=True, padding=True, max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def _resolve_languages(explicit: Optional[List[str]], subset_name: Optional[str]) -> Optional[List[str]]:
    if explicit and subset_name:
        raise RuntimeError("Use either --languages or --language_subset, not both.")
    if subset_name:
        if subset_name not in LANGUAGE_SUBSET_MAP:
            available = ", ".join(sorted(LANGUAGE_SUBSET_MAP))
            raise RuntimeError(f"{subset_name} not found. Available subsets: {available}")
        return LANGUAGE_SUBSET_MAP[subset_name]
    return explicit


def _assign_to_eval(language: str, text: str, fraction: float, seed: int) -> bool:
    key = f"{language}\u0000{seed}\u0000{text}".encode("utf-8", errors="ignore")
    digest = hashlib.sha256(key).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < fraction


def _should_keep_sample(language: str, text: str, keep_rate: float, seed: int) -> bool:
    if keep_rate >= 1.0:
        return True
    if keep_rate <= 0.0:
        return False
    key = f"{language}\u0000{seed}\u0000{text}".encode("utf-8", errors="ignore")
    digest = hashlib.sha256(key).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < keep_rate


def _load_language_rates(path: Optional[str]) -> Dict[str, float]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError("language_rates_json must contain a JSON object mapping language -> rate.")
    rates: Dict[str, float] = {}
    for lang, value in data.items():
        try:
            rate = float(value)
        except (TypeError, ValueError):
            raise RuntimeError(f"Invalid keep rate for language '{lang}': {value}")
        rates[lang] = max(0.0, min(1.0, rate))
    return rates


def main(args):
    shard_dir = Path(args.shard_dir)
    print(f"Tokenizer used for tokenization: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    rank, world_size = _resolve_rank_world(args.rank, args.world_size)
    languages = _resolve_languages(args.languages, args.language_subset)
    language_rates = _load_language_rates(args.language_rates_json)
    default_rate = max(0.0, min(1.0, args.sample_fraction))
    shards = _list_shards(shard_dir, languages, language_rates, default_rate)
    assigned = _assign_shards(shards, rank, world_size)
    if not assigned:
        raise RuntimeError(f"Rank {rank} received zero shard files. Check shard count vs. world size.")

    with tempfile.TemporaryDirectory(prefix="tok_simple_") as cache_dir:
        raw_dataset = Dataset.from_generator(
            lambda: _iter_shard_lines(
                assigned,
                args.eval_fraction,
                args.eval_seed,
                args.sample_seed,
                args.max_samples,
            ),
            cache_dir=cache_dir,
        )

        print(f"Rank {rank}: start tokenizing {len(assigned)} part(s)")
        tokenized_dataset = raw_dataset.map(
            lambda batch: tokenize_fn(batch, tokenizer),
            batched=True,
            remove_columns=["text"],
            num_proc=args.num_proc,
        )

        output_dir = Path(args.save_tokenized_data_dir)
        if world_size > 1:
            output_dir = output_dir / f"rank_{rank:05d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenized_dataset.save_to_disk(str(output_dir))
        print(f"Rank {rank}: tokenized data saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize pre-split corpus parts.")
    parser.add_argument("--shard_dir", type=str, required=True, help="Directory containing <lang>_part_*.jsonl(.gz) files.")
    parser.add_argument("--save_tokenized_data_dir", type=str, required=True, help="Directory to save tokenized shards.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Tokenizer model name or path.")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of workers for tokenizer.map.")
    parser.add_argument("--rank", type=int, default=None, help="Rank override (defaults to SLURM env vars).")
    parser.add_argument("--languages", nargs="+", default=None, help="Optional list of language directory names (e.g., eng_Latn deu_Latn).")
    parser.add_argument("--language_subset", type=str, default=None, help="Name of a language list defined in language_subsets.py (e.g., twenty_two_representatives_mediods).")
    parser.add_argument("--world_size", type=int, default=None, help="World size override (defaults to SLURM env vars).")
    parser.add_argument("--eval_fraction", type=float, default=0.05, help="Fraction per language to tag as validation data (set 0 to disable).")
    parser.add_argument("--eval_seed", type=int, default=42, help="Seed for the eval split hash.")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Default probability of keeping an example.")
    parser.add_argument("--sample_seed", type=int, default=0, help="Seed controlling deterministic sampling.")
    parser.add_argument("--language_rates_json", type=str, default=None, help="Optional JSON file mapping language -> keep_rate override.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional max number of records to emit per rank.")
    args = parser.parse_args()
    main(args)
