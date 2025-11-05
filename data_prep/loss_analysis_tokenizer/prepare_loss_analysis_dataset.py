"""
Quick helper to gather fixed-size language samples for tokenizer loss study.
Takes the first N docs per language (TODO: randomize sampling).
"""

import argparse, gzip, json, random
from pathlib import Path
from typing import Dict, Iterable, List

DEFAULT_WORST = ["dzo_Tibt", "ksw_Mymr", "lao_Laoo", "wuu_Hani", "ory_Orya"]
DEFAULT_BEST = ["rus_Cyrl", "zsm_Latn", "ile_Latn", "spa_Latn", "english"]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build JSONL splits.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--docs-per-lang", type=int, default=10000)
    p.add_argument("--text-key", default="text")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--allow-repeat", action="store_true")
    p.add_argument("--worst-langs", nargs="*", default=DEFAULT_WORST)
    p.add_argument("--best-langs", nargs="*", default=DEFAULT_BEST)
    return p.parse_args()

def sample_texts(lang_dir: Path, text_key: str, n: int, rng: random.Random, allow_repeat: bool) -> List[str]:
    """Take first n texts from .jsonl.gz"""
    path = next(lang_dir.glob("*.jsonl.gz"), None)
    if not path:
        raise FileNotFoundError(f"No .jsonl.gz in {lang_dir}")
    texts = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if len(texts) >= n: break
            try:
                rec = json.loads(line)
                t = rec.get(text_key)
                if isinstance(t, str) and t: texts.append(t)
            except json.JSONDecodeError:
                continue
    if len(texts) < n:
        if allow_repeat and texts:
            r, rem = divmod(n, len(texts))
            texts = texts * r + rng.sample(texts, rem)
        else:
            raise ValueError(f"Only {len(texts)} docs in {lang_dir}")
    # TODO: add random sampling instead of first-n
    return texts

def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    a = parse_args(); rng = random.Random(a.seed)
    langs = a.worst_langs + a.best_langs
    if len(set(langs)) != len(langs): raise SystemExit("Duplicate languages.")
    combined, stats = [], {"docs_per_lang": a.docs_per_lang, "seed": a.seed, "languages": {}}
    for lang in langs:
        print(f"Processing {lang}")
        d = a.root / lang
        if not d.is_dir(): raise FileNotFoundError(f"Missing {lang}")
        texts = sample_texts(d, a.text_key, a.docs_per_lang, rng, a.allow_repeat)
        rows = [{"text": t, "lang": lang, "group": "worst" if lang in a.worst_langs else "best"} for t in texts]
        combined.extend(rows); write_jsonl(a.out_dir / f"{lang}.jsonl", rows)
        stats["languages"][lang] = {"used": len(texts), "group": rows[0]["group"]}
    random.shuffle(combined); write_jsonl(a.out_dir / "combined.jsonl", combined)
    stats["combined_size"] = len(combined)
    (a.out_dir / "metadata.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\n✅Wrote {len(combined)} samples to {a.out_dir}")

if __name__ == "__main__":
    main()
