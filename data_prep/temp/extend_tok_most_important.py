import os
import gzip
import json
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BASE_TOKENIZER_PATH = "/scratch/hpc-prf-merlin/shared_cache/huggingface/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/"
MULTI_TOKENIZER_PATH = "/scratch/hpc-prf-merlin/project_data/moe_study/trained_multilingual_tokenizers/256k_vocab/llama-3.2-1B/"
LANG_ALLOC_CSV = "/scratch/hpc-prf-merlin/joel/moe-study/data_prep/metrics/vocab_extension_plan_ranking_based_all_llama_3.2-1b.csv"
DATA_DIR = "/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset/"
OUTPUT_PATH = "/scratch/hpc-prf-merlin/joel/moe-study/data_prep/temp/merged_extended_tokenizer"

base_tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER_PATH, use_fast=True)
base_vocab = set(base_tokenizer.get_vocab().keys())
alloc_df = pd.read_csv(LANG_ALLOC_CSV)

def load_texts(path, max_lines=50):
    texts = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            txt = json.loads(line).get("text")
            if txt:
                texts.append(txt)
    return texts

def process_language(args):
    lang, subset_path, alloc, base_vocab, multi_path = args
    from transformers import AutoTokenizer
    multi_tokenizer = AutoTokenizer.from_pretrained(multi_path, use_fast=True)
    texts = load_texts(subset_path)

    tokenized = [tok for t in texts for tok in multi_tokenizer.tokenize(t)]
    lang_tokens = set(tokenized)
    new_tokens = [t for t in lang_tokens if t not in base_vocab]

    selected = new_tokens[:alloc]
    print(f"[INFO] {lang}: requested={alloc}, found={len(new_tokens)}, added={len(selected)}")
    return lang, selected, alloc, len(selected)


tasks = []
for _, row in alloc_df.iterrows():
    lang = row["language"]
    alloc = int(row["token_alloc"])
    subset_path = os.path.join(DATA_DIR, f"{lang}.jsonl.gz")
    if os.path.exists(subset_path):
        tasks.append((lang, subset_path, alloc, base_vocab, MULTI_TOKENIZER_PATH))

# run per lang in parallel
extended_tokens = set()
lang_stats = []
with ProcessPoolExecutor(max_workers=32) as ex:
    for lang, toks, alloc, added in tqdm(ex.map(process_language, tasks), total=len(tasks)):
        extended_tokens.update(toks)
        lang_stats.append((lang, alloc, added))

print("\n \n Per-language token addition summary:")
for lang, alloc, added in sorted(lang_stats):
    print(f"{lang:<15} | requested: {alloc:<5} | added: {added:<5}")


print(f" \n \n Total new tokens to add: {len(extended_tokens)}")
print("merge vocab and merges")

base_json_path = Path(BASE_TOKENIZER_PATH) / "tokenizer.json"
multi_json_path = Path(MULTI_TOKENIZER_PATH) / "tokenizer.json"
base_data = json.load(open(base_json_path, encoding="utf-8"))
multi_data = json.load(open(multi_json_path, encoding="utf-8"))

base_vocab_map = base_data["model"]["vocab"]
multi_vocab_map = multi_data["model"]["vocab"]

# somehow the tokenizers merges objects have differnt formats, both need to be in HF format
base_merges_str = [m if isinstance(m, str) else " ".join(m) for m in base_data["model"]["merges"]]
multi_merges_str = [m if isinstance(m, str) else " ".join(m) for m in multi_data["model"]["merges"]]

base_set = set(base_merges_str)
new_merges = [m for m in multi_merges_str if m not in base_set]
print(f"[DEBUG] raw new merges: {len(new_merges)}")

parts, results = set(), set()
for m in new_merges:
    p = m.split()
    if len(p) == 2:
        parts.update(p)
        results.add("".join(p))

new_tokens = parts | results | extended_tokens
VOCAB_CAP = 256000
base_size = len(base_vocab_map)
max_new = VOCAB_CAP - base_size
print(f"[INFO] Base: {base_size}, Max new: {max_new}, Found: {len(new_tokens)}")

if len(new_tokens) > max_new:
    print(f"[WARN] {len(new_tokens)} > cap, truncating.")
    prio = sorted(extended_tokens)
    others = sorted(new_tokens - extended_tokens)
    new_tokens = set(prio + others[:max_new - len(prio)])

# extend vocab and keep current ids
vocab = base_vocab_map.copy()
next_id = max(vocab.values()) + 1
for t in sorted(new_tokens):
    vocab.setdefault(t, next_id)
    next_id += 1

# extend merges that are valid
valid_merges = []
for m in new_merges:
    parts = m.split()
    if len(parts) == 2:
        p1, p2 = parts
        res = "".join(parts)
        if p1 in vocab and p2 in vocab and res in vocab:
            valid_merges.append(m)

print(f"[DEBUG] valid merges: {len(valid_merges)}/{len(new_merges)}")

# save
base_data["model"].update({"vocab": vocab, "merges": base_merges_str + valid_merges})
out_dir = Path(OUTPUT_PATH)
out_dir.mkdir(parents=True, exist_ok=True)
(json.dumps(base_data, ensure_ascii=False, indent=2) 
    and (out_dir / "tokenizer.json").write_text(json.dumps(base_data, ensure_ascii=False, indent=2), encoding="utf-8"))

for f in ["tokenizer_config.json", "special_tokens_map.json"]:
    src = Path(BASE_TOKENIZER_PATH) / f
    if src.exists(): (out_dir / f).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

print(f"[INFO] Tokenizer saved to {out_dir.resolve()}")
print(f"[DEBUG] merges added: {len(valid_merges)}, tokens added: {len(new_tokens)}, final vocab: {len(vocab)}")

# Verify
try:
    tok = AutoTokenizer.from_pretrained(out_dir, use_fast=True)
    print(f"Loaded tokenizer ✓, size={len(tok)}")
except Exception as e:
    print(f"ERROR: {e}")
