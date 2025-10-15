import os
import gzip
import json
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BASE_TOKENIZER_PATH = "/data/shared_home_cache/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/"
MULTI_TOKENIZER_PATH = "/data/project_data/moe_study/trained_multilingual_tokenizers/256k_vocab/llama-3.2-1B"
LANG_ALLOC_CSV = "/upb/users/j/joeldag/profiles/unix/cs/MERLIN/moe-study/data_prep/metrics/vocab_extension_plan_ranking_based_all_llama_3.2-1b.csv"
DATA_DIR = "/data/project_data/moe_study/fw_samples/preview_subset"
OUTPUT_PATH = "/upb/users/j/joeldag/profiles/unix/cs/MERLIN/moe-study/data_prep/temp/merged_extended_tokenizer"

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

# --- Keep merges as strings (HF expects this format) ---
base_merges_str = [m if isinstance(m, str) else " ".join(m) for m in base_data["model"]["merges"]]
multi_merges_str = [m if isinstance(m, str) else " ".join(m) for m in multi_data["model"]["merges"]]

# --- Compute new merges and update vocab safely ---
base_merges_set = set(base_merges_str)
new_merges_raw = [m for m in multi_merges_str if m not in base_merges_set]
print(f"[DEBUG] raw new merges: {len(new_merges_raw)}")

# all tokens referenced by new merges + extended tokens
merge_part_tokens = set()
merge_result_tokens = set()
for m in new_merges_raw:
    parts = m.split()
    if len(parts) == 2:
        merge_part_tokens.update(parts)
        merge_result_tokens.add("".join(parts))
all_new_tokens = merge_part_tokens.union(merge_result_tokens).union(extended_tokens)

# Add any missing tokens to vocab first
next_id = max(base_vocab_map.values()) + 1
for tok in sorted(all_new_tokens):
    if tok not in base_vocab_map:
        base_vocab_map[tok] = next_id
        next_id += 1

# --- Keep only merges whose BOTH tokens exist in vocab ---
valid_new_merges = []
for m in new_merges_raw:
    parts = m.split()
    if len(parts) == 2 and all(p in base_vocab_map for p in parts):
        valid_new_merges.append(m)

print(f"[DEBUG] valid merges kept: {len(valid_new_merges)} / {len(new_merges_raw)}")
base_merges_str.extend(valid_new_merges)
base_data["model"]["merges"] = base_merges_str
base_data["model"]["vocab"] = base_vocab_map

out_dir = Path(OUTPUT_PATH)
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(base_data, f, ensure_ascii=False, indent=2)

for fname in ["tokenizer_config.json", "special_tokens_map.json"]:
    src = Path(BASE_TOKENIZER_PATH) / fname
    if src.exists():
        (out_dir / fname).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

print(f"Extended BPE tokenizer (vocab + merges) saved to: {out_dir.resolve()}")
print(f"[DEBUG] new merges added: {len(valid_new_merges)}")
print(f"[DEBUG] new tokens added: {len(extended_tokens)}")