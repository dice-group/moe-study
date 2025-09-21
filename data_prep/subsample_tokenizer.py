import os, json, gzip, random
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import load_dataset

random.seed(42)

tsv = "/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/tokenizer_lang_sample_sizes.tsv"
data_dir = "/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/samples"
out_dir = "/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(tsv, sep="\t")
total = len(df)

def process_language(row):
    subset, sample_size, final_count = row["subset"], int(row["sample_size"]), int(row["final_count"])
    lang_dir = os.path.join(data_dir, subset)
    if not os.path.exists(lang_dir):
        return f"skip {subset}, dir not found"

    ds = load_dataset("json", data_files=f"{lang_dir}/*.jsonl.gz", split="train")
    ds = ds.remove_columns([c for c in ["id", "metadata"] if c in ds.column_names])
    texts = list(ds["text"])

    if final_count <= sample_size:
        sampled = random.sample(texts, final_count)
    else:
        multiplier, rem = divmod(final_count, sample_size) # multiply and add remaining random
        sampled = texts * multiplier + random.sample(texts, rem)
        random.shuffle(sampled)

    out_file = os.path.join(out_dir, f"{subset}.jsonl.gz")
    with gzip.open(out_file, "wt", encoding="utf-8") as f:
        for line in sampled:
            f.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")

    return f"{subset}: {len(sampled)} lines"

# simple parallel
with ProcessPoolExecutor(max_workers=16) as ex:
    futures = {ex.submit(process_language, row): i for i, row in enumerate(df.to_dict(orient="records"), 1)}
    for future in as_completed(futures):
        idx = futures[future]
        try:
            msg = future.result()
            print(f"[{idx}/{total}] {msg}")
        except Exception as e:
            print(f"[{idx}/{total}] error: {e}")
