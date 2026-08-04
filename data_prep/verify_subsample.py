import os, gzip
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

tsv = "/scratch/anonymous-project/project_data/moe_study/fw_samples/tokenizer_lang_sample_sizes.tsv"
prep_dir = "/scratch/anonymous-project/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset"

df = pd.read_csv(tsv, sep="\t")

def verify(row):
    subset, final_count = row["subset"], int(row["final_count"])
    file = os.path.join(prep_dir, f"{subset}.jsonl.gz")
    if not os.path.exists(file):
        return f"{subset}: file missing"

    with gzip.open(file, "rt", encoding="utf-8") as f:
        count = sum(1 for _ in f)

    if count == final_count:
        return f"{subset}: {count} lines (ok)"
    else:
        return f"{subset}: {count} lines (expected {final_count})"
    
with ProcessPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(verify, row): row["subset"] for _, row in df.iterrows()}
    for fut in as_completed(futures):
        print(fut.result())
    
"""
srun --job-name=verify_subsample \
     --output=logs/verify_subsample_%j.out \
     --error=logs/verify_subsample_%j.err \
     --time=00:10:00 \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=8 \
     --mem=16G \
     -p normal \
     python verify_subsample.py
"""