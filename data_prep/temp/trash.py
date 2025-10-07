import pandas as pd

before = pd.read_csv("/scratch/hpc-prf-merlin/moe-study/data_prep/metrics/tokenizer_metrics_llama3.2-1b_all_.csv")
after  = pd.read_csv("/scratch/hpc-prf-merlin/moe-study/data_prep/metrics/test_after_tokenizer_metrics_llama3.2-1b_all_.csv")

merged = after.merge(before, on="language", suffixes=("_after", "_before"))
for col in ["chars_per_token", "pieces_per_word"]:
    merged[f"{col}_diff"] = merged[f"{col}_before"] - merged[f"{col}_after"]

summary = merged[["language", "chars_per_token_diff", "pieces_per_word_diff"]]
summary.to_csv("tokenizer_improvement_summary.csv", index=False)
print(summary.sort_values("pieces_per_word_diff", ascending=False).head(20))
