import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

TOTAL_NEW_TOKENS = 122_000
W_PPW, W_CPT, W_UNK = 0.5, 0.5, 0.0
GAMMA = 2.0
CSV_PATH = "/upb/users/j/joeldag/profiles/unix/cs/MERLIN/moe-study/data_prep/metrics/tokenizer_metric_llama3.2-1b.csv"

df = pd.read_csv(CSV_PATH)

scaler = MinMaxScaler()
dfn = df.copy()
dfn["ppw_norm"] = scaler.fit_transform(dfn[["pieces_per_word"]])
dfn["cpt_norm"] = 1 - scaler.fit_transform(dfn[["chars_per_token"]])
dfn["unk_norm"] = scaler.fit_transform(dfn[["unknown_rate"]])

dfn["I_linear"] = W_PPW * dfn["ppw_norm"] + W_CPT * dfn["cpt_norm"] + W_UNK * dfn["unk_norm"]
dfn["I_nonlinear"] = np.power(dfn["I_linear"], GAMMA)
dfn["inefficiency_score"] = MinMaxScaler().fit_transform(dfn[["I_nonlinear"]])
dfn = dfn.sort_values("inefficiency_score", ascending=False)

weights = dfn["inefficiency_score"].to_numpy()
total_w = weights.sum()

if total_w == 0:
    dfn["token_alloc"] = 0
else:
    ideal = weights / total_w * TOTAL_NEW_TOKENS
    floor = np.floor(ideal).astype(int)
    remainder = ideal - floor
    shortfall = TOTAL_NEW_TOKENS - floor.sum()
    if shortfall > 0:
        idx = np.argsort(-remainder)[:shortfall]
        floor[idx] += 1
    dfn["token_alloc"] = floor.astype(int)

dfn.to_csv("vocab_extension_plan_ranking_based.csv", index=False)
