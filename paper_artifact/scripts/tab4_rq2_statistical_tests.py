"""RQ2 omnibus and top-three post-hoc tests."""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import common as C


def _p_cell(p_value: float) -> str:
    text = "$" + C.fmt_p(p_value) + "$" + C.sig_stars(p_value)
    return rf"\textbf{{{text}}}" if p_value < 0.05 else text


def _pair_cell(row: pd.Series) -> str:
    if row["rank_biserial"] > 0:
        direction = r"$\leftarrow$"
    elif row["rank_biserial"] < 0:
        direction = r"$\rightarrow$"
    else:
        direction = r"$=$"
    content = direction + " " + "$" + C.fmt_p(row["p_holm"]) + "$" + C.sig_stars(row["p_holm"])
    return rf"\textbf{{{content}}}" if row["p_holm"] < 0.05 else content


def main(config_path: Path = C.CONFIG_PATH) -> None:
    cfg = C.load_config(config_path)
    data = C.load_primary(C.results_path(cfg, "mixed_followup_xlsx"), cfg)
    benchmarks = cfg["analysis"]["benchmarks"]
    finalists = cfg["rq2"]["finalists"]
    ranking, _ = C.followup_ranking(data, benchmarks)
    if ranking.head(3).index.tolist() != finalists:
        raise ValueError("The configured finalist set is no longer the follow-up top three")
    omnibus, pairs = C.followup_tests(data, benchmarks, finalists)
    expected_omnibus = np.array(
        [4.2497759484e-08, 3.4400658461e-29, 1.2831631670e-14, 4.2497759484e-08]
    )
    if not np.allclose(omnibus["p_holm"], expected_omnibus, rtol=1e-8, atol=1e-35):
        raise ValueError("RQ2 omnibus results no longer match the notebook")
    significant_pairs = pairs.loc[pairs["p_holm"].lt(0.05), ["benchmark", "config_a", "config_b"]]
    expected_pairs = {
        ("FLORES", "mtllora_r8_nup16", "moelora_tr32_e16_k8"),
        ("FLORES", "moelora_tr32_e16_k8", "hydralora_r2"),
    }
    if set(significant_pairs.itertuples(index=False, name=None)) != expected_pairs:
        raise ValueError("RQ2 top-three post-hoc conclusions no longer match the notebook")

    pair_order = list(combinations(finalists, 2))
    pair_headers = ["MTL-LoRA--MOELoRA", "MTL-LoRA--HydraLoRA", "MOELoRA--HydraLoRA"]
    body = []
    for omnibus_row in omnibus.itertuples(index=False):
        row_pairs = pairs[pairs["benchmark"].eq(omnibus_row.benchmark)].set_index(
            ["config_a", "config_b"]
        )
        cells = [
            C.benchmark_label(omnibus_row.benchmark),
            str(omnibus_row.n_units),
            f"{omnibus_row.kendalls_w:.3f}",
            _p_cell(omnibus_row.p_holm),
        ]
        for pair in pair_order:
            cells.append(_pair_cell(row_pairs.loc[pair]))
        body.append(" & ".join(cells) + r" \\")

    table_dir = C.output_dir(cfg, "tables")
    data_dir = C.output_dir(cfg, "data")
    C.write_table(
        table_dir / "tab4_rq2_statistical_tests.tex",
        "lrrrccc",
        # Header built from pair_headers (2026-07-31) rather than hardcoded, so
        # the two cannot drift apart. The old abbreviations shortened MOELoRA to
        # "MoE", which collides with Mixture-of-Experts as the paper uses it.
        [
            r"Benchmark & $n$ & Kendall's $W$ & Omnibus $p_{\mathrm{Holm}}$ & "
            + " & ".join(pair_headers)
            + r" \\"
        ],
        body,
        "Omnibus and pairwise tests for the mixed follow-up. Arrows point toward "
        "the descriptively better configuration, and bold marks "
        r"$p_{\mathrm{Holm}}<.05$.",
        "tab:rq2-tests",
        environment="table*",
        placement="tp",
    )
    C.write_csv(omnibus, data_dir / "tab4_rq2_omnibus.csv")
    C.write_csv(pairs, data_dir / "tab4_rq2_top3_pairwise.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=C.CONFIG_PATH)
    main(parser.parse_args().config)
