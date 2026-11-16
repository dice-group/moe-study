"""Resource accounting and language-sampling tables (added for the rebuttal round).

Reviewers asked for the quantities that a fixed wall-clock budget does not by
itself pin down: how many tokens each run actually processed, how many optimizer
updates it took, how fast it ran, how many parameters it trains, and how the
training text was split across languages. All of that is recorded per run in
``paper_resources/metadata/runs.csv`` and per language in
``paper_resources/metadata/languages.csv``, which this script turns into two
appendix tables.

Nothing here re-derives a result. The two tables describe the training budget,
so they are built from the run and language records rather than from the
benchmark workbooks the rest of the pipeline reads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import common as C

METADATA_DIR = Path(__file__).resolve().parents[1] / "metadata"

# Training-set directory fragments, in reporting order. The dataset column holds
# a full pipeline path, and these are the parts that identify the stage.
LANGUAGE_SETS = [
    ("10 (search)", "moe_study_profile_otus_cluster_10_langs"),
    ("10 (follow-up)", "mixed_followup_expanded_data_rerun_10_langs"),
    ("96", "96_langs_34p5h"),
    ("200", "200_langs_72h"),
]


def _routing_summary(approach: str, settings: str) -> str:
    """Expert count and top-$k$ as the paper names them, from the run record.

    Each method spells its routing options differently, so the mapping is
    explicit rather than guessed from key names. Methods that combine every
    expert densely report a dash for top-$k$, because there is no selection.
    """
    try:
        options = json.loads(settings) if settings and settings != "MISSING" else {}
    except json.JSONDecodeError:
        options = {}
    if approach == "lora":
        return "1", "--"
    if approach == "moelora":
        return str(options.get("expert_num", "--")), str(options.get("gate_top_k", "--"))
    if approach == "mola":
        return str(options.get("mola_num_experts", "--")), str(options.get("mola_top_k", "--"))
    if approach == "hydralora":
        # One shared down-projection with several up-projection heads, combined
        # by a dense gate, so every head contributes to every token.
        return str(options.get("lora_num", "--")), "dense"
    if approach == "mtllora":
        return str(options.get("num_up_projections", "--")), "dense"
    return "--", "--"


def _active_cell(run: pd.Series) -> str:
    """Active adapter parameters in millions, as a point value or as bounds.

    Most methods route a fixed number of experts, so the count is determined by
    the configuration. AdaMoE selects a variable number per token through its
    null experts, so the metadata gives a range instead and this returns both
    ends of it.
    """
    if run["active_parameters_basis"] == "derived_bounds":
        low = run["active_adapter_parameters_min"] / 1e6
        high = run["active_adapter_parameters_max"] / 1e6
        return rf"{low:.0f}--{high:.0f}"
    return f"{run['active_adapter_parameters'] / 1e6:.1f}"


def _load_runs() -> pd.DataFrame:
    runs = pd.read_csv(METADATA_DIR / "runs.csv")
    numeric = [
        "trainer_runtime_hours",
        "world_size",
        "optimizer_steps",
        "logged_epoch",
        "input_token_positions_seen",
        "input_token_positions_per_second",
        "trainable_parameters",
        "total_model_parameters",
        "active_adapter_parameters",
        "active_adapter_parameters_min",
        "active_adapter_parameters_max",
        "frozen_backbone_parameters",
        "implementation_rank",
        "alpha",
        "scaling_factor",
    ]
    for column in numeric:
        runs[column] = pd.to_numeric(runs[column], errors="coerce")
    # The metadata dropped its precomputed GPU-hour column. Every run holds one
    # process per GPU on a single node, so the product is the same quantity.
    runs["gpu_hours"] = runs["trainer_runtime_hours"] * runs["world_size"]
    return runs


def _resource_table(cfg: dict, runs: pd.DataFrame, table_dir: Path, data_dir: Path) -> None:
    labels = {cfg["rq3"]["baseline_run_id"]: cfg["rq3"]["baseline_label"]}
    order = []
    for entry in cfg["rq3"]["finalists"]:
        labels[entry["run_id"]] = entry["label"]
        order.append(entry["run_id"])

    rows: list[str] = []
    records: list[dict] = []
    for index, regime in enumerate(cfg["rq3"]["regimes"]):
        if index:
            rows.append(r"\addlinespace")
        rows.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{regime} training languages}}}} \\")
        stage = runs[runs["stage"].eq(f"final_{regime}")].set_index("run_id")
        for run_id in order:
            run = stage.loc[run_id]
            experts, top_k = _routing_summary(run["approach"], run["routing_settings"])
            cells = [
                C.latex_escape(labels[run_id]),
                experts,
                top_k,
                f"{run['trainable_parameters'] / 1e6:.1f}",
                _active_cell(run),
                f"{run['optimizer_steps'] / 1e3:.1f}",
                f"{run['input_token_positions_seen'] / 1e9:.2f}",
                f"{run['input_token_positions_per_second'] / 1e3:.1f}",
                f"{run['logged_epoch'] * 100:.0f}",
                f"{run['gpu_hours']:.0f}",
            ]
            rows.append(" & ".join(cells) + r" \\")
            record = {
                "regime": regime,
                "run_id": run_id,
                "label": labels[run_id],
                "experts": experts,
                "top_k": top_k,
                "per_expert_rank": run["implementation_rank"],
                "alpha": run["alpha"],
                "scaling_factor": run["scaling_factor"],
                "learning_rate": run["learning_rate"],
                "trainable_parameters": run["trainable_parameters"],
                "active_adapter_parameters": run["active_adapter_parameters"],
                "active_adapter_parameters_min": run["active_adapter_parameters_min"],
                "active_adapter_parameters_max": run["active_adapter_parameters_max"],
                "active_parameters_basis": run["active_parameters_basis"],
                "frozen_backbone_parameters": run["frozen_backbone_parameters"],
                "total_model_parameters": run["total_model_parameters"],
                "optimizer_steps": run["optimizer_steps"],
                "input_token_positions_seen": run["input_token_positions_seen"],
                "input_token_positions_per_second": run["input_token_positions_per_second"],
                "corpus_fraction_seen": run["logged_epoch"],
                "trainer_runtime_hours": run["trainer_runtime_hours"],
                "gpu_hours": run["gpu_hours"],
            }
            records.append(record)

    C.write_table(
        table_dir / "tabA5_resource_accounting.tex",
        "lccrrrrrrr",
        [
            r"Configuration & $E$ & $k$ & Train.\ & Active & Steps & Tokens & Tok./s & Corpus & GPU \\",
            r" & & & (M) & (M) & (k) & (B) & (k) & (\%) & (h) \\",
        ],
        rows,
        "Training budget actually consumed in each broader regime. $E$ is the "
        "expert count and $k$ the number of experts routed per token, \\emph{dense} "
        "where a method combines all of them. Train.\\ is trainable adapter "
        "parameters and Active those reached on a forward pass, given as a range "
        "for \\baseline{AdaMoE}, whose null experts make the count input-dependent. "
        "Tokens counts input token positions processed, Corpus the share of the "
        "training set seen once, and GPU the hours over the four A100 40GB GPUs of "
        "the run. Every run in a regime had the same wall clock, so the spread in "
        "steps and tokens is the throughput difference between methods. The frozen "
        "backbone adds 8.03B parameters to every row and is excluded throughout.",
        "tab:resource-accounting",
        environment="table*",
        placement="tp",
        col_sep="5pt",
    )
    C.write_csv(pd.DataFrame(records), data_dir / "tabA5_resource_accounting.csv")


def _language_table(table_dir: Path, data_dir: Path) -> None:
    languages = pd.read_csv(METADATA_DIR / "languages.csv")
    languages["share"] = pd.to_numeric(
        languages["dataset_train_example_share"], errors="coerce"
    )
    languages["examples"] = pd.to_numeric(languages["train_examples"], errors="coerce")

    rows: list[str] = []
    records: list[dict] = []
    for label, fragment in LANGUAGE_SETS:
        subset = languages[languages["dataset"].str.contains(fragment, regex=False)]
        subset = subset.dropna(subset=["share"]).sort_values("share", ascending=False)
        shares = subset["share"].to_numpy()
        cumulative = shares.cumsum()
        # How many language--script labels it takes to account for half and for
        # nine tenths of the training examples: the compact way to state how
        # uneven proportional sampling leaves the mixture.
        half = int((cumulative < 0.5).sum() + 1)
        ninety = int((cumulative < 0.9).sum() + 1)
        cells = [
            label,
            str(len(subset)),
            f"{subset['examples'].sum() / 1e6:.2f}",
            f"{shares[0] * 100:.1f}",
            f"{shares[:5].sum() * 100:.1f}",
            str(half),
            str(ninety),
            f"{shares.min() * 100:.4f}",
        ]
        rows.append(" & ".join(cells) + r" \\")
        records.append(
            {
                "training_set": label,
                "language_script_labels": len(subset),
                "train_examples": int(subset["examples"].sum()),
                "largest_label": subset.iloc[0]["language"],
                "largest_share": shares[0],
                "top5_share": shares[:5].sum(),
                "labels_for_50pct": half,
                "labels_for_90pct": ninety,
                "smallest_share": shares.min(),
                "smallest_label": subset.iloc[-1]["language"],
            }
        )

    C.write_table(
        table_dir / "tabA6_language_mixture.tex",
        "lrrrrrrr",
        [
            r"Training set & Labels & Ex.\ & Top-1 & Top-5 & 50\% & 90\% & Min \\",
            r" & & (M) & (\%) & (\%) & & & (\%) \\",
        ],
        rows,
        "Composition of each prepared training set. Labels counts language--script "
        "labels, which exceeds the nominal language count because a language can "
        "appear in more than one script. Ex.\\ is packed training sequences of "
        "1{,}024 tokens. Top-1 and Top-5 give the share of training sequences held "
        "by the largest labels, and the 50\\% and 90\\% columns the number of labels "
        "needed to reach that share. Construction draws the same fraction of every "
        "language's available documents, 0.0025 at 96 languages and 0.005 at 200, "
        "so the mixture follows FineWeb2's own distribution rather than being "
        "temperature-rebalanced. These are shares of the prepared split and not "
        "measured per-run consumption. Per-language rows are released with the "
        "analysis.",
        "tab:language-mixture",
        placement="tbp",
        col_sep="4pt",
    )
    C.write_csv(pd.DataFrame(records), data_dir / "tabA6_language_mixture.csv")


def main(config_path: Path = C.CONFIG_PATH) -> None:
    cfg = C.load_config(config_path)
    table_dir = C.output_dir(cfg, "tables")
    data_dir = C.output_dir(cfg, "data")
    _resource_table(cfg, _load_runs(), table_dir, data_dir)
    _language_table(table_dir, data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=C.CONFIG_PATH)
    main(parser.parse_args().config)
