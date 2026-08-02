"""Figure 1: how the shortlist reorders between the selection stage and scale.

A slope chart is the right form here because the quantity is an ordinal
position measured at three stages: the reader's question is which lines cross,
not what the values are. Only the two configurations that move furthest carry a
hue; the rest stay in recessive ink so the crossings, not the identities, are
what the eye picks up.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import common as C

ACCENTS = ["#2a78d6", "#eb6834"]  # validated categorical pair, all-pairs light mode
CONTEXT = "#8a8a85"
INK = "#0b0b0b"
MUTED = "#52514e"


def main(config_path: Path = C.CONFIG_PATH) -> None:
    cfg = C.load_config(config_path)
    data_dir = C.output_dir(cfg, "data")
    transfer = pd.read_csv(data_dir / "tab7_rq3_transfer.csv")
    stages = ["followup", *[str(regime) for regime in cfg["rq3"]["regimes"]]]
    stage_labels = ["10 languages\n(selection)", "96 languages", "200 languages"]

    # Emphasize the best and worst configuration of the largest regime. Both were
    # promoted by the same selection stage, so the pair of lines is the finding:
    # several configurations move, but these two swap ends.
    last_regime = stages[-1]
    highlighted = [
        int(transfer[last_regime].idxmin()),
        int(transfer[last_regime].idxmax()),
    ]

    figure, axes = plt.subplots(figsize=(7.0, 2.7))
    x = range(len(stages))
    for index, row in transfer.iterrows():
        values = [row[stage] for stage in stages]
        if index in highlighted:
            color = ACCENTS[highlighted.index(index)]
            width, marker, zorder, alpha = 2.0, 7.0, 3, 1.0
        else:
            color, width, marker, zorder, alpha = CONTEXT, 1.4, 5.5, 2, 0.9
        axes.plot(x, values, color=color, linewidth=width, marker="o", markersize=marker,
                  markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.2,
                  zorder=zorder, alpha=alpha, solid_capstyle="round")
        axes.annotate(
            row["label"],
            xy=(len(stages) - 1, values[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=color if index in highlighted else MUTED,
            fontweight="bold" if index in highlighted else "normal",
        )

    axes.set_yticks(range(1, len(transfer) + 1))
    axes.set_yticklabels([f"{place}" for place in range(1, len(transfer) + 1)], fontsize=9)
    axes.invert_yaxis()  # rank 1 on top, where readers expect the winner
    axes.set_ylabel("Position among the five", fontsize=9, color=MUTED)
    axes.set_xticks(list(x))
    axes.set_xticklabels(stage_labels, fontsize=9)
    axes.set_xlim(-0.25, len(stages) - 1 + 0.05)
    axes.tick_params(axis="both", colors=MUTED, length=0)
    axes.grid(axis="y", color="#e6e6e2", linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color("#d8d8d3")
    for label in axes.get_xticklabels():
        label.set_color(INK)

    figure.subplots_adjust(left=0.11, right=0.72, top=0.95, bottom=0.18)

    figures_dir = C.REPO_ROOT / cfg["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    output = figures_dir / "fig1_rq3_transfer.pdf"
    figure.savefig(output, format="pdf")
    figure.savefig(output.with_suffix(".png"), dpi=200)
    plt.close(figure)
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=C.CONFIG_PATH)
    main(parser.parse_args().config)
