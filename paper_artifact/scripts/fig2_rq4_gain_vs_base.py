"""Figure: adaptation gain against base-model competence, per language.

The claim this figure carries is that adaptation pays off in proportion to how
badly the base model already handles a language. A scatter is the right form
because both quantities are continuous and the question is the sign and strength
of a relationship, not the value of any one language.

Two things about the encoding. Resource level is an *ordinal* variable, so the
tiers take steps of one hue rather than three categorical hues, which keeps the
ordering legible and survives greyscale printing. And the vertical zero line is
the reference the reader actually needs, since a point above it is a language the
adaptation helped.

Scores arrive already oriented so that higher is better (``load_final_panel``
negates lower-is-better metrics), so a positive gain means adaptation helped on
every benchmark including FLORES.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

import common as C

# Ordinal ramp, blue steps 250/450/700 of the validated sequential scale. Tiers
# are ordered, so lightness carries the order and the panel still reads in
# greyscale. Step 700 rather than 600 at the dark end: at this marker size the
# two darker tiers are the confusable pair, and 600 left them the closest.
TIER_COLORS = {"HIGH": "#86b6ef", "MEDIUM": "#2a78d6", "LOW": "#0d366b"}
INK = "#0b0b0b"
MUTED = "#52514e"
TREND = "#eb6834"


def gain_frame(data: pd.DataFrame, baseline: str, benchmark: str) -> pd.DataFrame:
    """Per-language base score and mean gain over the adapted configurations.

    The two FLORES directions are averaged first, matching the aggregation used
    everywhere else in the paper.
    """
    subset = data[data["benchmark"].eq(benchmark)]
    pivot = subset.pivot_table(index="language", columns="run_id", values="score")
    pivot = pivot.dropna()
    tiers = (
        subset.drop_duplicates("language")
        .set_index("language")["resource_availability"]
        .reindex(pivot.index)
    )
    missing = tiers.index[tiers.isna()].tolist()
    if missing:
        raise ValueError(f"No resource tier for plotted languages: {missing}")
    unexpected = sorted(set(tiers) - set(TIER_COLORS))
    if unexpected:
        raise ValueError(f"Unexpected resource tiers in figure: {unexpected}")
    return pd.DataFrame(
        {
            "base": pivot[baseline],
            "gain": pivot.drop(columns=baseline).sub(pivot[baseline], axis=0).mean(axis=1),
            "tier": tiers,
        }
    )


def draw_panel(axes, frame: pd.DataFrame, title: str | None, flip_x: bool) -> tuple[float, float]:
    """One benchmark in one regime.

    ``flip_x`` is for FLORES, whose oriented score is negated bits per byte. We
    plot the raw bits per byte and invert the axis instead, so that a weaker base
    model sits on the left in every panel and the reader compares slopes without
    re-reading each axis.
    """
    rho, p_value = spearmanr(frame["base"], frame["gain"])
    x = -frame["base"] if flip_x else frame["base"]

    axes.axhline(0.0, color="#c3c2b7", linewidth=0.9, zorder=1)
    for tier, color in [("HIGH", TIER_COLORS["HIGH"]),
                        ("MEDIUM", TIER_COLORS["MEDIUM"]), ("LOW", TIER_COLORS["LOW"])]:
        mask = frame["tier"].eq(tier)
        if not mask.any():
            continue
        axes.scatter(x[mask], frame["gain"][mask], s=13, c=color, linewidths=0.4,
                     edgecolors="white", zorder=3)

    # Least-squares line for the eye; the quoted statistic stays the rank
    # correlation, which does not assume the relationship is linear.
    if len(frame) > 2:
        slope, intercept = np.polyfit(x, frame["gain"], 1)
        span = np.linspace(x.min(), x.max(), 2)
        axes.plot(span, slope * span + intercept, color=TREND, linewidth=1.6, zorder=4,
                  solid_capstyle="round")
    if flip_x:
        axes.invert_xaxis()

    # A negative slope empties the top-right corner, so the statistic goes there
    # rather than over the points.
    stars = "*" if p_value < 0.05 else ""
    axes.annotate(rf"$\rho={rho:.2f}$" + stars, xy=(0.96, 0.92), xycoords="axes fraction",
                  fontsize=7.5, color=INK, ha="right", va="top", zorder=6,
                  bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none",
                            alpha=0.85))
    if title:
        axes.set_title(title, fontsize=8.5, color=INK, pad=4)

    axes.tick_params(axis="both", labelsize=6.5, colors=MUTED, length=2)
    axes.grid(color="#eeeeea", linewidth=0.6, zorder=0)
    axes.set_axisbelow(True)
    for side in ["top", "right"]:
        axes.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        axes.spines[side].set_color("#d8d8d3")
    return rho, p_value


def main(config_path: Path = C.CONFIG_PATH) -> None:
    cfg = C.load_config(config_path)
    baseline = cfg["rq3"]["baseline_run_id"]
    regimes = cfg["rq3"]["regimes"]
    benchmarks = cfg["analysis"]["benchmarks"]

    # This figure carries the paper's headline claim, so it is deliberately the
    # one float that is not squeezed for the page limit. Height 3.25 (2026-07-31)
    # sits between the original 3.7 and the 2.85 the limit would have forced.
    figure, grid = plt.subplots(len(regimes), len(benchmarks), figsize=(7.1, 3.25))
    records = []
    for row, regime in enumerate(regimes):
        data = C.load_final_panel(C.results_path(cfg, f"final_{regime}_xlsx"), cfg)
        for column, benchmark in enumerate(benchmarks):
            frame = gain_frame(data, baseline, benchmark)
            rho, p_value = draw_panel(
                grid[row][column],
                frame,
                C.benchmark_label(benchmark) if row == 0 else None,
                flip_x=(benchmark == "FLORES"),
            )
            records.append({
                "regime": regime,
                "benchmark": C.benchmark_label(benchmark),
                "n_languages": len(frame),
                "spearman_rho": rho,
                "p_value": p_value,
            })
        # Row identity goes on the right edge, facet-strip style, so it cannot
        # crowd the shared "mean gain" label on the left. Kept to one line
        # (2026-07-31): wrapped to two, the strip reached back over the axes.
        grid[row][-1].annotate(
            f"{regime} train lang.", xy=(1.06, 0.5), xycoords="axes fraction",
            rotation=-90, va="center", ha="center", fontsize=8, color=INK,
        )

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=5, markerfacecolor=color,
               markeredgecolor="white", markeredgewidth=0.4, label=label)
        for label, color in [("High", TIER_COLORS["HIGH"]), ("Medium", TIER_COLORS["MEDIUM"]),
                             ("Low", TIER_COLORS["LOW"])]
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
                  bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.4,
                  labelcolor=MUTED)
    figure.supxlabel("Base-model performance on that language (better to the right)",
                     fontsize=8, color=MUTED, y=0.075)
    figure.supylabel("Mean gain over the base model", fontsize=8, color=MUTED, x=0.016)
    # Left and right margins widened (2026-07-31) so neither the shared y label
    # nor the row strip sits on top of the panels.
    figure.subplots_adjust(left=0.105, right=0.93, top=0.93, bottom=0.19, wspace=0.36, hspace=0.34)

    figures_dir = C.REPO_ROOT / cfg["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    output = figures_dir / "fig2_rq4_gain_vs_base.pdf"
    figure.savefig(output, format="pdf")
    figure.savefig(output.with_suffix(".png"), dpi=200)
    plt.close(figure)
    print(f"wrote {output}")

    C.write_csv(pd.DataFrame(records), C.output_dir(cfg, "data") / "fig2_rq4_gain_vs_base.csv")
    _cross_benchmark_table(cfg, baseline, regimes, benchmarks)


def _cross_benchmark_table(cfg: dict, baseline: str, regimes: list[int], benchmarks: list[str]) -> None:
    """Independent-noise check for the relationship in the figure.

    The base score appears on both axes of the figure, so measurement noise in it
    could manufacture a negative correlation on its own (regression to the mean).
    Predicting the gain on one benchmark from the base score on a *different*
    benchmark removes that path, because the two noise terms are independent.
    """
    rows: list[str] = []
    records: list[dict] = []
    for index, regime in enumerate(regimes):
        data = C.load_final_panel(C.results_path(cfg, f"final_{regime}_xlsx"), cfg)
        frames = {b: gain_frame(data, baseline, b) for b in benchmarks}
        if index:
            rows.append(r"\addlinespace")
        # The regime is a block heading rather than a repeated column, which
        # narrows the table enough to sit in a single text column.
        rows.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{regime} training languages}}}} \\")
        for target in benchmarks:
            cells = []
            for source in benchmarks:
                if source == target:
                    cells.append("--")
                    continue
                joined = pd.concat(
                    [frames[target]["gain"], frames[source]["base"].rename("x")], axis=1
                ).dropna()
                rho, p_value = spearmanr(joined["x"], joined["gain"])
                cells.append(f"{rho:+.2f}{C.sig_stars(p_value)}")
                records.append({
                    "regime": regime, "gain_on": target, "base_from": source,
                    "n_languages": len(joined), "spearman_rho": rho, "p_value": p_value,
                })
            rows.append(" & ".join([C.benchmark_label(target), *cells]) + r" \\")
    C.write_table(
        C.output_dir(cfg, "tables") / "tabA4_rq4_cross_benchmark.tex",
        "l" + "r" * len(benchmarks),
        [r"Gain on & \multicolumn{4}{c}{Base-model score taken from} \\",
         r"\cmidrule(lr){2-5}",
         " & " + " & ".join(C.benchmark_label(b) for b in benchmarks) + r" \\"],
        rows,
        "Spearman correlation between the gain on one benchmark and the "
        "base-model score on another. The diagonal is empty because it would "
        "take both values from the same benchmark.",
        "tab:rq4-cross-benchmark",
        # Placed with [H] at the end of the appendix text rather than floated.
        # A float deferred out of the first column is only reconsidered at the
        # start of the next page, never in the second column of the same one, so
        # the gap at the foot of that column is unreachable by any float
        # parameter. This is the table its closing paragraph ends on.
        placement="H",
        col_sep="3pt",
    )
    C.write_csv(pd.DataFrame(records), C.output_dir(cfg, "data") / "tabA4_rq4_cross_benchmark.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=C.CONFIG_PATH)
    main(parser.parse_args().config)
