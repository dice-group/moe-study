"""Regenerate every paper table, audit CSV, and figure."""
from __future__ import annotations

import argparse
from pathlib import Path

import common as C
import fig1_rq3_transfer
import fig2_rq4_gain_vs_base
import tabA0_method_coverage
import tabA0b_config_grid
import tab0_study_funnel
import tab1_rq1_design_factors
import tab2_rq2_family_comparison
import tab2_rq2_mixing_effects
import tab3_rq2_followup_ranking
import tab4_rq2_statistical_tests
import tab5_rq3_final

STEPS = [
    ("Method coverage", tabA0_method_coverage.main),
    ("Configuration grid", tabA0b_config_grid.main),
    ("Study funnel", tab0_study_funnel.main),
    ("RQ1 design factors", tab1_rq1_design_factors.main),
    ("RQ2 shared-panel family comparison", tab2_rq2_family_comparison.main),
    ("RQ2 mixing effects", tab2_rq2_mixing_effects.main),
    ("RQ2 follow-up ranking", tab3_rq2_followup_ranking.main),
    ("RQ2 statistical tests", tab4_rq2_statistical_tests.main),
    ("RQ3/RQ4 broader-regime tables", tab5_rq3_final.main),
    ("RQ3 transfer figure", fig1_rq3_transfer.main),
    ("RQ4 gain-vs-base figure", fig2_rq4_gain_vs_base.main),
]


def main(config_path: Path = C.CONFIG_PATH) -> None:
    for name, function in STEPS:
        print(f"\n=== {name} ===")
        function(config_path)
    print("\nAll paper tables, audit CSVs, and figures regenerated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=C.CONFIG_PATH)
    main(parser.parse_args().config)
