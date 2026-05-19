"""Select the next two-step PS experiments after single-step update."""

from __future__ import annotations

from pathlib import Path

from src.ps.eig_design import EigConfig, ROOT, design_next_batch


def main() -> None:
    design_next_batch(
        config=EigConfig(batch_size=18, min_single_step=0),
        mode_filter="two_step",
        next_experiments_path=ROOT / "data" / "ps" / "ps_eig_next_twostep_experiments.csv",
        ranking_path=ROOT / "results" / "ps" / "eig" / "ps_eig_twostep_candidate_ranking.csv",
        summary_path=ROOT / "results" / "ps" / "eig" / "ps_eig_twostep_selection_summary.json",
    )


if __name__ == "__main__":
    main()
