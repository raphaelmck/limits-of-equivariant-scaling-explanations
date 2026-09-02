# Figure provenance

Every panel of both figures, with the file it reads and the claim it visualizes (claim labels are
from `CLAIMS.md`). All panels are plotting-only: nothing is recomputed, refit, or re-bootstrapped
in `scripts/figures/`. Display transforms -- log axes, exponentiating the frozen power-law fit,
joining a compute value onto a row -- are transforms of already-validated numbers.

Generate both with `make figures`.

## Figure 1 (`scripts/figures/figure1.py` -> `figures/figure1.{pdf,png}`)

| Panel | Reads | Shows | Claim |
|---|---|---|---|
| A | `analysis_out/claim1/frontier_points.csv`, `force_scaling_recomputed.csv`, `dense_grid_ranking_comparison_recomputed.csv`, `manifests/checkpoints.csv` | Observed frontier checkpoints as scatter, the fitted power law per architecture as a dashed guide, the eight evaluation budgets as markers | K1, K2 |
| B | `analysis_out/claim1/dense_grid_ranking_comparison_recomputed.csv` | KRR learning-curve area per architecture at the eight budgets | K3 |
| C | `analysis_out/claim1/dense_grid_pairwise_gap_recomputed.csv` | Neural gap (top) and KRR gap with 95% intervals (bottom), sharing a compute axis and a zero line | K4 |

Two vertical references appear in all three panels: `C_obs`, the observed crossover compute
(labelled, long dash), and the power-law-implied crossover (unlabelled, fine dotted). The dashed
line in Panel A is a fit, not an observed boundary; the scatter points are the observations.

Two display filters, neither of which changes a plotted value:

- MPNN's highest budget reuses a checkpoint at about half that budget's compute, so MPNN has no
  checkpoint at the budget it nominally names. That marker is dropped rather than drawn at a
  compute the run did not reach.
- Four eSEN widths were checkpointed twice as often as every other width, which reads as a dense
  carpet of dots along the eSEN fit line rather than as any difference in scaling. They are
  thinned by a factor of two in Panel A's scatter so the plotted checkpoint cadence is uniform.

## Figure 2 (`scripts/figures/figure2.py` -> `figures/figure2.{pdf,png}`)

| Panel | Reads | Shows | Claim |
|---|---|---|---|
| A | `analysis_out/claim2/frontier_ell4_degree_balanced_recomputed.json` | Block-9 ell=4 dose response at the four frontier checkpoints, all measured alphas with 95% intervals | I1, I2 |
| B | `analysis_out/claim2/ood_lmax2_vs_lmax4_baseline.csv`, `data/claim2/matched_compute_lmax_owners.csv` | log(L_2 / L_4) at the four matched-compute pairs, on Neutral and support-matched validation, against the ell_max=4 checkpoint's own compute | I6 |

Panel B plots the two populations at a small multiplicative horizontal offset so their intervals
do not overlap; both series sit at the same actual compute. The points are not joined by lines:
they are four separately trained matched pairs, not one trajectory.

No ell_max=2 versus ell_max=4 crossover compute is drawn. No such fit exists in this repository --
only an ell_max=4 eSEN scaling fit does -- and none was constructed for the figure. Whether a
crossover lies beyond the tested range is a question this data cannot answer, so the figure does
not draw a line implying an answer.

The full 102-pair version of Panel B is reported numerically in the paper appendix and produced by
`scripts/claim2/build_lmax24_frontier.py`; the figure shows the four pairs that also have direct
intervention measurements, which is what lets Panels A and B be read together.
