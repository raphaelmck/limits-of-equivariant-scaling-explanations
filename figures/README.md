# figures/

Rendered by `make figures` from `code/analysis/figures/`, which read only from `analysis_out/`
after `make analysis` has regenerated and `make validate` has checked it. No scientific quantity
is computed in the plotting scripts; log axes and the exponentiation of the frozen power-law fit
are display transforms of already-validated numbers.

- `figure1.pdf` / `figure1.png` -- compute frontiers and power-law fits, force-NTK KRR
  learning-curve area, and the signed GemNet-OC/eSEN gap for both.
- `figure2.pdf` / `figure2.png` -- the block-9 ell=4 dose response, and log(L_2 / L_4) at the four
  matched-compute pairs on both evaluation populations.

## Panel-by-panel provenance

Every panel, with the file it reads and what it shows. All panels are plotting-only: nothing is
recomputed, refit, or re-bootstrapped in `code/analysis/figures/`. Display transforms -- log axes,
exponentiating the frozen power-law fit, joining a compute value onto a row -- are transforms of
already-validated numbers.

### Figure 1 (`code/analysis/figures/figure1.py` -> `figure1.{pdf,png}`)

| Panel | Reads | Shows |
|---|---|---|
| A | `analysis_out/claim1/frontier_points.csv`, `force_scaling_recomputed.csv`, `dense_grid_ranking_comparison_recomputed.csv`, `data/checkpoints.csv` | Observed frontier checkpoints as scatter, the fitted power law per architecture as a dashed guide, the eight evaluation budgets as markers |
| B | `analysis_out/claim1/dense_grid_ranking_comparison_recomputed.csv` | KRR learning-curve area per architecture at the eight budgets |
| C | `analysis_out/claim1/dense_grid_pairwise_gap_recomputed.csv` | Neural gap (top) and KRR gap with 95% intervals (bottom), sharing a compute axis and a zero line |

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

### Figure 2 (`code/analysis/figures/figure2.py` -> `figure2.{pdf,png}`)

| Panel | Reads | Shows |
|---|---|---|
| A | `analysis_out/claim2/frontier_ell4_degree_balanced_recomputed.json`, `data/claim2/matched_compute_lmax_checkpoints.csv` | Block-9 ell=4 dose response at the four frontier checkpoints, all measured alphas with 95% intervals; legend keyed to each checkpoint's own training compute |
| B | `data/claim2/lmax24_all_frontier_gaps.csv`, `lmax24_frontier_dependence.csv`, `lmax4_causal_vs_comparative.csv` | log(L_2 / L_4) on Neutral validation across the full dense ell_max=4-vs-ell_max=2 frontier (every valid matched pair), with the four Panel A checkpoints highlighted in the same colors as their Panel A curve |

Panel B shows the full dense frontier, not just the four matched-compute pairs with direct
intervention measurements: every `domain=="Neutral_val"`, `valid_match=="True"` row of
`lmax24_all_frontier_gaps.csv` is plotted as a light scatter point, with the four checkpoints
that also appear in Panel A drawn larger and colored to match, so a reader can connect a Panel B
point back to its Panel A curve without a separate legend. One training run is excluded from the
scatter: ell_max=4 width=16, checkpointed every 500 steps from step 500-50000
(`lmax24_frontier_dependence.csv`'s `run_name` column identifies it) -- a single run's own
checkpoint trajectory, not independent matched-compute observations, which would otherwise
dominate the low-compute end of the panel by sheer checkpoint count.

`lmax24_all_frontier_gaps.csv` and `lmax24_frontier_dependence.csv` are frozen inputs, not
`analysis_out/` outputs: they are the matched-pair table itself, the same files
`build_lmax24_frontier.py` reads to produce the appendix's 102-pair summary. The figure reads
them directly rather than through that summary, since it needs the per-pair rows, not the
aggregate counts.

Only the Neutral-validation series is shown; the two-domain (Neutral + support-matched)
comparison at the four canonical pairs is reported numerically in the appendix
(`code/analysis/claim2/build_lmax24_frontier.py`'s summary), not redrawn here.

No ell_max=2 versus ell_max=4 crossover compute is drawn. No such fit exists in this repository --
only an ell_max=4 eSEN scaling fit does -- and none was constructed for the figure. Whether a
crossover lies beyond the tested range is a question this data cannot answer, so the figure does
not draw a line implying an answer.
