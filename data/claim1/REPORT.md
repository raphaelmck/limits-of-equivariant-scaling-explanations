<title>Matched-Compute m=1024 Force-Tangent-Kernel Report</title>

# Claim 1 — matched-compute m=1024 force-tangent-kernel grid (2026-08-22)

## Purpose and scope

Fills the gap identified in `MISSING_CELLS.md`: none of the 12 LOW/MID/HIGH matched-compute
frontier owners (4 architectures x 3 budget tiers) previously had a force-tangent kernel built
beyond the frozen m=256 probe pool. The only prior m=1024 check
(`large_m_convergence_2026_08_19`) covered two UNMATCHED checkpoints (GemNet-OC width=256,
eSEN sphere=64, both step=500000) that are not any of the 12 owners (`MISSING_CELLS.md`). This
run builds real m=1024 kernels and KRR curves at the actual paper-relevant matched-compute
owners for the first time.

**Result: 12/12 owners succeeded end-to-end** (kernel construction, m=256 validation gate, KRR).
No owner was excluded.

## What was built (reused code paths, no new NTK/KRR implementation)

- `analysis_scripts/run_matched_compute_large_m_kernels.py` — composes, unchanged: owner
  loading / checkpoint provenance / strict state-dict load from
  `four_arch_matched_compute_frontier_kernel.py` (`OwnerSpec`, `OWNERS`,
  `instantiate_model_generic`, `load_checkpoint_strict_generic`), and Jacobian
  collection / chunked Gram assembly / numerics from `four_arch_initial_force_kernel_pilot.py`,
  `four_arch_trained_force_kernel_pilot.py`, `oak_schedulefree_kernel_falsification.py`. Adds
  only: generic 12-owner resolution, a SHA256+global_step provenance check against
  `owner_manifest.csv`, and the mandatory m=256 reproduction gate.
- `analysis_scripts/run_matched_compute_large_m_krr.py` — loops the exact `solve_and_score`
  block-KRR routine (atom-major rows, `RandomState(seed).permutation`, `split_seeds=range(20)`,
  ridge = rho*mean(diag(K_train)), rcond=1e-10 ridgeless, inner 75/25 ridge selection) imported
  unchanged from `run_fvs_controlD_large_m_krr.py` (the Control-D script), over all 12 new
  kernels.
- `analysis_scripts/build_matched_compute_m1024_comparison.py` — builds the four CSVs below.
- SLURM: `scripts/run_matched_compute_large_m_kernels.sbatch`, general `gpu:1` pool (not scarce
  `gpu:a100`), one job per owner, run concurrently.

## Compute accounting

- **Owners succeeded: 12/12.** No owner excluded from the final comparison.
- **Kernel-construction GPU time (12 successful jobs): 10,590s = 2.94 GPU-hours** (sacct
  elapsed sum). Per-owner: MPNN LOW/MID/HIGH 461s/1192s/5787s (mpnn_HIGH, 79M params, by far the
  most expensive single job at 1h37m); MC-EGNN 172s/173s/688s; GemNet-OC 140s/189s/332s; eSEN
  298s/360s/510s. Full per-owner breakdown, numerics, and timing in
  `matched_compute_m1024_kernel_inventory.csv`.
- **Debugging overhead (transparently reported, not hidden): 1.07 GPU-hours** were spent on 14
  job submissions that failed or were pre-emptively cancelled during pipeline debugging (see
  "Debugging narrative" below) — none of these numbers are used anywhere in the final results.
- **Total GPU time actually spent this run: 4.01 GPU-hours** (2.94 successful + 1.07 debugging).
- **KRR (CPU-only, all 12 owners): ~4 minutes wall-clock**, negligible next to kernel
  construction.
- **Wall-clock for the full pipeline** (first job submitted to last job COMPLETED): ~2h15m,
  dominated by queueing/priority delay for the first batch and by mpnn_HIGH's 1h37m single-job
  runtime for the last owner.

### Debugging narrative (for full transparency, not swept under the rug)

The first submission of all 12 owners used an incorrect m=256 validation-gate tolerance: an
absolute tolerance (1e-8) was applied both to the O(1)-scale alignment statistic AND to raw
kernel entries `K`, which have architecture-dependent scale (e.g. MC-EGNN's raw `K` has
max|K|~3e3). This produced false gate failures (actual relative error on `K` was ~7.6e-7,
consistent with this repo's documented fp32-Jacobian precision, well inside a correct relative
tolerance). A second, related bug: the alignment-statistic tolerance was mis-cited as 1e-8; the
actual reused code (`oak_large_m_convergence.py`) gates at `< 1e-6` — the "~1e-8" language
elsewhere in this repo's docs describes empirically-achieved precision, not the gate threshold.
Both were fixed (K held to 1e-4 relative tolerance per this repo's own fp32-precision precedent;
alignment held to the correct 1e-6 absolute tolerance). Several already-running jobs had already
imported the pre-fix module into memory and were cancelled/resubmitted rather than left to fail
on stale code. All 12 owners were then cleanly resubmitted against the corrected, verified
script and all 12 succeeded on the first attempt of that clean batch. This debugging overhead is
included in the 4.01 GPU-hour total above but excluded from the kernel/KRR numbers used in every
result below, which come only from the 12 clean, gated-and-passed runs.

## Mandatory m=256 validation gate: 12/12 PASS

Every owner's m=1024 kernel reproduces the existing frozen m=256 matched-compute kernel
(`four_arch_matched_compute_frontier_kernel_2026_08_15/kernels/{tag}_m256_kernel.npz`) on its
leading 768x768 (m=256) submatrix:

- target vector `y`: bit-identical (max abs diff = 0.0) for all 12 owners.
- raw kernel `K`: relative max abs diff between 3.3e-7 and 1.1e-5 across owners — consistent with
  this repo's fp32-Jacobian-precision precedent (~2e-7 relative), well inside the 1e-4 relative
  tolerance gate.
- alignment statistic `A(K_x,y)`: absolute diff between 2.3e-10 and 7.2e-8 across owners, well
  inside the 1e-6 absolute tolerance gate (matching the actual code precedent in
  `oak_large_m_convergence.py`).

Full per-owner numbers in `matched_compute_m1024_kernel_inventory.csv`
(`m256_gate_*` columns). No owner needed exclusion.

## Headline result: does m=1024 KRR change the standing kernel-vs-neural-hierarchy verdict?

**No — it confirms the standing verdict, with one caveat at HIGH.**

`ranking_comparison.csv` gives, per (architecture, budget), the NN rank (ascending
`force_mse_norm`, lower=better) and the KRR rank (ascending
`auc_log_nmse` = trapezoid of `log(ridgeless test_nmse_median)` over `log(n)` for
n in {32,64,128,256,512}, lower=better — the same AUC statistic definition used by
`analyze_four_arch_matched_compute_frontier_kernel.py::auc_summary`, evaluated on this run's
m=1024 n-grid instead of that report's m=256 n-grid {8,16,32,64}).

- **LOW and MID: perfect rank agreement (4/4 architectures, both tiers).** KRR rank order
  matches NN rank order exactly: GemNet-OC < eSEN < MC-EGNN < MPNN (best to worst) at both
  budgets.
- **HIGH: 3/4 agree, GemNet-OC/eSEN swap.** NN order at HIGH is eSEN < GemNet-OC < MC-EGNN <
  MPNN (eSEN has overtaken GemNet-OC neurally, per the standing `force_scaling.csv` crossover:
  eSEN gamma=0.663 > GemNet-OC gamma=0.331 beyond ~1.64e17 FLOPs). KRR order at HIGH remains
  GemNet-OC < eSEN < MC-EGNN < MPNN — KRR has NOT caught up to the neural crossover even at
  m=1024 and the actual HIGH-budget matched-compute owners.

`pairwise_concordance.csv` makes this precise across all 6 pairwise architecture comparisons x 3
budgets = 18 rows, using paired bootstrap CIs (n_boot=2000, resampling the 20 split-level paired
log-ratios at n=512 — the largest common train size — per the project's established paired
per-split-seed convention, e.g. `run_fvs_controlD_large_m_krr.py`'s paired comparison and the
n_boot=2000 config-level paired-resampling precedent in
`final_validation_sprint_2026_08_20/INVENTORY.md` sec. 3):

- **17/18 (94.4%) concordant** — NN and KRR agree on which architecture is more learnable, CI
  excludes zero in every one of these 17.
- **1/18 (5.6%) discordant** — GemNet-OC vs eSEN at HIGH: NN favors eSEN (post-crossover), KRR
  favors GemNet-OC (log-ratio mean = -0.387, 95% CI [-0.652, -0.123], excludes zero — the KRR
  learner robustly and significantly still favors GemNet-OC here).
- **0/18 unresolved** (no CI includes zero) — every one of the 18 comparisons has a
  statistically clear direction at m=1024, n_boot=2000.

This is the single sharpest confirmation yet of the project's standing anomaly
(`CURRENT_PIPELINE.md` component 6/9/10/11/12): **the one place kernel-learnability and true
neural learnability disagree is exactly the GemNet-OC-vs-eSEN pair, and exactly at the point
where the neural hierarchy has crossed over** (eSEN overtaking GemNet-OC beyond ~1.64e17 FLOPs).
Every other pairwise comparison, at every budget, in the real matched-compute owners, at the
largest probe size (m=1024) and largest train size (n=512) tested anywhere in this project's
KRR line, is concordant. This sharpens rather than muddies the picture from
`four_arch_matched_compute_frontier_kernel_2026_08_15` (which found pooled Spearman rho=+0.769
with a persistent GemNet-OC>eSEN crossover anomaly at HIGH under 0.84-1.00 utilization, m=256):
at m=1024, with genuine matched-compute owners (not the unmatched width256/sphere64 checkpoints
the earlier m=1024 check used), the anomaly is isolated to exactly one pair at exactly one tier,
CI-significant, not a broad kernel-vs-neural disagreement.

## Comparison with the earlier (unmatched-checkpoint) m=1024 check

`large_m_convergence_2026_08_19` found GemNet-OC > eSEN alignment at every m from 64 to 1024 on
two UNMATCHED, single fixed-width checkpoints (width=256/sphere=64, both step=500000) — not any
LOW/MID/HIGH owner. This run replaces that with the real matched-compute owners and finds: the
GemNet-OC > eSEN kernel-learnability anomaly is real, m=1024-confirmed, and now precisely
localized to the HIGH budget tier and only that pair — not present at LOW or MID, where KRR and
NN now agree perfectly. This is new information the earlier unmatched-checkpoint check could not
provide (it had no budget-tier axis at all).

## m=2048 recommendation: **NO — do not run it.**

**(a) Trend/stability.** The alignment and KRR-AUC numbers at m=1024 show no sign of an
imminent crossover that a further m increase might resolve. The one discordant pair
(GemNet-OC/eSEN at HIGH) has a CI that excludes zero with real margin (mean log-ratio -0.387,
CI half-width ~0.26, i.e., the CI is roughly 1.5 log-ratio-widths away from zero) — this is a
stable, resolved disagreement, not a borderline case sitting near zero that more probe mass
could plausibly flip. This is consistent with the non-monotonic-but-bounded pattern already
established in `large_m_convergence_2026_08_19/REPORT.md` (alignment ratio staying in a
1.1-2.1x band from m=64 to m=1024 with no convergence trend) — more probe mass has not resolved
that disagreement's *direction* across four prior probe-size doublings, and there is no
mechanistic reason m=2048 would be different.

**(b) Which of the 18 pairwise comparisons are "unresolved"?** None. All 18 have CIs that
exclude zero at m=1024, n_boot=2000. There is no comparison in this grid that more probe mass
could plausibly resolve from "unresolved" to "resolved" — every comparison is already
CI-significant in one direction. Doubling m further would only sharpen already-clear answers,
which is low marginal scientific value for the cost below.

**(c) Cost estimate for m=2048**, extrapolated from this run's own OWNER-MATCHED m=256-vs-m=1024
timing (not the earlier m=64-vs-m=1024 unmatched-checkpoint proxy, which is a worse match for
this cost model since it isn't on the same 12 owners): for each of the 12 owners, comparing
total kernel-construction wall-clock at m=256 (from the existing
`four_arch_matched_compute_frontier_kernel_2026_08_15` kernels) against this run's m=1024
wall-clock (4x more probe atoms) gives an empirically observed scaling exponent
`total_seconds(m1024)/total_seconds(m256) = 4^alpha`:

| owner | alpha | owner | alpha |
|---|---|---|---|
| mpnn_LOW | 0.91 | mpnn_MID | 1.03 |
| mpnn_HIGH | 1.05 | egnn_LOW | 0.58 |
| egnn_MID | 0.54 | egnn_HIGH | 0.80 |
| gemnet_oc_LOW | 0.25 | gemnet_oc_MID | 0.42 |
| gemnet_oc_HIGH | 0.50 | esen_LOW | 0.64 |
| esen_MID | 0.84 | esen_HIGH | 0.82 |

Mean alpha = 0.70, max alpha (largest model, mpnn_HIGH) = 1.05 — worse than linear for the
biggest owner but noticeably better than the naive O(m^2) asymptotic one would expect from Gram
assembly alone (chunked-matmul overhead and jacobian-collection batching efficiency both grow
with m, partially offsetting the quadratic Gram cost at these sizes). Using the conservative
(larger) end of the observed range, alpha~1.0-1.1, a further m=1024-to-2048 doubling (factor 2)
would cost roughly `2^1.05 ~ 2.07x` the current per-owner time, i.e., total kernel-construction
GPU time for a 12-owner m=2048 grid would be **roughly 2.94 GPU-hours * 2.07 ~ 6.1 GPU-hours**,
dominated by an mpnn_HIGH single job that would grow from 1h37m to roughly 3h20m (still inside
a 5-hour sbatch ceiling, but not by a lot — a real risk of needing a longer time limit or chunked
resubmission for exactly the owner that already cost the most GPU time this run). KRR cost stays
negligible (CPU, seconds-to-minutes even at m=2048 given the O(n^3) train-block eigendecomposition
is capped by TRAIN_SIZES, not by m directly).

**Conclusion:** ~6 GPU-hours of further spend to sharpen 18 comparisons that are already all
CI-significant, with no case where the point estimate is trending toward a sign flip, is not
justified. The m=1024 matched-compute grid built in this run is the recommended stopping point
for this branch of Claim 1's kernel-vs-neural-hierarchy diagnostic.

## Output files (this directory, additive — the 6 pre-existing consolidation files were not
touched)

- `matched_compute_m1024_kernel_inventory.csv` — 12 rows, one per owner: numerics, provenance
  (SHA256/global_step verified against `owner_manifest.csv`), m=256 gate pass + numeric diffs,
  timing.
- `matched_compute_m1024_krr_results.csv` — 420 rows, tidy long format matching
  `existing_krr_inventory.csv`'s schema (architecture, checkpoint_state, budget_tier, m=1024,
  n_train, ridge_setting, target="common", test_nmse median/mean, split_count).
- `ranking_comparison.csv` — 12 rows: NN rank vs KRR rank (AUC statistic) per
  (architecture, budget).
- `pairwise_concordance.csv` — 18 rows: all pairwise architecture comparisons x 3 budgets,
  concordant/discordant/unresolved status with paired bootstrap CIs (n_boot=2000).

Source scripts: `analysis_scripts/run_matched_compute_large_m_kernels.py`,
`analysis_scripts/run_matched_compute_large_m_krr.py`,
`analysis_scripts/build_matched_compute_m1024_comparison.py`, all in
`dev-equivariant-scaling-laws-kernel-pilot-clean/`.
