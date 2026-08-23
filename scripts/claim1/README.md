<title>scripts/claim1/ — Claim-1 derived-table entry points</title>

# scripts/claim1/

Thin entry points that load `data/claim1/*.csv`, call `src/` functions, and write regenerated
tables to `analysis_out/claim1/`. Read ONLY from `paper_repro/data/`; never import from either
historical repo checkout.

- **`build_ranking_comparison.py`** — regenerates `ranking_comparison.csv` (NN rank vs KRR-AUC
  rank per architecture/budget) from `owner_manifest.csv` (force_mse_norm) +
  `matched_compute_m1024_krr_results.csv` (ridgeless/common-target KRR curves). Category (a),
  bit-exact reproduction (no randomness involved).
- **`build_pairwise_concordance.py`** — PARTIAL: recomputes the point-estimate log-ratio
  direction (which architecture KRR favors at n=512) and checks it against the frozen file's own
  `nn_favors`/`krr_favors_point_estimate` columns. Does NOT regenerate the bootstrap CI
  (`bootstrap_ci_lo/hi_*pct`) — the raw per-split (20 splits) test_nmse values that CI would need
  are not present in `data/claim1/` (only their median/mean are). See AUDIT_STAGE2.md category (b).

Run via `make analysis` (from the `paper_repro/` root) or invoke a script directly with
`python3 scripts/claim1/<script>.py`.
