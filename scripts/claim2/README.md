<title>scripts/claim2/ — Claim-2 derived-table entry points</title>

# scripts/claim2/

Thin entry points that load `data/claim2/*.jsonl`/`*.csv`/`*.json`, call `src/` functions, and
write regenerated tables to `analysis_out/claim2/`. Read ONLY from `paper_repro/data/`; never
import from either historical repo checkout.

- **`build_dose_response.py`** — recomputes the (budget, ell, alpha) Delta/CI grid from
  `dose_response.jsonl`'s per-config arrays. Category (a).
- **`build_compensation.py`** — recomputes the (pair, ell, alpha) `C_ell = Delta_l2 - Delta_l4`
  contrast from `compensation.jsonl`'s per-config arrays, using a shared (paired) bootstrap
  resample across the l2/l4 baseline+intervened cells. Category (a).
- **`build_same_width_control_B.py`** — same as above, keyed by same-width pair instead of
  matched-compute pair, from `same_width_control_B.jsonl`. Category (a).
- **`build_run_replication_control_C.py`** — per-run Delta/CI plus the paired runA-vs-runB
  difference/CI, from `run_replication_control_C.jsonl`. Category (a).
- **`build_frontier_ell4_degree_balanced.py`** — linear interpolation (no extrapolation) of the
  already-bootstrapped per-alpha (P_bal, delta, CI) points in
  `frontier_ell4_degree_balanced.csv` to the common-support P_bal edge. Category (a) for the
  interpolation step itself (the upstream per-config bootstrap that produced those points is not
  redone — no raw per-config array for this degree-balanced metric exists in `data/claim2/`).
- **`build_seed_replication_result.py`** — same interpolation rule, applied over P_total to
  P*=0.15, to recompute `G_by_seed` from `seed_replication_result.csv`'s measured-alpha rows.
  Category (a) for the Delta/G point estimate; the frozen CI at P* is passthrough-only (no raw
  per-config array in this file either).

**Not implemented in this pass** (documented explicitly in AUDIT_STAGE2.md, not silently
skipped):
- `gap_erasure_summary.json`, `pathway_rescue*.{jsonl,json}` — category (b): no
  `per_config_sum`/`per_config_natoms` raw arrays exist in these files, only already-aggregated
  scalars.
- The OOD files (`ood_G_m_convergence.csv`, `ood_interaction_w24_vs_w10.csv`, and the other
  `ood_*.csv`) — raw per-config arrays DO exist (`ood_raw_per_config/*.jsonl`, at the full
  M=16384 pool), and G_m_convergence's nested-M values are in principle recoverable by
  prefix-truncating those arrays, but the exact "matched degree-balanced representation
  perturbation" alignment convention that turns per-alpha deltas into a single G(M) per tag was
  not successfully reverse-engineered from the copied summary/CSV files in this pass. Treated as
  a documented gap rather than guessed at. See AUDIT_STAGE2.md.

Run via `make analysis` (from the `paper_repro/` root) or invoke a script directly with
`python3 scripts/claim2/<script>.py`.
