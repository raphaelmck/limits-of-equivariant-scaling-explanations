<title>src/ — Stage-2 analysis library</title>

# src/ — reusable analysis functions

Requires numpy (see `pyproject.toml` / `AUDIT_STAGE2.md` §0 — a Stage-2 policy change from the
first pass's pure-stdlib-only convention). Import-only: no file I/O and no side effects at import
time. Every function reads its input as plain Python data (lists/dicts/numpy arrays) already
loaded by the caller in `scripts/claim1/` or `scripts/claim2/`.

- **`io_utils.py`** — CSV/JSON/JSONL read/write helpers, and `data_path()`/`manifest_path()`/
  `out_path()` — the only sanctioned way any script constructs a path into `data/`,
  `manifests/`, or `analysis_out/`.
- **`bootstrap.py`** — configuration-level paired bootstrap, numpy-backed
  (`numpy.random.RandomState`). `weighted_L` (per-atom-weighted mean loss from
  `per_config_sum`/`per_config_natoms`), `log_delta` (Delta = log(L_int/L_base), verified against
  every claim2 file's own `delta`/`Delta` field), `paired_config_bootstrap` (generic resampling
  of configuration indices, same resample shared across every array passed in so
  cross-cell/cross-architecture contrasts stay paired), `bootstrap_delta_ci` (the common
  single-Delta case), `bootstrap_paired_log_ratio_ci` (Claim-1 KRR pairwise bootstrap — supports
  threading a single shared `RandomState` across calls for bit-exact upstream reproduction),
  `ci95` (numpy `np.percentile`-based).
- **`interpolate.py`** — `interpolate_bracket`: linear interpolation between the two measured
  points bracketing a target x, no extrapolation. Implements the exact rule used repeatedly
  across Claim 2's dose-response / degree-balanced / seed-replication summaries.
- **`krr_auc.py`** — `auc_log_nmse` (Claim 1's `auc_log_nmse` statistic: trapezoidal AUC of
  log(nmse) over log(n)) and `rank_ascending` (rank-by-lower-is-better).
- **`ood_analysis.py`** — numpy port (`numpy.random.default_rng`) of the OOD/ID re-analysis;
  P_bal-matched interpolation, bit-exact bootstrap machinery shared by the OOD curve/contrast,
  absolute-effect-robustness, and domain-decomposition scripts.
- **`seed_replication.py`** — standalone module for the seed-replication experiment's RAW
  `P_total` metric (deliberately NOT sharing code with `ood_analysis.py`'s degree-balanced `P_bal`
  metric — see its own module docstring for why the two must stay separate).
- **`depth_localization.py`** — degree-balanced `P_bal`/`denom_bal` helpers for the block-3/6/9
  depth-localization analysis (block-9's own bootstrap now lives in
  `scripts/claim2/build_depth_localization.py` directly, sharing one `default_rng` index matrix
  across all three depths — see `AUDIT_STAGE2.md`).

See `paper_repro/AUDIT_STAGE2.md` for exactly which frozen tables each function reproduces, and
which numbers remain out of reach given what's in `data/`.
