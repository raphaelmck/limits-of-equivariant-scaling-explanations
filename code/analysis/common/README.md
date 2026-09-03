# code/analysis/common/ — reusable analysis functions

Import-only: no file I/O and no side effects at import time. Every function takes plain Python
data or numpy arrays already loaded by a caller in `code/analysis/`.

- **`io_utils.py`** — CSV/JSON/JSONL helpers, and `data_path()`, `out_path()`: the only sanctioned
  way to build a path into `data/` or `analysis_out/`.
- **`compute_axis.py`** — training compute of each (architecture, budget) checkpoint, including
  the resolution rule for budgets that reuse a neighbour's checkpoint. Shared by the figures and
  the tests so the compute axis has one definition.
- **`bootstrap.py`** — configuration-level paired bootstrap: `weighted_L`, `log_delta`,
  `paired_config_bootstrap`, `bootstrap_delta_ci`, `bootstrap_paired_log_ratio_ci`, `ci95`. One
  resample of configuration indices is shared across every array in a comparison, so cross-cell
  and cross-architecture contrasts stay paired.
- **`interpolate.py`** — `interpolate_bracket`: linear interpolation between the two measured
  points bracketing a target, never extrapolating. This is the read-off rule used wherever a
  damage curve is evaluated at a common perturbation magnitude.
- **`krr_auc.py`** — `auc_log_nmse`, the trapezoidal area of log NMSE over log training size, and
  `rank_ascending`.
- **`force_scaling_fit.py`** — frontier construction and the power-law fit behind the scaling
  exponents.
- **`domain_analysis.py`** — re-aggregation of the intervention runs on the two evaluation
  populations, including the degree-balanced perturbation magnitude and its matched interpolation.
- **`seed_replication.py`** — the independent-run replication, which uses the raw total
  perturbation norm rather than the degree-balanced one. Deliberately shares no code with
  `domain_analysis.py` so the two metrics cannot be conflated.
- **`depth_interventions.py`** — degree-balanced helpers for the block-3, block-6, and block-9
  comparison.
