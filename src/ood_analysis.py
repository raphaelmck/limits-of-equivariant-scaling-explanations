"""numpy port of the CPU-only OOD/ID re-analysis in
analysis_outputs/esen_irrep_ood_2026_08_22/eval/analyze_and_report.py and
.../eval/robustness_analysis.py.

GPU/CPU boundary (confirmed by reading run_gpu_eval.py, referenced in analyze_and_report.py's own
module docstring): run_gpu_eval.py does the model forward passes and writes ONE JSONL row per
(tag, domain, alpha) with already-reduced per_config_sum/per_config_natoms arrays (length
m_eval=16384) plus ss_by_ell_centered summaries into eval/raw/*.jsonl and eval/raw/run_meta.json.
analyze_and_report.py and robustness_analysis.py never touch a model or a checkpoint again --
they open those saved JSONL/JSON files and do only numpy re-aggregation (sum/log/bootstrap) over
the saved arrays. This module re-implements that same post-processing, reading from
data/claim2/ood_raw_per_config/*.

STAGE-2 UPGRADE (see AUDIT_STAGE2.md): a prior pass ported this module to pure-stdlib `random`
to avoid a numpy dependency. Both source files use `np.random.default_rng(seed)` (the numpy
Generator/PCG64 API, NOT RandomState -- confirmed by reading analyze_and_report.py's boot_L()
and robustness_analysis.py's boot_L(), both of which do
`rng = np.random.default_rng(seed); idx = rng.integers(0, m, size=(N_BOOT, m))`). Crucially, a
FRESH `default_rng(seed)` is constructed on EVERY call, always with the SAME fixed seed constant
regardless of (tag, domain, alpha, m) -- so, unlike the Claim-1 KRR pairwise bootstrap (which
threads a single shared RandomState across calls, see src/bootstrap.py), no cross-call ordering
matters here for bit-exactness: any two calls with the same m draw an identical index matrix
because each call re-seeds from scratch. This module now reproduces that exactly with numpy,
giving BIT-EXACT (not merely overlapping) CIs wherever the source used one of these two documented
seeds (BOOT_SEED_ANALYZE=20260822 for analyze_and_report.py's tables; BOOT_SEED_ROBUST=20260823
for robustness_analysis.py's tables).

Every function below is a direct port of the corresponding source code; see the docstring of each
for the exact file:line it reproduces. Nothing here is a guessed or reverse-engineered convention.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

ALPHAS = [1.00, 0.75, 0.50, 0.25, 0.00]
POOL_SIZES = [1024, 2048, 4096, 8192, 16384]
OWNER_ORDER = ["w10/50k", "w16/150k", "w20/200k", "w24/300k"]
N_BOOT = 2000
# BOOT_SEED_ANALYZE matches eval/analyze_and_report.py's module-level BOOT_SEED (line 27).
BOOT_SEED_ANALYZE = 20260822
# BOOT_SEED_ROBUST matches eval/robustness_analysis.py's module-level BOOT_SEED (line 39).
BOOT_SEED_ROBUST = 20260823


# ---------------------------------------------------------------------------
# weighted_L / P_bal -- ports of the identical helper duplicated verbatim in both source files
# (analyze_and_report.py:31-36,70-73 and robustness_analysis.py:44-49,76-79).
# ---------------------------------------------------------------------------

def weighted_L(sums: Sequence[float], natoms: Sequence[float]) -> float:
    """L = sum(per_config_sum) / sum(per_config_natoms). Ports analyze_and_report.py's L()
    (lines 31-36) / robustness_analysis.py's L() (lines 44-49) for the un-resampled case."""
    s = np.asarray(sums, dtype=np.float64)
    c = np.asarray(natoms, dtype=np.float64)
    return float(s.sum() / c.sum())


def compute_p_bal(ss_centered: Mapping[str, float], alpha: float, degree_weights: Mapping[int, float]) -> float:
    """Degree-balanced perturbation magnitude. Ports analyze_and_report.py:70-73 /
    robustness_analysis.py:76-79 EXACTLY:
        num = DEGREE_WEIGHTS[4] * (1 - alpha) ** 2 * ss_centered["4"]
        den = sum(DEGREE_WEIGHTS[e] * ss_centered[str(e)] for e in range(5))
        return sqrt(num / den)
    """
    num = degree_weights[4] * (1 - alpha) ** 2 * ss_centered["4"]
    den = sum(degree_weights[e] * ss_centered[str(e)] for e in range(5))
    return float(np.sqrt(num / den))


def compute_p_bal_curves(run_meta: dict) -> dict:
    """Ports analyze_and_report.py:83-87. Only owners carrying 'ss_by_ell_centered' (the lmax4
    block-9 owners) get a curve -- the lmax2 baseline-only owners have no ss_by_ell, exactly as
    the source comment states (line 85)."""
    degree_weights = {int(k): v for k, v in run_meta["degree_weights"].items()}
    curves = {}
    for key, om in run_meta["owners"].items():
        if "ss_by_ell_centered" not in om:
            continue
        tag, domain = key.split("::")
        curves[(tag, domain)] = {
            a: compute_p_bal(om["ss_by_ell_centered"], a, degree_weights) for a in ALPHAS
        }
    return curves


def common_support_hi(p_bal_curves: dict) -> float:
    """Ports analyze_and_report.py:89: common_hi = min(max(curve.values()) for curve in
    p_bal_curves.values())."""
    return min(max(curve.values()) for curve in p_bal_curves.values())


# ---------------------------------------------------------------------------
# Paired bootstrap over configuration indices -- numpy, bit-exact to the source's boot_L.
# ---------------------------------------------------------------------------

def gen_boot_indices(m: int, n_boot: int, seed: int) -> np.ndarray:
    """Bit-exact port of analyze_and_report.py:39-47 / robustness_analysis.py:52-63's
    `rng = np.random.default_rng(seed); idx = rng.integers(0, m, size=(N_BOOT, m))`.

    Both source files construct a FRESH default_rng(seed) on every call, always with the same
    fixed seed constant -- so this function, called once per m and reused for every (tag, domain,
    alpha, lmax) cell at that m, exactly reproduces that behaviour (every call in the source with
    the same m draws an identical matrix; sharing one matrix across cells at fixed m is
    equivalent, not an approximation)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, m, size=(n_boot, m))


def boot_L_all(sums: Sequence[float], natoms: Sequence[float], idx_matrix: np.ndarray) -> np.ndarray:
    """Vectorized port of `s[idx].sum(axis=1) / c[idx].sum(axis=1)`
    (analyze_and_report.py:46 / robustness_analysis.py:62)."""
    s = np.asarray(sums, dtype=np.float64)
    c = np.asarray(natoms, dtype=np.float64)
    return s[idx_matrix].sum(axis=1) / c[idx_matrix].sum(axis=1)


def ci95(values: np.ndarray) -> tuple[float, float]:
    """np.quantile(values, [0.025, 0.975]) -- matches every CI in both source files."""
    lo, hi = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
    return float(lo), float(hi)


def ci_excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


# ---------------------------------------------------------------------------
# P_bal-matched interpolation of Delta = log(L_int) - log(L_base) at the common support edge.
# Ports analyze_and_report.py:126-137 / the byte-identical robustness_analysis.py:82-92
# (interp_boot_at_x) EXACTLY.
# ---------------------------------------------------------------------------

def interp_at_x(
    x_vals: Sequence[float],
    y_pt: Sequence[float],
    y_boot_rows: Sequence[np.ndarray],
    x_target: float,
) -> tuple[float, np.ndarray]:
    """Ports:
        order = np.argsort(x_vals)
        ... (sort) ...
        d_read_pt = float(np.interp(read_x, x_sorted, d_pt_sorted))
        j = np.clip(np.searchsorted(x_sorted, read_x, side="right") - 1, 0, len(x_sorted) - 2)
        x0, x1 = x_sorted[j], x_sorted[j + 1]
        w_ = 0.0 if x1 == x0 else (read_x - x0) / (x1 - x0)
        d_read_boot = d_boot_sorted[:, j] + w_ * (d_boot_sorted[:, j + 1] - d_boot_sorted[:, j])
    (analyze_and_report.py:126-137, identical formulas in robustness_analysis.py:82-92).
    By construction x_target (the common P_bal support edge) always lies within
    [min(x_vals), max(x_vals)] for every curve it is applied to (common_hi = min over curves of
    each curve's own max), so no extrapolation ever occurs here."""
    x_vals = np.asarray(x_vals, dtype=np.float64)
    y_pt = np.asarray(y_pt, dtype=np.float64)
    y_boot_rows = np.stack([np.asarray(r, dtype=np.float64) for r in y_boot_rows], axis=0)  # [n_alpha, n_boot]

    order = np.argsort(x_vals)
    x_sorted = x_vals[order]
    y_pt_sorted = y_pt[order]
    y_boot_sorted = y_boot_rows[order].T  # [n_boot, n_alpha], matches source's [:, j] indexing

    y_read_pt = float(np.interp(x_target, x_sorted, y_pt_sorted))
    j = int(np.clip(np.searchsorted(x_sorted, x_target, side="right") - 1, 0, len(x_sorted) - 2))
    x0, x1 = x_sorted[j], x_sorted[j + 1]
    w = 0.0 if x1 == x0 else (x_target - x0) / (x1 - x0)
    y_read_boot = y_boot_sorted[:, j] + w * (y_boot_sorted[:, j + 1] - y_boot_sorted[:, j])
    return y_read_pt, y_read_boot


# ---------------------------------------------------------------------------
# Loaders for the raw per-configuration JSONL files.
# ---------------------------------------------------------------------------

def index_lmax4(records: list[dict]) -> dict:
    """Ports analyze_and_report.py:50-57 / robustness_analysis.py:66-73 load_lmax4():
    {(tag, domain): {alpha: record}}."""
    out: dict = {}
    for r in records:
        out.setdefault((r["tag"], r["domain"]), {})[round(r["alpha"], 2)] = r
    return out


def index_lmax2(records: list[dict]) -> dict:
    """Ports analyze_and_report.py:60-67 load_lmax2(): {(tag, domain): record}."""
    out = {}
    for r in records:
        out[(r["tag"], r["domain"])] = r
    return out


def find_matched_lmax2_tag(lmax2_by_key: dict, lmax4_tag: str) -> str | None:
    """Ports the `matched_to` lookup used at analyze_and_report.py:204-208/240-244."""
    for (t2, _d2), r2 in lmax2_by_key.items():
        if r2.get("matched_to") == lmax4_tag:
            return t2
    return None
