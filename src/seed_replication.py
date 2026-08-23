"""Bit-exact numpy port of
dev-equivariant-scaling-laws-kernel-pilot-clean/analysis_scripts/analyze_seed_replication.py.

DELIBERATELY SEPARATE from src/ood_analysis.py / the frontier_ell4_degree_balanced code path
(see scripts/claim2/build_seed_replication_result.py and AUDIT_STAGE2.md): this experiment's
pre-registered estimand (PRE_REGISTRATION.md Sec 8) uses the RAW P_total = RMS(h_int-h_base) /
RMS(h_base) metric read directly off each row's own `P_total`/`q_power` field, NOT the
degree-balanced-RMS P_bal metric used by frontier_ell4_degree_balanced.csv/Task 3. The two
metrics are numerically different (P_total is an unweighted-by-degree RMS ratio; P_bal reweights
each spherical-harmonic degree by 1/(2l+1) after l=0-centering) even though both are computed from
the same underlying per-configuration block-9 ell=4 NORM_FIXED intervention runs. This module must
never import from src/ood_analysis.py's P_bal machinery, and vice versa, to keep that distinction
structurally enforced, not just documented.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

N_BOOT = 2000
BOOTSTRAP_SEED = 0  # documented in G_s_analysis.json's own "bootstrap_seed" field
P_STAR = 0.15
ALPHAS = [1.00, 0.75, 0.50, 0.25, 0.00]


def bootstrap_index_sets(m: int, n_boot: int, seed: int) -> list[np.ndarray]:
    """Bit-exact port of analyze_seed_replication.py:30-32:
        rng = np.random.default_rng(seed)
        return [rng.integers(0, m, size=m) for _ in range(n_boot)]
    ONE shared rng instance, n_boot SEPARATE `.integers()` calls in a loop (not one batched call)
    -- generated ONCE and reused for every seed/tier/cell (paired bootstrap)."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, m, size=m) for _ in range(n_boot)]


def flat_mean_for_indices(per_config_sum: Sequence[float], per_config_natoms: Sequence[float], idx: np.ndarray) -> float:
    """Ports flat_mean_for_indices() lines 34-37 exactly, including max(n,1)."""
    s = np.asarray(per_config_sum, dtype=np.float64)[idx].sum()
    n = np.asarray(per_config_natoms, dtype=np.int64)[idx].sum()
    return float(s / max(n, 1))


def ci95(arr: np.ndarray) -> tuple[float, float]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


def bracket(cells: list[dict], p_star: float):
    """Ports bracket() lines 106-116: cells sorted by DESCENDING alpha (== ascending P_total)."""
    ps = [c["P_total"] for c in cells]
    for i in range(len(ps) - 1):
        assert ps[i] <= ps[i + 1] + 1e-12, f"P_total not monotonic: {ps}"
    if p_star < ps[0] or p_star > ps[-1]:
        return None, None, False
    for i in range(len(cells) - 1):
        if ps[i] <= p_star <= ps[i + 1]:
            return cells[i], cells[i + 1], True
    return None, None, False


def interp_delta_and_ci(cells: list[dict], base_cell: dict, p_star: float, idx_sets: list[np.ndarray]):
    """Bit-exact port of interp_delta_and_ci() lines 126-149."""
    lo_cell, hi_cell, in_support = bracket(cells, p_star)
    result = {"in_support": in_support}
    if not in_support:
        result.update({
            "delta_point": None, "delta_ci95": [None, None], "bracket_alphas": None,
            "bracket_P": [cells[0]["P_total"], cells[-1]["P_total"]],
        })
        return result, None
    p_lo, p_hi = lo_cell["P_total"], hi_cell["P_total"]
    frac = 0.0 if p_hi == p_lo else (p_star - p_lo) / (p_hi - p_lo)
    delta_point = lo_cell["delta"] + frac * (hi_cell["delta"] - lo_cell["delta"])

    boot = np.empty(len(idx_sets))
    for b, idx in enumerate(idx_sets):
        Lo_base = flat_mean_for_indices(base_cell["per_config_sum"], base_cell["per_config_natoms"], idx)
        Lo_lo = flat_mean_for_indices(lo_cell["per_config_sum"], lo_cell["per_config_natoms"], idx)
        Lo_hi = flat_mean_for_indices(hi_cell["per_config_sum"], hi_cell["per_config_natoms"], idx)
        d_lo = np.log(Lo_lo / Lo_base) if Lo_base > 0 and Lo_lo > 0 else np.nan
        d_hi = np.log(Lo_hi / Lo_base) if Lo_base > 0 and Lo_hi > 0 else np.nan
        boot[b] = d_lo + frac * (d_hi - d_lo)
    lo_ci, hi_ci = ci95(boot)
    result.update({
        "delta_point": delta_point, "delta_ci95": [lo_ci, hi_ci],
        "bracket_alphas": [lo_cell["alpha"], hi_cell["alpha"]],
        "bracket_P": [p_lo, p_hi], "frac": frac,
    })
    return result, boot
