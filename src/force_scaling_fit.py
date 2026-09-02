"""Force compute-scaling frontier construction and power-law fit.

Implements the headline fit only; the two sensitivity variants recorded in the frozen table
(a fit over the unevenly spaced frontier events, and nonlinear least squares on unlogged loss)
are not reimplemented here -- see scripts/claim1/build_force_scaling.py.

Procedure:

  1. Frontier construction, per architecture, over its `metric_valid` checkpoint rows: sort by
     (compute, loss) ascending with a stable sort, so at equal compute the lowest loss comes
     first; drop duplicate compute values keeping that first row; then walk in increasing
     compute and keep only rows that set a new running minimum in loss. The result is the
     right-continuous step minimum L*(C).
  2. Common compute interval: the intersection across architectures of each one's frontier
     support, so every exponent is fitted on the same interval.
  3. Fit: evaluate L*(C) on 256 points equally spaced in log10(C) across that interval and take
     the ordinary-least-squares fit of log L on log C. The slope is -gamma; the intercept is
     logA. Reported alongside are r2, the root-mean-square residual in log space, the interval
     bounds, its width in decades, and how many frontier checkpoints fall inside it.

Pure standard library: no numpy is needed for this fit.
"""


from __future__ import annotations

import math
from typing import Sequence

N_GRID = 256


def log_grid(c_lo: float, c_hi: float, n: int = N_GRID) -> list[float]:
    """n points equally spaced in log10(C) over [c_lo, c_hi], endpoints pinned exactly."""
    if n < 2:
        raise ValueError("n must be >= 2")
    log_lo, log_hi = math.log10(c_lo), math.log10(c_hi)
    step = (log_hi - log_lo) / (n - 1)
    grid = [10 ** (log_lo + i * step) for i in range(n)]
    grid[0], grid[-1] = c_lo, c_hi
    return grid


def build_frontier(rows: Sequence[dict], loss_col: str, c_col: str = "C") -> list[dict]:
    """Return the frontier-owner rows (right-continuous step-minimum), in increasing C."""
    valid_rows = [r for r in rows if r.get(c_col) is not None and r.get(loss_col) is not None]
    # rule: sort by (C, loss) ascending, stable -- at exact-C ties the lowest loss sorts first
    ordered = sorted(valid_rows, key=lambda r: (r[c_col], r[loss_col]))
    # drop_duplicates(subset=C, keep="first")
    deduped = []
    seen_c = set()
    for r in ordered:
        if r[c_col] not in seen_c:
            seen_c.add(r[c_col])
            deduped.append(r)
    keep = []
    best = math.inf
    for r in deduped:
        if r[loss_col] < best:
            best = r[loss_col]
            keep.append(r)
    return keep


def frontier_eval(owners: Sequence[dict], c_query: float, loss_col: str, c_col: str = "C"):
    """Right-continuous step-frontier lookup at a single query point C. Returns None if
    c_query is below the frontier's domain (no owner established yet)."""
    best = None
    for r in owners:
        if r[c_col] <= c_query:
            if best is None or r[c_col] > best[c_col]:
                best = r
    return best[loss_col] if best is not None else None


def common_interval(domains: dict) -> tuple[float, float]:
    """domains: {arch -> (lo, hi)}. Returns [max_a lo_a, min_a hi_a]."""
    lo = max(d[0] for d in domains.values())
    hi = min(d[1] for d in domains.values())
    return lo, hi


def truncate_upper(c_lo: float, c_hi: float, frac: float) -> tuple[float, float]:
    """Keep the lowest `frac` of the log-compute span, anchored at c_lo."""
    return c_lo, c_lo * (c_hi / c_lo) ** frac


def ols_loglog(xs: Sequence[float], ys: Sequence[float]) -> dict:
    """Unweighted OLS of y on x (both already in log space): y = slope*x + intercept.
    Returns slope, intercept, r2, rmse (of residuals in y)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = math.sqrt(ss_res / n)
    return {"slope": slope, "intercept": intercept, "r2": r2, "rmse": rmse, "n": n}


def fit_loglog(owners: Sequence[dict], c_lo: float, c_hi: float, loss_col: str,
               c_col: str = "C", n_grid: int = N_GRID) -> dict | None:
    """OLS of log(loss) on log(C) over n_grid equally spaced log-C points in [c_lo, c_hi],
    evaluated against the step frontier `owners`. Matches frontier_fit.py's fit_loglog."""
    if not (c_hi > c_lo):
        return None
    grid = log_grid(c_lo, c_hi, n_grid)
    xs, ys = [], []
    for c in grid:
        lv = frontier_eval(owners, c, loss_col, c_col)
        if lv is not None and math.isfinite(lv) and lv > 0:
            xs.append(math.log(c))
            ys.append(math.log(lv))
    if len(xs) < 3:
        return None
    fit = ols_loglog(xs, ys)
    n_owners_in_range = sum(1 for r in owners if c_lo <= r[c_col] <= c_hi)
    return {
        "gamma": -fit["slope"],
        "logA": fit["intercept"],
        "r2": fit["r2"],
        "rmse_log": fit["rmse"],
        "C_min": c_lo,
        "C_max": c_hi,
        "decades": math.log10(c_hi / c_lo),
        "n_grid_used": len(xs),
        "n_frontier_owners_in_range": n_owners_in_range,
    }
