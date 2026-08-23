"""Claim-1 KRR-curve statistics: auc_log_nmse, matching the frozen convention documented in
data/claim1/REPORT.md ('trapezoid(log(ridgeless_nmse_median), log(n)) / (log(n_max)-log(n_min))')
and reproduced verbatim in ranking_comparison.csv's own auc_statistic_definition column.
"""
from __future__ import annotations

import math
from typing import Sequence


def auc_log_nmse(n_values: Sequence[float], nmse_values: Sequence[float]) -> float:
    """Trapezoidal AUC of log(nmse) over log(n), normalized by the log(n) span. n_values and
    nmse_values must be parallel and both strictly positive. Lower AUC = better (learns faster)."""
    pairs = sorted(zip(n_values, nmse_values), key=lambda p: p[0])
    xs = [math.log(p[0]) for p in pairs]
    ys = [math.log(p[1]) for p in pairs]
    area = 0.0
    for i in range(len(xs) - 1):
        area += 0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i])
    span = xs[-1] - xs[0]
    return area / span


def rank_ascending(items: Sequence[tuple[str, float]]) -> dict[str, int]:
    """items: [(key, value), ...], lower value = better = rank 1. Returns {key: rank}."""
    ordered = sorted(items, key=lambda p: p[1])
    return {k: i + 1 for i, (k, _v) in enumerate(ordered)}
