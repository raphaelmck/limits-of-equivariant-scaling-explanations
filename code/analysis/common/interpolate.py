"""Linear interpolation to a target value between the two bracketing measured points, no
extrapolation. This exact rule is used repeatedly across Claim-2's frozen summaries
(dose-response / degree-balanced / seed-replication) -- implemented once here and reused.
"""
from __future__ import annotations

from typing import Sequence


def interpolate_bracket(
    x_measured: Sequence[float],
    rows: Sequence[dict],
    x_target: float,
    value_keys: Sequence[str],
) -> dict:
    """Given parallel x_measured and rows (each row a dict with numeric fields named in
    value_keys), find the two measured x points bracketing x_target and linearly interpolate
    every field in value_keys. Raises ValueError if x_target is outside [min(x_measured),
    max(x_measured)] (no extrapolation, per the frozen convention).

    x_measured need not be pre-sorted; this function sorts internally and carries `rows`
    along with it.
    """
    pairs = sorted(zip(x_measured, rows), key=lambda p: p[0])
    xs = [p[0] for p in pairs]
    if x_target < xs[0] or x_target > xs[-1]:
        raise ValueError(
            f"x_target={x_target} is outside measured support [{xs[0]}, {xs[-1]}] -- "
            "no extrapolation allowed."
        )
    # exact hit
    for x, row in pairs:
        if x == x_target:
            return {k: row[k] for k in value_keys}
    # find bracket
    for i in range(len(xs) - 1):
        x_lo, x_hi = xs[i], xs[i + 1]
        if x_lo <= x_target <= x_hi:
            row_lo, row_hi = pairs[i][1], pairs[i + 1][1]
            frac = (x_target - x_lo) / (x_hi - x_lo)
            out = {}
            for k in value_keys:
                v_lo, v_hi = row_lo[k], row_hi[k]
                out[k] = v_lo + (v_hi - v_lo) * frac
            return out
    raise ValueError(f"could not bracket x_target={x_target} in {xs}")
