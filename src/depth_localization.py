"""Depth-localization helpers for the block-3, block-6, and block-9 interventions
(data/claim2/depth_localization/).

Same degree-balanced perturbation magnitude P_bal and matched-support readout used for the
block-9 result in data/claim2/frontier_ell4_degree_balanced*, applied at the two additional
depths.

Normalized block depth. Block indices in this codebase are 0-based (blocks 0 through 11), so
block k sits at fractional depth z = (k + 1) / 12, which is the convention this module treats as
canonical. Some frozen artifacts carry an older z = k / 12 label; that label is retained below
only as an explicitly superseded field. The choice affects how a block position is reported, not
any damage value, perturbation magnitude, or confidence interval.
"""
from __future__ import annotations

import math
from typing import Sequence

from src.bootstrap import bootstrap_delta_ci
from src.interpolate import interpolate_bracket

N_BLOCKS = 12
PROBED_BLOCKS = (3, 6, 9)


def block_z_canonical(k: int) -> float:
    """Canonical normalized depth location of eSEN block k (0-indexed, 12 blocks total):
    z = (k+1)/12. This is the formula this module treats as correct."""
    return (k + 1) / N_BLOCKS


def block_z_superseded_label(k: int) -> float:
    """Superseded: z = k/12, the off-by-one convention carried by some frozen artifacts
    (block 3 -> 0.25, block 6 -> 0.5, block 9 -> 0.75). Kept only so those labels can be read
    back; do not use it in a new computation or figure."""
    return k / N_BLOCKS


def degree_weight(ell: int) -> float:
    """w_l = 1/(2l+1), the degree-balancing weight used throughout the frontier_ell4_degree_
    balanced / depth_localization P_bal metric."""
    return 1.0 / (2 * ell + 1)


def denom_bal(ss_by_ell_raw: dict, ss_ell0_centered: float) -> float:
    """sum_l w_l * SS_base_l(k), with the l=0 term replaced by its ell=0-per-atom-per-channel-
    centered value (`rms_norm_sh`'s own centering convention) and l=1..4 taken raw -- matches
    the `denom_bal` field already stored in depth_localization_summary.json's `curves` entries
    (verified numerically to 1e-9 relative against the frozen w10/50k_L3 curve)."""
    total = degree_weight(0) * ss_ell0_centered
    for ell in (1, 2, 3, 4):
        total += degree_weight(ell) * ss_by_ell_raw[str(ell)]
    return total


def p_bal(alpha: float, ss4_base: float, denom: float, ell: int = 4) -> float:
    """P_bal = sqrt( w_ell * SS_base_ell(k) * (1-alpha)^2 / denom_bal(k) )."""
    w_ell = degree_weight(ell)
    return math.sqrt(w_ell * ss4_base * (1.0 - alpha) ** 2 / denom)


def alpha_grid_deltas(rows_for_owner_layer: Sequence[dict], n_boot: int = 2000, seed: int = 0) -> list[dict]:
    """Given the raw per-alpha rows (each carrying baseline_per_config_sum/natoms and
    per_config_sum/natoms) for one (owner, layer), return one dict per alpha with the bootstrap
    Delta/CI, matching `src.bootstrap.bootstrap_delta_ci`'s convention (also used by
    scripts/claim2/build_dose_response.py)."""
    out = []
    for r in rows_for_owner_layer:
        result = bootstrap_delta_ci(
            r["baseline_per_config_sum"],
            r["baseline_per_config_natoms"],
            r["per_config_sum"],
            r["per_config_natoms"],
            n_boot=n_boot,
            seed=seed,
        )
        out.append({"alpha": r["alpha"], **result})
    return out


def interpolate_curve_to_support(
    curve_rows: Sequence[dict], p_bal_key: str, target_p_bal: float
) -> dict:
    """Linear interpolation (no extrapolation) of delta/delta_ci95 to target_p_bal, using the
    shared src/interpolate.py bracketing rule -- same convention as
    scripts/claim2/build_frontier_ell4_degree_balanced.py."""
    x_measured = [r[p_bal_key] for r in curve_rows]
    typed = [
        {"delta": r["delta"], "ci_lo": r["delta_ci95"][0], "ci_hi": r["delta_ci95"][1]}
        for r in curve_rows
    ]
    interp = interpolate_bracket(x_measured, typed, target_p_bal, ["delta", "ci_lo", "ci_hi"])
    on_measured_point = any(abs(x - target_p_bal) < 1e-9 for x in x_measured)
    return {
        "delta": interp["delta"],
        "loss_multiplier": math.exp(interp["delta"]),
        "ci95": [interp["ci_lo"], interp["ci_hi"]],
        "on_measured_point": on_measured_point,
    }


def depth_contrast(delta_a: float, delta_b: float) -> dict:
    """delta_difference = delta_a - delta_b (e.g. a=deeper block, b=shallower block), matching
    depth_localization_summary.json's `depth_contrasts` naming (`{deep}_minus_{shallow}`)."""
    diff = delta_a - delta_b
    sign = "increases_with_depth" if diff > 0 else "decreases_with_depth"
    return {"delta_difference": diff, "sign": sign}
