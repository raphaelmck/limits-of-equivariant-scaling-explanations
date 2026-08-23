#!/usr/bin/env python3
"""Regenerate data/claim2/frontier_ell4_degree_balanced_summary.json's per-alpha delta/CI,
interpolated-to-common-support-edge Delta/CI per owner, and the 6 pairwise Delta-difference
contrasts -- ALL from raw per-configuration arrays, with a REAL numpy bootstrap (Stage-2 fix;
see AUDIT_STAGE2.md). A prior pass only interpolated the frozen CSV's own already-bootstrapped
delta_ci_lo/delta_ci_hi columns (a passthrough of an upstream bootstrap it could not itself
redo, since it believed no raw per-config array existed for this metric). Two things changed:

1. This Stage-2 pass discovered (numerically, not by assumption) that the degree-balanced-RMS
   pass's "delta" field at every (owner, alpha) is IDENTICAL to log(L_intervened/L_baseline)
   computed directly on the SAME raw per-atom-weighted force-MSE per_config_sum/per_config_natoms
   arrays already sitting in data/claim2/dose_response.jsonl (for w10/50k=LOW, w16/150k=MID) and
   the newly-copied data/claim2/frontier_ell4_sensitivity_raw.jsonl (for w20/200k=UPPERMID,
   w24/300k=NEARMAX, copied from
   dev-equivariant-scaling-laws-kernel-pilot-clean/analysis_outputs/frontier_ell4_sensitivity_2026_08_20/
   frontier_ell4_M1024.jsonl). The "degree-balanced" part of this table is ONLY the X-AXIS
   (P_bal instead of alpha, already given per-row in frontier_ell4_degree_balanced.csv's own
   P_bal column -- a deterministic function of ss_by_ell, not something to bootstrap); the Y-AXIS
   quantity (delta) is the plain block-9 ell=4 NORM_FIXED loss-ratio metric this whole project
   uses everywhere else. Verified to full float precision for all 4 owners at multiple alphas.

2. The frozen summary documents `"n_boot": 2000, "boot_seed": 20260820` at its top level. Trying
   `np.random.default_rng(20260820)` (the same Generator/PCG64 API confirmed for the OOD scripts,
   see src/ood_analysis.py), constructed FRESH per (owner, alpha) cell with
   `idx = rng.integers(0, 1024, size=(2000, 1024))` applied identically to baseline and
   intervened per_config_sum/per_config_natoms, reproduces every one of the frozen per-alpha
   delta_ci_lo/delta_ci_hi columns in frontier_ell4_degree_balanced.csv to full float precision
   (< 1e-12 absolute) -- BIT-EXACT, not merely overlapping. The interpolated-to-common-support
   read-off CI is reproduced the same way src/ood_analysis.interp_at_x does it: linearly
   interpolate the BOOTSTRAP REPLICATE ARRAYS themselves (not just their quantiles) at the target
   P_bal using the same bracket/weight as the point-estimate interpolation, then take the 2.5/97.5
   percentile of the interpolated replicate array -- also verified bit-exact against
   read_off_at_common_support_edge's loss_multiplier_ci95. The 6 pairwise Delta-difference
   contrasts are then a THIRD, correlated use of the exact same idx matrices: because
   default_rng(20260820) is re-seeded identically for every cell regardless of owner/alpha, two
   different owners' bootstrap replicate arrays are implicitly a PAIRED resample of the SAME
   underlying M=1024 configuration-index draws (position i in every owner's per_config_sum array
   is the same frozen-pool molecule) -- exactly the paired-bootstrap convention documented
   throughout this project (AUDIT_STAGE1.md / src/bootstrap.py's module docstring). Taking
   (owner_B's interpolated replicate array - owner_A's) per replicate and its 2.5/97.5 percentile
   reproduces all 6 frozen pairwise_contrasts_at_common_support_edge entries bit-exactly.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.io_utils import data_path, out_path, read_csv, read_json, read_jsonl, write_json
from src.ood_analysis import interp_at_x  # shared bracket-interpolation-of-replicates helper

N_BOOT = 2000
BOOT_SEED = 20260820  # documented in frontier_ell4_degree_balanced_summary.json; matches OOD's
                        # default_rng convention -- confirmed bit-exact against the frozen CIs.

ALPHAS = [1.0, 0.75, 0.5, 0.25, 0.0]
OWNER_ORDER = ["w10/50k", "w16/150k", "w20/200k", "w24/300k"]


def _boot_delta(base_sum, base_natoms, int_sum, int_natoms, seed=BOOT_SEED, n_boot=N_BOOT):
    bs = np.asarray(base_sum, dtype=np.float64)
    bc = np.asarray(base_natoms, dtype=np.float64)
    is_ = np.asarray(int_sum, dtype=np.float64)
    ic = np.asarray(int_natoms, dtype=np.float64)
    n = len(bs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    Lb_boot = bs[idx].sum(axis=1) / bc[idx].sum(axis=1)
    Li_boot = is_[idx].sum(axis=1) / ic[idx].sum(axis=1)
    d_boot = np.log(Li_boot) - np.log(Lb_boot)
    d_pt = float(math.log(is_.sum() / ic.sum()) - math.log(bs.sum() / bc.sum()))
    return d_pt, d_boot


def _load_owner_curve(owner: str, dose_rows: list[dict], sens_rows: list[dict]):
    """Returns (d_pt: list[5], d_boot: list[np.ndarray[2000]]) aligned with ALPHAS, plus the
    per-alpha P_bal values from the frozen degree-balanced CSV (deterministic geometry, not
    bootstrapped -- see module docstring item 1)."""
    if owner == "w10/50k":
        budget = "LOW"
        base = next(r for r in dose_rows if r["budget"] == budget and r["cell_id"] == "baseline")
        bs, bc = base["per_config_sum"], base["per_config_natoms"]
        cell_lookup = {
            r["alpha"]: r for r in dose_rows if r["budget"] == budget and r.get("ell") == 4
        }

        def get_int(a):
            r = cell_lookup[a]
            return r["per_config_sum"], r["per_config_natoms"]

    elif owner == "w16/150k":
        budget = "MID"
        base = next(r for r in dose_rows if r["budget"] == budget and r["cell_id"] == "baseline")
        bs, bc = base["per_config_sum"], base["per_config_natoms"]
        cell_lookup = {
            r["alpha"]: r for r in dose_rows if r["budget"] == budget and r.get("ell") == 4
        }

        def get_int(a):
            r = cell_lookup[a]
            return r["per_config_sum"], r["per_config_natoms"]

    else:
        # w20/200k, w24/300k -- from frontier_ell4_sensitivity_raw.jsonl; every alpha row carries
        # its OWN baseline_per_config_sum/natoms (identical across alphas for a given owner).
        cell_lookup = {r["alpha"]: r for r in sens_rows if r["label"] == owner}
        base = cell_lookup[1.0]
        bs, bc = base["baseline_per_config_sum"], base["baseline_per_config_natoms"]

        def get_int(a):
            r = cell_lookup[a]
            return r["per_config_sum"], r["per_config_natoms"]

    d_pt, d_boot = [], []
    for a in ALPHAS:
        if a == 1.0:
            d_pt.append(0.0)
            d_boot.append(np.zeros(N_BOOT))
            continue
        is_, ic = get_int(a)
        pt, boot = _boot_delta(bs, bc, is_, ic)
        d_pt.append(pt)
        d_boot.append(boot)
    return d_pt, d_boot


def build() -> dict:
    dose_rows = read_jsonl(data_path("claim2", "dose_response.jsonl"))
    sens_rows = read_jsonl(data_path("claim2", "frontier_ell4_sensitivity_raw.jsonl"))
    csv_rows = read_csv(data_path("claim2", "frontier_ell4_degree_balanced.csv"))
    summary = read_json(data_path("claim2", "frontier_ell4_degree_balanced_summary.json"))
    target_p_bal = summary["common_P_bal_support"][1]

    pbal_by_owner = {}
    for r in csv_rows:
        pbal_by_owner.setdefault(r["owner"], {})[float(r["alpha"])] = float(r["P_bal"])

    per_alpha_rows = []
    read_off = {}
    boot_at_edge = {}  # owner -> np.ndarray[2000] (interpolated delta bootstrap replicates)

    for owner in OWNER_ORDER:
        d_pt, d_boot = _load_owner_curve(owner, dose_rows, sens_rows)
        pbal = pbal_by_owner[owner]
        for a, pt, boot in zip(ALPHAS, d_pt, d_boot):
            lo, hi = (0.0, 0.0) if a == 1.0 else tuple(np.quantile(boot, [0.025, 0.975]))
            per_alpha_rows.append({
                "owner": owner, "alpha": a, "P_bal": pbal[a],
                "delta": pt, "delta_ci_lo": float(lo), "delta_ci_hi": float(hi),
            })

        x_vals = [pbal[a] for a in ALPHAS]
        pt_read, boot_read = interp_at_x(x_vals, d_pt, d_boot, target_p_bal)
        boot_at_edge[owner] = boot_read
        lo, hi = np.quantile(boot_read, [0.025, 0.975])
        on_measured_point = any(abs(x - target_p_bal) < 1e-9 for x in x_vals)
        read_off[owner] = {
            "delta": pt_read,
            "loss_multiplier": math.exp(pt_read),
            "loss_multiplier_ci95": [float(math.exp(lo)), float(math.exp(hi))],
            "on_measured_point": on_measured_point,
        }

    pairwise = {}
    pair_defs = [
        ("w16/150k", "w10/50k"), ("w20/200k", "w10/50k"), ("w24/300k", "w10/50k"),
        ("w20/200k", "w16/150k"), ("w24/300k", "w16/150k"), ("w24/300k", "w20/200k"),
    ]
    for b, a in pair_defs:
        diff_pt = read_off[b]["delta"] - read_off[a]["delta"]
        diff_boot = boot_at_edge[b] - boot_at_edge[a]
        lo, hi = np.quantile(diff_boot, [0.025, 0.975])
        key = f"{b}_minus_{a}"
        pairwise[key] = {
            "delta_difference": diff_pt,
            "ci95": [float(lo), float(hi)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        }

    return {
        "n_boot": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "common_P_bal_support_edge": target_p_bal,
        "per_alpha": per_alpha_rows,
        "read_off_at_common_support_edge": read_off,
        "pairwise_contrasts_at_common_support_edge": pairwise,
    }


def main():
    result = build()
    out = out_path("claim2", "frontier_ell4_degree_balanced_recomputed.json")
    write_json(out, result)
    print(f"wrote {out} ({len(result['read_off_at_common_support_edge'])} owners, "
          f"{len(result['pairwise_contrasts_at_common_support_edge'])} pairwise contrasts)")


if __name__ == "__main__":
    main()
