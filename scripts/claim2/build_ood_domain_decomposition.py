#!/usr/bin/env python3
"""Regenerate the chemistry-family decomposition of the intervention effect and the shared-source
positive control (paper appendix, chemistry-family transfer).

The per-configuration chemistry-family label is not a field on the per-configuration rows; it
comes from the evaluation-pool manifests, aligned by pool order -- position i in the label list
is position i in the per_config_sum and per_config_natoms arrays, both pools being verified
nested, complete, and duplicate-free. The labels for the 16384-configuration pool are checked in
as data/claim2/ood_domain_labels.json. Their per-family counts (electrolytes 10867, biomolecules
3286, reactivity 1749, metal complexes 32, and 450 shared-source configurations) match the frozen
table's own counts for every stratum, which confirms the alignment.

The bootstrap is the same machinery as the sibling scripts, default_rng(20260823) constructed
fresh per (pool size, stratum) cell, restricted to the stratum mask before resampling.

Writes into analysis_out/claim2/:
  ood_domain_decomposition.csv
  ood_shared_family_positive_control.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.io_utils import data_path, out_path, read_json, read_jsonl, write_csv
from src.ood_analysis import (
    ALPHAS,
    BOOT_SEED_ROBUST,
    N_BOOT,
    OWNER_ORDER,
    ci95,
    ci_excludes_zero,
    common_support_hi,
    compute_p_bal_curves,
    index_lmax4,
    interp_at_x,
    weighted_L,
)

RAW = lambda *p: data_path("claim2", "ood_raw_per_config", *p)
MIN_STRATUM_N = 200  # ports robustness_analysis.py's MIN_STRATUM_N

NEUTRAL_LIKE = {"ani2x", "spice", "geom_orca6", "orbnet_denali"}


def delta_at_read_x_masked(cells, p_bal_curve, read_x, seed, m, mask):
    """Ports robustness_analysis.py's delta_at_read_x(), restricted to a boolean stratum mask
    (applied identically to baseline and every alpha cell, then bootstrapped within that
    restricted set -- mask is applied BEFORE the m-slice/bootstrap, matching
    robustness_analysis.py:96-107's `if idx_mask is not None: s, c = s[idx_mask], c[idx_mask]`
    inside boot_L, called with m=16384 for all domain-decomposition uses)."""
    base = cells[1.00]
    bs = np.asarray(base["baseline_per_config_sum"], dtype=np.float64)[:m][mask]
    bc = np.asarray(base["baseline_per_config_natoms"], dtype=np.float64)[:m][mask]
    n = len(bs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    Lb_boot = bs[idx].sum(axis=1) / bc[idx].sum(axis=1)
    Lb_pt = weighted_L(bs, bc)

    x_vals, d_boot_rows, d_pt = [], [], []
    for a in ALPHAS:
        r = cells[a]
        si = np.asarray(r["per_config_sum"], dtype=np.float64)[:m][mask]
        ci = np.asarray(r["per_config_natoms"], dtype=np.float64)[:m][mask]
        rng_a = np.random.default_rng(seed)
        idx_a = rng_a.integers(0, n, size=(N_BOOT, n))
        Li_boot = si[idx_a].sum(axis=1) / ci[idx_a].sum(axis=1)
        Li_pt = weighted_L(si, ci)
        d_boot_rows.append(np.log(Li_boot) - np.log(Lb_boot))
        d_pt.append(float(np.log(Li_pt) - np.log(Lb_pt)))
        x_vals.append(p_bal_curve[a])

    d_read_pt, d_read_boot = interp_at_x(x_vals, d_pt, d_boot_rows, read_x)
    return Lb_pt, d_read_pt, d_read_boot, n


def main():
    run_meta = read_json(RAW("run_meta.json"))
    lmax4 = index_lmax4(read_jsonl(RAW("lmax4_block9_intervention.jsonl")))
    p_bal_curves = compute_p_bal_curves(run_meta)
    read_x = common_support_hi(p_bal_curves)

    labels = read_json(data_path("claim2", "ood_domain_labels.json"))
    ood_data_id = np.array(labels["ood_data_id_M16384"])
    id_data_id = np.array(labels["id_data_id_M16384"])

    strata_defs = {
        "elytes": ("OOD", ood_data_id == "elytes"),
        "biomolecules": ("OOD", ood_data_id == "biomolecules"),
        "reactivity": ("OOD", ood_data_id == "reactivity"),
        "metal_complexes": ("OOD", ood_data_id == "metal_complexes"),
        "pooled_neutral_like_sources_in_ValComp": ("OOD", np.isin(ood_data_id, list(NEUTRAL_LIKE))),
        "Neutral_val_pooled_(ID_reference)": ("ID", np.ones(len(id_data_id), dtype=bool)),
    }

    strata_rows = []
    strata_delta_boot = {}  # (stratum, tag) -> np.ndarray delta bootstrap replicates
    for name, (domain, mask) in strata_defs.items():
        n = int(mask.sum())
        insufficient = n < MIN_STRATUM_N
        for tag in OWNER_ORDER:
            if insufficient:
                strata_rows.append({
                    "stratum": name, "domain": domain, "tag": tag, "n_configs": n,
                    "insufficient_n": True, "L_base": "", "delta": "",
                    "delta_ci_lo": "", "delta_ci_hi": "",
                    "A": "", "A_ci_lo": "", "A_ci_hi": "",
                })
                continue
            cells = lmax4[(tag, domain)]
            p_bal_curve = p_bal_curves[(tag, domain)]
            Lb_pt, d_pt, d_boot, n_b = delta_at_read_x_masked(
                cells, p_bal_curve, read_x, BOOT_SEED_ROBUST, 16384, mask)
            Li_pt = Lb_pt * float(np.exp(d_pt))
            A_pt = Li_pt - Lb_pt
            Lb_boot_for_A = None  # A's CI needs Lb_boot; recompute cheaply below
            # Recompute Lb_boot once more (same seed/mask -> identical draw) to get A_boot.
            base = cells[1.00]
            bs = np.asarray(base["baseline_per_config_sum"], dtype=np.float64)[:16384][mask]
            bc = np.asarray(base["baseline_per_config_natoms"], dtype=np.float64)[:16384][mask]
            rng = np.random.default_rng(BOOT_SEED_ROBUST)
            idx = rng.integers(0, n_b, size=(N_BOOT, n_b))
            Lb_boot = bs[idx].sum(axis=1) / bc[idx].sum(axis=1)
            A_boot = Lb_boot * np.exp(d_boot) - Lb_boot
            d_lo, d_hi = ci95(d_boot)
            a_lo, a_hi = ci95(A_boot)
            strata_delta_boot[(name, tag)] = d_boot
            strata_rows.append({
                "stratum": name, "domain": domain, "tag": tag, "n_configs": n_b,
                "insufficient_n": False, "L_base": Lb_pt, "delta": d_pt,
                "delta_ci_lo": d_lo, "delta_ci_hi": d_hi,
                "A": A_pt, "A_ci_lo": a_lo, "A_ci_hi": a_hi,
            })
    write_csv(out_path("claim2", "ood_domain_decomposition.csv"), strata_rows)

    # ---------------- w24-vs-w10 contrast per stratum (shared-family positive control) --------
    contrast_rows = []
    for name, (domain, mask) in strata_defs.items():
        n = int(mask.sum())
        if n < MIN_STRATUM_N:
            contrast_rows.append({
                "stratum": name, "n_configs": n, "insufficient_n": True,
                "contrast_w24_minus_w10": "", "ci_lo": "", "ci_hi": "", "ci_excludes_zero": "",
            })
            continue
        d_hi = strata_delta_boot[(name, "w24/300k")]
        d_lo = strata_delta_boot[(name, "w10/50k")]
        diff_boot = d_hi - d_lo
        row_hi = next(r for r in strata_rows if r["stratum"] == name and r["tag"] == "w24/300k")
        row_lo = next(r for r in strata_rows if r["stratum"] == name and r["tag"] == "w10/50k")
        diff_pt = row_hi["delta"] - row_lo["delta"]
        lo, hi = ci95(diff_boot)
        contrast_rows.append({
            "stratum": name, "n_configs": n, "insufficient_n": False,
            "contrast_w24_minus_w10": diff_pt, "ci_lo": lo, "ci_hi": hi,
            "ci_excludes_zero": ci_excludes_zero(lo, hi),
        })
    write_csv(out_path("claim2", "ood_shared_family_positive_control.csv"), contrast_rows)

    print("wrote ood_domain_decomposition.csv, ood_shared_family_positive_control.csv")


if __name__ == "__main__":
    main()
