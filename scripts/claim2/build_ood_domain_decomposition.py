#!/usr/bin/env python3
"""Regenerate the last 2 of the 8 frozen OOD tables in data/claim2/: a numpy port of
analysis_outputs/esen_irrep_ood_2026_08_22/eval/robustness_analysis.py's PART 2 (domain
decomposition by chemistry family, and the w24-vs-w10 "shared family" positive control).

STAGE-2 GAP CLOSED (see AUDIT_STAGE2.md): a prior pass concluded this was NOT reproducible,
because the per-configuration `data_id` (chemistry-family) label these two tables stratify by
is not a field inside ood_raw_per_config/lmax4_block9_intervention.jsonl's rows -- that pass
checked those rows' own field names and, finding no `data_id`, stopped there.

A second, harder look (per this Stage-2 pass's explicit instruction to re-check before declaring
something unreproducible) found `data_id` does exist on disk, one level up: robustness_analysis.py
itself reads it not from the per-config JSONL rows but from the POOL MANIFESTS --
dataset/ood_pool_manifest.json["pools"]["16384"] (list of {dataset_index, source, data_id,
natoms}, OOD pool) and id_pool/id_pool_manifest.json["pools"]["16384"] (ID/Neutral-val pool) --
aligned by POOL ORDER (position i in that list == position i in the per_config_sum/natoms arrays;
ood_pool_provenance.json already documents both pools as verified nested/complete/unique, i.e.
the M=1024 prefix of the M=16384 list is exactly the M=1024 pool, etc.). Neither manifest's
"pools" field had been copied into paper_repro/data/ (ood_pool_provenance.json explicitly listed
them as "referenced, not copied" due to their raw size, 6.4-6.8MB each, when a prior pass copied
that provenance file) -- so the field was never missing, only its container had not yet been
brought into the clean layer. This Stage-2 pass extracted JUST the `data_id` list for the M=16384
pool from each manifest (16384 short strings each, ~400KB total) into
data/claim2/ood_domain_labels.json, with provenance to the exact source path/field. Sanity check:
the per-family counts in that extraction (elytes=10867, biomolecules=3286, reactivity=1749,
metal_complexes=32, orbnet_denali=212+spice=104+geom_orca6=86+ani2x=48=450 "neutral-like") exactly
match the frozen ood_domain_decomposition.csv's own n_configs column for every stratum -- this is
the correct alignment, not a guess.

Both tables use src/ood_analysis.py's SAME bit-exact bootstrap machinery as
build_ood_curves_and_contrasts.py / build_ood_absolute_effect_robustness.py
(np.random.default_rng(BOOT_SEED_ROBUST=20260823), fresh per (m, mask) cell, matching
robustness_analysis.py's boot_L() exactly), restricted to the boolean stratum mask BEFORE
resampling (robustness_analysis.py:52-63's `idx_mask` parameter) -- ports delta_at_read_x()
(lines 96-118) and the two PART-2 loops (lines 165-249) directly.

Writes into analysis_out/claim2/:
  - ood_domain_decomposition.csv          (ports strata_rows / robustness/domain_decomposition_M16384.csv)
  - ood_shared_family_positive_control.csv (ports contrast_rows / robustness/w24_vs_w10_contrast_by_stratum.csv)
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
