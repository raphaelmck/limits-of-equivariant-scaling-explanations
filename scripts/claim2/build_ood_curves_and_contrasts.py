#!/usr/bin/env python3
"""Regenerate the intervention curves and contrasts on the two evaluation populations, from the
per-configuration arrays in data/claim2/ood_raw_per_config/.

Writes into analysis_out/claim2/:
  ood_full_alpha_curves.csv        damage curve per checkpoint, population, and alpha
  ood_G_m_convergence.csv          high-minus-low contrast as a function of pool size
  ood_interaction_w24_vs_w10.csv   the same contrast across populations
  ood_lmax2_vs_lmax4_baseline.csv  matched-compute log(L_2 / L_4), the source of Figure 2B

The contrasts are taken at the common P_bal support edge (src/ood_analysis.interp_at_x), not at
a fixed alpha: two checkpoints reach different perturbation magnitudes at the same alpha, so a
fixed-alpha contrast would compare unequal interventions.

The bootstrap index matrix depends only on the pool size and the seed, never on the checkpoint,
population, or alpha, so it is generated once per pool size and reused, which keeps the pure
re-aggregation tractable at pools up to 16384 configurations.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.io_utils import data_path, out_path, read_json, read_jsonl, write_csv
from src.ood_analysis import (
    ALPHAS,
    BOOT_SEED_ANALYZE,
    N_BOOT,
    OWNER_ORDER,
    POOL_SIZES,
    boot_L_all,
    ci95,
    ci_excludes_zero,
    common_support_hi,
    compute_p_bal_curves,
    find_matched_lmax2_tag,
    gen_boot_indices,
    index_lmax2,
    index_lmax4,
    interp_at_x,
    weighted_L,
)

RAW = lambda *p: data_path("claim2", "ood_raw_per_config", *p)


def main():
    run_meta = read_json(RAW("run_meta.json"))
    lmax4 = index_lmax4(read_jsonl(RAW("lmax4_block9_intervention.jsonl")))
    lmax2 = index_lmax2(read_jsonl(RAW("lmax2_baseline.jsonl")))

    p_bal_curves = compute_p_bal_curves(run_meta)
    read_x = common_support_hi(p_bal_curves)
    print(f"P_bal common support: [0, {read_x:.5f}]")

    full_curve_rows = []
    # (tag, domain, m) -> (delta_point_at_read_x, delta_boot_at_read_x)
    delta_at_read_x: dict[tuple[str, str, int], tuple[float, list[float]]] = {}
    h_rows = []

    for m in POOL_SIZES:
        idx = gen_boot_indices(m, N_BOOT, BOOT_SEED_ANALYZE)

        for tag in OWNER_ORDER:
            # --- matched lmax2 tag for this owner (used below for the H_force comparison) ---
            l2_tag = find_matched_lmax2_tag(lmax2, tag)

            for domain in ("ID", "OOD"):
                cells = lmax4[(tag, domain)]
                base = cells[1.00]
                bs = base["baseline_per_config_sum"][:m]
                bc = base["baseline_per_config_natoms"][:m]
                Lb_boot = boot_L_all(bs, bc, idx)
                Lb_pt = weighted_L(bs, bc)
                p_bal_curve = p_bal_curves[(tag, domain)]

                x_vals, d_boot_rows, d_pt = [], [], []
                for a in ALPHAS:
                    r = cells[a]
                    si = r["per_config_sum"][:m]
                    ci = r["per_config_natoms"][:m]
                    Li_boot = boot_L_all(si, ci, idx)
                    Li_pt = weighted_L(si, ci)
                    d_boot = [math.log(li) - math.log(lb) for li, lb in zip(Li_boot, Lb_boot)]
                    d_pt_val = math.log(Li_pt) - math.log(Lb_pt)
                    d_boot_rows.append(d_boot)
                    d_pt.append(d_pt_val)
                    x_vals.append(p_bal_curve[a])
                    lo, hi = ci95(d_boot)
                    full_curve_rows.append({
                        "tag": tag, "domain": domain, "M": m, "alpha": a,
                        "P_bal": p_bal_curve[a], "L_baseline": Lb_pt, "L_int": Li_pt,
                        "delta": d_pt_val, "delta_ci_lo": lo, "delta_ci_hi": hi,
                        "loss_multiplier": math.exp(d_pt_val),
                    })

                d_read_pt, d_read_boot = interp_at_x(x_vals, d_pt, d_boot_rows, read_x)
                delta_at_read_x[(tag, domain, m)] = (d_read_pt, d_read_boot)

            # --- lmax2 vs lmax4 baseline (H_force), both domains, this tag/m ---
            for domain in ("ID", "OOD"):
                l4rec = lmax4[(tag, domain)][1.00]
                l2rec = lmax2[(l2_tag, domain)]
                l4_s, l4_c = l4rec["baseline_per_config_sum"][:m], l4rec["baseline_per_config_natoms"][:m]
                l2_s, l2_c = l2rec["per_config_sum"][:m], l2rec["per_config_natoms"][:m]
                L4_boot = boot_L_all(l4_s, l4_c, idx)
                L4_pt = weighted_L(l4_s, l4_c)
                L2_boot = boot_L_all(l2_s, l2_c, idx)
                L2_pt = weighted_L(l2_s, l2_c)
                h_boot = [math.log(l2) - math.log(l4) for l2, l4 in zip(L2_boot, L4_boot)]
                h_pt = math.log(L2_pt) - math.log(L4_pt)
                h_lo, h_hi = ci95(h_boot)
                h_rows.append({
                    "lmax4_tag": tag, "lmax2_tag": l2_tag, "domain": domain, "M": m,
                    "L_lmax4": L4_pt, "L_lmax2": L2_pt, "H_force": h_pt,
                    "H_ci_lo": h_lo, "H_ci_hi": h_hi,
                    "lmax4_advantage_significant": h_lo > 0,
                })

    write_csv(out_path("claim2", "ood_full_alpha_curves.csv"), full_curve_rows)
    write_csv(out_path("claim2", "ood_lmax2_vs_lmax4_baseline.csv"), h_rows)

    # ---------------- G_m = Delta_OOD - Delta_ID, per owner, per nested M ----------------
    g_rows = []
    for tag in OWNER_ORDER:
        for m in POOL_SIZES:
            d_ood_pt, d_ood_boot = delta_at_read_x[(tag, "OOD", m)]
            d_id_pt, d_id_boot = delta_at_read_x[(tag, "ID", m)]
            g_pt = d_ood_pt - d_id_pt
            g_boot = [o - i for o, i in zip(d_ood_boot, d_id_boot)]
            lo, hi = ci95(g_boot)
            g_rows.append({
                "tag": tag, "M": m, "G": g_pt, "G_ci_lo": lo, "G_ci_hi": hi,
                "ci_excludes_zero": ci_excludes_zero(lo, hi),
            })
    write_csv(out_path("claim2", "ood_G_m_convergence.csv"), g_rows)

    # ---------------- low-vs-high interaction ----------------
    interaction_rows = []
    for m in POOL_SIZES:
        ood_hi_pt, ood_hi_boot = delta_at_read_x[("w24/300k", "OOD", m)]
        ood_lo_pt, ood_lo_boot = delta_at_read_x[("w10/50k", "OOD", m)]
        id_hi_pt, id_hi_boot = delta_at_read_x[("w24/300k", "ID", m)]
        id_lo_pt, id_lo_boot = delta_at_read_x[("w10/50k", "ID", m)]
        inter_pt = (ood_hi_pt - ood_lo_pt) - (id_hi_pt - id_lo_pt)
        inter_boot = [
            (oh - ol) - (ih - il)
            for oh, ol, ih, il in zip(ood_hi_boot, ood_lo_boot, id_hi_boot, id_lo_boot)
        ]
        lo, hi = ci95(inter_boot)
        interaction_rows.append({
            "M": m, "interaction": inter_pt, "ci_lo": lo, "ci_hi": hi,
            "ci_excludes_zero": ci_excludes_zero(lo, hi),
        })
    write_csv(out_path("claim2", "ood_interaction_w24_vs_w10.csv"), interaction_rows)

    print("wrote ood_full_alpha_curves.csv, ood_G_m_convergence.csv, "
          "ood_interaction_w24_vs_w10.csv, ood_lmax2_vs_lmax4_baseline.csv")


if __name__ == "__main__":
    main()
