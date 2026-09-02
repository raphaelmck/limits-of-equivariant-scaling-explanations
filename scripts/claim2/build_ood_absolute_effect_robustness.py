#!/usr/bin/env python3
"""Regenerate the absolute-scale robustness check for the intervention effect: the same
comparison expressed as an absolute loss increase rather than a log ratio.

A = L_base * (exp(Delta) - 1), where Delta is the same P_bal-matched interpolated log ratio used
throughout (src/ood_analysis.py), so the absolute statistic rests on the identical
interpolation as the relative one.

Writes into analysis_out/claim2/:
  ood_absolute_effect_robustness.csv
  ood_absolute_effect_A_OOD_minus_A_Neutral.csv

The chemistry-family decomposition is built by build_ood_domain_decomposition.py.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.io_utils import data_path, out_path, read_json, read_jsonl, write_csv
from src.ood_analysis import (
    ALPHAS,
    BOOT_SEED_ROBUST,
    N_BOOT,
    OWNER_ORDER,
    POOL_SIZES,
    boot_L_all,
    ci95,
    ci_excludes_zero,
    common_support_hi,
    compute_p_bal_curves,
    gen_boot_indices,
    index_lmax4,
    interp_at_x,
    weighted_L,
)

RAW = lambda *p: data_path("claim2", "ood_raw_per_config", *p)


def main():
    run_meta = read_json(RAW("run_meta.json"))
    lmax4 = index_lmax4(read_jsonl(RAW("lmax4_block9_intervention.jsonl")))

    p_bal_curves = compute_p_bal_curves(run_meta)
    read_x = common_support_hi(p_bal_curves)

    part1_rows = []
    # (tag, domain, m) -> (A_pt, A_boot)
    A_at_read_x: dict[tuple[str, str, int], tuple[float, list[float]]] = {}

    for m in POOL_SIZES:
        idx = gen_boot_indices(m, N_BOOT, BOOT_SEED_ROBUST)
        for tag in OWNER_ORDER:
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
                    d_boot_rows.append([math.log(li) - math.log(lb) for li, lb in zip(Li_boot, Lb_boot)])
                    d_pt.append(math.log(Li_pt) - math.log(Lb_pt))
                    x_vals.append(p_bal_curve[a])

                d_pt_read, d_boot_read = interp_at_x(x_vals, d_pt, d_boot_rows, read_x)

                Li_pt_read = Lb_pt * math.exp(d_pt_read)
                Li_boot_read = [Lb_boot[b] * math.exp(d_boot_read[b]) for b in range(N_BOOT)]
                A_pt = Li_pt_read - Lb_pt
                A_boot = [Li_boot_read[b] - Lb_boot[b] for b in range(N_BOOT)]
                A_at_read_x[(tag, domain, m)] = (A_pt, A_boot)

                rel_lo, rel_hi = ci95(d_boot_read)
                a_lo, a_hi = ci95(A_boot)
                part1_rows.append({
                    "tag": tag, "domain": domain, "M": m, "n_configs": m,
                    "P_bal_read_x": read_x,
                    "L_base": Lb_pt, "L_intervened": Li_pt_read,
                    "relative_effect_log_ratio": d_pt_read,
                    "relative_effect_ci_lo": rel_lo, "relative_effect_ci_hi": rel_hi,
                    "absolute_excess_A": A_pt,
                    "absolute_excess_A_ci_lo": a_lo, "absolute_excess_A_ci_hi": a_hi,
                })

    write_csv(out_path("claim2", "ood_absolute_effect_robustness.csv"), part1_rows)

    # ---------------- A_OOD - A_Neutral(ID), nested M, paired per owner ----------------
    a_diff_rows = []
    for tag in OWNER_ORDER:
        for m in POOL_SIZES:
            a_ood_pt, a_ood_boot = A_at_read_x[(tag, "OOD", m)]
            a_id_pt, a_id_boot = A_at_read_x[(tag, "ID", m)]
            diff_pt = a_ood_pt - a_id_pt
            diff_boot = [o - i for o, i in zip(a_ood_boot, a_id_boot)]
            lo, hi = ci95(diff_boot)
            a_diff_rows.append({
                "tag": tag, "M": m, "A_OOD": a_ood_pt, "A_ID_Neutral": a_id_pt,
                "A_OOD_minus_A_Neutral": diff_pt, "ci_lo": lo, "ci_hi": hi,
                "ci_excludes_zero": ci_excludes_zero(lo, hi),
            })
    write_csv(out_path("claim2", "ood_absolute_effect_A_OOD_minus_A_Neutral.csv"), a_diff_rows)

    print("wrote ood_absolute_effect_robustness.csv, ood_absolute_effect_A_OOD_minus_A_Neutral.csv")


if __name__ == "__main__":
    main()
