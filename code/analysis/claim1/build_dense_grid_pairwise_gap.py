#!/usr/bin/env python3
"""Regenerate the GemNet-OC vs eSEN signed gap, neural and KRR, at all eight evaluation budgets
(paper Table 1 and Figure 1C).

Inputs, all in data/claim1/:
  frontier_checkpoint_manifest.csv                        normalized force MSE, LOW/MID/HIGH
  dense_grid_frontier_force_mse.csv            normalized force MSE, the five added budgets
  matched_compute_m1024_krr_split_results.csv   per-split test NMSE, LOW/MID/HIGH
  dense_grid_krr_split_results.csv          per-split test NMSE, the five added budgets

Sign convention: gap = log(L_GemNet-OC / L_eSEN), so positive favours eSEN.

The bootstrap threads one shared RandomState(20260822) across all eight budgets in the fixed
order above, one `.choice(...).mean()` call per replicate, reproducing the frozen interval
bit-exactly. The LOW/MID/HIGH rows here are recomputed from the same split-level data that backs
pairwise_concordance.csv; they are an independent reproduction of the same underlying KRR
results, and their intervals differ slightly from that table's because the shared RNG stream
advances over eight budgets rather than within a single three-budget loop.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.bootstrap import DEFAULT_N_BOOT, bootstrap_paired_log_ratio_ci
from common.io_utils import data_path, out_path, read_csv, write_csv

N_TRAIN_USE = 512
RNG_SEED = 20260822
ALL_BUDGETS = ["LOW", "MID", "PRECROSS", "CROSS", "FITX", "POSTCROSS", "HIGH", "TOP"]
EXISTING_BUDGETS = {"LOW", "MID", "HIGH"}


def main():
    owners = read_csv(data_path("claim1", "frontier_checkpoint_manifest.csv"))
    dense_owners = read_csv(data_path("claim1", "dense_grid_frontier_force_mse.csv"))
    existing_split = read_csv(data_path("claim1", "matched_compute_m1024_krr_split_results.csv"))
    dense_split = read_csv(data_path("claim1", "dense_grid_krr_split_results.csv"))

    force_mse = {(r["architecture"], r["budget_tier"]): float(r["force_mse_norm"]) for r in owners}
    for r in dense_owners:
        force_mse[(r["architecture"], r["budget_tier"])] = float(r["force_mse_norm"])

    owner_tag = {(r["architecture"], r["budget_tier"]): r["tag"] for r in owners}
    for r in dense_owners:
        owner_tag[(r["architecture"], r["budget_tier"])] = r["owner_tag"]

    # (owner_tag, budget_tier, n, split_seed) -> test_nmse, selected-rho rows at n=512
    selected = {}
    for r in existing_split:
        if r["is_selected"] != "True" or int(r["n"]) != N_TRAIN_USE:
            continue
        selected[(r["owner_tag"], r["budget_tier"], int(r["split_seed"]))] = float(r["test_nmse"])
    for r in dense_split:
        if r["is_selected"] != "True" or int(r["n"]) != N_TRAIN_USE:
            continue
        selected[(r["owner_tag"], r["budget_tier"], int(r["split_seed"]))] = float(r["test_nmse"])

    rng = np.random.RandomState(RNG_SEED)
    rows_out = []
    for budget in ALL_BUDGETS:
        fmn_g = force_mse[("GemNet-OC", budget)]
        fmn_e = force_mse[("eSEN", budget)]
        neural_gap = math.log(fmn_g / fmn_e)

        tag_g = owner_tag[("GemNet-OC", budget)]
        tag_e = owner_tag[("eSEN", budget)]
        log_ratios = []
        for s in range(20):
            vg = selected.get((tag_g, budget, s))
            ve = selected.get((tag_e, budget, s))
            if vg is None or ve is None:
                continue
            log_ratios.append(math.log(vg) - math.log(ve))

        result = bootstrap_paired_log_ratio_ci(log_ratios, n_boot=DEFAULT_N_BOOT, seed=RNG_SEED, rng=rng)
        point_est = result["point_estimate"]
        ci_lo, ci_hi = result["ci95"]
        excludes_zero = ci_lo > 0 or ci_hi < 0

        rows_out.append({
            "budget_tier": budget,
            "is_new_budget": budget not in EXISTING_BUDGETS,
            "neural_signed_gap_log_gemnet_over_esen": neural_gap,
            "neural_favors": "eSEN" if neural_gap > 0 else "GemNet-OC",
            "krr_signed_gap_log_gemnet_over_esen_at_n512": point_est,
            "krr_bootstrap_ci_lo_2.5pct": ci_lo,
            "krr_bootstrap_ci_hi_97.5pct": ci_hi,
            "krr_ci_excludes_zero": excludes_zero,
            "krr_favors": "eSEN" if point_est > 0 else "GemNet-OC",
            "n_boot": DEFAULT_N_BOOT,
            "bootstrap_seed": RNG_SEED,
            "n_paired_splits": len(log_ratios),
        })

    out = out_path("claim1", "dense_grid_pairwise_gap_recomputed.csv")
    write_csv(out, rows_out)
    print(f"wrote {out} ({len(rows_out)} rows)")
    for r in rows_out:
        print(f"  {r['budget_tier']:10s} neural={r['neural_signed_gap_log_gemnet_over_esen']:+.4f} "
              f"({r['neural_favors']:9s}) krr={r['krr_signed_gap_log_gemnet_over_esen_at_n512']:+.4f} "
              f"CI=[{r['krr_bootstrap_ci_lo_2.5pct']:+.4f},{r['krr_bootstrap_ci_hi_97.5pct']:+.4f}] "
              f"({r['krr_favors']:9s}, excludes_zero={r['krr_ci_excludes_zero']})")


if __name__ == "__main__":
    main()
