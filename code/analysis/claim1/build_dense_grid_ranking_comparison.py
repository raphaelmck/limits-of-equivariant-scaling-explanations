#!/usr/bin/env python3
"""Regenerate the four-architecture neural-versus-KRR ranking at all eight evaluation budgets,
extending build_ranking_comparison.py's LOW/MID/HIGH procedure to the five added budgets.

Same learning-curve-area statistic (common/krr_auc.py), same ascending-rank convention, same
training sizes {32, 64, 128, 256, 512}, ridgeless curves against the common force target.

Inputs, all in data/claim1/:
  frontier_checkpoint_manifest.csv                      normalized force MSE, LOW/MID/HIGH
  dense_grid_frontier_force_mse.csv          normalized force MSE, the five added budgets
  matched_compute_m1024_krr_results.csv   KRR curves, LOW/MID/HIGH
  dense_grid_krr_results.csv              KRR curves, the five added budgets

Deterministic: no bootstrap is involved, so this reproduces the frozen table exactly.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.io_utils import data_path, out_path, read_csv, write_csv
from common.krr_auc import auc_log_nmse, rank_ascending

AUC_N_GRID = {32, 64, 128, 256, 512}
ALL_BUDGETS = ["LOW", "MID", "PRECROSS", "CROSS", "FITX", "POSTCROSS", "HIGH", "TOP"]
AUC_DEFINITION = (
    "trapezoid(log(ridgeless_nmse_median), log(n)) / (log(n_max)-log(n_min)), "
    "n in {32,64,128,256,512}; lower=better; reused from "
    "analyze_four_arch_matched_compute_frontier_kernel.py::auc_summary (same statistic as "
    "ranking_comparison.csv, extended here to all 8 budgets)"
)


def build() -> list[dict]:
    owners = read_csv(data_path("claim1", "frontier_checkpoint_manifest.csv"))
    dense_owners = read_csv(data_path("claim1", "dense_grid_frontier_force_mse.csv"))
    krr = read_csv(data_path("claim1", "matched_compute_m1024_krr_results.csv"))
    dense_krr = read_csv(data_path("claim1", "dense_grid_krr_results.csv"))

    force_mse = {}
    for row in owners:
        key = (row["architecture"], row["budget_tier"])
        force_mse[key] = {"force_mse_norm": float(row["force_mse_norm"]), "owner_tag": row["tag"]}
    for row in dense_owners:
        key = (row["architecture"], row["budget_tier"])
        force_mse[key] = {"force_mse_norm": float(row["force_mse_norm"]), "owner_tag": row["owner_tag"]}

    curves = {}  # (arch, budget) -> {n: nmse_median}
    for row in list(krr) + list(dense_krr):
        if row["ridge_setting"] != "ridgeless" or row["target"] != "common":
            continue
        n = int(row["n_train"])
        if n not in AUC_N_GRID:
            continue
        key = (row["architecture"], row["budget_tier"])
        curves.setdefault(key, {})[n] = float(row["test_nmse_median"])

    auc_by_key = {}
    for key, ns in curves.items():
        if set(ns.keys()) != AUC_N_GRID:
            continue  # incomplete curve for this key (shouldn't happen for the 8 budgets x 4 archs)
        auc_by_key[key] = auc_log_nmse(list(ns.keys()), list(ns.values()))

    rows_out = []
    for budget in ALL_BUDGETS:
        keys_this_budget = [k for k in force_mse if k[1] == budget and k in auc_by_key]
        if len(keys_this_budget) != 4:
            raise ValueError(f"budget {budget}: expected 4 architectures, found {len(keys_this_budget)}")
        nn_ranked = rank_ascending([(k, force_mse[k]["force_mse_norm"]) for k in keys_this_budget])
        krr_ranked = rank_ascending([(k, auc_by_key[k]) for k in keys_this_budget])
        for k in keys_this_budget:
            arch, _budget = k
            rows_out.append({
                "budget_tier": budget,
                "architecture": arch,
                "owner_tag": force_mse[k]["owner_tag"],
                "force_mse_norm": force_mse[k]["force_mse_norm"],
                "nn_rank": nn_ranked[k],
                "krr_auc_log_nmse_n32_512": auc_by_key[k],
                "krr_rank": krr_ranked[k],
                "concordant_with_nn": nn_ranked[k] == krr_ranked[k],
                "auc_statistic_definition": AUC_DEFINITION,
            })
    return rows_out


def main():
    rows = build()
    fieldnames = [
        "budget_tier", "architecture", "owner_tag", "force_mse_norm",
        "nn_rank", "krr_auc_log_nmse_n32_512", "krr_rank", "concordant_with_nn",
        "auc_statistic_definition",
    ]
    out = out_path("claim1", "dense_grid_ranking_comparison_recomputed.csv")
    write_csv(out, rows, fieldnames)
    print(f"wrote {out} ({len(rows)} rows)")
    for budget in ALL_BUDGETS:
        tier = [r for r in rows if r["budget_tier"] == budget]
        tier.sort(key=lambda r: r["nn_rank"])
        line = " ".join(f"{r['architecture']}(nn={r['nn_rank']},krr={r['krr_rank']})" for r in tier)
        print(f"  {budget:10s} {line}")


if __name__ == "__main__":
    main()
