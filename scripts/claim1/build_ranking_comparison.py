#!/usr/bin/env python3
"""Regenerate data/claim1/ranking_comparison.csv from RAW inputs already in data/claim1/:

  - owner_manifest.csv          -> force_mse_norm (the real neural-network metric) per
                                    (architecture, budget_tier)
  - matched_compute_m1024_krr_results.csv -> ridgeless, target="common" KRR test_nmse_median
                                    at n in {32,64,128,256,512}, per (architecture, budget_tier)

This is category (a): directly re-derivable. See AUDIT_STAGE2.md.

Reads ONLY from paper_repro/data/. Writes to paper_repro/analysis_out/claim1/ranking_comparison.csv.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.io_utils import data_path, out_path, read_csv, write_csv
from src.krr_auc import auc_log_nmse, rank_ascending

AUC_N_GRID = {32, 64, 128, 256, 512}
AUC_DEFINITION = (
    "trapezoid(log(ridgeless_nmse_median), log(n)) / (log(n_max)-log(n_min)), "
    "n in {32,64,128,256,512}; lower=better; reused from "
    "analyze_four_arch_matched_compute_frontier_kernel.py::auc_summary"
)


def build() -> list[dict]:
    owners = read_csv(data_path("claim1", "owner_manifest.csv"))
    krr = read_csv(data_path("claim1", "matched_compute_m1024_krr_results.csv"))

    force_mse = {}
    for row in owners:
        key = (row["architecture"], row["budget_tier"])
        force_mse[key] = {
            "force_mse_norm": float(row["force_mse_norm"]),
            "owner_tag": row["tag"],
        }

    # Collect ridgeless, target=common test_nmse_median at n in AUC_N_GRID
    curves = {}  # (arch, budget) -> {n: nmse_median}
    for row in krr:
        if row["ridge_setting"] != "ridgeless":
            continue
        if row["target"] != "common":
            continue
        n = int(row["n_train"])
        if n not in AUC_N_GRID:
            continue
        key = (row["architecture"], row["budget_tier"])
        curves.setdefault(key, {})[n] = float(row["test_nmse_median"])

    auc_by_key = {}
    for key, ns in curves.items():
        if set(ns.keys()) != AUC_N_GRID:
            raise ValueError(f"incomplete AUC n-grid for {key}: have {sorted(ns.keys())}")
        auc_by_key[key] = auc_log_nmse(list(ns.keys()), list(ns.values()))

    if set(force_mse.keys()) != set(auc_by_key.keys()):
        raise ValueError(
            "owner_manifest and matched_compute_m1024_krr_results disagree on "
            f"(architecture,budget) coverage: {set(force_mse) ^ set(auc_by_key)}"
        )

    rows_out = []
    for budget in ["LOW", "MID", "HIGH"]:
        keys_this_budget = [k for k in force_mse if k[1] == budget]
        nn_ranked = rank_ascending([(k, force_mse[k]["force_mse_norm"]) for k in keys_this_budget])
        krr_ranked = rank_ascending([(k, auc_by_key[k]) for k in keys_this_budget])
        for k in keys_this_budget:
            arch, _budget = k
            rows_out.append(
                {
                    "budget_tier": budget,
                    "architecture": arch,
                    "owner_tag": force_mse[k]["owner_tag"],
                    "force_mse_norm": force_mse[k]["force_mse_norm"],
                    "nn_rank": nn_ranked[k],
                    "krr_auc_log_nmse_n32_512": auc_by_key[k],
                    "krr_rank": krr_ranked[k],
                    "concordant_with_nn": nn_ranked[k] == krr_ranked[k],
                    "auc_statistic_definition": AUC_DEFINITION,
                }
            )
    return rows_out


def main():
    rows = build()
    fieldnames = [
        "budget_tier",
        "architecture",
        "owner_tag",
        "force_mse_norm",
        "nn_rank",
        "krr_auc_log_nmse_n32_512",
        "krr_rank",
        "concordant_with_nn",
        "auc_statistic_definition",
    ]
    out = out_path("claim1", "ranking_comparison.csv")
    write_csv(out, rows, fieldnames)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
