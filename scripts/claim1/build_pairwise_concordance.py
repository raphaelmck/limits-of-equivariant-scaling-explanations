#!/usr/bin/env python3
"""Reproduce the pairwise neural-versus-KRR concordance table, including its bootstrap intervals.

Procedure, for each (budget, architecture A, architecture B) pair among the twelve frontier
checkpoints:
  1. Take the selected-ridge row at n_train=512 for each of the 20 split seeds, for both
     architectures.
  2. log_ratio[s] = log(NMSE_A[s]) - log(NMSE_B[s]) for each split where both arms have a
     selected row, matched by split seed -- this is what makes the comparison paired.
  3. The point estimate is the mean over splits; KRR favours whichever architecture has the
     lower NMSE.
  4. The interval comes from one RandomState(20260822) created before the loop and threaded
     through all 18 pairs in file order, with one `.choice(...).mean()` call per replicate. The
     shared stream is what makes this bit-exact against the frozen table; re-seeding per pair or
     vectorizing across replicates would both desynchronize it.
  5. A pair is concordant or discordant when the interval excludes zero and the sign agrees or
     disagrees with the neural ordering, and unresolved otherwise.

Writes both a direction-only check against the frozen file and the full recomputed table with
intervals.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.bootstrap import DEFAULT_N_BOOT, bootstrap_paired_log_ratio_ci
from src.io_utils import data_path, out_path, read_csv, write_csv

N_TRAIN_USE = 512
RNG_SEED = 20260822  # documented upstream seed -- see module docstring; now bit-exactly matched


def build_direction_check() -> list[dict]:
    owners = read_csv(data_path("claim1", "owner_manifest.csv"))
    krr = read_csv(data_path("claim1", "matched_compute_m1024_krr_results.csv"))
    frozen = read_csv(data_path("claim1", "pairwise_concordance.csv"))

    force_mse = {(r["architecture"], r["budget_tier"]): float(r["force_mse_norm"]) for r in owners}

    nmse_512 = {}
    for row in krr:
        if row["ridge_setting"] != "ridgeless" or row["target"] != "common":
            continue
        if int(row["n_train"]) != 512:
            continue
        key = (row["architecture"], row["budget_tier"])
        nmse_512[key] = float(row["test_nmse_median"])

    rows_out = []
    for frow in frozen:
        budget = frow["budget_tier"]
        a, b = frow["architecture_a"], frow["architecture_b"]
        nn_favors_recomputed = a if force_mse[(a, budget)] < force_mse[(b, budget)] else b
        log_ratio_recomputed = math.log(nmse_512[(a, budget)] / nmse_512[(b, budget)])
        krr_favors_recomputed = a if log_ratio_recomputed < 0 else b
        rows_out.append(
            {
                "budget_tier": budget,
                "architecture_a": a,
                "architecture_b": b,
                "nn_favors_frozen": frow["nn_favors"],
                "nn_favors_recomputed": nn_favors_recomputed,
                "nn_favors_match": nn_favors_recomputed == frow["nn_favors"],
                "krr_favors_point_estimate_frozen": frow["krr_favors_point_estimate"],
                "krr_favors_point_estimate_recomputed": krr_favors_recomputed,
                "krr_favors_match": krr_favors_recomputed == frow["krr_favors_point_estimate"],
                "log_ratio_recomputed_from_median_at_n512": log_ratio_recomputed,
                "log_ratio_mean_frozen_at_n512": frow["log_ratio_mean_a_over_b_at_n512"],
                "note": (
                    "Direction-only check from aggregated medians (kept for backward "
                    "compatibility); see pairwise_concordance_recomputed.csv for the full "
                    "split-level paired-bootstrap CI."
                ),
            }
        )
    return rows_out


def build_recomputed_ci() -> list[dict]:
    owners = read_csv(data_path("claim1", "owner_manifest.csv"))
    frozen = read_csv(data_path("claim1", "pairwise_concordance.csv"))
    split_rows = read_csv(data_path("claim1", "matched_compute_m1024_krr_split_results.csv"))

    # (architecture, budget_tier) -> owner_tag
    owner_tag = {(r["architecture"], r["budget_tier"]): r["tag"] for r in owners}
    force_mse = {(r["architecture"], r["budget_tier"]): float(r["force_mse_norm"]) for r in owners}

    # (owner_tag, n_train, split_seed) -> test_nmse, restricted to selected-rho rows at n=512
    selected = {}
    for r in split_rows:
        if r["is_selected"] != "True":
            continue
        if int(r["n"]) != N_TRAIN_USE:
            continue
        key = (r["owner_tag"], int(r["split_seed"]))
        selected[key] = float(r["test_nmse"])

    # Single shared RandomState, created once, threaded through all 18 pairs IN FILE ORDER
    # (confirmed == upstream's budget/itertools.combinations(OWNER_TAGS order) order) so its
    # state advances identically to the upstream script's -- required for bit-exactness (see
    # module docstring).
    rng = np.random.RandomState(RNG_SEED)

    rows_out = []
    for frow in frozen:
        budget = frow["budget_tier"]
        a, b = frow["architecture_a"], frow["architecture_b"]
        tag_a, tag_b = owner_tag[(a, budget)], owner_tag[(b, budget)]
        nn_favors = a if force_mse[(a, budget)] < force_mse[(b, budget)] else b

        log_ratios = []
        for s in range(20):
            va = selected.get((tag_a, s))
            vb = selected.get((tag_b, s))
            if va is None or vb is None:
                continue
            log_ratios.append(math.log(va) - math.log(vb))

        result = bootstrap_paired_log_ratio_ci(log_ratios, n_boot=DEFAULT_N_BOOT, seed=RNG_SEED, rng=rng)
        point_est = result["point_estimate"]
        ci_lo, ci_hi = result["ci95"]
        krr_favors = b if point_est > 0 else a
        excludes_zero = ci_lo > 0 or ci_hi < 0
        status = "unresolved"
        if excludes_zero:
            status = "concordant" if nn_favors == krr_favors else "discordant"

        rows_out.append(
            {
                "budget_tier": budget,
                "architecture_a": a,
                "architecture_b": b,
                "nn_favors": nn_favors,
                "krr_favors_point_estimate": krr_favors,
                "log_ratio_mean_a_over_b_at_n512": point_est,
                "bootstrap_ci_lo_2.5pct": ci_lo,
                "bootstrap_ci_hi_97.5pct": ci_hi,
                "n_boot": DEFAULT_N_BOOT,
                "bootstrap_seed": RNG_SEED,
                "n_paired_splits": len(log_ratios),
                "status": status,
                "frozen_log_ratio_mean_a_over_b_at_n512": frow["log_ratio_mean_a_over_b_at_n512"],
                "frozen_bootstrap_ci_lo_2.5pct": frow["bootstrap_ci_lo_2.5pct"],
                "frozen_bootstrap_ci_hi_97.5pct": frow["bootstrap_ci_hi_97.5pct"],
                "frozen_status": frow["status"],
            }
        )
    return rows_out


def main():
    direction_rows = build_direction_check()
    out1 = out_path("claim1", "pairwise_concordance_direction_check.csv")
    write_csv(out1, direction_rows)
    n_mismatch = sum(1 for r in direction_rows if not r["nn_favors_match"] or not r["krr_favors_match"])
    print(f"wrote {out1} ({len(direction_rows)} rows, {n_mismatch} direction mismatches)")

    ci_rows = build_recomputed_ci()
    out2 = out_path("claim1", "pairwise_concordance_recomputed.csv")
    write_csv(out2, ci_rows)
    n_concordant = sum(1 for r in ci_rows if r["status"] == "concordant")
    n_discordant = sum(1 for r in ci_rows if r["status"] == "discordant")
    n_unresolved = sum(1 for r in ci_rows if r["status"] == "unresolved")
    n_status_match = sum(1 for r in ci_rows if r["status"] == r["frozen_status"])
    print(
        f"wrote {out2} ({len(ci_rows)} rows): "
        f"concordant={n_concordant} discordant={n_discordant} unresolved={n_unresolved}; "
        f"status matches frozen for {n_status_match}/{len(ci_rows)} rows"
    )


if __name__ == "__main__":
    main()
