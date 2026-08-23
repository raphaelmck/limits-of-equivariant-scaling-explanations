#!/usr/bin/env python3
"""Reproduction of data/claim1/pairwise_concordance.csv, INCLUDING its bootstrap CI columns.

Originally this script could only recompute the point-estimate log-ratio direction (see git
history / AUDIT_STAGE2.md category (b) item 3), because the 20 individual per-split test_nmse
values that data/claim1/pairwise_concordance.csv's bootstrap CI is built from were not present
anywhere in data/claim1/ -- only test_nmse_median/test_nmse_mean (already aggregated over the 20
splits) were. That gap has since been closed: the raw per-split file was copied in as
data/claim1/matched_compute_m1024_krr_split_results.csv (source:
dev-equivariant-scaling-laws-kernel-pilot-clean/analysis_outputs/matched_compute_large_m_krr_2026_08_22/
krr_split_results_m1024.csv), so the full CI is now recomputable from data/ alone.

Procedure (confirmed by reading
dev-equivariant-scaling-laws-kernel-pilot-clean/analysis_scripts/build_matched_compute_m1024_comparison.py
lines ~189-231, the exact script that produced the frozen pairwise_concordance.csv):
  1. For each (budget_tier, architecture_a, architecture_b) pair among the 12 validated owners,
     take the SELECTED-rho row (is_selected == True; rho chosen by inner validation NMSE) at
     n_train=512 for each of the 20 split_seeds, for both architectures.
  2. log_ratio[s] = log(test_nmse_a[s]) - log(test_nmse_b[s]) for each split s where both arms
     have a selected row (matched by split_seed -- this is what makes it "paired").
  3. point_estimate = mean(log_ratio); krr_favors = architecture_b if point_estimate > 0 else
     architecture_a (i.e. whichever has the lower nmse).
  4. Bootstrap CI, BIT-EXACT reproduction (Stage-2 fix -- see AUDIT_STAGE2.md): upstream created
     ONE `rng = np.random.RandomState(RNG_SEED=20260822)` BEFORE any per-pair loop, then for
     `budget in ["LOW","MID","HIGH"]`, for `a_tag,b_tag in itertools.combinations(tier_tags, 2)`
     (tier_tags in the fixed OWNER_TAGS order: mpnn, egnn, gemnet_oc, esen within each budget --
     verified by reading OWNER_TAGS in the upstream script, and confirmed this exactly matches
     the frozen pairwise_concordance.csv's own row order), ran
     `boot_means = np.array([rng.choice(log_ratios, size=len(log_ratios), replace=True).mean()
     for _ in range(N_BOOT)])` -- i.e. ONE rng.choice() call PER bootstrap replicate, in a loop,
     with the SAME RandomState instance carried across all 18 pairs in sequence (not
     re-seeded per pair, not vectorized across replicates in one call -- both would desync the
     draw sequence from upstream's). This script reproduces that exactly: a single
     `np.random.RandomState(20260822)` is created once, then threaded via `rng=` through
     `src/bootstrap.py::bootstrap_paired_log_ratio_ci` for each of the 18 frozen rows IN FILE
     ORDER (== upstream's budget/combinations order, verified above), so the shared RNG state
     advances identically to upstream's. CI = [2.5th, 97.5th] percentile of the 2000 replicate
     means, via numpy's default 'linear' interpolation (matches np.quantile's default).
  5. status = "concordant"/"discordant" if CI excludes zero (sign of point estimate agrees or
     disagrees with nn_favors), else "unresolved" -- identical logic to the frozen script.

Outputs (both retained -- the direction-only check is still valid and cheap):
  - analysis_out/claim1/pairwise_concordance_direction_check.csv (unchanged from before: point
    estimate + nn/krr-favors direction consistency against the frozen file, no CI).
  - analysis_out/claim1/pairwise_concordance_recomputed.csv (full CI, one row per pair, now
    BIT-EXACT -- see validate.py's tolerance, tightened from overlap-only to relative 1e-6).
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
