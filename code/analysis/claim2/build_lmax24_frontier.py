#!/usr/bin/env python3
"""Matched-frontier ell_max=2 vs ell_max=4 comparison (paper appendix "Matched-frontier
ell_max=2 versus ell_max=4 comparison").

Reads the frozen matched-pair tables in data/claim2/ and re-derives every summary quantity the
appendix reports: how many ell_max=4 frontier checkpoints exist, how many have a valid
ell_max=2 match under the |log10(C_2/C_4)| <= 0.06 rule, how many training trajectories those
checkpoints come from, the sign tally of log(L_2/L_4) in each evaluation population, and the
highest-compute comparison covered by both model families.

The pairing itself (nearest ell_max=2 frontier owner in log10 compute) is frozen upstream in
lmax24_all_frontier_matches.csv; this script re-checks the tolerance rule against the recorded
compute ratios rather than re-running the match.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.io_utils import data_path, out_path, read_csv, write_json

TOL_LOG10 = 0.06  # matching tolerance on |log10(C_2 / C_4)|, ~15% in compute


def _truthy(value: str) -> bool:
    return str(value).strip().lower() == "true"


def summarize_domain(gap_rows, domain):
    rows = [r for r in gap_rows if r["domain"] == domain and _truthy(r["valid_match"])]
    favors_lmax2 = [r for r in rows if float(r["H_force"]) < 0.0]
    favors_lmax4 = [r for r in rows if float(r["H_force"]) > 0.0]

    ci_excludes_zero = 0
    ci_available = 0
    for r in rows:
        lo, hi = r["ci_low"], r["ci_high"]
        if lo == "" or hi == "":
            continue
        ci_available += 1
        if float(lo) * float(hi) > 0.0:
            ci_excludes_zero += 1

    return {
        "n_valid_pairs": len(rows),
        "n_favoring_lmax2": len(favors_lmax2),
        "n_favoring_lmax4": len(favors_lmax4),
        "H_force_min": min(float(r["H_force"]) for r in rows),
        "H_force_max": max(float(r["H_force"]) for r in rows),
        "n_pairs_with_ci": ci_available,
        "n_ci_excluding_zero": ci_excludes_zero,
        "lmax4_point_estimates": sorted(round(float(r["H_force"]), 6) for r in favors_lmax4),
    }


def highest_common_support_pair(gap_rows, matches):
    """The highest-compute matched pair inside the compute range covered by both families, with
    its bootstrap confidence interval (the appendix's decisive comparison)."""
    beyond = {
        r["l4_id"] for r in matches if _truthy(r["l4_is_beyond_C_common_max"])
    }
    id_by_wstep = {(r["l4_width"], r["l4_step"]): r["l4_id"] for r in matches}
    candidates = []
    for r in gap_rows:
        if r["domain"] != "Neutral_val" or not _truthy(r["valid_match"]):
            continue
        if r["ci_low"] == "" or r["ci_high"] == "":
            continue
        if id_by_wstep.get((r["l4_width"], r["l4_step"])) in beyond:
            continue
        candidates.append(r)
    top = max(candidates, key=lambda r: float(r["l4_flops"]))
    return {
        "C_lmax4_flops": float(top["l4_flops"]),
        "C_lmax2_flops": float(top["l2_flops"]),
        "H_force": float(top["H_force"]),
        "H_force_note": "paired configuration-level bootstrap point estimate over the full "
                        "Neutral validation population; the CI below is from the same bootstrap",
        "ci95": [float(top["ci_low"]), float(top["ci_high"])],
        "ci_excludes_zero": float(top["ci_low"]) * float(top["ci_high"]) > 0.0,
    }


def main():
    matches = read_csv(data_path("claim2", "lmax24_all_frontier_matches.csv"))
    gaps = read_csv(data_path("claim2", "lmax24_all_frontier_gaps.csv"))
    dependence = read_csv(data_path("claim2", "lmax24_frontier_dependence.csv"))

    valid = [r for r in matches if _truthy(r["valid_match"])]
    tol_violations = [
        r["l4_id"] for r in valid
        if abs(float(r["log10_compute_difference"])) > TOL_LOG10
    ]
    non_frontier_partners = [
        r["l4_id"] for r in valid if not _truthy(r["l2_is_frontier_owner"])
    ]

    def trajectories(arch, only_used=False):
        rows = [r for r in dependence if r["arch"] == arch]
        if only_used:
            rows = [r for r in rows if _truthy(r["used_in_valid_pair"])]
        return {
            "n_checkpoints": len(rows),
            # A trajectory is one (width, training-run) pair: a run that was preempted and
            # requeued keeps its name but gets a new run id, and is a separate trajectory.
            "n_trajectories": len({(r["width"], r["run_id"]) for r in rows}),
            "n_widths": len({r["width"] for r in rows}),
        }

    matched_flops = [float(r["l4_flops"]) for r in valid]

    summary = {
        "matching_rule": "nearest ell_max=2 frontier owner in log10 compute, "
                         "|log10(C_2 / C_4)| <= %.2f" % TOL_LOG10,
        "n_lmax4_frontier_checkpoints": len(matches),
        "n_valid_matches": len(valid),
        "matching_tolerance_violations": tol_violations,
        "matches_whose_partner_is_not_a_frontier_owner": non_frontier_partners,
        "matched_compute_range_flops": [min(matched_flops), max(matched_flops)],
        "lmax4_checkpoints_and_trajectories": trajectories("lmax4"),
        "lmax4_matched_checkpoints_and_trajectories": trajectories("lmax4", only_used=True),
        "lmax2_matched_checkpoints_and_trajectories": trajectories("lmax2", only_used=True),
        "neutral_validation": summarize_domain(gaps, "Neutral_val"),
        "support_matched_validation": summarize_domain(gaps, "Val_Comp_support_matched"),
        "highest_common_support_pair_neutral": highest_common_support_pair(gaps, matches),
    }

    dest = out_path("claim2", "lmax24_all_frontier_summary.json")
    write_json(dest, summary)
    print("wrote", dest)
    print("  %d ell_max=4 frontier checkpoints, %d valid matches"
          % (summary["n_lmax4_frontier_checkpoints"], summary["n_valid_matches"]))
    for key in ("neutral_validation", "support_matched_validation"):
        d = summary[key]
        print("  %-28s %d/%d favor ell_max=2, %d/%d CIs exclude zero"
              % (key, d["n_favoring_lmax2"], d["n_valid_pairs"],
                 d["n_ci_excluding_zero"], d["n_pairs_with_ci"]))


if __name__ == "__main__":
    main()
