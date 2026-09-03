#!/usr/bin/env python3
"""Checks every number reported in the paper against the tables this repository regenerates.

Run `make analysis` first, then `make test` (or `python3 tests/test_paper_numbers.py`). Each
check names the section, figure, or table of the paper it covers, so a discrepancy points at the
sentence that has to change. Values are compared at the precision the paper prints them to.

`make validate` answers a different question: that the regenerated tables match the frozen ones.
This file answers whether the paper's prose matches those tables.
"""
from __future__ import annotations

import math
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "code", "analysis"))

from common.compute_axis import budget_flops
from common.io_utils import data_path, out_path, read_csv, read_json

failures = []
checks = 0


def check(label, actual, expected, tol):
    """Compare a reported value against the value in the data, at the reported precision."""
    global checks
    checks += 1
    ok = actual == expected if isinstance(expected, (int, str)) else \
        math.isclose(actual, expected, abs_tol=tol)
    status = "ok  " if ok else "FAIL"
    print(f"[{status}] {label}: paper {expected}, data {actual}")
    if not ok:
        failures.append(label)


def rel_check(label, actual, expected, rel_tol=5e-4):
    check(label, actual, expected, tol=abs(expected) * rel_tol)


# ------------------------------------------------------------------ scaling exponents (Sec. 2)

def test_scaling_exponents():
    rows = {r["architecture"].replace(" (lmax=4)", ""): r
            for r in read_csv(out_path("claim1", "force_scaling_recomputed.csv"))}
    for arch, gamma in [("MPNN", 0.227), ("MC-EGNN", 0.296),
                        ("GemNet-OC", 0.331), ("eSEN", 0.663)]:
        check(f"Sec. 2 scaling exponent, {arch}",
              round(float(rows[arch]["gamma"]), 3), gamma, tol=0)

    row = next(iter(rows.values()))
    rel_check("Sec. 2 / App. compute, fit interval lower bound",
              float(row["C_min_flops"]), 8.4905e15)
    rel_check("Sec. 2 / App. compute, fit interval upper bound",
              float(row["C_max_flops"]), 1.2243e18)


# ------------------------------------------------- GemNet-OC / eSEN crossover (Sec. 3, Table 1)

DENSE_GRID = [
    # budget, C_GemNet-OC, C_eSEN, neural gap, KRR gap, CI low, CI high
    ("LOW",       2.105e16, 2.106e16, -0.611, -0.873, -1.060, -0.692),
    ("MID",       8.570e16, 9.988e16, -0.070, -0.336, -0.560, -0.141),
    ("PRECROSS",  1.151e17, 1.189e17, -0.079, -0.496, -0.686, -0.302),
    ("CROSS",     1.151e17, 1.244e17, +0.046, -0.545, -0.787, -0.296),
    ("FITX",      1.612e17, 1.658e17, +0.125, -0.376, -0.655, -0.130),
    ("POSTCROSS", 1.916e17, 2.056e17, +0.130, -0.925, -1.235, -0.646),
    ("HIGH",      3.874e17, 4.523e17, +0.379, -0.387, -0.661, -0.128),
    ("TOP",       9.530e17, 9.181e17, +0.497, -0.786, -1.102, -0.444),
]


def test_dense_crossover_grid():
    gaps = {r["budget_tier"]: r
            for r in read_csv(out_path("claim1", "dense_grid_pairwise_gap_recomputed.csv"))}
    flops = budget_flops()

    for budget, c_gemnet, c_esen, neural, krr, lo, hi in DENSE_GRID:
        row = gaps[budget]
        rel_check(f"Table 1 [{budget}] GemNet-OC compute",
                  flops[("GemNet-OC", budget)], c_gemnet, rel_tol=1e-3)
        rel_check(f"Table 1 [{budget}] eSEN compute",
                  flops[("eSEN", budget)], c_esen, rel_tol=1e-3)
        check(f"Table 1 [{budget}] neural gap",
              round(float(row["neural_signed_gap_log_gemnet_over_esen"]), 3), neural, tol=0)
        check(f"Table 1 [{budget}] KRR gap at n=512",
              round(float(row["krr_signed_gap_log_gemnet_over_esen_at_n512"]), 3), krr, tol=0)
        check(f"Table 1 [{budget}] KRR CI low",
              round(float(row["krr_bootstrap_ci_lo_2.5pct"]), 3), lo, tol=0)
        check(f"Table 1 [{budget}] KRR CI high",
              round(float(row["krr_bootstrap_ci_hi_97.5pct"]), 3), hi, tol=0)

    # Sec. 3: the neural gap changes sign exactly once, at the reported crossover.
    signs = [float(gaps[b]["neural_signed_gap_log_gemnet_over_esen"]) > 0
             for b, *_ in DENSE_GRID]
    check("Sec. 3 neural gap changes sign exactly once",
          sum(1 for a, b in zip(signs, signs[1:]) if a != b), 1, tol=0)
    check("Sec. 3 KRR favors GemNet-OC at every budget",
          all(float(gaps[b]["krr_signed_gap_log_gemnet_over_esen_at_n512"]) < 0
              and float(gaps[b]["krr_bootstrap_ci_hi_97.5pct"]) < 0 for b, *_ in DENSE_GRID),
          True, tol=0)


# ------------------------------------------------------ intervened checkpoints (App. models)

INTERVENED = [
    # width, step, realized compute
    (10, 50000, 2.399e16),
    (16, 150000, 1.127e17),
    (20, 200000, 1.865e17),
    (24, 300000, 3.347e17),
]


def test_intervened_checkpoints():
    rows = [r for r in read_csv(data_path("claim2", "matched_compute_lmax_checkpoints.csv"))
            if r["lmax"] == "4"]
    rows.sort(key=lambda r: float(r["C_flops"]))
    check("App. models, number of intervened checkpoints", len(rows), 4, tol=0)
    for row, (width, step, flops) in zip(rows, INTERVENED):
        check(f"App. models, width of the {flops:.3g}-FLOP checkpoint",
              int(row["width"]), width, tol=0)
        check(f"App. models, step of the {flops:.3g}-FLOP checkpoint",
              int(row["global_step"]), step, tol=0)
        rel_check(f"App. models, compute of width-{width} checkpoint",
                  float(row["C_flops"]), flops, rel_tol=1e-3)


# ------------------------------------------------------------------- depth table (App. depth)

DEPTH = {  # block -> damage at each of the four checkpoints, in compute order
    3: [0.235, 0.532, 0.680, 0.853],
    6: [0.312, 0.654, 1.021, 1.047],
    9: [0.097, 0.681, 0.907, 0.678],
}
OWNERS_BY_COMPUTE = ["w10/50k", "w16/150k", "w20/200k", "w24/300k"]


def test_depth_table():
    depth = read_json(out_path("claim2", "depth_interventions_recomputed.json"))
    rel_check("App. depth, common perturbation magnitude",
              depth["common_P_bal_support"], 0.07770, rel_tol=1e-3)
    at_support = depth["read_at_common_support"]
    for block, values in DEPTH.items():
        for owner, expected in zip(OWNERS_BY_COMPUTE, values):
            key = f"{owner}_L{block}"
            check(f"App. depth, block {block} at {owner}",
                  round(at_support[key]["delta"], 3), expected, tol=0)


# ----------------------------------------------------------- independent runs (App. seeds)

SEEDS = {1: (0.477, 0.429, 0.534), 2: (0.715, 0.658, 0.778), 3: (0.134, 0.107, 0.166)}


def test_seed_table():
    summary = read_json(out_path("claim2", "seed_replication_result_recomputed.json"))
    rel_check("App. seeds, perturbation magnitude read-off",
              summary["P_star"], 0.15, rel_tol=1e-9)
    for seed, (g, lo, hi) in SEEDS.items():
        check(f"App. seeds, run {seed} contrast",
              round(summary["G_by_seed"][str(seed)], 3), g, tol=0)
        ci = summary["G_ci95_by_seed"][str(seed)]
        check(f"App. seeds, run {seed} CI low", round(ci[0], 3), lo, tol=0)
        check(f"App. seeds, run {seed} CI high", round(ci[1], 3), hi, tol=0)
    check("App. seeds, sign replicates in all three runs",
          all(v > 0 for v in summary["G_by_seed"].values()), True, tol=0)


# --------------------------------------------------- chemistry-family transfer (App. transfer)

FAMILIES = {
    "Neutral_val_pooled_(ID_reference)": (16384, 0.565, 0.551, 0.579),
    "pooled_neutral_like_sources_in_ValComp": (450, 0.588, 0.501, 0.676),
    "biomolecules": (3286, 0.427, 0.413, 0.440),
    "elytes": (10867, 0.344, 0.332, 0.356),
    "reactivity": (1749, 0.149, 0.138, 0.160),
}


def test_family_transfer_table():
    rows = {r["stratum"]: r
            for r in read_csv(out_path("claim2", "chemistry_family_positive_control.csv"))}
    for stratum, (n_cfg, contrast, lo, hi) in FAMILIES.items():
        row = rows[stratum]
        check(f"App. transfer, {stratum} n_cfg", int(row["n_configs"]), n_cfg, tol=0)
        check(f"App. transfer, {stratum} contrast",
              round(float(row["contrast_w24_minus_w10"]), 3), contrast, tol=0)
        check(f"App. transfer, {stratum} CI low", round(float(row["ci_lo"]), 3), lo, tol=0)
        check(f"App. transfer, {stratum} CI high", round(float(row["ci_hi"]), 3), hi, tol=0)
    check("App. transfer, metal complexes excluded from inference",
          rows.get("metal_complexes", {}).get("insufficient_n"), "True", tol=0)


# ------------------------------------------------- matched frontier comparison (App. frontier)

def test_matched_frontier():
    s = read_json(out_path("claim2", "lmax24_all_frontier_summary.json"))
    check("App. frontier, ell_max=4 frontier checkpoints",
          s["n_lmax4_frontier_checkpoints"], 126, tol=0)
    check("App. frontier, valid matches", s["n_valid_matches"], 102, tol=0)
    check("App. frontier, ell_max=4 training trajectories",
          s["lmax4_checkpoints_and_trajectories"]["n_trajectories"], 11, tol=0)
    check("App. frontier, trajectories among matched checkpoints",
          s["lmax4_matched_checkpoints_and_trajectories"]["n_trajectories"], 9, tol=0)
    check("App. frontier, ell_max=2 checkpoints used",
          s["lmax2_matched_checkpoints_and_trajectories"]["n_checkpoints"], 55, tol=0)
    check("App. frontier, ell_max=2 trajectories",
          s["lmax2_matched_checkpoints_and_trajectories"]["n_trajectories"], 8, tol=0)

    lo, hi = s["matched_compute_range_flops"]
    rel_check("App. frontier, lowest matched compute", lo, 3.75e14, rel_tol=2e-3)
    rel_check("App. frontier, highest matched compute", hi, 4.16e17, rel_tol=2e-3)

    neutral = s["neutral_validation"]
    check("App. frontier, Neutral pairs favoring ell_max=2",
          neutral["n_favoring_lmax2"], 100, tol=0)
    check("App. frontier, Neutral pairs favoring ell_max=4",
          neutral["n_favoring_lmax4"], 2, tol=0)
    for value, expected in zip(neutral["lmax4_point_estimates"], [0.0233, 0.0288]):
        check(f"App. frontier, ell_max=4-favoring point estimate {expected}",
              round(value, 4), expected, tol=0)

    matched = s["support_matched_validation"]
    check("App. frontier, support-matched pairs favoring ell_max=2",
          matched["n_favoring_lmax2"], 102, tol=0)
    check("App. frontier, support-matched CIs excluding zero",
          matched["n_ci_excluding_zero"], 102, tol=0)

    top = s["highest_common_support_pair_neutral"]
    rel_check("App. frontier, decisive pair ell_max=4 compute",
              top["C_lmax4_flops"], 3.700e17, rel_tol=1e-3)
    rel_check("App. frontier, decisive pair ell_max=2 compute",
              top["C_lmax2_flops"], 3.724e17, rel_tol=1e-3)
    check("App. frontier, decisive pair log ratio", round(top["H_force"], 4), -0.0181, tol=0)
    check("App. frontier, decisive pair CI low", round(top["ci95"][0], 4), -0.0626, tol=0)
    check("App. frontier, decisive pair CI high", round(top["ci95"][1], 4), 0.0285, tol=0)


# ---------------------------------------------------- evaluation populations (App. data/metric)

def test_evaluation_populations():
    pools = read_json(data_path("claim2", "evaluation_pool_provenance.json"))
    check("App. data, Neutral validation configurations",
          pools["neutral_domain_pool"]["n_eligible_total"], 27697, tol=0)
    check("App. data, broader validation configurations",
          pools["valcomp_ood_pool"]["total_val_structures"], 2762021, tol=0)
    check("App. data, support-matched eligible configurations",
          pools["valcomp_ood_pool"]["n_eligible_total"], 1011426, tol=0)


def main():
    for test in (test_scaling_exponents, test_dense_crossover_grid, test_intervened_checkpoints,
                 test_depth_table, test_seed_table, test_family_transfer_table,
                 test_matched_frontier, test_evaluation_populations):
        print(f"\n== {test.__name__}")
        test()

    print(f"\n{checks - len(failures)}/{checks} paper values match the data.")
    if failures:
        print("MISMATCHED:")
        for name in failures:
            print("  " + name)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
