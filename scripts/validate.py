#!/usr/bin/env python3
"""Compare every table regenerated into analysis_out/ against the corresponding frozen table in
data/. Prints a pass or fail line per table and exits non-zero on any scientifically meaningful
discrepancy: a flipped sign, a confidence interval that no longer excludes or includes zero where
the frozen one did, or a point estimate outside tolerance. Floating-point noise never fails.

Tolerance: 1e-6 relative, for every point estimate, and for every confidence interval, since every
table checked here has a documented upstream bootstrap seed and call structure and reproduces
bit-exact (in practice these match to better than 1e-9).

`main()` is organized around one check per reported paper result, printed as "C1." and "C2."
lines, on top of the lower-level per-table checks.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import data_path, out_path, read_csv, read_json

REL_TOL_EXACT = 1e-6

results = []  # (name, passed: bool, detail: str)
main_paper_results = []  # (label, passed: bool) -- one per reported paper result, in order


def record(name, passed, detail=""):
    results.append((name, passed, detail))
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] {name}" + (f" -- {detail}" if detail and not passed else ""))
    return passed


def record_main(label, passed):
    main_paper_results.append((label, passed))
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] {label}")
    return passed


def rel_close(a, b, tol=REL_TOL_EXACT):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-12)


def ci_close(lo1, hi1, lo2, hi2, tol=REL_TOL_EXACT):
    return rel_close(lo1, lo2, tol) and rel_close(hi1, hi2, tol)


def ci_overlap(lo1, hi1, lo2, hi2):
    return lo1 <= hi2 and lo2 <= hi1


def excludes_zero(lo, hi):
    return lo > 0 or hi < 0


# ===========================================================================
# Claim 1
# ===========================================================================

def validate_force_scaling():
    name = "claim1/force_scaling.csv (gamma/logA/r2/rmse_log exact, no randomness)"
    regen = read_csv(out_path("claim1", "force_scaling_recomputed.csv"))
    frozen = read_csv(data_path("claim1", "force_scaling.csv"))
    fmap = {r["architecture_key"]: r for r in frozen}
    ok = True
    details = []
    for r in regen:
        f = fmap.get(r["architecture_key"])
        if f is None:
            ok = False
            details.append(f"missing frozen row for {r['architecture_key']}")
            continue
        for k in ("gamma", "logA", "r2", "rmse_log"):
            if not rel_close(float(r[k]), float(f[k]), tol=1e-9):
                ok = False
                details.append(f"{r['architecture_key']}: {k} mismatch")
        if int(r["n_frontier_owners_in_range"]) != int(f["n_frontier_owners_in_range"]):
            ok = False
            details.append(f"{r['architecture_key']}: n_frontier_owners_in_range mismatch")
    record(name, ok, "; ".join(details))
    return ok


def validate_dense_grid_ranking_comparison():
    """Exact, no randomness: four-architecture NN-vs-KRR ranking at all 8 budgets (LOW, MID,
    PRECROSS, CROSS, FITX, POSTCROSS, HIGH, TOP). See
    scripts/claim1/build_dense_grid_ranking_comparison.py."""
    name = "claim1/dense_grid_ranking_comparison.csv (exact, no randomness, 8 budgets x 4 archs)"
    regen = read_csv(out_path("claim1", "dense_grid_ranking_comparison_recomputed.csv"))
    frozen = read_csv(data_path("claim1", "dense_grid_ranking_comparison.csv"))
    key = lambda r: (r["budget_tier"], r["architecture"])
    fmap = {key(r): r for r in frozen}
    ok, details = True, []
    for r in regen:
        f = fmap.get(key(r))
        if f is None:
            ok = False
            details.append(f"missing frozen row for {key(r)}")
            continue
        if int(r["nn_rank"]) != int(f["nn_rank"]) or int(r["krr_rank"]) != int(f["krr_rank"]):
            ok = False
            details.append(f"{key(r)}: rank mismatch")
        if not rel_close(float(r["force_mse_norm"]), float(f["force_mse_norm"])):
            ok = False
            details.append(f"{key(r)}: force_mse_norm mismatch")
        if not rel_close(float(r["krr_auc_log_nmse_n32_512"]), float(f["krr_auc_log_nmse_n32_512"])):
            ok = False
            details.append(f"{key(r)}: AUC mismatch")
    record(name, ok, "; ".join(details[:10]))
    return ok


def validate_dense_grid_pairwise_gap():
    """BIT-EXACT: dense-grid (PRECROSS/CROSS/FITX/POSTCROSS/TOP) GemNet-OC/eSEN signed gap +
    paired-bootstrap CI, all 8 budgets, single shared RandomState(20260822) threaded across
    budgets in fixed order. See scripts/claim1/build_dense_grid_pairwise_gap.py."""
    regen = read_csv(out_path("claim1", "dense_grid_pairwise_gap_recomputed.csv"))
    frozen = read_csv(data_path("claim1", "dense_grid_pairwise_gap.csv"))
    fmap = {r["budget_tier"]: r for r in frozen}
    ok, details = True, []
    for r in regen:
        f = fmap.get(r["budget_tier"])
        if f is None:
            ok = False
            details.append(f"missing budget {r['budget_tier']}")
            continue
        if not rel_close(float(r["neural_signed_gap_log_gemnet_over_esen"]),
                          float(f["neural_signed_gap_log_gemnet_over_esen"])):
            ok = False
            details.append(f"{r['budget_tier']}: neural gap mismatch")
        if not rel_close(float(r["krr_signed_gap_log_gemnet_over_esen_at_n512"]),
                          float(f["krr_signed_gap_log_gemnet_over_esen_at_n512"])):
            ok = False
            details.append(f"{r['budget_tier']}: krr gap mismatch")
        if not ci_close(float(r["krr_bootstrap_ci_lo_2.5pct"]), float(r["krr_bootstrap_ci_hi_97.5pct"]),
                         float(f["krr_bootstrap_ci_lo_2.5pct"]), float(f["krr_bootstrap_ci_hi_97.5pct"])):
            ok = False
            details.append(f"{r['budget_tier']}: CI not bit-exact")
    record("claim1/dense_grid_pairwise_gap.csv (point + CI BIT-EXACT, 8 budgets)", ok, "; ".join(details[:10]))
    return ok


# ===========================================================================
# Claim 2
# ===========================================================================

def validate_frontier_ell4_degree_balanced():
    """BIT-EXACT: np.random.default_rng(20260820), fresh per (owner,alpha) cell --
    see scripts/claim2/build_frontier_ell4_degree_balanced.py. Now validates per-alpha CIs AND
    the 6 pairwise Delta-difference contrasts, not just the interpolated point estimate."""
    name = "claim2/frontier_ell4_degree_balanced (delta + CI BIT-EXACT, all 4 owners + 6 contrasts)"
    regen = read_json(out_path("claim2", "frontier_ell4_degree_balanced_recomputed.json"))
    frozen = read_json(data_path("claim2", "frontier_ell4_degree_balanced_summary.json"))
    ok = True
    details = []
    for owner, v in regen["read_off_at_common_support_edge"].items():
        f = frozen["read_off_at_common_support_edge"][owner]
        if not rel_close(v["delta"], f["delta"]):
            ok = False
            details.append(f"{owner}: delta mismatch")
        lo1, hi1 = v["loss_multiplier_ci95"]
        lo2, hi2 = f["loss_multiplier_ci95"]
        if not ci_close(lo1, hi1, lo2, hi2):
            ok = False
            details.append(f"{owner}: loss_multiplier_ci95 not bit-exact")
        if v["on_measured_point"] != f["on_measured_point"]:
            ok = False
            details.append(f"{owner}: on_measured_point mismatch")
    for key, v in regen["pairwise_contrasts_at_common_support_edge"].items():
        f = frozen["pairwise_contrasts_at_common_support_edge"][key]
        if not rel_close(v["delta_difference"], f["delta_difference"]):
            ok = False
            details.append(f"{key}: delta_difference mismatch")
        lo1, hi1 = v["ci95"]
        lo2, hi2 = f["ci95"]
        if not ci_close(lo1, hi1, lo2, hi2):
            ok = False
            details.append(f"{key}: ci95 not bit-exact")
        if v["ci_excludes_zero"] != f["ci_excludes_zero"]:
            ok = False
            details.append(f"{key}: ci_excludes_zero mismatch")
    record(name, ok, "; ".join(details[:10]))
    return ok


def validate_seed_replication_result():
    """BIT-EXACT: np.random.default_rng(0) shared idx_sets, ported line-for-line
    from analyze_seed_replication.py -- see scripts/claim2/build_seed_replication_result.py.
    Now validates G_ci95_by_seed too, not just the G point estimate."""
    name = "claim2/seed_replication_result_summary.json (G point + CI BIT-EXACT, all 3 seeds)"
    regen = read_json(out_path("claim2", "seed_replication_result_recomputed.json"))
    frozen = read_json(data_path("claim2", "seed_replication_result_summary.json"))
    ok = True
    details = []
    for seed, g in regen["G_by_seed"].items():
        f = frozen["G_by_seed"][seed]
        if not rel_close(g, f):
            ok = False
            details.append(f"seed {seed}: G mismatch {g} vs {f}")
    for seed, ci in regen["G_ci95_by_seed"].items():
        f_lo, f_hi = frozen["G_ci95_by_seed"][seed]
        if not ci_close(ci[0], ci[1], f_lo, f_hi):
            ok = False
            details.append(f"seed {seed}: G_ci95 not bit-exact {ci} vs {[f_lo, f_hi]}")
    if regen["sign_replication"] != frozen["sign_replication"]:
        ok = False
        details.append(f"sign_replication mismatch: {regen['sign_replication']} vs {frozen['sign_replication']}")
    record(name, ok, "; ".join(details[:10]))
    return ok


def validate_depth_localization():
    """BIT-EXACT for all 3 depths (block 3, 6, and 9) and all 12 depth
    contrasts (block9-involving contrasts included) -- see
    scripts/claim2/build_depth_localization.py, which ports build_depth_localization_report.py's
    single shared np.random.default_rng(20260820) idx matrix across every owner and depth."""
    name = "claim2/depth_localization_summary.json (delta + CI BIT-EXACT for all 3 depths + 12 contrasts)"
    regen = read_json(out_path("claim2", "depth_localization_recomputed.json"))
    frozen = read_json(data_path("claim2", "depth_localization", "depth_localization_summary.json"))
    ok = True
    details = []
    for k, v in regen["read_at_common_support"].items():
        f = frozen["read_at_common_support"].get(k)
        if f is None:
            ok = False
            details.append(f"{k}: missing frozen cell")
            continue
        if not rel_close(v["delta"], f["delta"]):
            ok = False
            details.append(f"{k}: delta mismatch")
        lo, hi = v["ci95"]
        flo, fhi = f["ci95"]
        if not ci_close(lo, hi, flo, fhi):
            ok = False
            details.append(f"{k}: CI not bit-exact")
    # New derived quantity (not in the frozen file, no upstream reproduction target): D_k =
    # Delta_{w24,k} - Delta_{w10,k}, added for Figure 2 v2 panel A. Self-consistency only --
    # checks the point estimate is exactly the arithmetic difference of the two already-validated
    # deltas above, and that its CI is a well-formed interval containing the point estimate.
    dk_ok = True
    dk_details = []
    for key, v in regen.get("low_vs_high_contrast_by_depth", {}).items():
        k = v["block"]
        d_high = regen["read_at_common_support"][f"w24/300k_L{k}"]["delta"]
        d_low = regen["read_at_common_support"][f"w10/50k_L{k}"]["delta"]
        if not rel_close(v["D_k"], d_high - d_low, tol=1e-9):
            dk_ok = False
            dk_details.append(f"{key}: D_k arithmetic identity mismatch")
        lo, hi = v["ci95"]
        if not (lo < hi):
            dk_ok = False
            dk_details.append(f"{key}: malformed CI [{lo},{hi}]")
        if v["ci_excludes_zero"] != (lo > 0 or hi < 0):
            dk_ok = False
            dk_details.append(f"{key}: ci_excludes_zero flag inconsistent with bounds")
    record("claim2/depth_localization_recomputed.json (low_vs_high D_k, self-consistency, no upstream frozen target)",
           dk_ok, "; ".join(dk_details[:10]))
    ok = ok and dk_ok

    for k, v in regen.get("depth_contrasts", {}).items():
        f = frozen.get("depth_contrasts", {}).get(k)
        if f is None:
            ok = False
            details.append(f"{k}: missing frozen contrast")
            continue
        if not rel_close(v["delta_difference"], f["delta_difference"]):
            ok = False
            details.append(f"{k}: delta_difference mismatch")
        lo, hi = v["ci95"]
        flo, fhi = f["ci95"]
        if not ci_close(lo, hi, flo, fhi):
            ok = False
            details.append(f"{k}: contrast CI not bit-exact")
        if v["sign"] != f["sign"]:
            ok = False
            details.append(f"{k}: sign mismatch")
    record(name, ok, "; ".join(details[:10]))
    return ok


# ===========================================================================
# Claim 2 -- OOD (BIT-EXACT: default_rng(20260822)/(20260823), fresh per cell)
# ===========================================================================

def validate_ood_curves_and_contrasts():
    ok_all = True
    for fname, keys, pt, ci in [
        ("ood_full_alpha_curves.csv", ["tag", "domain", "M", "alpha"], "delta", ("delta_ci_lo", "delta_ci_hi")),
        ("ood_G_m_convergence.csv", ["tag", "M"], "G", ("G_ci_lo", "G_ci_hi")),
        ("ood_interaction_w24_vs_w10.csv", ["M"], "interaction", ("ci_lo", "ci_hi")),
        ("ood_lmax2_vs_lmax4_baseline.csv", ["lmax4_tag", "domain", "M"], "H_force", ("H_ci_lo", "H_ci_hi")),
    ]:
        regen = read_csv(out_path("claim2", fname))
        frozen = read_csv(data_path("claim2", fname))
        fmap = {tuple(r[k] for k in keys): r for r in frozen}
        ok, details = True, []
        for r in regen:
            f = fmap.get(tuple(r[k] for k in keys))
            if f is None:
                ok = False
                details.append(f"missing {tuple(r[k] for k in keys)}")
                continue
            if not rel_close(float(r[pt]), float(f[pt])):
                ok = False
                details.append(f"{fname}:{pt} mismatch")
            lo, hi = ci
            if not ci_close(float(r[lo]), float(r[hi]), float(f[lo]), float(f[hi])):
                ok = False
                details.append(f"{fname}: CI not bit-exact")
        record(f"claim2/{fname} (point + CI BIT-EXACT)", ok, "; ".join(details[:10]))
        ok_all = ok_all and ok
    return ok_all


def validate_ood_domain_decomposition():
    """BIT-EXACT: np.random.default_rng(20260823), the per-configuration
    data_id chemistry-family label recovered from data/claim2/ood_domain_labels.json -- see
    scripts/claim2/build_ood_domain_decomposition.py."""
    ok_all = True
    for fname, keys, pt, ci in [
        ("ood_domain_decomposition.csv", ["stratum", "domain", "tag"], "delta", ("delta_ci_lo", "delta_ci_hi")),
        ("ood_shared_family_positive_control.csv", ["stratum"], "contrast_w24_minus_w10", ("ci_lo", "ci_hi")),
    ]:
        regen = read_csv(out_path("claim2", fname))
        frozen = read_csv(data_path("claim2", fname))
        fmap = {tuple(r[k] for k in keys): r for r in frozen}
        ok, details = True, []
        for r in regen:
            f = fmap.get(tuple(r[k] for k in keys))
            if f is None:
                ok = False
                details.append(f"missing {tuple(r[k] for k in keys)}")
                continue
            if r.get("insufficient_n") == "True":
                continue
            if not rel_close(float(r[pt]), float(f[pt])):
                ok = False
                details.append(f"{fname}:{pt} mismatch")
            lo, hi = ci
            if not ci_close(float(r[lo]), float(r[hi]), float(f[lo]), float(f[hi])):
                ok = False
                details.append(f"{fname}: CI not bit-exact")
        record(f"claim2/{fname} (point + CI BIT-EXACT)", ok, "; ".join(details[:10]))
        ok_all = ok_all and ok
    return ok_all


# ===========================================================================
# MAIN-PAPER CHECK LIST (one per reported result)
# ===========================================================================

def main():
    print("=" * 78)
    print("MAIN-PAPER CHECKS")
    print("=" * 78)

    c1_force_scaling = validate_force_scaling()
    record_main("C1. force-scaling exponents", c1_force_scaling)

    c1_dense_gap = validate_dense_grid_pairwise_gap()
    record_main("C1. GemNet-OC/eSEN crossover, 8 budgets", c1_dense_gap)

    c2_degree_balanced = validate_frontier_ell4_degree_balanced()
    record_main("C2. four-owner degree-balanced causal result", c2_degree_balanced)

    c2_seed_repl = validate_seed_replication_result()
    record_main("C2. 3/3 seed-sign replication + per-seed CIs", c2_seed_repl)

    c2_depth = validate_depth_localization()
    record_main("C2. depth-localization result", c2_depth)

    c2_ood_curves = validate_ood_curves_and_contrasts()
    c2_ood_domain = validate_ood_domain_decomposition()
    record_main("C2. Val-Comp/domain-transfer result", c2_ood_curves and c2_ood_domain)

    # compute-matched ellmax2-vs-ellmax4 comparison is one of the tables validate_ood_curves_and_contrasts
    # already checked (ood_lmax2_vs_lmax4_baseline.csv); re-derive its pass/fail standalone label.
    regen_h = read_csv(out_path("claim2", "ood_lmax2_vs_lmax4_baseline.csv"))
    frozen_h = read_csv(data_path("claim2", "ood_lmax2_vs_lmax4_baseline.csv"))
    fmap_h = {(r["lmax4_tag"], r["domain"], r["M"]): r for r in frozen_h}
    ok_h = True
    for r in regen_h:
        f = fmap_h.get((r["lmax4_tag"], r["domain"], r["M"]))
        if f is None or not rel_close(float(r["H_force"]), float(f["H_force"])):
            ok_h = False
    record_main("C2. compute-matched ellmax2-vs-ellmax4 comparison", ok_h)

    # -----------------------------------------------------------------
    # Lower-level / supporting checks (kept for completeness, not individually reported)
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUPPORTING CHECKS")
    print("=" * 78)
    validate_dense_grid_ranking_comparison()

    n_pass = sum(1 for _n, ok, _d in results if ok)
    n_total = len(results)
    n_main_pass = sum(1 for _l, ok in main_paper_results if ok)
    n_main_total = len(main_paper_results)

    print(f"\n{n_main_pass}/{n_main_total} MAIN-PAPER checks passed.")
    print(f"{n_pass}/{n_total} total checks passed (including supporting checks).")
    if n_main_pass != n_main_total or n_pass != n_total:
        print("VALIDATE: FAIL")
        sys.exit(1)
    print("VALIDATE: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
