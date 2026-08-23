#!/usr/bin/env python3
"""make validate: compares every table `make analysis` regenerates in analysis_out/ against the
corresponding frozen table already in data/claim1/ or data/claim2/. Prints a pass/fail summary
per table and exits non-zero if any scientifically meaningful discrepancy is found (wrong sign,
a CI that no longer excludes/includes zero where the frozen one did, a point-estimate off by more
than tolerance). Passing floating-point noise never fails.

STAGE-2 REWRITE (see AUDIT_STAGE2.md): every bootstrap in this repo now uses real numpy
(numpy.random.default_rng or numpy.random.RandomState, matching whichever API the corresponding
upstream analysis script actually used), not a same-shape stdlib substitute. Wherever the
upstream bootstrap's seed AND exact call structure could be recovered (which turned out to be
almost everywhere -- see AUDIT_STAGE2.md sec. 1), this reproduces the frozen CI BIT-EXACTLY, and
this file's tolerance for that table is tightened accordingly from "CI overlap" to
REL_TOL_EXACT-relative bound matching. Overlap-only checks remain ONLY for the small number of
tables where the frozen file's own bootstrap seed is genuinely undocumented (dose_response,
compensation, same_width_control_B, run_replication_control_C -- see AUDIT_STAGE2.md sec. 3);
those are explicitly labeled below.

Tolerances:
  - REL_TOL_EXACT = 1e-6 relative, used for every point estimate with NO randomness, and for
    every CI this pass established is a BIT-EXACT bootstrap reproduction (in practice every one
    of these matches to < 1e-9 relative -- pure floating-point noise from a different summation
    order, not from a different RNG draw).
  - CI OVERLAP + zero-exclusion-sign agreement, used ONLY for the 4 tables whose frozen bootstrap
    seed is genuinely undocumented (see AUDIT_STAGE2.md sec. 3) -- their point estimates (always
    deterministic) are still checked at REL_TOL_EXACT.

This file's `main()` is organized around the ordered list of 8 MAIN-PAPER checks the user
requested by name (see MAIN_PAPER_CHECKS below and AUDIT_STAGE2.md's final section) -- each one
is printed as "C1./C2. <label>" so they are unambiguously identifiable in the output, in addition
to (and built on top of) the lower-level per-table checks that were already here.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import data_path, out_path, read_csv, read_json

REL_TOL_EXACT = 1e-6

results = []  # (name, passed: bool, detail: str)
main_paper_results = []  # (label, passed: bool) -- the 8 named checks, in order


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

def validate_ranking_comparison():
    name = "claim1/ranking_comparison.csv (exact, no randomness)"
    regen = read_csv(out_path("claim1", "ranking_comparison.csv"))
    frozen = read_csv(data_path("claim1", "ranking_comparison.csv"))
    key = lambda r: (r["budget_tier"], r["architecture"])
    fmap = {key(r): r for r in frozen}
    ok = True
    details = []
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
    record(name, ok, "; ".join(details))
    return ok


def validate_pairwise_concordance_direction():
    name = "claim1/pairwise_concordance.csv (direction check only)"
    regen = read_csv(out_path("claim1", "pairwise_concordance_direction_check.csv"))
    ok = all(r["nn_favors_match"] == "True" and r["krr_favors_match"] == "True" for r in regen)
    n_mismatch = sum(1 for r in regen if r["nn_favors_match"] != "True" or r["krr_favors_match"] != "True")
    record(name, ok, f"{n_mismatch} direction mismatches")
    return ok


def validate_pairwise_concordance():
    """BIT-EXACT (Stage-2 fix): np.random.RandomState(20260822), single shared RNG threaded
    through all 18 pairs in upstream's exact call order -- see
    scripts/claim1/build_pairwise_concordance.py. Tolerance tightened from overlap to exact-bound
    match."""
    name = "claim1/pairwise_concordance.csv (log_ratio + CI BIT-EXACT, status match)"
    regen = read_csv(out_path("claim1", "pairwise_concordance_recomputed.csv"))
    frozen = read_csv(data_path("claim1", "pairwise_concordance.csv"))
    key = lambda r: (r["budget_tier"], r["architecture_a"], r["architecture_b"])
    fmap = {key(r): r for r in frozen}
    ok = True
    details = []
    for r in regen:
        f = fmap.get(key(r))
        if f is None:
            ok = False
            details.append(f"missing frozen row for {key(r)}")
            continue
        if not rel_close(float(r["log_ratio_mean_a_over_b_at_n512"]), float(f["log_ratio_mean_a_over_b_at_n512"])):
            ok = False
            details.append(f"{key(r)}: log_ratio mismatch")
        lo, hi = float(r["bootstrap_ci_lo_2.5pct"]), float(r["bootstrap_ci_hi_97.5pct"])
        flo, fhi = float(f["bootstrap_ci_lo_2.5pct"]), float(f["bootstrap_ci_hi_97.5pct"])
        if not ci_close(lo, hi, flo, fhi):
            ok = False
            details.append(f"{key(r)}: CI not bit-exact (mine=[{lo},{hi}] frozen=[{flo},{fhi}])")
        if r["status"] != f["status"]:
            ok = False
            details.append(f"{key(r)}: status mismatch ({r['status']} vs {f['status']})")
    record(name, ok, "; ".join(details[:10]))
    return ok


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


# ===========================================================================
# Claim 2 -- dose-response family (overlap-only: frozen bootstrap seed genuinely undocumented)
# ===========================================================================

def validate_dose_response():
    name = "claim2/dose_response_summary.json (Delta exact, CI overlap [undocumented frozen seed])"
    regen = read_json(out_path("claim2", "dose_response_recomputed.json"))
    frozen = read_json(data_path("claim2", "dose_response_summary.json"))
    fmap = {(g["budget"], g["ell"], g["alpha"]): g for g in frozen["grid"]}
    ok = True
    details = []
    for g in regen["grid"]:
        key = (g["budget"], g["ell"], g["alpha"])
        f = fmap.get(key)
        if f is None:
            ok = False
            details.append(f"missing frozen cell {key}")
            continue
        if not rel_close(g["Delta"], f["Delta"]):
            ok = False
            details.append(f"{key}: Delta mismatch {g['Delta']} vs {f['Delta']}")
        lo, hi = g["Delta_ci95"]
        flo, fhi = f["Delta_ci95"]
        if not ci_overlap(lo, hi, flo, fhi):
            ok = False
            details.append(f"{key}: CI no overlap")
        if excludes_zero(lo, hi) != excludes_zero(flo, fhi):
            ok = False
            details.append(f"{key}: CI zero-exclusion sign differs")
    record(name, ok, "; ".join(details[:10]))
    return ok


def validate_compensation():
    name = "claim2/compensation_summary.json (C_ell exact, CI overlap [undocumented frozen seed])"
    regen = read_json(out_path("claim2", "compensation_recomputed.json"))
    frozen = read_json(data_path("claim2", "compensation_summary.json"))
    ok = True
    details = []
    for pair, pd in regen["pairs"].items():
        for ell, ad in pd["ells"].items():
            for alpha, cell in ad.items():
                try:
                    f = frozen["pairs"][pair]["ells"][ell]["alphas"][alpha]
                except KeyError:
                    ok = False
                    details.append(f"missing frozen cell {pair}/{ell}/{alpha}")
                    continue
                if not rel_close(cell["C_ell"], f["C_ell"]):
                    ok = False
                    details.append(f"{pair}/{ell}/{alpha}: C_ell mismatch")
                lo, hi = cell["C_ell_ci95"]
                flo, fhi = f["C_ell_ci95"]
                if not ci_overlap(lo, hi, flo, fhi):
                    ok = False
                    details.append(f"{pair}/{ell}/{alpha}: CI no overlap")
                if excludes_zero(lo, hi) != f["C_ell_excludes_zero"]:
                    ok = False
                    details.append(f"{pair}/{ell}/{alpha}: zero-exclusion sign differs")
    record(name, ok, "; ".join(details[:10]))
    return ok


def validate_same_width_control_B():
    name = "claim2/same_width_control_B_summary.json (C_ell exact, CI overlap [undocumented frozen seed])"
    regen = read_json(out_path("claim2", "same_width_control_B_recomputed.json"))
    frozen = read_json(data_path("claim2", "same_width_control_B_summary.json"))
    ok = True
    details = []
    for pair, pd in regen["same_width"].items():
        for cid, cell in pd["cells"].items():
            try:
                f = frozen["same_width"][pair]["cells"][cid]
            except KeyError:
                ok = False
                details.append(f"missing frozen cell {pair}/{cid}")
                continue
            if not rel_close(cell["C_ell"], f["C_ell"]):
                ok = False
                details.append(f"{pair}/{cid}: C_ell mismatch")
            lo, hi = cell["C_ell_ci95"]
            flo, fhi = f["C_ell_ci95"]
            if not ci_overlap(lo, hi, flo, fhi):
                ok = False
                details.append(f"{pair}/{cid}: CI no overlap")
            if cell["sign"] != f["sign"]:
                ok = False
                details.append(f"{pair}/{cid}: sign mismatch")
    record(name, ok, "; ".join(details[:10]))
    return ok


def validate_run_replication_control_C():
    name = "claim2/run_replication_control_C_summary.json (delta/diff exact, CI overlap [undocumented frozen seed])"
    regen = read_json(out_path("claim2", "run_replication_control_C_recomputed.json"))
    frozen = read_json(data_path("claim2", "run_replication_control_C_summary.json"))
    ok = True
    details = []
    for run, rd in regen["runs"].items():
        for cid, cell in rd["cells"].items():
            f = frozen["runs"][run]["cells"][cid]
            if not rel_close(cell["delta"], f["delta"]):
                ok = False
                details.append(f"{run}/{cid}: delta mismatch")
            lo, hi = cell["delta_ci95"]
            flo, fhi = f["delta_ci95"]
            if not ci_overlap(lo, hi, flo, fhi):
                ok = False
                details.append(f"{run}/{cid}: CI no overlap")
    for cid, c in regen["across_run_contrasts"].items():
        f = frozen["across_run_contrasts"][cid]
        if not rel_close(c["difference"], f["difference"]):
            ok = False
            details.append(f"contrast/{cid}: difference mismatch")
        lo, hi = c["difference_ci95"]
        flo, fhi = f["difference_ci95"]
        if not ci_overlap(lo, hi, flo, fhi):
            ok = False
            details.append(f"contrast/{cid}: CI no overlap")
        if excludes_zero(lo, hi) != f["difference_ci_excludes_zero"]:
            ok = False
            details.append(f"contrast/{cid}: zero-exclusion sign differs")
    record(name, ok, "; ".join(details[:10]))
    return ok


# ===========================================================================
# Claim 2 -- BIT-EXACT tables (documented upstream seed recovered and matched)
# ===========================================================================

def validate_frontier_ell4_degree_balanced():
    """BIT-EXACT (Stage-2 fix): np.random.default_rng(20260820), fresh per (owner,alpha) cell --
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
    """BIT-EXACT (Stage-2 fix): np.random.default_rng(0) shared idx_sets, ported line-for-line
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
    """BIT-EXACT (Stage-2 fix) for ALL 3 depths (block 3, 6, AND 9) and ALL 6x4=... 12 depth
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


def validate_ood_absolute_effect_robustness():
    ok_all = True
    for fname, keys, pt, ci in [
        (
            "ood_absolute_effect_robustness.csv",
            ["tag", "domain", "M"],
            "absolute_excess_A",
            ("absolute_excess_A_ci_lo", "absolute_excess_A_ci_hi"),
        ),
        (
            "ood_absolute_effect_A_OOD_minus_A_Neutral.csv",
            ["tag", "M"],
            "A_OOD_minus_A_Neutral",
            ("ci_lo", "ci_hi"),
        ),
    ]:
        regen = read_csv(out_path("claim2", fname))
        frozen = read_csv(data_path("claim2", fname))
        fmap = {tuple(r[k] for k in keys): r for r in frozen}
        ok, details = True, []
        for r in regen:
            f = fmap.get(tuple(r[k] for k in keys))
            if f is None:
                ok = False
                details.append("missing row")
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
    """BIT-EXACT (Stage-2 gap closed): np.random.default_rng(20260823), the per-configuration
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
        record(f"claim2/{fname} (point + CI BIT-EXACT; Stage-2 gap closed)", ok, "; ".join(details[:10]))
        ok_all = ok_all and ok
    return ok_all


# ===========================================================================
# MAIN-PAPER CHECK LIST (8 named checks the user asked for by name)
# ===========================================================================

def main():
    print("=" * 78)
    print("MAIN-PAPER CHECKS (8, as named in the Stage-2 task)")
    print("=" * 78)

    c1_force_scaling = validate_force_scaling()
    record_main("C1. force-scaling exponents", c1_force_scaling)

    c1_ranking = validate_ranking_comparison()
    record_main("C1. 17/18 NN/KRR concordance", c1_ranking)

    validate_pairwise_concordance_direction()
    c1_pairwise = validate_pairwise_concordance()
    # the HIGH GemNet-OC/eSEN discordant pair specifically:
    regen_pw = read_csv(out_path("claim1", "pairwise_concordance_recomputed.csv"))
    high_row = next(
        r for r in regen_pw
        if r["budget_tier"] == "HIGH" and r["architecture_a"] == "GemNet-OC" and r["architecture_b"] == "eSEN"
    )
    high_discordant_ok = (
        high_row["status"] == "discordant"
        and rel_close(float(high_row["log_ratio_mean_a_over_b_at_n512"]), -0.38667807569008344)
        and ci_close(
            float(high_row["bootstrap_ci_lo_2.5pct"]), float(high_row["bootstrap_ci_hi_97.5pct"]),
            -0.6518903518249106, -0.12342702923828595,
        )
    )
    record_main(
        "C1. HIGH GemNet-OC/eSEN discordance + CI "
        f"(log_ratio={high_row['log_ratio_mean_a_over_b_at_n512']}, "
        f"CI=[{high_row['bootstrap_ci_lo_2.5pct']},{high_row['bootstrap_ci_hi_97.5pct']}])",
        c1_pairwise and high_discordant_ok,
    )

    c2_degree_balanced = validate_frontier_ell4_degree_balanced()
    record_main("C2. four-owner degree-balanced causal result", c2_degree_balanced)

    c2_seed_repl = validate_seed_replication_result()
    record_main("C2. 3/3 seed-sign replication + per-seed CIs", c2_seed_repl)

    c2_depth = validate_depth_localization()
    record_main("C2. depth-localization result", c2_depth)

    c2_ood_curves = validate_ood_curves_and_contrasts()
    c2_ood_domain = validate_ood_domain_decomposition()
    c2_ood_abs = validate_ood_absolute_effect_robustness()
    record_main("C2. Val-Comp/domain-transfer result", c2_ood_curves and c2_ood_domain and c2_ood_abs)

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
    # Lower-level / supporting checks (kept for completeness, not part of the 8 named checks)
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUPPORTING CHECKS (dose-response family, controls, appendix-only)")
    print("=" * 78)
    validate_dose_response()
    validate_compensation()
    validate_same_width_control_B()
    validate_run_replication_control_C()

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
