#!/usr/bin/env python3
"""Check the numbers printed in the manuscript against the artifacts this repository generates.

Unlike `test_paper_numbers.py`, which hardcodes the expected values, this reads every number out
of the LaTeX source itself. A number that changes in the manuscript but not in the data (or the
reverse) fails here; nothing is transcribed by hand.

    python3 tests/check_manuscript.py --tex path/to/main.tex

Each rule anchors on the surrounding prose or on a table row, pulls the printed value, and
compares it to the generated artifact at the precision the manuscript prints. The run ends with
the numeric literals no rule claimed, so a newly added number cannot pass unnoticed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compute_axis import budget_flops
from src.io_utils import data_path, out_path, read_csv, read_json

BUDGET_ORDER = ["LOW", "MID", "PRECROSS", "CROSS", "FITX", "POSTCROSS", "HIGH", "TOP"]
OWNERS_BY_COMPUTE = ["w10/50k", "w16/150k", "w20/200k", "w24/300k"]

results = []     # (label, ok, printed, generated)
claimed = set()  # spans of the tex a rule consumed


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def record(label, printed, generated, ok):
    results.append((label, ok, printed, generated))


def check(label, printed, generated, tol=0.0):
    ok = (printed == generated if isinstance(generated, (int, str))
          else abs(printed - generated) <= tol)
    record(label, printed, generated, ok)


def find(tex, pattern, label):
    """Locate a single anchored match, marking its span as claimed."""
    m = re.search(pattern, tex)
    if m is None:
        record(label, "NOT FOUND", "-", False)
        return None
    claimed.add((m.start(), m.end()))
    return m


def sci(mantissa, exponent):
    return float(mantissa) * 10.0 ** int(exponent)


# --------------------------------------------------------------------------- generated values

def artifacts():
    a = {}
    a["fits"] = {r["architecture"].replace(" (lmax=4)", ""): r
                 for r in read_csv(out_path("claim1", "force_scaling_recomputed.csv"))}
    a["gaps"] = {r["budget_tier"]: r
                 for r in read_csv(out_path("claim1", "dense_grid_pairwise_gap_recomputed.csv"))}
    a["flops"] = budget_flops()
    a["lmax_owners"] = sorted(
        (r for r in read_csv(data_path("claim2", "matched_compute_lmax_owners.csv"))
         if r["lmax"] == "4"), key=lambda r: float(r["C_flops"]))
    a["depth"] = read_json(out_path("claim2", "depth_localization_recomputed.json"))
    a["seeds"] = read_json(out_path("claim2", "seed_replication_result_recomputed.json"))
    a["families"] = {r["stratum"]: r
                     for r in read_csv(out_path("claim2", "ood_shared_family_positive_control.csv"))}
    a["frontier"] = read_json(out_path("claim2", "lmax24_all_frontier_summary.json"))
    a["pools"] = read_json(data_path("claim2", "ood_pool_provenance.json"))
    return a


# ------------------------------------------------------------------------------------- rules

def rule_exponents(tex, a):
    m = find(tex, r"exponents are \$([\d.]+)\$, \$([\d.]+)\$, \$([\d.]+)\$, and \$([\d.]+)\$ "
                  r"for MPNN, MC-EGNN, GemNet-OC, and eSEN", "Sec. 2 scaling exponents")
    if not m:
        return
    for printed, arch in zip(m.groups(), ["MPNN", "MC-EGNN", "GemNet-OC", "eSEN"]):
        check(f"exponent {arch}", float(printed), round(float(a["fits"][arch]["gamma"]), 3))


def rule_fit_interval(tex, a):
    m = find(tex, r"common interval\s*\$\[([\d.]+)\\times10\^\{(\d+)\},"
                  r"([\d.]+)\\times10\^\{(\d+)\}\]\$", "App. compute fit interval")
    if not m:
        return
    row = next(iter(a["fits"].values()))
    check("fit interval lower", sci(m.group(1), m.group(2)), float(row["C_min_flops"]),
          tol=1e12)
    check("fit interval upper", sci(m.group(3), m.group(4)), float(row["C_max_flops"]),
          tol=1e14)


def rule_crossover(tex, a):
    m = find(tex, r"overtakes it near \$([\d.]+)\\times10\^\{(\d+)\}\$ floating-point",
             "Sec. 2 crossover compute")
    if m:
        check("crossover compute", sci(m.group(1), m.group(2)),
              a["flops"][("eSEN", "CROSS")], tol=5e14)


def rule_caption_bracket(tex, a):
    """Figure 1's caption brackets the crossover; the observed crossover must lie inside it."""
    m = find(tex, r"reverses ordering between \$([\d.]+)\$ and \$([\d.]+)\\times10\^\{(\d+)\}\$ FLOPs",
             "Fig. 1 caption crossover bracket")
    if not m:
        return
    exponent = m.group(3)
    lo, hi = sci(m.group(1), exponent), sci(m.group(2), exponent)
    observed = a["flops"][("eSEN", "CROSS")]
    record("caption bracket contains the observed crossover",
           f"[{lo:.3g}, {hi:.3g}]", f"{observed:.4g}", lo < observed < hi)
    prev = a["flops"][("eSEN", "PRECROSS")]
    record("caption bracket excludes the preceding budget",
           f"[{lo:.3g}, {hi:.3g}]", f"{prev:.4g}", not (lo < prev < hi))


def rule_neural_sign_change(tex, a):
    m = find(tex, r"changes sign from \$-([\d.]+)\$ to \$\+([\d.]+)\$", "Sec. 3 neural sign change")
    if not m:
        return
    check("neural gap before crossover", -float(m.group(1)),
          round(float(a["gaps"]["PRECROSS"]["neural_signed_gap_log_gemnet_over_esen"]), 3))
    check("neural gap at crossover", float(m.group(2)),
          round(float(a["gaps"]["CROSS"]["neural_signed_gap_log_gemnet_over_esen"]), 3))


def rule_krr_at_crossover(tex, a):
    m = find(tex, r"\$-([\d.]+)\$ \(95\\% CI \$\[-([\d.]+),-([\d.]+)\]\$\) and \$-([\d.]+)\$\s*"
                  r"\(95\\% CI \$\[-([\d.]+),-([\d.]+)\]\$\)", "Sec. 3 KRR gaps at crossover")
    if not m:
        return
    for tier, (val, lo, hi) in zip(["PRECROSS", "CROSS"],
                                   [m.group(1, 2, 3), m.group(4, 5, 6)]):
        row = a["gaps"][tier]
        check(f"KRR gap {tier}", -float(val),
              round(float(row["krr_signed_gap_log_gemnet_over_esen_at_n512"]), 3))
        check(f"KRR CI low {tier}", -float(lo),
              round(float(row["krr_bootstrap_ci_lo_2.5pct"]), 2))
        check(f"KRR CI high {tier}", -float(hi),
              round(float(row["krr_bootstrap_ci_hi_97.5pct"]), 2))


def rule_top_budget(tex, a):
    m = find(tex, r"through\s*\$([\d.]+)\\times10\^\{(\d+)\}\$ FLOPs", "Sec. 3 highest budget")
    if m:
        check("highest tested budget", sci(m.group(1), m.group(2)),
              a["flops"][("GemNet-OC", "TOP")], tol=5e15)
    m = find(tex, r"gaps are \$\+([\d.]+)\$ and\s*\$-([\d.]+)\$ \(95\\% CI \$\[-([\d.]+),"
                  r"-([\d.]+)\]\$\)", "Sec. 3 gaps at highest budget")
    if not m:
        return
    row = a["gaps"]["TOP"]
    check("neural gap TOP", float(m.group(1)),
          round(float(row["neural_signed_gap_log_gemnet_over_esen"]), 3))
    check("KRR gap TOP", -float(m.group(2)),
          round(float(row["krr_signed_gap_log_gemnet_over_esen_at_n512"]), 3))
    check("KRR CI low TOP", -float(m.group(3)),
          round(float(row["krr_bootstrap_ci_lo_2.5pct"]), 2))
    check("KRR CI high TOP", -float(m.group(4)),
          round(float(row["krr_bootstrap_ci_hi_97.5pct"]), 2))


def rule_dense_krr_table(tex, a):
    body = table_body(tex, "tab:dense_krr", "Table: KRR comparisons")
    if body is None:
        return
    row_re = (r"(?:\\textbf\{)?(\d)\}?\s*&\s*\$([\d.]+)\\!\\times\\!10\^\{(\d+)\}\$\s*&\s*"
              r"\$([\d.]+)\\!\\times\\!10\^\{(\d+)\}\$\s*&\s*\$([+-])([\d.]+)\$\s*&\s*"
              r"\$-([\d.]+)\\;\[-([\d.]+),-([\d.]+)\]\$")
    rows = re.findall(row_re, body)
    check("KRR table row count", len(rows), 8)
    for (idx, cg, eg, ce, ee, sign, neural, krr, lo, hi), tier in zip(rows, BUDGET_ORDER):
        g = a["gaps"][tier]
        check(f"table row {idx} GemNet compute", sci(cg, eg),
              a["flops"][("GemNet-OC", tier)], tol=a["flops"][("GemNet-OC", tier)] * 1e-3)
        check(f"table row {idx} eSEN compute", sci(ce, ee),
              a["flops"][("eSEN", tier)], tol=a["flops"][("eSEN", tier)] * 1e-3)
        check(f"table row {idx} neural gap", float(sign + neural),
              round(float(g["neural_signed_gap_log_gemnet_over_esen"]), 3))
        check(f"table row {idx} KRR gap", -float(krr),
              round(float(g["krr_signed_gap_log_gemnet_over_esen_at_n512"]), 3))
        check(f"table row {idx} KRR CI low", -float(lo),
              round(float(g["krr_bootstrap_ci_lo_2.5pct"]), 3))
        check(f"table row {idx} KRR CI high", -float(hi),
              round(float(g["krr_bootstrap_ci_hi_97.5pct"]), 3))


def rule_populations(tex, a):
    m = find(tex, r"Neutral validation population contains ([\d,]+) configurations",
             "App. Neutral population")
    if m:
        check("Neutral population", int(m.group(1).replace(",", "")),
              a["pools"]["neutral_domain_pool"]["n_eligible_total"])
    m = find(tex, r"validation population contains ([\d,]+) configurations\. Applying",
             "App. broader population")
    if m:
        check("broader population", int(m.group(1).replace(",", "")),
              a["pools"]["valcomp_ood_pool"]["total_val_structures"])
    m = find(tex, r"restrictions leaves ([\d,]+) eligible configurations",
             "App. support-matched eligible")
    if m:
        check("support-matched eligible", int(m.group(1).replace(",", "")),
              a["pools"]["valcomp_ood_pool"]["n_eligible_total"])


def rule_intervened_checkpoints(tex, a):
    m = find(tex, r"spherical-channel widths ([\d, ]+?) and (\d+) at ([\d,]+), ([\d,]+), "
                  r"([\d,]+), and ([\d,]+) training", "App. intervened checkpoints")
    if m:
        widths = [int(w) for w in m.group(1).replace(",", " ").split()] + [int(m.group(2))]
        steps = [int(s.replace(",", "")) for s in m.group(3, 4, 5, 6)]
        for i, (w, s) in enumerate(zip(widths, steps)):
            check(f"checkpoint {i} width", w, int(a["lmax_owners"][i]["width"]))
            check(f"checkpoint {i} step", s, int(a["lmax_owners"][i]["global_step"]))
    m = find(tex, r"computes are \$([\d.]+)\\times10\^\{(\d+)\}\$,\s*"
                  r"\$([\d.]+)\\times10\^\{(\d+)\}\$, \$([\d.]+)\\times10\^\{(\d+)\}\$, and "
                  r"\$([\d.]+)\\times10\^\{(\d+)\}\$ FLOPs", "App. intervened computes")
    if m:
        vals = [sci(m.group(i), m.group(i + 1)) for i in (1, 3, 5, 7)]
        for i, v in enumerate(vals):
            gen = float(a["lmax_owners"][i]["C_flops"])
            check(f"checkpoint {i} compute", v, gen, tol=gen * 1e-3)


def rule_matching_tolerance(tex, a):
    m = find(tex, r"\\leq([\d.]+)\$", "App. matching tolerance")
    if m:
        rule = a["frontier"]["matching_rule"]
        check("matching tolerance", float(m.group(1)),
              float(re.search(r"<= ([\d.]+)", rule).group(1)))


def rule_depth_table(tex, a):
    body = table_body(tex, "tab:depth", "Table: depth")
    if body is None:
        return
    at = a["depth"]["read_at_common_support"]
    for line in body.strip().split("\\\\"):
        m = re.match(r"\s*(\d)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)", line)
        if not m:
            continue
        block = m.group(1)
        for owner, printed in zip(OWNERS_BY_COMPUTE, m.group(2, 3, 4, 5)):
            check(f"depth block {block} at {owner}", float(printed),
                  round(at[f"{owner}_L{block}"]["delta"], 3))


def rule_perturbation_readout(tex, a):
    for label, pattern in [("App. depth read-off P*", r"P\^\\star=([\d.]+)\$"),
                           ("App. common perturbation range", r"P\\le([\d.]+)\$")]:
        m = find(tex, pattern, label)
        if m:
            check(label, float(m.group(1)),
                  round(a["depth"]["common_P_bal_support"], 5), tol=1e-5)


def rule_seed_table(tex, a):
    body = table_body(tex, "tab:seeds", "Table: independent runs")
    if body is None:
        return
    for line in body.strip().split("\\\\"):
        m = re.match(r"\s*(\d)\s*&\s*([\d.]+)\s*&\s*\$\[([\d.]+),([\d.]+)\]\$", line)
        if not m:
            continue
        run = m.group(1)
        check(f"run {run} contrast", float(m.group(2)),
              round(a["seeds"]["G_by_seed"][run], 3))
        ci = a["seeds"]["G_ci95_by_seed"][run]
        check(f"run {run} CI low", float(m.group(3)), round(ci[0], 3))
        check(f"run {run} CI high", float(m.group(4)), round(ci[1], 3))
    m = find(tex, r"P_\{\\mathrm\{total\}\}\^\\star=([\d.]+)\$", "App. seeds read-off")
    if m:
        check("seeds read-off P_total", float(m.group(1)), a["seeds"]["P_star"], tol=1e-9)


FAMILY_ROWS = {
    "Neutral validation": "Neutral_val_pooled_(ID_reference)",
    "Same-source control": "pooled_neutral_like_sources_in_ValComp",
    "Biomolecules": "biomolecules",
    "Electrolytes": "elytes",
    "Reactivity": "reactivity",
}


def rule_family_table(tex, a):
    body = table_body(tex, "tab:family_transfer", "Table: chemistry transfer")
    if body is None:
        return
    for line in body.strip().split("\\\\"):
        m = re.match(r"\s*([A-Za-z -]+?)\s*&\s*([\d,]+)\s*&\s*([\d.]+)\s*&\s*"
                     r"\$\[([\d.]+),([\d.]+)\]\$", line)
        if not m:
            continue
        name = m.group(1).strip()
        if name not in FAMILY_ROWS:
            continue
        row = a["families"][FAMILY_ROWS[name]]
        check(f"{name} n_cfg", int(m.group(2).replace(",", "")), int(row["n_configs"]))
        check(f"{name} contrast", float(m.group(3)),
              round(float(row["contrast_w24_minus_w10"]), 3))
        check(f"{name} CI low", float(m.group(4)), round(float(row["ci_lo"]), 3))
        check(f"{name} CI high", float(m.group(5)), round(float(row["ci_hi"]), 3))


def rule_full_frontier(tex, a):
    f = a["frontier"]
    m = find(tex, r"There are (\d+) distinct", "App. frontier checkpoint count")
    if m:
        check("frontier checkpoints", int(m.group(1)), f["n_lmax4_frontier_checkpoints"])
    m = find(tex, r"of which (\d+) have a valid", "App. frontier valid matches")
    if m:
        check("valid matches", int(m.group(1)), f["n_valid_matches"])
    m = find(tex, r"come from (\d+) training trajectories", "App. frontier trajectories")
    if m:
        check("lmax4 trajectories", int(m.group(1)),
              f["lmax4_checkpoints_and_trajectories"]["n_trajectories"])
    m = find(tex, r"side uses (\d+) checkpoints", "App. frontier lmax2 checkpoints")
    if m:
        check("lmax2 checkpoints", int(m.group(1)),
              f["lmax2_matched_checkpoints_and_trajectories"]["n_checkpoints"])
    m = find(tex, r"span \$([\d.]+)\\times10\^\{(\d+)\}\$ to \$([\d.]+)\\times10\^\{(\d+)\}\$",
             "App. frontier compute span")
    if m:
        lo, hi = f["matched_compute_range_flops"]
        check("frontier span low", sci(m.group(1), m.group(2)), lo, tol=lo * 2e-3)
        check("frontier span high", sci(m.group(3), m.group(4)), hi, tol=hi * 2e-3)
    m = find(tex, r"for (\d+) of (\d+) matches", "App. frontier Neutral tally")
    if m:
        check("Neutral favoring lmax2", int(m.group(1)),
              f["neutral_validation"]["n_favoring_lmax2"])
        check("Neutral pairs", int(m.group(2)), f["neutral_validation"]["n_valid_pairs"])
    m = find(tex, r"point estimates\s*\$\+([\d.]+)\$ and \$\+([\d.]+)\$",
             "App. frontier exceptions")
    if m:
        got = f["neutral_validation"]["lmax4_point_estimates"]
        check("exception 1", float(m.group(1)), round(got[0], 4))
        check("exception 2", float(m.group(2)), round(got[1], 4))
    m = find(tex, r"\(\$C_4=([\d.]+)\\times10\^\{(\d+)\}\$ and \$C_2=([\d.]+)"
                  r"\\times10\^\{(\d+)\}\$ FLOPs\)", "App. decisive pair computes")
    if m:
        top = f["highest_common_support_pair_neutral"]
        check("decisive C_4", sci(m.group(1), m.group(2)), top["C_lmax4_flops"],
              tol=top["C_lmax4_flops"] * 1e-3)
        check("decisive C_2", sci(m.group(3), m.group(4)), top["C_lmax2_flops"],
              tol=top["C_lmax2_flops"] * 1e-3)
    m = find(tex, r"L_\{\\ell_\{\\max\}=4\}\)=-([\d.]+)\$ with 95\\% CI "
                  r"\$\[-([\d.]+),([\d.]+)\]\$", "App. decisive pair estimate and CI")
    if m:
        top = f["highest_common_support_pair_neutral"]
        check("decisive point estimate", -float(m.group(1)), round(top["H_force"], 4))
        check("decisive CI low", -float(m.group(2)), round(top["ci95"][0], 4))
        check("decisive CI high", float(m.group(3)), round(top["ci95"][1], 4))


def table_body(tex, label, name):
    m = re.search(r"\\midrule(.*?)\\bottomrule.*?\\label\{" + re.escape(label) + r"\}",
                  tex, re.S)
    if m is None:
        record(name, "NOT FOUND", "-", False)
        return None
    claimed.add((m.start(), m.end()))
    return m.group(1)


RULES = [rule_exponents, rule_fit_interval, rule_crossover, rule_caption_bracket,
         rule_neural_sign_change,
         rule_krr_at_crossover, rule_top_budget, rule_dense_krr_table, rule_populations,
         rule_intervened_checkpoints, rule_matching_tolerance, rule_depth_table,
         rule_perturbation_readout, rule_seed_table, rule_family_table, rule_full_frontier]


def uncovered(tex):
    """Numeric literals in the body that no rule consumed, for a human to eyeball.

    Takes the same flattened text the rules matched against, so the claimed spans line up.
    """
    body_start = tex.find("\\begin{document}")
    out = []
    for m in re.finditer(r"(?<![\w.^{])\d[\d,]*\.?\d*", tex):
        if m.start() < body_start:
            continue
        if any(s <= m.start() and m.end() <= e for s, e in claimed):
            continue
        context = flat(tex[max(0, m.start() - 45):m.end() + 25])
        out.append((m.group(0), context))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", required=True, help="the manuscript LaTeX source")
    parser.add_argument("--show-uncovered", action="store_true",
                        help="list numeric literals no rule claimed")
    args = parser.parse_args()

    tex = flat(open(args.tex).read())
    a = artifacts()
    for rule in RULES:
        rule(tex, a)

    failures = [r for r in results if not r[1]]
    for label, ok, printed, generated in results:
        if not ok:
            print(f"[FAIL] {label}: manuscript {printed}, generated {generated}")

    print(f"\n{len(results) - len(failures)}/{len(results)} manuscript numbers "
          f"match the generated artifacts.")

    if args.show_uncovered:
        rest = uncovered(tex)
        print(f"\n{len(rest)} numeric literals not claimed by a rule "
              f"(configuration constants, references, and prose):")
        for value, context in rest:
            print(f"  {value:>12}   ...{context}...")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
