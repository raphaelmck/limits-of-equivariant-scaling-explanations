#!/usr/bin/env python3
"""Reproduction of data/claim1/force_scaling.csv's headline gamma/logA/r2/rmse_log fits.

AUDIT_STAGE2.md category (b) item 1 flagged force_scaling.csv as NOT reproducible from
data/claim1/ alone, because only the already-fitted summary had been copied into the clean
layer -- the per-checkpoint (FLOPs, force_mse_norm) table backing the fit was absent. This gap
is now closed: data/claim1/four_arch_force_checkpoint_table.csv (the 501-row REPAIRED
per-checkpoint table, copied verbatim from
dev-equivariant-scaling-laws/analysis_outputs/four_arch_force_scaling_repaired_2026_08_14/) is
the raw input, and this script re-derives the fit from it using the exact procedure implemented
in src/force_scaling_fit.py (itself a line-for-line pure-stdlib port of that repo's
frontier_fit.py + run_analysis.py collect_fits()/main() logic for the primary force_mse_norm
metric, fit_domain="common", truncation_frac=1.0 row -- the row that appears in the frozen
force_scaling.csv).

What IS reproduced: for each architecture with >=3 distinct widths among its metric_valid==True
checkpoints (all four architectures here: MPNN, MC-EGNN, GemNet-OC, eSEN), the step-frontier
L*_a(C), the common compute interval [max_a C_min_a, min_a C_max_a] intersected across all
four, and the headline OLS fit (gamma, logA, r2, rmse_log, C_min, C_max, decades,
n_frontier_owners_in_range) over 256 equally-spaced log-C grid points spanning that interval
(truncation_frac=1.0, i.e. untruncated).

What is NOT reproduced: the frozen file's gamma_event_point_ols and gamma_nls_rawloss
sensitivity-variant columns (fit over the unequally-spaced frontier event points themselves,
and a nonlinear least-squares fit on raw un-logged loss respectively) -- these are documented
in frontier_fit.py as "SENSITIVITY ONLY", not the headline number, and the NLS variant requires
scipy.optimize.curve_fit, which is out of scope for this pure-stdlib repro layer. Both columns
are written as empty in this script's output so the row shape still lines up with the frozen
file's columns for comparison purposes.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.io_utils import data_path, out_path, read_csv, write_csv
from src.force_scaling_fit import build_frontier, common_interval, fit_loglog, truncate_upper

ARCH_LABEL = {"mpnn": "MPNN", "egnn": "MC-EGNN", "gemnet_oc": "GemNet-OC", "esen": "eSEN (lmax=4)"}
ARCH_ORDER = ["mpnn", "egnn", "gemnet_oc", "esen"]
LOSS_COL = "force_mse_norm"
C_COL = "C"
MIN_WIDTHS = 3
TRUNCATION_FRAC = 1.0  # the only truncation the frozen force_scaling.csv reports


def _load_valid_rows():
    raw = read_csv(data_path("claim1", "four_arch_force_checkpoint_table.csv"))
    rows = []
    for r in raw:
        if r["metric_valid"] != "True":
            continue
        r = dict(r)
        r["width"] = int(r["width"])
        r[C_COL] = float(r[C_COL])
        r[LOSS_COL] = float(r[LOSS_COL])
        rows.append(r)
    return rows


def build() -> list[dict]:
    rows = _load_valid_rows()
    by_arch = {a: [r for r in rows if r["architecture"] == a] for a in ARCH_ORDER}

    prim_arches = [a for a in ARCH_ORDER
                   if len({r["width"] for r in by_arch[a]}) >= MIN_WIDTHS]

    frontiers = {a: build_frontier(by_arch[a], LOSS_COL, C_COL) for a in prim_arches}
    domains = {a: (min(r[C_COL] for r in by_arch[a]), max(r[C_COL] for r in by_arch[a]))
               for a in prim_arches}

    clo, chi = common_interval(domains)
    # clamp into every architecture's own observed domain (matches run_analysis.py; a no-op
    # here since each frontier's own C range equals its architecture's observed domain)
    clo = max(clo, max(d[0] for d in domains.values()))
    chi = min(chi, min(d[1] for d in domains.values()))

    tlo, thi = truncate_upper(clo, chi, TRUNCATION_FRAC)

    out_rows = []
    for a in prim_arches:
        fit = fit_loglog(frontiers[a], tlo, thi, LOSS_COL, C_COL)
        if fit is None:
            continue
        out_rows.append({
            "architecture": ARCH_LABEL[a],
            "architecture_key": a,
            "metric": LOSS_COL,
            "fit_domain": "common",
            "truncation_frac": TRUNCATION_FRAC,
            "gamma": fit["gamma"],
            "gamma_event_point_ols": "",
            "gamma_nls_rawloss": "",
            "logA": fit["logA"],
            "r2": fit["r2"],
            "rmse_log": fit["rmse_log"],
            "C_min_flops": fit["C_min"],
            "C_max_flops": fit["C_max"],
            "decades": fit["decades"],
            "n_frontier_owners_in_range": fit["n_frontier_owners_in_range"],
            "source_file": "recomputed from data/claim1/four_arch_force_checkpoint_table.csv "
                            "via src/force_scaling_fit.py",
        })
    return out_rows


def main():
    rows = build()
    out = out_path("claim1", "force_scaling_recomputed.csv")
    write_csv(out, rows)
    print(f"wrote {out} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['architecture']:<14} gamma={r['gamma']:.6f} logA={r['logA']:.6f} "
              f"r2={r['r2']:.6f} n_owners_in_range={r['n_frontier_owners_in_range']}")


if __name__ == "__main__":
    main()
