#!/usr/bin/env python3
"""Regenerate per-run Delta/CI and across-run (runA - runB) difference/CI for Control C
(fixed-init trajectory replicate -- NOT an independent seed draw, see
run_replication_type field) from RAW per-config rows in
data/claim2/run_replication_control_C.jsonl.

Category (a). Cross-run contrasts use the SAME bootstrap resample of configuration indices
for both runs (paired), matching the pairing convention used throughout Claim-2's bootstrap
files.

Determinism caveat: run_replication_control_C_summary.json documents no n_boot/seed -> n_boot=2000
convention with seed=0 fixed here; make validate checks CI overlap, not exact match.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.bootstrap import bootstrap_delta_ci, log_delta, paired_config_bootstrap, weighted_L, ci95
from src.io_utils import data_path, out_path, read_jsonl, write_json

N_BOOT = 2000
SEED = 0


def build() -> dict:
    rows = read_jsonl(data_path("claim2", "run_replication_control_C.jsonl"))
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)

    runs_out = {}
    per_run_cells = {}
    for run, rrows in by_run.items():
        baseline = next(r for r in rrows if r["kind"] == "baseline")
        cells_out = {}
        cell_rows = {}
        for r in rrows:
            if r["kind"] == "baseline":
                continue
            cell_id = f"ell{r['ell']}_alpha{r['alpha']:.2f}"
            res = bootstrap_delta_ci(
                baseline["per_config_sum"],
                baseline["per_config_natoms"],
                r["per_config_sum"],
                r["per_config_natoms"],
                n_boot=N_BOOT,
                seed=SEED,
            )
            cells_out[cell_id] = {
                "L_int": res["L_intervened"],
                "delta": res["delta"],
                "delta_ci95": res["delta_ci95"],
            }
            cell_rows[cell_id] = r
        runs_out[run] = {"baseline_L": weighted_L(baseline["per_config_sum"], baseline["per_config_natoms"]), "cells": cells_out}
        per_run_cells[run] = (baseline, cell_rows)

    run_names = list(per_run_cells.keys())
    contrasts = {}
    if len(run_names) == 2:
        run_a, run_b = run_names
        base_a, cells_a = per_run_cells[run_a]
        base_b, cells_b = per_run_cells[run_b]
        for cell_id in cells_a:
            if cell_id not in cells_b:
                continue
            ra, rb = cells_a[cell_id], cells_b[cell_id]
            delta_a_point = runs_out[run_a]["cells"][cell_id]["delta"]
            delta_b_point = runs_out[run_b]["cells"][cell_id]["delta"]
            diff_point = delta_a_point - delta_b_point

            def stat(Ls):
                return log_delta(Ls["base_a"], Ls["int_a"]) - log_delta(Ls["base_b"], Ls["int_b"])

            reps = paired_config_bootstrap(
                {
                    "base_a": (base_a["per_config_sum"], base_a["per_config_natoms"]),
                    "int_a": (ra["per_config_sum"], ra["per_config_natoms"]),
                    "base_b": (base_b["per_config_sum"], base_b["per_config_natoms"]),
                    "int_b": (rb["per_config_sum"], rb["per_config_natoms"]),
                },
                stat,
                n_boot=N_BOOT,
                seed=SEED,
            )
            lo, hi = ci95(reps)
            contrasts[cell_id] = {
                "delta_runA": delta_a_point,
                "delta_runB": delta_b_point,
                "difference": diff_point,
                "difference_ci95": [lo, hi],
                "difference_ci_excludes_zero": (lo > 0 or hi < 0),
            }
    return {"runs": runs_out, "across_run_contrasts": contrasts, "n_boot": N_BOOT, "bootstrap_seed": SEED}


def main():
    result = build()
    out = out_path("claim2", "run_replication_control_C_recomputed.json")
    write_json(out, result)
    print(f"wrote {out} ({len(result['runs'])} runs, {len(result['across_run_contrasts'])} contrasts)")


if __name__ == "__main__":
    main()
