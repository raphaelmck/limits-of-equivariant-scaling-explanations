#!/usr/bin/env python3
"""Regenerate the independent-training-run replication reported in the paper appendix: per-(run,
checkpoint) damage interpolated to P_total = 0.15 with its interval, and the high-minus-low
contrast G per run.

Raw inputs: data/claim2/seed_replication_raw.jsonl for runs 2 and 3; for run 1, the same arrays
the rest of the analysis uses, dose_response.jsonl for the low-compute checkpoint and
frontier_ell4_sensitivity_raw.jsonl for the high-compute one.

Metric. This experiment's pre-registered estimand uses the raw total perturbation norm
P_total = RMS(h_int - h_base) / RMS(h_base), read off each row, not the degree-balanced P_bal
used in Figure 2A. src/seed_replication.py exists as a separate module to keep the two apart.

Procedure:
  1. One shared list of bootstrap index sets, default_rng(0) over 2000 replicates, reused for
     every run and checkpoint, so contrasts stay paired.
  2. Per (run, checkpoint), bracket the two measured alphas whose P_total straddles 0.15,
     interpolate the point estimate, and interpolate each resampled damage value at the same
     weight before taking percentiles.
  3. G = damage(high) - damage(low), with its interval taken from the paired replicate arrays.

Reproduces every frozen per-run interval to within 1e-9.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.io_utils import data_path, out_path, read_jsonl, write_json
from src.seed_replication import (
    ALPHAS,
    BOOTSTRAP_SEED,
    N_BOOT,
    P_STAR,
    bootstrap_index_sets,
    ci95,
    interp_delta_and_ci,
)


def load_seed1_low(dose_rows: list[dict]):
    rows = {}
    baseline = None
    for d in dose_rows:
        if d["budget"] != "LOW":
            continue
        if d["cell_id"] == "baseline":
            baseline = d
        elif d.get("ell") == 4:
            rows[d["alpha"]] = d
    cells = []
    for a in ALPHAS:
        r = rows[a]
        p_total = float(np.sqrt(r["q_power"])) if r["q_power"] is not None else 0.0
        cells.append(dict(alpha=a, P_total=p_total, delta=r["log_ratio"],
                           per_config_sum=r["per_config_sum"], per_config_natoms=r["per_config_natoms"],
                           L_baseline=r["L_original"]))
    base_cell = dict(per_config_sum=baseline["per_config_sum"], per_config_natoms=baseline["per_config_natoms"],
                      L_baseline=baseline["L_original"])
    return cells, base_cell


def load_seed1_high(sens_rows: list[dict]):
    cells = []
    base_cell = None
    for d in sens_rows:
        if d["tag"] != "esen_lmax4_NEARMAX_w24_s300000":
            continue
        p_total = float(np.sqrt(d["q_power"])) if d["q_power"] is not None else 0.0
        cells.append(dict(alpha=d["alpha"], P_total=p_total, delta=d["delta"],
                           per_config_sum=d["per_config_sum"], per_config_natoms=d["per_config_natoms"],
                           L_baseline=d["L_baseline"]))
        if base_cell is None:
            base_cell = dict(per_config_sum=d["baseline_per_config_sum"],
                              per_config_natoms=d["baseline_per_config_natoms"],
                              L_baseline=d["L_baseline"])
    cells.sort(key=lambda c: -c["alpha"])
    return cells, base_cell


def load_seedrepl(seedrepl_rows: list[dict], tag: str):
    cells = []
    base_cell = None
    for d in seedrepl_rows:
        if d["tag"] != tag:
            continue
        p_total = d["P_total"] if d["P_total"] is not None else 0.0
        cells.append(dict(alpha=d["alpha"], P_total=p_total, delta=d["delta"],
                           per_config_sum=d["per_config_sum"], per_config_natoms=d["per_config_natoms"],
                           L_baseline=d["L_baseline"]))
        if base_cell is None:
            base_cell = dict(per_config_sum=d["baseline_per_config_sum"],
                              per_config_natoms=d["baseline_per_config_natoms"],
                              L_baseline=d["L_baseline"])
    cells.sort(key=lambda c: -c["alpha"])
    return cells, base_cell


def build() -> dict:
    dose_rows = read_jsonl(data_path("claim2", "dose_response.jsonl"))
    sens_rows = read_jsonl(data_path("claim2", "frontier_ell4_sensitivity_raw.jsonl"))
    seedrepl_rows = read_jsonl(data_path("claim2", "seed_replication_raw.jsonl"))

    idx_sets = bootstrap_index_sets(1024, N_BOOT, BOOTSTRAP_SEED)

    seed_data = {
        1: {"LOW": load_seed1_low(dose_rows), "HIGH": load_seed1_high(sens_rows)},
        2: {
            "LOW": load_seedrepl(seedrepl_rows, "seedrepl_LOW_w10_seed2"),
            "HIGH": load_seedrepl(seedrepl_rows, "seedrepl_HIGH_w24_seed2"),
        },
        3: {
            "LOW": load_seedrepl(seedrepl_rows, "seedrepl_LOW_w10_seed3"),
            "HIGH": load_seedrepl(seedrepl_rows, "seedrepl_HIGH_w24_seed3"),
        },
    }

    out = {"P_star": P_STAR, "n_boot": N_BOOT, "bootstrap_seed": BOOTSTRAP_SEED, "seeds": {}}
    g_by_seed, g_ci95_by_seed = {}, {}
    for s in (1, 2, 3):
        low_cells, low_base = seed_data[s]["LOW"]
        high_cells, high_base = seed_data[s]["HIGH"]
        low_res, low_boot = interp_delta_and_ci(low_cells, low_base, P_STAR, idx_sets)
        high_res, high_boot = interp_delta_and_ci(high_cells, high_base, P_STAR, idx_sets)
        entry = {"LOW": low_res, "HIGH": high_res}
        if low_res["in_support"] and high_res["in_support"]:
            g_val = high_res["delta_point"] - low_res["delta_point"]
            g_boot = high_boot - low_boot
            g_lo, g_hi = ci95(g_boot)
            entry["G"] = g_val
            entry["G_ci95"] = [g_lo, g_hi]
            g_by_seed[str(s)] = g_val
            g_ci95_by_seed[str(s)] = [g_lo, g_hi]
        out["seeds"][str(s)] = entry

    out["G_by_seed"] = g_by_seed
    out["G_ci95_by_seed"] = g_ci95_by_seed
    n_pos = sum(1 for v in g_by_seed.values() if v > 0)
    out["sign_replication"] = f"{n_pos}/{len(g_by_seed)} positive"
    return out


def main():
    result = build()
    out = out_path("claim2", "seed_replication_result_recomputed.json")
    write_json(out, result)
    print(f"wrote {out} (G_by_seed: {result['G_by_seed']}, {result['sign_replication']})")


if __name__ == "__main__":
    main()
