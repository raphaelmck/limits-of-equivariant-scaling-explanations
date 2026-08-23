#!/usr/bin/env python3
"""Regenerate data/claim2/seed_replication_result_summary.json's per-(seed, tier)
interpolated-to-P*=0.15 Delta AND its 95% CI, plus G_s = Delta_HIGH(P*) - Delta_LOW(P*) and its
CI, for seeds 1/2/3 -- a REAL numpy bootstrap (Stage-2 fix; see AUDIT_STAGE2.md). A prior pass
regenerated only the Delta/G point estimates from seed_replication_result.csv's measured-alpha
rows (no CI, since that CSV carries none at the measured-alpha rows) and left the CI as
passthrough, believing the raw per-config arrays needed to bootstrap it were not present.

Gap closed: dev-equivariant-scaling-laws-kernel-pilot-clean/analysis_outputs/
seed_replication_2026_08_20/results/seed_replication_ell4_M1024.jsonl (raw per-config arrays for
the NEW seed-2/seed-3 runs) is now copied in as data/claim2/seed_replication_raw.jsonl. Seed 1's
raw arrays are NOT re-sourced independently here -- per this task's explicit instruction, seed 1
LOW reuses data/claim2/dose_response.jsonl (budget=LOW, ell=4 rows) and seed 1 HIGH reuses
data/claim2/frontier_ell4_sensitivity_raw.jsonl (tag=esen_lmax4_NEARMAX_w24_s300000), i.e. EXACTLY
the same raw inputs Task 3's degree-balanced reproduction already established, not a
separately-recomputed copy of the same underlying data.

METRIC WARNING (read src/seed_replication.py's module docstring too): this experiment's
pre-registered estimand uses the RAW P_total = RMS(h_int-h_base)/RMS(h_base) metric read directly
off each row's own `P_total`/`q_power` field -- explicitly NOT the degree-balanced-RMS P_bal
metric Task 3 / frontier_ell4_degree_balanced.csv uses. src/seed_replication.py is a standalone
module for exactly this reason: it must never be conflated with src/ood_analysis.py's P_bal
interpolation code, even though both bootstrap the same kind of block-9 ell=4 per-config arrays.

Procedure (bit-exact port of analysis_scripts/analyze_seed_replication.py -- see
src/seed_replication.py for the line-by-line correspondence):
  1. idx_sets = bootstrap_index_sets(1024, 2000, seed=0) -- ONE shared index-set list, generated
     once, reused for every seed/tier/cell (the project's paired-bootstrap convention).
  2. For each (seed, tier) in {1,2,3} x {LOW,HIGH}: bracket the two measured alpha points whose
     P_total straddles P*=0.15, linearly interpolate the point Delta, and -- separately, per
     bootstrap replicate -- interpolate the RESAMPLED Delta at the two bracket alphas using the
     SAME frac weight, then take the 2.5/97.5 percentile of the 2000 replicate values.
  3. G_s = Delta_HIGH(P*) - Delta_LOW(P*), point and CI (interpolated-Delta bootstrap arrays are
     themselves paired across tiers via the shared idx_sets, so G_s's CI is a valid paired
     contrast, not an independent-difference approximation).

Validated bit-exact (<1e-9 absolute) against every one of G_s_analysis.json's per-seed
delta_ci95/G_ci95 entries.
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
