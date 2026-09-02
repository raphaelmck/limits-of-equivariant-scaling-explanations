#!/usr/bin/env python3
"""Regenerate the (budget, degree, alpha) intervention-damage grid from the per-configuration
arrays in data/claim2/dose_response.jsonl.

That file carries per_config_sum and per_config_natoms for the baseline and every intervened
cell, which is what is needed to recompute L = sum(sum) / sum(natoms), Delta = log(L_int /
L_base), and its paired configuration-level bootstrap interval.

The frozen summary records no bootstrap seed, so 2000 replicates are used here with a fixed
seed stated at the call site: the intervals match the frozen ones statistically rather than
bit-for-bit. The point estimates involve no randomness and are checked exactly.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.bootstrap import bootstrap_delta_ci
from src.io_utils import data_path, out_path, read_jsonl, write_json

N_BOOT = 2000
SEED = 0  # not documented in the frozen file; fixed here for our own determinism


def build() -> dict:
    rows = read_jsonl(data_path("claim2", "dose_response.jsonl"))
    baselines = {r["budget"]: r for r in rows if r["cell_id"] == "baseline"}

    grid = []
    for r in rows:
        if r["cell_id"] == "baseline":
            continue
        budget = r["budget"]
        base = baselines[budget]
        result = bootstrap_delta_ci(
            base["per_config_sum"],
            base["per_config_natoms"],
            r["per_config_sum"],
            r["per_config_natoms"],
            n_boot=N_BOOT,
            seed=SEED,
        )
        grid.append(
            {
                "budget": budget,
                "ell": r["ell"],
                "alpha": r["alpha"],
                "L_baseline": result["L_baseline"],
                "L_intervened": result["L_intervened"],
                "Delta": result["delta"],
                "Delta_ci95": result["delta_ci95"],
            }
        )
    return {"grid": grid, "n_boot": N_BOOT, "bootstrap_seed": SEED}


def main():
    result = build()
    out = out_path("claim2", "dose_response_recomputed.json")
    write_json(out, result)
    print(f"wrote {out} ({len(result['grid'])} grid cells)")


if __name__ == "__main__":
    main()
