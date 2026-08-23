#!/usr/bin/env python3
"""Regenerate the (budget, ell, alpha) grid of data/claim2/dose_response_summary.json's Delta/CI
from RAW per-config rows in data/claim2/dose_response.jsonl.

Category (a): dose_response.jsonl carries per_config_sum/per_config_natoms for every row
(baseline and every alpha/ell cell), which is exactly what's needed to recompute
L = sum(sum)/sum(natoms) and Delta = log(L_intervened/L_baseline) plus its paired
configuration-level bootstrap CI.

Determinism caveat: dose_response_summary.json does NOT document its own bootstrap
seed (checked: no n_boot/seed key anywhere in the file). AUDIT_STAGE1.md's prose says n_boot=2000
is the project-wide convention, so n_boot=2000 is used here, but with an unrecorded seed we
cannot bit-reproduce the frozen Delta_ci95 bounds exactly. make validate therefore checks
statistical consistency (point-estimate closeness + CI overlap) rather than exact match for the
CI bounds; the point-estimate Delta itself IS deterministic (no randomness) and is checked exactly.
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
