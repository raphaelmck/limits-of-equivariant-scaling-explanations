#!/usr/bin/env python3
"""Regenerate the per-(pair=same-width, ell, alpha) C_ell (=Delta_l2-Delta_l4) grid of
data/claim2/same_width_control_B_summary.json from RAW per-config rows in
data/claim2/same_width_control_B.jsonl.

Category (a) -- same structure/schema as compensation.jsonl (see build_compensation.py's
module docstring for the shared bootstrap/pairing convention and sign-convention verification
method), just keyed by same-width pair (SW_w10_s50000 etc.) instead of matched-compute pair.

Determinism caveat: same_width_control_B_summary.json documents no n_boot/seed at all -> the
project-wide n_boot=2000 convention is used with a fixed seed=0 here, and make validate checks
CI overlap rather than exact match.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.bootstrap import log_delta, paired_config_bootstrap, weighted_L, ci95
from src.io_utils import data_path, out_path, read_jsonl, write_json

N_BOOT = 2000
SEED = 0


def build() -> dict:
    rows = read_jsonl(data_path("claim2", "same_width_control_B.jsonl"))
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair"], []).append(r)

    pairs_out = {}
    for pair, prows in by_pair.items():
        baseline = {r["arch"]: r for r in prows if r["kind"] == "baseline"}
        cells = [r for r in prows if r["kind"] != "baseline"]
        by_cell_id = {}
        for r in cells:
            group_key = r["cell_id"].split("_", 1)[1]
            by_cell_id.setdefault(group_key, {})[r["arch"]] = r

        cells_out = {}
        for group_key, archs in by_cell_id.items():
            ell_label, alpha_raw = group_key.split("_alpha")
            alpha_label = f"{float(alpha_raw):.2f}"  # matches frozen "ell0_alpha0.50" key style
            r4, r2 = archs["l4"], archs["l2"]
            b4, b2 = baseline["l4"], baseline["l2"]

            l4_base = weighted_L(b4["per_config_sum"], b4["per_config_natoms"])
            l4_int = weighted_L(r4["per_config_sum"], r4["per_config_natoms"])
            l2_base = weighted_L(b2["per_config_sum"], b2["per_config_natoms"])
            l2_int = weighted_L(r2["per_config_sum"], r2["per_config_natoms"])
            delta_l4_point = log_delta(l4_base, l4_int)
            delta_l2_point = log_delta(l2_base, l2_int)
            c_ell_point = delta_l2_point - delta_l4_point

            def stat(Ls):
                return log_delta(Ls["l2_base"], Ls["l2_int"]) - log_delta(Ls["l4_base"], Ls["l4_int"])

            reps = paired_config_bootstrap(
                {
                    "l4_base": (b4["per_config_sum"], b4["per_config_natoms"]),
                    "l4_int": (r4["per_config_sum"], r4["per_config_natoms"]),
                    "l2_base": (b2["per_config_sum"], b2["per_config_natoms"]),
                    "l2_int": (r2["per_config_sum"], r2["per_config_natoms"]),
                },
                stat,
                n_boot=N_BOOT,
                seed=SEED,
            )
            lo, hi = ci95(reps)
            cells_out[f"ell{ell_label[3:]}_alpha{alpha_label}"] = {
                "delta_l4": delta_l4_point,
                "delta_l2": delta_l2_point,
                "C_ell": c_ell_point,
                "C_ell_ci95": [lo, hi],
                "ci_excludes_zero": (lo > 0 or hi < 0),
                "sign": "positive" if c_ell_point > 0 else "negative",
            }
        pairs_out[pair] = {"cells": cells_out}
    return {"same_width": pairs_out, "n_boot": N_BOOT, "bootstrap_seed": SEED}


def main():
    result = build()
    out = out_path("claim2", "same_width_control_B_recomputed.json")
    write_json(out, result)
    n_cells = sum(len(p["cells"]) for p in result["same_width"].values())
    print(f"wrote {out} ({n_cells} cells across {len(result['same_width'])} same-width pairs)")


if __name__ == "__main__":
    main()
