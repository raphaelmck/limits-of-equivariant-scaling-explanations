#!/usr/bin/env python3
"""Regenerate the per-(pair, ell, alpha) Delta_l4 / Delta_l2 / C_ell (=Delta_l4-Delta_l2) grid
of data/claim2/compensation_summary.json from RAW per-config rows in
data/claim2/compensation.jsonl.

Category (a). compensation.jsonl carries per_config_sum/per_config_natoms for baseline and every
(arch in {l4,l2}, ell in {0,1,2,012}, alpha in {0.5,0.0}) cell, for each of the 4 matched-compute
pairs (LOW/MID/UPPERMID/HIGH). C_ell is a *paired* contrast (Delta_l4 - Delta_l2 using the SAME
bootstrap resample of configuration indices for both architectures), matching
compensation_summary.json's own bootstrap_unit convention documented in the sibling
perturbation_control_A_summary.json ("SAME resample applied to every tier/cell so that
cross-tier contrasts are paired").

Determinism caveat: compensation_summary.json documents n_boot=2000 in its own "config" block but
no seed -> statistical-consistency (CI overlap) check in make validate, not exact match.
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
    rows = read_jsonl(data_path("claim2", "compensation.jsonl"))
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair"], []).append(r)

    pairs_out = {}
    for pair, prows in by_pair.items():
        baseline = {r["arch"]: r for r in prows if r["kind"] == "baseline"}
        cells = [r for r in prows if r["kind"] != "baseline"]
        by_cell_id = {}
        for r in cells:
            # cell_id like "l4_ell0_alpha0.50" -> group key "ell0_alpha0.50"
            group_key = r["cell_id"].split("_", 1)[1]
            by_cell_id.setdefault(group_key, {})[r["arch"]] = r

        ells_out = {}
        for group_key, archs in by_cell_id.items():
            ell_label, alpha_raw = group_key.split("_alpha")
            alpha_label = str(float(alpha_raw))  # normalize "0.50" -> "0.5" to match frozen keys
            r4, r2 = archs["l4"], archs["l2"]
            b4, b2 = baseline["l4"], baseline["l2"]

            l4_base = weighted_L(b4["per_config_sum"], b4["per_config_natoms"])
            l4_int = weighted_L(r4["per_config_sum"], r4["per_config_natoms"])
            l2_base = weighted_L(b2["per_config_sum"], b2["per_config_natoms"])
            l2_int = weighted_L(r2["per_config_sum"], r2["per_config_natoms"])
            delta_l4_point = log_delta(l4_base, l4_int)
            delta_l2_point = log_delta(l2_base, l2_int)
            # NOTE: C_ell = Delta_l2 - Delta_l4 (verified numerically against the frozen
            # compensation_summary.json, e.g. LOW/ell0/alpha0.5: 0.8431601... - 0.3454782... ==
            # 0.4976819..., matching the frozen C_ell exactly) -- l2 relies MORE on the shared
            # lower-order irreps than l4 does, hence the positive sign convention throughout.
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
            ells_out.setdefault(ell_label, {})[alpha_label] = {
                "Delta_l4": delta_l4_point,
                "Delta_l2": delta_l2_point,
                "C_ell": c_ell_point,
                "C_ell_ci95": [lo, hi],
                "C_ell_excludes_zero": (lo > 0 or hi < 0),
            }
        pairs_out[pair] = {"ells": ells_out}
    return {"pairs": pairs_out, "n_boot": N_BOOT, "bootstrap_seed": SEED}


def main():
    result = build()
    out = out_path("claim2", "compensation_recomputed.json")
    write_json(out, result)
    n_cells = sum(len(a) for p in result["pairs"].values() for a in p["ells"].values())
    print(f"wrote {out} ({n_cells} (ell,alpha) cells across {len(result['pairs'])} pairs)")


if __name__ == "__main__":
    main()
