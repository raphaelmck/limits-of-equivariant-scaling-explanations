#!/usr/bin/env python3
"""Write the per-architecture empirical step frontier, (compute, normalized force MSE), used as
the scatter in Figure 1A.

This is plotting support, not a new quantity: it is the frontier-owner list that
common/force_scaling_fit.py already builds as an internal step of the exponent fit and then
discards. Selection rule, over the `metric_valid` rows of
data/claim1/four_arch_force_checkpoint_table.csv: sort by (compute, loss) ascending, drop
duplicate compute values keeping the lowest loss, and keep only rows that set a new running
minimum in loss. Writing it out keeps frontier selection out of the plotting scripts.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.io_utils import data_path, out_path, read_csv, write_csv
from common.force_scaling_fit import build_frontier

ARCH_LABEL = {"mpnn": "MPNN", "egnn": "MC-EGNN", "gemnet_oc": "GemNet-OC", "esen": "eSEN"}
ARCH_ORDER = ["mpnn", "egnn", "gemnet_oc", "esen"]
LOSS_COL = "force_mse_norm"
C_COL = "C"


def _load_valid_rows():
    raw = read_csv(data_path("claim1", "four_arch_force_checkpoint_table.csv"))
    rows = []
    for r in raw:
        if r["metric_valid"] != "True":
            continue
        r = dict(r)
        r[C_COL] = float(r[C_COL])
        r[LOSS_COL] = float(r[LOSS_COL])
        rows.append(r)
    return rows


def build() -> list[dict]:
    rows = _load_valid_rows()
    by_arch = {a: [r for r in rows if r["architecture"] == a] for a in ARCH_ORDER}

    out_rows = []
    for a in ARCH_ORDER:
        frontier = build_frontier(by_arch[a], LOSS_COL, C_COL)
        for r in frontier:
            out_rows.append({
                "architecture": ARCH_LABEL[a],
                "architecture_key": a,
                "C_flops": r[C_COL],
                "force_mse_norm": r[LOSS_COL],
                "width": r.get("width", ""),
                "global_step": r.get("global_step", ""),
            })
    return out_rows


def _self_check(rows: list[dict], valid_rows: list[dict]) -> None:
    """Inline self-consistency assertions (no frozen upstream file exists to diff against, since
    build_force_scaling.py never persisted its own intermediate frontier list) -- these assert
    the two invariants build_frontier() is defined to guarantee: strictly increasing C and
    strictly decreasing loss along each architecture's frontier, and that every emitted point is
    literally one of the input checkpoint rows (no interpolation/synthesis)."""
    valid_keys = {(r["architecture"], float(r["C"]), float(r["force_mse_norm"])) for r in valid_rows}
    for a in ARCH_ORDER:
        pts = [r for r in rows if r["architecture_key"] == a]
        pts.sort(key=lambda r: r["C_flops"])
        for r in pts:
            assert (r["architecture_key"], r["C_flops"], r["force_mse_norm"]) in valid_keys, \
                f"{a}: frontier point not found verbatim in input checkpoint table"
        for prev, cur in zip(pts, pts[1:]):
            assert cur["C_flops"] > prev["C_flops"], f"{a}: frontier C_flops not strictly increasing"
            assert cur["force_mse_norm"] < prev["force_mse_norm"], \
                f"{a}: frontier force_mse_norm not strictly decreasing"


def main():
    valid_rows = _load_valid_rows()
    rows = build()
    _self_check(rows, valid_rows)
    out = out_path("claim1", "frontier_points.csv")
    write_csv(out, rows, fieldnames=["architecture", "architecture_key", "C_flops",
                                      "force_mse_norm", "width", "global_step"])
    print(f"wrote {out} ({len(rows)} rows) -- self-consistency checks passed")
    for a in ARCH_ORDER:
        n = sum(1 for r in rows if r["architecture_key"] == a)
        print(f"  {ARCH_LABEL[a]:<10} {n} frontier points")


if __name__ == "__main__":
    main()
