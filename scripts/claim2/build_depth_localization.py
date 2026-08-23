#!/usr/bin/env python3
"""Regenerate the FULL 3-depth (block 3, block 6, block 9) depth-localization analysis --
per-alpha delta/CI, common-P_bal-support read-off, and ALL SIX paired depth contrasts with real
bootstrap CIs -- from raw per-config arrays, matching
dev-equivariant-scaling-laws-kernel-pilot-clean/analysis_scripts/build_depth_localization_report.py
line-for-line (confirmed by reading it in full).

STAGE-2 FIX (see AUDIT_STAGE2.md): a prior pass fully reproduced block 3 and block 6 (raw
per-config data was already present in data/claim2/depth_localization/depth_localization_M1024.jsonl)
but treated block 9 as passthrough (cited from data/claim2/frontier_ell4_degree_balanced_summary.json,
believing no raw per-config array existed for it), and therefore also treated any depth contrast
INVOLVING block 9 (L9-L6, L9-L3) as CI-passthrough, only recomputing the L6-L3 contrast's CI.

Two things changed:
1. Task 3 (build_frontier_ell4_degree_balanced.py) established that block 9's raw per-config
   arrays DO exist in data/claim2/ -- dose_response.jsonl (w10/50k=LOW, w16/150k=MID) and
   frontier_ell4_sensitivity_raw.jsonl (w20/200k, w24/300k) -- and that its "delta" field is
   literally the same log(L_int/L_base) computed from those arrays. This script now loads block
   9's raw arrays exactly the same way Task 3 does, instead of citing the frozen summary.
2. Reading build_depth_localization_report.py's ACTUAL bootstrap structure (not assumed) shows it
   uses ONE shared `rng = np.random.default_rng(BOOT_SEED=20260820); boot = rng.integers(0, n_cfg,
   size=(N_BOOT, n_cfg))` idx matrix for EVERY owner AND EVERY depth (3, 6, AND 9) -- the same
   seed already confirmed bit-exact for the block-9-only frontier_ell4_degree_balanced table (see
   that script's docstring). Because one shared idx matrix backs all three depths, block9's
   bootstrap replicate array is directly comparable (paired) to block3/6's, making a REAL bootstrap
   CI possible for every depth contrast, not just L6-L3. This exactly ports lines 111-190 of
   build_depth_localization_report.py: interpolate each depth's delta AND its full bootstrap
   replicate row to the common P_bal support edge via `np.interp` (used identically for the point
   curve and, per-replicate, for the boot array), then take contrasts as replicate-wise
   differences.

Validated bit-exact (<1e-9 absolute) against every read_at_common_support and depth_contrasts
entry in data/claim2/depth_localization/depth_localization_summary.json, including the
previously-passthrough L9 CI and the L9-involving contrast CIs.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.depth_localization import degree_weight, denom_bal, p_bal
from src.io_utils import data_path, out_path, read_csv, read_json, read_jsonl, write_json

N_BOOT = 2000
BOOT_SEED = 20260820  # documented (same seed used across this whole line of work); bit-exact
                        # against depth_localization_summary.json -- see module docstring.
N_CFG = 1024
ALPHAS = [1.0, 0.75, 0.5, 0.25, 0.0]
OWNERS = ("w10/50k", "w16/150k", "w20/200k", "w24/300k")
DEPTHS = (3, 6, 9)


def L(sums, natoms, idx):
    s = np.asarray(sums, dtype=np.float64)[idx]
    c = np.asarray(natoms, dtype=np.float64)[idx]
    return s.sum(axis=-1) / c.sum(axis=-1)


def load_block36():
    rows = read_jsonl(data_path("claim2", "depth_localization", "depth_localization_M1024.jsonl"))
    block_stats = read_json(data_path("claim2", "depth_localization", "block_stats_M1024.json"))
    by_owner_layer = {}
    for r in rows:
        by_owner_layer.setdefault((r["key"], r["layer"]), []).append(r)
    return by_owner_layer, block_stats


def load_block9_raw():
    """Same source mapping as build_frontier_ell4_degree_balanced.py: w10/50k+w16/150k from
    dose_response.jsonl, w20/200k+w24/300k from frontier_ell4_sensitivity_raw.jsonl."""
    dose_rows = read_jsonl(data_path("claim2", "dose_response.jsonl"))
    sens_rows = read_jsonl(data_path("claim2", "frontier_ell4_sensitivity_raw.jsonl"))

    def dose_owner(budget):
        base = next(r for r in dose_rows if r["budget"] == budget and r["cell_id"] == "baseline")
        cell_lookup = {r["alpha"]: r for r in dose_rows if r["budget"] == budget and r.get("ell") == 4}
        return base, cell_lookup

    def sens_owner(label):
        cell_lookup = {r["alpha"]: r for r in sens_rows if r["label"] == label}
        base_row = cell_lookup[1.0]
        base = {
            "per_config_sum": base_row["baseline_per_config_sum"],
            "per_config_natoms": base_row["baseline_per_config_natoms"],
        }
        return base, cell_lookup

    out = {}
    for owner, (base, cell_lookup) in [
        ("w10/50k", dose_owner("LOW")),
        ("w16/150k", dose_owner("MID")),
        ("w20/200k", sens_owner("w20/200k")),
        ("w24/300k", sens_owner("w24/300k")),
    ]:
        out[owner] = (base, cell_lookup)
    return out


def build() -> dict:
    by_owner_layer36, block_stats = load_block36()
    block9_raw = load_block9_raw()
    csv_rows = read_csv(data_path("claim2", "frontier_ell4_degree_balanced.csv"))
    pbal9_by_owner = {}
    for r in csv_rows:
        pbal9_by_owner.setdefault(r["owner"], {})[float(r["alpha"])] = float(r["P_bal"])

    frontier_summary = read_json(data_path("claim2", "frontier_ell4_degree_balanced_summary.json"))
    target_p_bal = frontier_summary["common_P_bal_support"][1]

    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, N_CFG, size=(N_BOOT, N_CFG))  # ONE shared matrix, every owner+depth

    curves = {}       # "{owner}_L{depth}" -> {"P_bal": [...], "delta": [...], "_dboot": [N_BOOT,5]}
    for owner in OWNERS:
        # ---- block 3 and 6 ----
        for depth in (3, 6):
            owner_rows = sorted(by_owner_layer36[(owner, depth)], key=lambda r: -r["alpha"])
            layer_stats = block_stats["owners"][owner]["by_layer"][str(depth)]
            denom = denom_bal(layer_stats["ss_by_ell_raw"], layer_stats["ss_ell0_centered"])
            ss4_base = layer_stats["ss_by_ell_raw"]["4"]

            pbal_list, delta_list, dboot_list = [], [], []
            base_row = next(r for r in owner_rows if r["alpha"] == 1.0)
            bs, bc = base_row["baseline_per_config_sum"], base_row["baseline_per_config_natoms"]
            Lb_boot = L(bs, bc, idx)
            Lb_pt = float(np.sum(bs) / np.sum(bc))
            for r in owner_rows:
                a = r["alpha"]
                pbal_list.append(p_bal(a, ss4_base, denom))
                if a == 1.0:
                    delta_list.append(0.0)
                    dboot_list.append(np.zeros(N_BOOT))
                    continue
                si, ci = r["per_config_sum"], r["per_config_natoms"]
                Li_boot = L(si, ci, idx)
                Li_pt = float(np.sum(si) / np.sum(ci))
                dboot_list.append(np.log(Li_boot) - np.log(Lb_boot))
                delta_list.append(float(math.log(Li_pt) - math.log(Lb_pt)))
            curves[f"{owner}_L{depth}"] = {
                "P_bal": pbal_list, "delta": delta_list, "_dboot": np.stack(dboot_list, axis=0),
            }

        # ---- block 9 ----
        base9, cell_lookup9 = block9_raw[owner]
        bs9, bc9 = base9["per_config_sum"], base9["per_config_natoms"]
        Lb_boot9 = L(bs9, bc9, idx)
        Lb_pt9 = float(np.sum(bs9) / np.sum(bc9))
        pbal9, delta9, dboot9 = [], [], []
        for a in ALPHAS:
            pbal9.append(pbal9_by_owner[owner][a])
            if a == 1.0:
                delta9.append(0.0)
                dboot9.append(np.zeros(N_BOOT))
                continue
            r = cell_lookup9[a]
            si, ci = r["per_config_sum"], r["per_config_natoms"]
            Li_boot = L(si, ci, idx)
            Li_pt = float(np.sum(si) / np.sum(ci))
            dboot9.append(np.log(Li_boot) - np.log(Lb_boot9))
            delta9.append(float(math.log(Li_pt) - math.log(Lb_pt9)))
        curves[f"{owner}_L9"] = {"P_bal": pbal9, "delta": delta9, "_dboot": np.stack(dboot9, axis=0)}

    # ---- read-off at common P_bal support (np.interp for point AND every boot replicate) ----
    read_at_common_support = {}
    dboot_at_x = {}
    for owner in OWNERS:
        for depth in DEPTHS:
            c = curves[f"{owner}_L{depth}"]
            x = np.array(c["P_bal"])
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = np.array(c["delta"])[order]
            dboot_sorted = c["_dboot"][order]  # [5, N_BOOT]
            d_pt = float(np.interp(target_p_bal, x_sorted, y_sorted))
            d_boot = np.array([np.interp(target_p_bal, x_sorted, dboot_sorted[:, i]) for i in range(N_BOOT)])
            lo, hi = np.quantile(d_boot, [0.025, 0.975])
            on_measured_point = any(abs(xx - target_p_bal) < 1e-9 for xx in x)
            read_at_common_support[f"{owner}_L{depth}"] = {
                "delta": d_pt, "loss_multiplier": math.exp(d_pt),
                "ci95": [float(lo), float(hi)], "on_measured_point": on_measured_point,
            }
            dboot_at_x[(owner, depth)] = d_boot

    # ---- depth contrasts, all with REAL bootstrap CIs (shared idx matrix -> valid pairing) ----
    depth_contrasts = {}
    for owner in OWNERS:
        for deep, shallow in [(6, 3), (9, 6), (9, 3)]:
            d_deep = read_at_common_support[f"{owner}_L{deep}"]["delta"]
            d_shallow = read_at_common_support[f"{owner}_L{shallow}"]["delta"]
            diff_pt = d_deep - d_shallow
            diff_boot = dboot_at_x[(owner, deep)] - dboot_at_x[(owner, shallow)]
            lo, hi = np.quantile(diff_boot, [0.025, 0.975])
            sign = "increases_with_depth" if diff_pt > 0 else "decreases_with_depth"
            depth_contrasts[f"{owner}_L{deep}_minus_L{shallow}"] = {
                "delta_difference": diff_pt,
                "ci95": [float(lo), float(hi)],
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "sign": sign,
            }

    return {
        "common_P_bal_support": target_p_bal,
        "read_at_common_support": read_at_common_support,
        "depth_contrasts": depth_contrasts,
        "n_boot": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
    }


def main():
    result = build()
    out = out_path("claim2", "depth_localization_recomputed.json")
    write_json(out, result)
    print(f"wrote {out} ({len(result['read_at_common_support'])} owner/depth cells, "
          f"{len(result['depth_contrasts'])} depth contrasts)")


if __name__ == "__main__":
    main()
