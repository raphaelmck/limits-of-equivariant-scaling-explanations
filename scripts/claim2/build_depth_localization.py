#!/usr/bin/env python3
"""Regenerate the three-depth (blocks 3, 6, and 9) intervention analysis from per-configuration
arrays: per-alpha damage and interval, the read-off at the common perturbation-magnitude
support edge, and all six paired depth contrasts with real bootstrap intervals.

Raw inputs, all in data/claim2/: depth_localization/depth_localization_M1024.jsonl for blocks 3
and 6; dose_response.jsonl and frontier_ell4_sensitivity_raw.jsonl for block 9, whose `delta`
field is the same log(L_int / L_base) computed from those arrays.

One shared index matrix, default_rng(20260820) drawing (2000, n_cfg) indices, backs every
checkpoint and every depth. Because all three depths resample the same configuration positions,
their replicate arrays are paired and every depth contrast gets a real interval rather than a
passthrough. Each depth's damage curve and its full replicate array are interpolated to the
common support edge with the same bracket weight, then contrasts are taken replicate-wise.

Reproduces every read-off and contrast in the frozen summary to within 1e-9.
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

    # ---- low-vs-high (w24/300k minus w10/50k) contrast at each depth, D_k ----
    # Reuses the SAME shared bootstrap idx matrix as depth_contrasts above (no new bootstrap
    # logic): dboot_at_x[("w24/300k", k)] and dboot_at_x[("w10/50k", k)] are draws from identical
    # resampled config indices at every k, so their replicate-wise difference is a valid paired
    # bootstrap CI for D_k = Delta_{w24,k} - Delta_{w10,k}.
    low_vs_high_by_depth = {}
    for k in DEPTHS:
        d_high = read_at_common_support[f"w24/300k_L{k}"]["delta"]
        d_low = read_at_common_support[f"w10/50k_L{k}"]["delta"]
        diff_pt = d_high - d_low
        diff_boot = dboot_at_x[("w24/300k", k)] - dboot_at_x[("w10/50k", k)]
        lo, hi = np.quantile(diff_boot, [0.025, 0.975])
        low_vs_high_by_depth[f"L{k}"] = {
            "D_k": diff_pt,
            "ci95": [float(lo), float(hi)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "block": k,
            "z": (k + 1) / 12,
        }

    return {
        "common_P_bal_support": target_p_bal,
        "read_at_common_support": read_at_common_support,
        "depth_contrasts": depth_contrasts,
        "low_vs_high_contrast_by_depth": low_vs_high_by_depth,
        "n_boot": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
    }


def main():
    result = build()
    out = out_path("claim2", "depth_localization_recomputed.json")
    write_json(out, result)

    # Small derived CSV, purely a flat re-serialization of low_vs_high_contrast_by_depth above
    # (same numbers, no recomputation) -- convenient for the figure script and caption text.
    dk_path = out_path("claim2", "depth_low_high_contrast.csv")
    import csv as _csv
    with open(dk_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["block", "z", "D_k", "ci_lo", "ci_hi", "ci_excludes_zero"])
        for key, v in sorted(result["low_vs_high_contrast_by_depth"].items(), key=lambda kv: kv[1]["block"]):
            w.writerow([v["block"], v["z"], v["D_k"], v["ci95"][0], v["ci95"][1], v["ci_excludes_zero"]])

    print(f"wrote {out} ({len(result['read_at_common_support'])} owner/depth cells, "
          f"{len(result['depth_contrasts'])} depth contrasts, "
          f"{len(result['low_vs_high_contrast_by_depth'])} low-vs-high D_k contrasts)")
    print(f"wrote {dk_path}")


if __name__ == "__main__":
    main()
