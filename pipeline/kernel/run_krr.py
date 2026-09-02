#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/run_matched_compute_large_m_krr.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 a58a1dbf22b10b9004edbdfce7b5777ca71e127a2d6bb08e93bb41be369174cc
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/run_matched_compute_large_m_krr.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 a58a1dbf22b10b9004edbdfce7b5777ca71e127a2d6bb08e93bb41be369174cc
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Matched-compute owner block-KRR at m=1024, generalized to all 12 LOW/MID/HIGH owners
(2026-08-22).

Reuses the EXACT canonical block-KRR protocol and m=256 reproduction-gate logic from
`run_fvs_controlD_large_m_krr.py` (Control D, which established this repo's m=1024 KRR
convention: atom-major xyz rows, ATOM-level splits with RandomState(seed).permutation,
split_seeds=range(20), ridgeless via rcond=1e-10 spectral truncation, ridge = rho *
mean(diag(K_train)), inner 75/25 train/val ridge selection). The `solve_and_score` /
`atom_rows` functions are imported UNCHANGED, not reimplemented.

Runs KRR on the 12 new matched-compute-owner m=1024 kernels written by
`run_matched_compute_large_m_kernels.py` to
analysis_outputs/matched_compute_large_m_kernels_2026_08_22/kernels/{tag}_m1024_kernel.npz,
for every owner whose m256_validation_gate PASSED (see that script's metadata json).

CPU-only (pure linear algebra on already-materialized kernels); no GPU / model needed.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from pipeline.kernel.krr_solver import atom_rows, solve_and_score

KERNEL_DIR = REPO_ROOT / "analysis_outputs/matched_compute_large_m_kernels_2026_08_22/kernels"
OUT_DIR = REPO_ROOT / "analysis_outputs/matched_compute_large_m_krr_2026_08_22"

OWNER_TAGS = [
    "mpnn_LOW_w607_s400000", "mpnn_MID_w1150_s500000", "mpnn_HIGH_w2557_s450000",
    "egnn_LOW_w96_s100000", "egnn_MID_w96_s450000", "egnn_HIGH_w320_s250000",
    "gemnet_oc_LOW_w48_s200000", "gemnet_oc_MID_w64_s500000", "gemnet_oc_HIGH_w144_s500000",
    "esen_LOW_w10_s50000", "esen_MID_w16_s150000", "esen_HIGH_w40_s275000",
]
ARCH_LABEL = {"mpnn": "MPNN", "egnn": "MC-EGNN", "gemnet_oc": "GemNet-OC", "esen": "eSEN"}

M = 1024
RCOND = 1e-10
SPLIT_SEEDS = list(range(20))
TEST_ATOMS = 256
TRAIN_SIZES = [32, 64, 128, 256, 512]
RHO_GRID = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
INNER_VAL_FRACTION = 0.25


def load_validated_owners() -> list[str]:
    validated = []
    for tag in OWNER_TAGS:
        meta_path = KERNEL_DIR / f"{tag}_m1024_metadata.json"
        kernel_path = KERNEL_DIR / f"{tag}_m1024_kernel.npz"
        if not meta_path.exists() or not kernel_path.exists():
            print(f"SKIP {tag}: missing kernel/metadata (job not complete or failed)", flush=True)
            continue
        meta = json.loads(meta_path.read_text())
        gate = meta.get("m256_validation_gate", {})
        if not gate.get("pass"):
            print(f"SKIP {tag}: m256 validation gate did not pass: {gate}", flush=True)
            continue
        validated.append(tag)
    return validated


def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    validated = load_validated_owners()
    print(f"validated owners ({len(validated)}/12): {validated}", flush=True)
    if not validated:
        raise RuntimeError("no validated owners; nothing to run")

    K, y, owner_meta = {}, {}, {}
    for tag in validated:
        z = np.load(KERNEL_DIR / f"{tag}_m1024_kernel.npz")
        K[tag] = z["K_raw"].astype(np.float64)
        y[tag] = z["y"].astype(np.float64)
        assert K[tag].shape == (3 * M, 3 * M) and y[tag].shape == (3 * M,)
        meta = json.loads((KERNEL_DIR / f"{tag}_m1024_metadata.json").read_text())
        owner_meta[tag] = meta

    perms = {s: np.random.RandomState(s).permutation(M) for s in SPLIT_SEEDS}
    split_rows = []
    for tag in validated:
        arch_key = owner_meta[tag]["architecture"]
        budget = owner_meta[tag]["budget_label"]
        for seed, perm in perms.items():
            test_atoms = perm[:TEST_ATOMS]
            train_pool = perm[TEST_ATOMS:]
            assert len(train_pool) >= max(TRAIN_SIZES)
            for n in TRAIN_SIZES:
                train_atoms = train_pool[:n]
                scores, rank, diag_mean = solve_and_score(K[tag], y[tag], train_atoms, test_atoms, RHO_GRID)
                n_val = max(int(round(INNER_VAL_FRACTION * n)), 1)
                inner_val = train_atoms[-n_val:]
                inner_train = train_atoms[:-n_val]
                inner_scores, _, _ = solve_and_score(K[tag], y[tag], inner_train, inner_val, RHO_GRID)
                rho_sel = min(RHO_GRID, key=lambda r: inner_scores[r])
                for rho in RHO_GRID:
                    split_rows.append({
                        "architecture": ARCH_LABEL[arch_key], "budget_tier": budget, "owner_tag": tag,
                        "split_seed": seed, "n": n, "rho": rho,
                        "ridge_setting": "ridgeless" if rho == 0.0 else f"rho_{rho:g}",
                        "ridge_value": rho * diag_mean, "train_diag_mean": diag_mean,
                        "test_atoms": TEST_ATOMS, "train_block_dimension": 3 * n,
                        "effective_numerical_rank_train": rank, "rcond": RCOND,
                        "test_nmse": scores[rho],
                        "selected_rho": rho_sel, "is_selected": rho == rho_sel,
                        "inner_val_nmse": inner_scores[rho], "n_inner_val_atoms": n_val,
                    })
        print(f"done {tag}", flush=True)

    with (OUT_DIR / "krr_split_results_m1024.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(split_rows[0]))
        w.writeheader()
        w.writerows(split_rows)

    curves = []
    split_rows.sort(key=lambda r: (r["owner_tag"], r["n"], r["ridge_setting"], r["split_seed"]))
    for key, grp in itertools.groupby(split_rows, key=lambda r: (r["owner_tag"], r["n"], r["ridge_setting"], r["rho"])):
        g = list(grp)
        v = np.array([x["test_nmse"] for x in g])
        q25, med, q75 = np.quantile(v, [0.25, 0.5, 0.75])
        curves.append({
            "architecture": g[0]["architecture"], "budget_tier": g[0]["budget_tier"], "owner_tag": key[0],
            "n": key[1], "ridge_setting": key[2], "rho": key[3],
            "split_count": len(v), "nmse_median": float(med), "nmse_mean": float(v.mean()),
            "nmse_q25": float(q25), "nmse_q75": float(q75),
            "train_rank_median": float(np.median([x["effective_numerical_rank_train"] for x in g])),
        })
    sel_rows = [r for r in split_rows if r["is_selected"]]
    sel_rows.sort(key=lambda r: (r["owner_tag"], r["n"], r["split_seed"]))
    for key, grp in itertools.groupby(sel_rows, key=lambda r: (r["owner_tag"], r["n"])):
        g = list(grp)
        v = np.array([x["test_nmse"] for x in g])
        q25, med, q75 = np.quantile(v, [0.25, 0.5, 0.75])
        curves.append({
            "architecture": g[0]["architecture"], "budget_tier": g[0]["budget_tier"], "owner_tag": key[0],
            "n": key[1], "ridge_setting": "SELECTED", "rho": None,
            "split_count": len(v), "nmse_median": float(med), "nmse_mean": float(v.mean()),
            "nmse_q25": float(q25), "nmse_q75": float(q75),
            "train_rank_median": float(np.median([x["effective_numerical_rank_train"] for x in g])),
        })
    with (OUT_DIR / "krr_learning_curves_m1024.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0]))
        w.writeheader()
        w.writerows(curves)

    summary = {
        "validated_owners": validated,
        "n_validated": len(validated),
        "n_missing_or_failed": 12 - len(validated),
        "protocol": {"M": M, "test_atoms": TEST_ATOMS, "train_sizes": TRAIN_SIZES,
                     "split_seeds": SPLIT_SEEDS, "rho_grid": RHO_GRID, "rcond": RCOND,
                     "inner_val_fraction": INNER_VAL_FRACTION,
                     "reused_from": "run_fvs_controlD_large_m_krr.py (solve_and_score, atom_rows)"},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"DONE -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
