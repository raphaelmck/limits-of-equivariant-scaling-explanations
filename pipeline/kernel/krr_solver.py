#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/run_fvs_controlD_large_m_krr.py
# source revision withheld for anonymous review, original SHA256 8f2657f21635832aa4852d291330b5ce2436f53d56a95da80ecf85c982c444a6
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/run_fvs_controlD_large_m_krr.py
# source revision withheld for anonymous review, original SHA256 8f2657f21635832aa4852d291330b5ce2436f53d56a95da80ecf85c982c444a6
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Kernel ridge regression on the force NTK: the block-KRR solver and its scoring.

Question: does the ACTUAL learner induced by the frozen m=1024 force tangent kernel preserve the
GemNet-OC > eSEN comparative ordering that the spectral target-alignment diagnostic A(K_x, y)
reports, or does the kernel's own learner give the opposite (neural-hierarchy-consistent) story?

NO new Jacobians, NO new Gram matrices, NO model inference. Inputs are exactly the two frozen
m=1024 kernels and the single common target produced by
`analysis_outputs/large_m_convergence_2026_08_19/` (job 10415557 / 10415558, both at
global_step=500000):

    kernels/esen_large_sphere64_500k_Kx_m1024.npy          (3072 x 3072, float64)
    kernels/gemnet_oc_large_width256_500k_Kx_m1024.npy     (3072 x 3072, float64)
    kernels/{arch}_500k_y_m1024.npy                        (3072,)   -- verified BIT-IDENTICAL

PREDECLARED PROTOCOL (fixed before any m=1024 KRR number was computed). Everything except the
three quantities marked [NEW] is copied verbatim from this repo's established block-KRR protocol
(`the force-NTK KRR routine`, reused unchanged by
`analyze_four_arch_trained_force_kernel_pilot.krr_for_target`):

  * rows are atom-major xyz:  atom a -> rows {3a, 3a+1, 3a+2}
  * splits are at ATOM level; per split_seed, perm = RandomState(seed).permutation(M),
    test = perm[:TEST_ATOMS], train_pool = perm[TEST_ATOMS:], train = train_pool[:n]
    (train sets are therefore nested in n within a split -- the frozen convention)
  * split_seeds = range(20)
  * error metric: test_nmse = ||K_ts K_tt^+ y_tr - y_te||^2 / ||y_te||^2
  * ridgeless solve = eigendecomposition with rcond=1e-10 spectral truncation
  * ridge convention: ridge_value = rho * mean(diag(K_train))   (i.e. rho * trace(K_tt)/(3n))
  * float64 throughout
  [NEW] TEST_ATOMS = 256 (was 128 at M=256; scaled with the 4x larger pool so that
        train_pool = 768 >= max n = 512)
  [NEW] TRAIN_SIZES = [32, 64, 128, 256, 512] (the sizes the sprint task specifies)
  [NEW] RHO_GRID extended upward to [0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1] so that the grid brackets
        the optimum for BOTH architectures rather than only the small-ridge end. The grid is
        COMMON to both architectures and is never tuned on the test set.

  Ridge SELECTION protocol (identical for both architectures): within each (split, n) cell the
  n training atoms are split 75/25 into inner-train / inner-val by the already-drawn permutation
  order (no new randomness); rho is chosen to minimise inner-val nmse; the model is then refit on
  all n training atoms with that rho and scored once on the held-out test atoms. The full
  per-rho grid is also reported for sensitivity.

MANDATORY VALIDATION, run and required to pass BEFORE any m=1024 result is written:
  V1 index/target alignment: the two architectures' y vectors are bit-identical; the m=1024
     probe pool's first 256 atoms/configs are bit-identical to the frozen m=256 pool.
  V2 numerics: both kernels finite, exactly symmetric, PSD to the recorded tolerance.
  V3 reference reproduction: re-running the FROZEN protocol (M=256, TEST_ATOMS=128,
     TRAIN_SIZES=[8,16,32,64,96,128], RIDGES=[ridgeless,1e-8,1e-6,1e-4]) on
     (a) the frozen m=256 kernel npz  -> must match
         `four_arch_trained_force_kernel_pilot_2026_08_15/krr_learning_curves_common_target.csv`
         (K500k rows) to <1e-9 relative, and
     (b) the leading 768x768 block of the m=1024 kernel -> must match the same reference to
         <1e-4 relative (the two kernels agree only to fp32 Jacobian precision, ~2e-7 relative).
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

SPRINT_DIR = REPO_ROOT / "analysis_outputs" / "final_validation_sprint_2026_08_20"
OUT_DIR = SPRINT_DIR / "control_D_large_m_krr"
LARGE_M_DIR = REPO_ROOT / "analysis_outputs" / "large_m_convergence_2026_08_19"
TRAINED_PILOT_DIR = REPO_ROOT / "analysis_outputs" / "four_arch_trained_force_kernel_pilot_2026_08_15"

ARCHS = {
    "GemNet-OC": "gemnet_oc_large_width256_500k",
    "eSEN": "esen_large_sphere64_500k",
}
FROZEN_M256_NPZ = {
    "GemNet-OC": TRAINED_PILOT_DIR / "kernels" / "gemnet_oc_large_width256_K500k_m256_kernel.npz",
    "eSEN": TRAINED_PILOT_DIR / "kernels" / "esen_large_sphere64_K500k_m256_kernel.npz",
}

M = 1024
RCOND = 1e-10
SPLIT_SEEDS = list(range(20))
TEST_ATOMS = 256
TRAIN_SIZES = [32, 64, 128, 256, 512]
RHO_GRID = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
INNER_VAL_FRACTION = 0.25

# frozen reference protocol
REF_M = 256
REF_TEST_ATOMS = 128
REF_TRAIN_SIZES = [8, 16, 32, 64, 96, 128]
REF_RIDGES = [("ridgeless", 0.0), ("rho_1e-8", 1e-8), ("rho_1e-6", 1e-6), ("rho_1e-4", 1e-4)]


def atom_rows(atom_idx):
    a = np.asarray(atom_idx, dtype=np.int64)
    return (3 * a[:, None] + np.arange(3)[None, :]).reshape(-1)


def solve_and_score(K, y, train_atoms, test_atoms, rhos):
    """Returns {rho: test_nmse} plus the train-block numerical rank. rho=0.0 => ridgeless."""
    tr, te = atom_rows(train_atoms), atom_rows(test_atoms)
    k_tt = K[np.ix_(tr, tr)]
    k_tt = 0.5 * (k_tt + k_tt.T)
    k_st = K[np.ix_(te, tr)]
    y_tr, y_te = y[tr], y[te]
    den = float(np.dot(y_te, y_te))
    evals, evecs = np.linalg.eigh(k_tt)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    keep = evals > RCOND * max(float(evals[0]), 0.0)
    diag_mean = float(np.mean(np.diag(k_tt)))
    coeff = evecs.T @ y_tr
    out = {}
    for rho in rhos:
        if rho == 0.0:
            alpha = evecs[:, keep] @ (coeff[keep] / evals[keep])
        else:
            alpha = evecs @ (coeff / (np.clip(evals, 0.0, None) + rho * diag_mean))
        pred = k_st @ alpha
        out[rho] = float(np.dot(pred - y_te, pred - y_te)) / den
    return out, int(keep.sum()), diag_mean


def run_frozen_reference(K, y, m):
    """The exact frozen protocol, for the m=256 reproduction gate."""
    perms = {s: np.random.RandomState(s).permutation(m) for s in SPLIT_SEEDS}
    rows = []
    for seed, perm in perms.items():
        test_atoms = perm[:REF_TEST_ATOMS]
        train_pool = perm[REF_TEST_ATOMS:]
        for n in REF_TRAIN_SIZES:
            scores, rank, _ = solve_and_score(K, y, train_pool[:n], test_atoms,
                                              [rho for _, rho in REF_RIDGES])
            for name, rho in REF_RIDGES:
                rows.append({"split_seed": seed, "n": n, "ridge_setting": name,
                             "test_nmse": scores[rho], "rank": rank})
    curves = {}
    rows.sort(key=lambda r: (r["n"], r["ridge_setting"], r["split_seed"]))
    for key, grp in itertools.groupby(rows, key=lambda r: (r["n"], r["ridge_setting"])):
        vals = np.array([g["test_nmse"] for g in grp])
        curves[key] = float(np.median(vals))
    return curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-reference", action="store_true",
                    help="debug only; the reference gate is mandatory for a trusted result")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    validation = {}

    # ---------------- load ----------------
    K, y = {}, {}
    for label, stem in ARCHS.items():
        K[label] = np.load(LARGE_M_DIR / "kernels" / f"{stem}_Kx_m1024.npy")
        y[label] = np.load(LARGE_M_DIR / "kernels" / f"{stem}_y_m1024.npy")
        assert K[label].shape == (3 * M, 3 * M) and y[label].shape == (3 * M,)

    # ---------------- V1 index / target alignment ----------------
    y_common = y["eSEN"]
    probe = np.load(LARGE_M_DIR / "probe_indices_ext.npz")
    manifest = json.loads((LARGE_M_DIR / "probe_manifest_ext.json").read_text())
    z256 = np.load(FROZEN_M256_NPZ["eSEN"])
    v1 = {
        "y_bit_identical_across_architectures": bool(np.array_equal(y["eSEN"], y["GemNet-OC"])),
        "y_max_abs_diff": float(np.abs(y["eSEN"] - y["GemNet-OC"]).max()),
        "m1024_first256_config_indices_match_frozen_m256": bool(np.array_equal(
            probe["config_indices"][:REF_M].astype(np.int64), z256["config_indices"].astype(np.int64))),
        "m1024_first256_atom_indices_match_frozen_m256": bool(np.array_equal(
            probe["atom_indices"][:REF_M].astype(np.int64), z256["atom_indices"].astype(np.int64))),
        "m1024_first768_y_match_frozen_m256_y": float(
            np.abs(y_common[:3 * REF_M] - z256["y"].astype(np.float64)).max()),
        "probe_manifest_nesting_validation": manifest["nesting_validation"],
        "n_unique_config_indices": int(len(set(probe["config_indices"].tolist()))),
        "m": M,
    }
    v1["passed"] = (v1["y_bit_identical_across_architectures"]
                    and v1["m1024_first256_config_indices_match_frozen_m256"]
                    and v1["m1024_first256_atom_indices_match_frozen_m256"]
                    and v1["m1024_first768_y_match_frozen_m256_y"] == 0.0)
    validation["V1_index_target_alignment"] = v1
    print("V1", json.dumps(v1, indent=1), flush=True)
    assert v1["passed"], "V1 failed"

    # ---------------- V2 numerics ----------------
    v2 = {}
    for label in ARCHS:
        w = np.linalg.eigvalsh(K[label])
        v2[label] = {
            "finite": bool(np.all(np.isfinite(K[label]))),
            "symmetry_max_abs_error": float(np.abs(K[label] - K[label].T).max()),
            "min_eigenvalue": float(w.min()), "max_eigenvalue": float(w.max()),
            "psd_tolerance": float(1e-8 * w.max()),
            "psd_pass": bool(w.min() > -1e-8 * w.max()),
            "trace": float(np.trace(K[label])), "trace_over_dim": float(np.trace(K[label]) / (3 * M)),
        }
        v2[label]["passed"] = v2[label]["finite"] and v2[label]["symmetry_max_abs_error"] == 0.0 and v2[label]["psd_pass"]
    validation["V2_numerics"] = v2
    print("V2", json.dumps(v2, indent=1), flush=True)
    assert all(v["passed"] for v in v2.values()), "V2 failed"

    # ---------------- V3 frozen-reference reproduction ----------------
    if not args.skip_reference:
        ref_rows = list(csv.DictReader((TRAINED_PILOT_DIR / "krr_learning_curves_common_target.csv").open()))
        ref = {}
        for r in ref_rows:
            if r["time"] != "K500k" or r["target"] != "common":
                continue
            ref[(r["architecture"], int(r["n"]), r["ridge_setting"])] = float(r["nmse_median"])
        v3 = {}
        for label in ARCHS:
            z = np.load(FROZEN_M256_NPZ[label])
            K256_frozen = z["K_raw"].astype(np.float64)
            y256 = z["y"].astype(np.float64)
            cur_a = run_frozen_reference(K256_frozen, y256, REF_M)
            cur_b = run_frozen_reference(K[label][:3 * REF_M, :3 * REF_M], y_common[:3 * REF_M], REF_M)
            errs_a, errs_b = [], []
            cells = []
            for n in REF_TRAIN_SIZES:
                for name, _ in REF_RIDGES:
                    tgt = ref[(label, n, name)]
                    ea = abs(cur_a[(n, name)] - tgt) / tgt
                    eb = abs(cur_b[(n, name)] - tgt) / tgt
                    errs_a.append(ea)
                    errs_b.append(eb)
                    cells.append({"n": n, "ridge_setting": name, "frozen_reference": tgt,
                                  "recomputed_from_frozen_m256_kernel": cur_a[(n, name)], "rel_err_a": ea,
                                  "recomputed_from_m1024_leading_block": cur_b[(n, name)], "rel_err_b": eb})
            v3[label] = {
                "max_rel_err_frozen_m256_kernel": max(errs_a),
                "max_rel_err_m1024_leading_block": max(errs_b),
                "passed_a_tol_1e-9": max(errs_a) < 1e-9,
                "passed_b_tol_1e-4": max(errs_b) < 1e-4,
                "cells": cells,
            }
            print(f"V3 {label}: max_rel_err(a)={max(errs_a):.3e} max_rel_err(b)={max(errs_b):.3e}", flush=True)
        validation["V3_reference_reproduction"] = v3
        assert all(v["passed_a_tol_1e-9"] and v["passed_b_tol_1e-4"] for v in v3.values()), "V3 failed"

    validation["V4_ridge_convention"] = {
        "ridge_value_formula": "rho * mean(diag(K_train))  == rho * trace(K_train)/(3n)",
        "rho_grid": RHO_GRID,
        "ridgeless_definition": f"eigendecomposition with spectral truncation at rcond={RCOND} * lambda_max",
        "grid_is_common_to_both_architectures": True,
        "selection": f"inner {1 - INNER_VAL_FRACTION:.2f}/{INNER_VAL_FRACTION:.2f} train/val split of the "
                     "n training atoms taken in the already-drawn permutation order, rho chosen to minimise "
                     "inner-val nmse, refit on all n and scored once on the common held-out test atoms",
        "test_set_never_used_for_selection": True,
        "train_diag_mean_by_arch_note": "diag_mean is computed per (arch, split, n) from that cell's own "
                                        "K_train, so the same rho corresponds to the same RELATIVE ridge in "
                                        "both architectures despite their ~4x different kernel scale",
    }

    # ---------------- primary: m=1024 learning curves ----------------
    perms = {s: np.random.RandomState(s).permutation(M) for s in SPLIT_SEEDS}
    split_rows = []
    for label in ARCHS:
        for seed, perm in perms.items():
            test_atoms = perm[:TEST_ATOMS]
            train_pool = perm[TEST_ATOMS:]
            assert len(train_pool) >= max(TRAIN_SIZES)
            for n in TRAIN_SIZES:
                train_atoms = train_pool[:n]
                scores, rank, diag_mean = solve_and_score(K[label], y_common, train_atoms, test_atoms, RHO_GRID)
                # ridge selection on an inner split (no test information used)
                n_val = max(int(round(INNER_VAL_FRACTION * n)), 1)
                inner_val = train_atoms[-n_val:]
                inner_train = train_atoms[:-n_val]
                inner_scores, _, _ = solve_and_score(K[label], y_common, inner_train, inner_val, RHO_GRID)
                rho_sel = min(RHO_GRID, key=lambda r: inner_scores[r])
                for rho in RHO_GRID:
                    split_rows.append({
                        "architecture": label, "split_seed": seed, "n": n, "rho": rho,
                        "ridge_setting": "ridgeless" if rho == 0.0 else f"rho_{rho:g}",
                        "ridge_value": rho * diag_mean, "train_diag_mean": diag_mean,
                        "test_atoms": TEST_ATOMS, "train_block_dimension": 3 * n,
                        "effective_numerical_rank_train": rank, "rcond": RCOND,
                        "test_nmse": scores[rho],
                        "selected_rho": rho_sel, "is_selected": rho == rho_sel,
                        "inner_val_nmse": inner_scores[rho], "n_inner_val_atoms": n_val,
                    })
        print(f"done {label}", flush=True)

    with (OUT_DIR / "krr_split_results_m1024.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(split_rows[0]))
        w.writeheader()
        w.writerows(split_rows)

    curves = []
    split_rows.sort(key=lambda r: (r["architecture"], r["n"], r["ridge_setting"], r["split_seed"]))
    for key, grp in itertools.groupby(split_rows, key=lambda r: (r["architecture"], r["n"], r["ridge_setting"], r["rho"])):
        g = list(grp)
        v = np.array([x["test_nmse"] for x in g])
        q25, med, q75 = np.quantile(v, [0.25, 0.5, 0.75])
        curves.append({"architecture": key[0], "n": key[1], "ridge_setting": key[2], "rho": key[3],
                       "split_count": len(v), "nmse_median": float(med), "nmse_mean": float(v.mean()),
                       "nmse_q25": float(q25), "nmse_q75": float(q75),
                       "train_rank_median": float(np.median([x["effective_numerical_rank_train"] for x in g]))})
    # selected-rho curve
    sel_rows = [r for r in split_rows if r["is_selected"]]
    sel_rows.sort(key=lambda r: (r["architecture"], r["n"], r["split_seed"]))
    for key, grp in itertools.groupby(sel_rows, key=lambda r: (r["architecture"], r["n"])):
        g = list(grp)
        v = np.array([x["test_nmse"] for x in g])
        q25, med, q75 = np.quantile(v, [0.25, 0.5, 0.75])
        curves.append({"architecture": key[0], "n": key[1], "ridge_setting": "SELECTED", "rho": None,
                       "split_count": len(v), "nmse_median": float(med), "nmse_mean": float(v.mean()),
                       "nmse_q25": float(q25), "nmse_q75": float(q75),
                       "train_rank_median": float(np.median([x["effective_numerical_rank_train"] for x in g]))})
    with (OUT_DIR / "krr_learning_curves_m1024.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0]))
        w.writeheader()
        w.writerows(curves)

    # ---------------- paired per-split comparison (same splits for both archs) ----------------
    by = {(r["architecture"], r["n"], r["split_seed"]): r for r in sel_rows}
    paired = []
    for n in TRAIN_SIZES:
        d = np.array([np.log(by[("GemNet-OC", n, s)]["test_nmse"]) - np.log(by[("eSEN", n, s)]["test_nmse"])
                      for s in SPLIT_SEEDS])
        wins_esen = int((d > 0).sum())
        paired.append({"n": n, "mean_log_ratio_gemnet_over_esen": float(d.mean()),
                       "median_log_ratio": float(np.median(d)),
                       "splits_where_eSEN_better": wins_esen, "n_splits": len(SPLIT_SEEDS),
                       "eSEN_better_in_all_splits": wins_esen == len(SPLIT_SEEDS),
                       "GemNet_better_in_all_splits": wins_esen == 0})
    # per-rho ordering sensitivity
    med = {(c["architecture"], c["n"], c["ridge_setting"]): c["nmse_median"] for c in curves}
    sensitivity = []
    for rho in RHO_GRID:
        name = "ridgeless" if rho == 0.0 else f"rho_{rho:g}"
        row = {"ridge_setting": name}
        for n in TRAIN_SIZES:
            row[f"n{n}_esen_better"] = bool(med[("eSEN", n, name)] < med[("GemNet-OC", n, name)])
            row[f"n{n}_ratio_gemnet_over_esen"] = med[("GemNet-OC", n, name)] / med[("eSEN", n, name)]
        sensitivity.append(row)

    summary = {
        "purpose": "Control D: does the frozen m=1024 force-tangent kernel's own KRR learner "
                   "reproduce the spectral A(K_x,y) GemNet-OC > eSEN ordering?",
        "kernels": {label: str(LARGE_M_DIR / "kernels" / f"{stem}_Kx_m1024.npy") for label, stem in ARCHS.items()},
        "frozen_spectral_alignment_m1024": {"GemNet-OC": 0.3213, "eSEN": 0.1763,
                                            "note": "A(K_x,y) from large_m_convergence_2026_08_19; "
                                                    "HIGHER alignment = kernel spectrally better aligned to y"},
        "protocol": {"M": M, "test_atoms": TEST_ATOMS, "train_sizes": TRAIN_SIZES,
                     "split_seeds": SPLIT_SEEDS, "rho_grid": RHO_GRID, "rcond": RCOND,
                     "inner_val_fraction": INNER_VAL_FRACTION},
        "validation": validation,
        "learning_curves_selected": [c for c in curves if c["ridge_setting"] == "SELECTED"],
        "paired_split_comparison_selected": paired,
        "ridge_sensitivity_ordering": sensitivity,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"paired": paired, "sensitivity": sensitivity}, indent=1), flush=True)
    print(f"DONE -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
