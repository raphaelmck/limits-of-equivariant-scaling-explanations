#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/run_dense_grid_large_m_krr_2026_08_23.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 0223ac42a6faf93208278d43e0ddce0f3a2e28ee38d2e8a855d68c0e1911aa1e
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/run_dense_grid_large_m_krr_2026_08_23.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 0223ac42a6faf93208278d43e0ddce0f3a2e28ee38d2e8a855d68c0e1911aa1e
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Dense-grid (PRECROSS/CROSS/FITX/POSTCROSS/TOP) m=1024 block-KRR (2026-08-23).

Reuses the EXACT canonical block-KRR protocol (`solve_and_score`, `atom_rows`, imported
UNCHANGED from `run_fvs_controlD_large_m_krr.py`) and the exact loop/CSV-writing structure of
`run_matched_compute_large_m_krr.py` (2026-08-22), extended to:
  (a) the 16 new dense-grid kernels written by `run_dense_grid_large_m_kernels_2026_08_23.py`
      (self-consistency-gated, not m256-gated -- see that script's docstring for why), and
  (b) the pre-existing `mpnn_MID_w1150_s500000` kernel (2026-08-22 run), REUSED VERBATIM for the
      PRECROSS budget's MPNN cell (same checkpoint as MID, per DENSER_GRID_PROPOSAL_V2.md sec.3).

A kernel that serves two budgets (e.g. egnn_PRECROSS_w160_s300000_dense23 also serves CROSS)
produces ONE set of split-level KRR rows per (kernel, budget_tier) pair it is mapped to in
`owners_dense_grid_2026_08_23.BUDGET_TO_TAGS` -- the KRR computation itself is identical (same
K, same y), only the `budget_tier` label differs per output row, so no KRR is literally re-run,
the same numbers are just also labeled under the second budget (documented, not hidden).
"""
from __future__ import annotations

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
from pipeline.kernel.dense_grid_checkpoints import BUDGET_TO_TAGS

DENSE_KERNEL_DIR = REPO_ROOT / "analysis_outputs/dense_grid_claim1_2026_08_23/kernels"
EXISTING_KERNEL_DIR = REPO_ROOT / "analysis_outputs/matched_compute_large_m_kernels_2026_08_22/kernels"
OUT_DIR = REPO_ROOT / "analysis_outputs/dense_grid_claim1_2026_08_23/krr"

ARCH_LABEL = {"mpnn": "MPNN", "egnn": "MC-EGNN", "gemnet_oc": "GemNet-OC", "esen": "eSEN"}
NEW_BUDGETS = ["PRECROSS", "CROSS", "FITX", "POSTCROSS", "TOP"]

M = 1024
RCOND = 1e-10
SPLIT_SEEDS = list(range(20))
TEST_ATOMS = 256
TRAIN_SIZES = [32, 64, 128, 256, 512]
RHO_GRID = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
INNER_VAL_FRACTION = 0.25


def kernel_path_for(tag: str) -> Path:
    if tag == "mpnn_MID_w1150_s500000":
        return EXISTING_KERNEL_DIR / f"{tag}_m1024_kernel.npz"
    return DENSE_KERNEL_DIR / f"{tag}_m1024_kernel.npz"


def meta_path_for(tag: str) -> Path:
    if tag == "mpnn_MID_w1150_s500000":
        return EXISTING_KERNEL_DIR / f"{tag}_m1024_metadata.json"
    return DENSE_KERNEL_DIR / f"{tag}_m1024_metadata.json"


def load_gate_pass(tag: str) -> bool:
    meta = json.loads(meta_path_for(tag).read_text())
    if tag == "mpnn_MID_w1150_s500000":
        return bool(meta.get("m256_validation_gate", {}).get("pass"))
    return bool(meta.get("self_consistency_gate", {}).get("pass"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # unique kernel tags actually needed (dedup across budgets)
    unique_tags = sorted({tag for tags in BUDGET_TO_TAGS.values() for tag in tags})
    present = []
    for tag in unique_tags:
        kp, mp = kernel_path_for(tag), meta_path_for(tag)
        if not kp.exists() or not mp.exists():
            print(f"SKIP {tag}: missing kernel/metadata (job not complete or failed)", flush=True)
            continue
        if not load_gate_pass(tag):
            print(f"SKIP {tag}: validation gate did not pass", flush=True)
            continue
        present.append(tag)
    print(f"validated unique kernels ({len(present)}/{len(unique_tags)}): {present}", flush=True)
    if not present:
        raise RuntimeError("no validated kernels; nothing to run")

    K, y, meta = {}, {}, {}
    for tag in present:
        z = np.load(kernel_path_for(tag))
        K[tag] = z["K_raw"].astype(np.float64)
        y[tag] = z["y"].astype(np.float64)
        assert K[tag].shape == (3 * M, 3 * M) and y[tag].shape == (3 * M,)
        meta[tag] = json.loads(meta_path_for(tag).read_text())

    perms = {s: np.random.RandomState(s).permutation(M) for s in SPLIT_SEEDS}

    # Compute the raw per-(tag, n, seed, rho) KRR scores ONCE per unique tag (not per budget it
    # serves) -- identical protocol/RNG as run_matched_compute_large_m_krr.py.
    per_tag_rows = {}
    for tag in present:
        rows = []
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
                    rows.append({
                        "owner_tag": tag, "split_seed": seed, "n": n, "rho": rho,
                        "ridge_setting": "ridgeless" if rho == 0.0 else f"rho_{rho:g}",
                        "ridge_value": rho * diag_mean, "train_diag_mean": diag_mean,
                        "test_atoms": TEST_ATOMS, "train_block_dimension": 3 * n,
                        "effective_numerical_rank_train": rank, "rcond": RCOND,
                        "test_nmse": scores[rho],
                        "selected_rho": rho_sel, "is_selected": rho == rho_sel,
                        "inner_val_nmse": inner_scores[rho], "n_inner_val_atoms": n_val,
                    })
        per_tag_rows[tag] = rows
        print(f"done {tag}", flush=True)

    # expand into (budget_tier, architecture) labeled rows -- one copy of the KRR numbers per
    # budget the tag serves (documented dedup, not a re-run)
    split_rows = []
    skipped_budgets = []
    for budget in NEW_BUDGETS:
        needed = BUDGET_TO_TAGS[budget]
        missing = [t for t in needed if t not in per_tag_rows]
        if missing:
            print(f"SKIP budget {budget}: missing kernels for {missing} (job(s) not yet complete)", flush=True)
            skipped_budgets.append(budget)
            continue
        for tag in needed:
            arch_key = meta[tag]["architecture"]
            for r in per_tag_rows[tag]:
                row = dict(r)
                row["architecture"] = ARCH_LABEL[arch_key]
                row["budget_tier"] = budget
                split_rows.append(row)

    fieldnames = ["architecture", "budget_tier", "owner_tag", "split_seed", "n", "rho",
                  "ridge_setting", "ridge_value", "train_diag_mean", "test_atoms",
                  "train_block_dimension", "effective_numerical_rank_train", "rcond",
                  "test_nmse", "selected_rho", "is_selected", "inner_val_nmse", "n_inner_val_atoms"]
    with (OUT_DIR / "krr_split_results_dense_m1024.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(split_rows)

    curves = []
    split_rows.sort(key=lambda r: (r["budget_tier"], r["owner_tag"], r["n"], r["ridge_setting"], r["split_seed"]))
    for key, grp in itertools.groupby(split_rows, key=lambda r: (r["budget_tier"], r["owner_tag"], r["n"], r["ridge_setting"], r["rho"])):
        g = list(grp)
        v = np.array([x["test_nmse"] for x in g])
        q25, med, q75 = np.quantile(v, [0.25, 0.5, 0.75])
        curves.append({
            "architecture": g[0]["architecture"], "budget_tier": key[0], "owner_tag": key[1],
            "n": key[2], "ridge_setting": key[3], "rho": key[4],
            "split_count": len(v), "nmse_median": float(med), "nmse_mean": float(v.mean()),
            "nmse_q25": float(q25), "nmse_q75": float(q75),
            "train_rank_median": float(np.median([x["effective_numerical_rank_train"] for x in g])),
        })
    sel_rows = [r for r in split_rows if r["is_selected"]]
    sel_rows.sort(key=lambda r: (r["budget_tier"], r["owner_tag"], r["n"], r["split_seed"]))
    for key, grp in itertools.groupby(sel_rows, key=lambda r: (r["budget_tier"], r["owner_tag"], r["n"])):
        g = list(grp)
        v = np.array([x["test_nmse"] for x in g])
        q25, med, q75 = np.quantile(v, [0.25, 0.5, 0.75])
        curves.append({
            "architecture": g[0]["architecture"], "budget_tier": key[0], "owner_tag": key[1],
            "n": key[2], "ridge_setting": "SELECTED", "rho": None,
            "split_count": len(v), "nmse_median": float(med), "nmse_mean": float(v.mean()),
            "nmse_q25": float(q25), "nmse_q75": float(q75),
            "train_rank_median": float(np.median([x["effective_numerical_rank_train"] for x in g])),
        })
    with (OUT_DIR / "krr_learning_curves_dense_m1024.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0]))
        w.writeheader()
        w.writerows(curves)

    summary = {
        "validated_unique_kernels": present,
        "n_validated_unique": len(present),
        "n_unique_needed": len(unique_tags),
        "skipped_budgets": skipped_budgets,
        "budget_to_tags": BUDGET_TO_TAGS,
        "protocol": {"M": M, "test_atoms": TEST_ATOMS, "train_sizes": TRAIN_SIZES,
                     "split_seeds": SPLIT_SEEDS, "rho_grid": RHO_GRID, "rcond": RCOND,
                     "inner_val_fraction": INNER_VAL_FRACTION,
                     "reused_from": "run_fvs_controlD_large_m_krr.py (solve_and_score, atom_rows), "
                                    "identical call structure to run_matched_compute_large_m_krr.py"},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"DONE -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
