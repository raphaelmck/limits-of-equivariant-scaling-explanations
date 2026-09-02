#!/usr/bin/env python3
# Adapted for anonymous release from esen_irrep_ood_2026_08_22/dataset/scan_omol_ood.py
# source revision unversioned at port time, original SHA256 90d81cb5cd33a92823d228292b430efacf6110d33be0fd2b26ac2acfc79d0b2b
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from esen_irrep_ood_2026_08_22/dataset/scan_omol_ood.py
# source revision unversioned at port time, original SHA256 90d81cb5cd33a92823d228292b430efacf6110d33be0fd2b26ac2acfc79d0b2b
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Freeze the support-matched evaluation population: scan the OMol25 validation split, apply the
charge, spin, and elemental-support criteria, and draw the nested stratified pools.

Phase 1: derive the atomic-number support of the eSEN Neutral training set
          (open_mol/neutral_train) by a full parallel scan of every shard.
Phase 2: scan the full OMol25 val split (<CHECKPOINT_ROOT>/omol25/val)
          and flag structures with charge==0, spin==1, and all Z in the
          Neutral training support.
Phase 3: freeze a stratified, nested OOD pool (1024/2048/4096/8192/16384) from
          the eligible population with a fixed RNG seed.
Phase 4: write all audit/report/manifest artifacts.

Run with many CPUs (see the accompanying .sbatch wrapper) -- this does a full
row-by-row scan of both datasets, not a heuristic/sampled shortcut.
"""
from __future__ import annotations

import json
import glob
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import ase.db

NEUTRAL_TRAIN_SRC = "<DATA_ROOT>/open_mol/neutral_train"
VAL_SRC = "<CHECKPOINT_ROOT>/omol25/val"
OUT_DIR = "<PROJECT_ROOT>/analysis_outputs/esen_irrep_ood_2026_08_22/dataset"
SEED = 42
POOL_SIZES = [1024, 2048, 4096, 8192, 16384]

N_WORKERS = int(os.environ.get("SCAN_NWORKERS", "8"))


def list_shards(src_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(src_dir, "*")))
    # AseDBDataset attempts every glob hit as a db and silently skips
    # anything that fails to connect (e.g. the .aselmdb-lock sidecar files).
    shards = [p for p in paths if p.endswith(".aselmdb")]
    return shards


# ---------------------------------------------------------------------------
# Phase 1: Neutral-train atomic-number support
# ---------------------------------------------------------------------------

def _scan_shard_elements(path: str) -> set[int]:
    db = ase.db.connect(path, readonly=True, use_lock_file=False)
    seen: set[int] = set()
    for rid in db.ids:
        row = db._get_row(rid)
        seen.update(int(z) for z in row.numbers)
    return seen


def phase1_neutral_support() -> dict:
    shards = list_shards(NEUTRAL_TRAIN_SRC)
    print(f"[phase1] {len(shards)} neutral_train shards, {N_WORKERS} workers", flush=True)
    t0 = time.time()
    support: set[int] = set()
    with mp.Pool(N_WORKERS) as pool:
        for i, s in enumerate(pool.imap_unordered(_scan_shard_elements, shards)):
            support |= s
            print(f"[phase1] shard {i+1}/{len(shards)} done, support so far ({len(support)}): {sorted(support)}", flush=True)
    dt = time.time() - t0
    print(f"[phase1] done in {dt:.1f}s, final support ({len(support)} elements): {sorted(support)}", flush=True)
    return {"support": sorted(support), "n_shards": len(shards), "scan_seconds": dt, "src": NEUTRAL_TRAIN_SRC}


# ---------------------------------------------------------------------------
# Phase 2: Val eligibility scan
# ---------------------------------------------------------------------------

def _scan_shard_val(args) -> dict:
    path, shard_idx, offset, support_set = args
    support_set = set(support_set)
    db = ase.db.connect(path, readonly=True, use_lock_file=False)
    ids = db.ids
    n = len(ids)

    total = 0
    n_charge0 = 0
    n_spin1 = 0
    n_charge0_spin1 = 0
    eligible_local_idx = []
    eligible_source = []
    eligible_data_id = []
    eligible_natoms = []
    eligible_composition = []
    data_id_counts_all = Counter()
    element_freq_eligible = Counter()

    for local_i, rid in enumerate(ids):
        row = db._get_row(rid)
        total += 1
        d = row.data if isinstance(row.data, dict) else {}
        charge = d.get("charge", None)
        spin = d.get("spin", None)
        data_id = d.get("data_id", "<missing>")
        source = d.get("source", "<missing>")
        composition = d.get("composition", "<missing>")
        data_id_counts_all[data_id] += 1

        c0 = charge == 0
        s1 = spin == 1
        if c0:
            n_charge0 += 1
        if s1:
            n_spin1 += 1
        if c0 and s1:
            n_charge0_spin1 += 1
            numbers = row.numbers
            if all(int(z) in support_set for z in numbers):
                eligible_local_idx.append(local_i)
                eligible_source.append(source)
                eligible_data_id.append(data_id)
                eligible_natoms.append(int(row.natoms))
                eligible_composition.append(composition)
                element_freq_eligible.update(int(z) for z in numbers)

    global_idx = [offset + i for i in eligible_local_idx]

    return {
        "shard_idx": shard_idx,
        "path": path,
        "total": total,
        "n_charge0": n_charge0,
        "n_spin1": n_spin1,
        "n_charge0_spin1": n_charge0_spin1,
        "n_eligible": len(global_idx),
        "global_idx": global_idx,
        "source": eligible_source,
        "data_id": eligible_data_id,
        "natoms": eligible_natoms,
        "composition": eligible_composition,
        "data_id_counts_all": dict(data_id_counts_all),
        "element_freq_eligible": dict(element_freq_eligible),
    }


def phase2_val_eligibility(neutral_support: list[int]) -> dict:
    shards = list_shards(VAL_SRC)
    print(f"[phase2] {len(shards)} val shards, {N_WORKERS} workers", flush=True)

    # Reproduce AseDBDataset global index assignment exactly: shards sorted,
    # ids within each shard already 1..n_shard sequential -> cumulative offset.
    shard_lens = []
    for s in shards:
        db = ase.db.connect(s, readonly=True, use_lock_file=False)
        shard_lens.append(len(db.ids))
    offsets = np.cumsum([0] + shard_lens[:-1]).tolist()

    support_set = set(neutral_support)
    jobs = [(s, i, offsets[i], support_set) for i, s in enumerate(shards)]

    t0 = time.time()
    results = [None] * len(shards)
    with mp.Pool(N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(_scan_shard_val, jobs)):
            results[res["shard_idx"]] = res
            print(
                f"[phase2] shard {i+1}/{len(shards)} (idx {res['shard_idx']}) done: "
                f"total={res['total']} eligible={res['n_eligible']}", flush=True
            )
    dt = time.time() - t0
    print(f"[phase2] done in {dt:.1f}s", flush=True)

    total = sum(r["total"] for r in results)
    n_charge0 = sum(r["n_charge0"] for r in results)
    n_spin1 = sum(r["n_spin1"] for r in results)
    n_charge0_spin1 = sum(r["n_charge0_spin1"] for r in results)
    n_eligible = sum(r["n_eligible"] for r in results)

    global_idx = np.concatenate([np.array(r["global_idx"], dtype=np.int64) for r in results]) if n_eligible else np.array([], dtype=np.int64)
    source = sum((r["source"] for r in results), [])
    data_id = sum((r["data_id"] for r in results), [])
    natoms = np.concatenate([np.array(r["natoms"], dtype=np.int32) for r in results]) if n_eligible else np.array([], dtype=np.int32)
    composition = sum((r["composition"] for r in results), [])

    data_id_counts_all = Counter()
    for r in results:
        data_id_counts_all.update(r["data_id_counts_all"])

    data_id_counts_eligible = Counter(data_id)

    element_freq_eligible = Counter()
    for r in results:
        element_freq_eligible.update(r["element_freq_eligible"])

    return {
        "n_shards": len(shards),
        "shard_lens": shard_lens,
        "scan_seconds": dt,
        "total_val": total,
        "n_charge0": n_charge0,
        "n_spin1": n_spin1,
        "n_charge0_spin1": n_charge0_spin1,
        "n_eligible": n_eligible,
        "fraction_eligible": n_eligible / total if total else 0.0,
        "global_idx": global_idx,
        "source": source,
        "data_id": data_id,
        "natoms": natoms,
        "composition": composition,
        "data_id_counts_all": dict(data_id_counts_all),
        "data_id_counts_eligible": dict(data_id_counts_eligible),
        "element_freq_eligible": dict(element_freq_eligible),
    }


# ---------------------------------------------------------------------------
# Phase 3: stratified nested pool
# ---------------------------------------------------------------------------

def size_bin(n: int, edges: list[float]) -> int:
    for k, e in enumerate(edges):
        if n <= e:
            return k
    return len(edges)


def phase3_stratified_pool(elig: dict, seed: int = SEED, pool_sizes: list[int] = POOL_SIZES) -> dict:
    idx = elig["global_idx"]
    natoms = elig["natoms"]
    data_id = np.array(elig["data_id"])
    n_elig = len(idx)
    max_pool = min(pool_sizes[-1], n_elig)

    # size regime via quartiles of the eligible natoms distribution
    qs = np.quantile(natoms, [0.25, 0.5, 0.75]).tolist()
    bins = np.array([size_bin(int(n), qs) for n in natoms])

    strata_key = [f"{d}|size{b}" for d, b in zip(data_id, bins)]
    strata = defaultdict(list)
    for i, k in enumerate(strata_key):
        strata[k].append(i)  # local position into idx/natoms/data_id arrays

    rng = np.random.default_rng(seed)
    for k in strata:
        arr = np.array(strata[k])
        rng.shuffle(arr)
        strata[k] = arr.tolist()

    strata_names = sorted(strata.keys())
    strata_target_frac = {k: len(strata[k]) / n_elig for k in strata_names}
    strata_cursor = {k: 0 for k in strata_names}
    strata_picked = {k: 0 for k in strata_names}

    order = []  # local positions, in pool order
    for step in range(max_pool):
        # largest-remainder / Sainte-Lague style deficit: pick stratum whose
        # (target * (step+1) - picked) deficit is largest, skipping exhausted strata
        best_k = None
        best_deficit = -1e18
        for k in strata_names:
            if strata_cursor[k] >= len(strata[k]):
                continue
            deficit = strata_target_frac[k] * (step + 1) - strata_picked[k]
            if deficit > best_deficit:
                best_deficit = deficit
                best_k = k
        if best_k is None:
            break
        pos = strata[best_k][strata_cursor[best_k]]
        strata_cursor[best_k] += 1
        strata_picked[best_k] += 1
        order.append(pos)

    order = np.array(order, dtype=np.int64)
    pool_global_idx = idx[order]
    pool_natoms = natoms[order]
    pool_data_id = data_id[order]
    pool_source = [elig["source"][i] for i in order]
    pool_composition = [elig["composition"][i] for i in order]

    return {
        "seed": seed,
        "pool_sizes": [p for p in pool_sizes if p <= max_pool] + ([max_pool] if max_pool not in pool_sizes and max_pool < pool_sizes[-1] else []),
        "requested_pool_sizes": pool_sizes,
        "max_pool": max_pool,
        "n_eligible": n_elig,
        "size_bin_edges_natoms_quartiles": qs,
        "strata_names": strata_names,
        "strata_sizes": {k: len(strata[k]) for k in strata_names},
        "order_local_positions": order,
        "pool_global_idx": pool_global_idx,
        "pool_natoms": pool_natoms,
        "pool_data_id": pool_data_id,
        "pool_source": pool_source,
        "pool_composition": pool_composition,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== PHASE 1: neutral-train element support ===", flush=True)
    p1 = phase1_neutral_support()
    with open(os.path.join(OUT_DIR, "neutral_element_support.json"), "w") as f:
        json.dump(p1, f, indent=2)

    print("=== PHASE 2: val eligibility scan ===", flush=True)
    p2 = phase2_val_eligibility(p1["support"])

    # eligible_summary.json (drop the huge arrays into a separate npz)
    natoms = p2["natoms"]
    natoms_hist = {}
    if len(natoms):
        edges = [0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 10_000]
        hist, _ = np.histogram(natoms, bins=edges)
        natoms_hist = {f"{edges[i]}-{edges[i+1]}": int(hist[i]) for i in range(len(hist))}

    eligible_summary = {
        "total_val": p2["total_val"],
        "n_charge0": p2["n_charge0"],
        "n_spin1": p2["n_spin1"],
        "n_charge0_spin1": p2["n_charge0_spin1"],
        "n_eligible": p2["n_eligible"],
        "fraction_eligible": p2["fraction_eligible"],
        "neutral_element_support": p1["support"],
        "eligible_counts_by_data_id": p2["data_id_counts_eligible"],
        "val_counts_by_data_id_all": p2["data_id_counts_all"],
        "eligible_natoms_stats": {
            "min": int(natoms.min()) if len(natoms) else None,
            "max": int(natoms.max()) if len(natoms) else None,
            "mean": float(natoms.mean()) if len(natoms) else None,
            "median": float(np.median(natoms)) if len(natoms) else None,
            "p10": float(np.percentile(natoms, 10)) if len(natoms) else None,
            "p90": float(np.percentile(natoms, 90)) if len(natoms) else None,
        },
        "eligible_natoms_histogram": natoms_hist,
        "eligible_element_frequency": p2["element_freq_eligible"],
        "shard_lens_val": p2["shard_lens"],
        "scan_seconds_phase1": p1["scan_seconds"],
        "scan_seconds_phase2": p2["scan_seconds"],
    }
    with open(os.path.join(OUT_DIR, "eligible_summary.json"), "w") as f:
        json.dump(eligible_summary, f, indent=2)

    # persist full eligible index table for reproducibility / future use
    np.savez_compressed(
        os.path.join(OUT_DIR, "eligible_full_index.npz"),
        global_idx=p2["global_idx"],
        natoms=p2["natoms"],
        data_id=np.array(p2["data_id"]),
        source=np.array(p2["source"]),
        composition=np.array(p2["composition"]),
    )

    print("=== PHASE 3: stratified nested OOD pool ===", flush=True)
    p3 = phase3_stratified_pool(p2)

    np.save(os.path.join(OUT_DIR, "ood_pool_indices.npy"), p3["pool_global_idx"])

    manifest = {
        "seed": p3["seed"],
        "requested_pool_sizes": p3["requested_pool_sizes"],
        "achieved_pool_sizes": p3["pool_sizes"],
        "n_eligible_total": p3["n_eligible"],
        "max_pool": p3["max_pool"],
        "stratification": {
            "method": "sequential proportional (largest-remainder) interleaving across (data_id x natoms-quartile-bin) strata, seeded per-stratum shuffle",
            "size_bin_edges_natoms_quartiles_of_eligible_pop": p3["size_bin_edges_natoms_quartiles"],
            "strata_sizes_in_eligible_pop": p3["strata_sizes"],
        },
        "pools": {},
    }
    for k in POOL_SIZES:
        if k > p3["max_pool"]:
            continue
        sl = slice(0, k)
        manifest["pools"][str(k)] = [
            {
                "dataset_index": int(p3["pool_global_idx"][j]),
                "source": p3["pool_source"][j],
                "data_id": str(p3["pool_data_id"][j]),
                "natoms": int(p3["pool_natoms"][j]),
            }
            for j in range(k)
        ]
    with open(os.path.join(OUT_DIR, "ood_pool_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # subset_summary.csv: per nested prefix, breakdown by data_id and size bin
    import csv
    edges = p3["size_bin_edges_natoms_quartiles"]
    with open(os.path.join(OUT_DIR, "subset_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool_size", "data_id", "size_bin", "count", "frac_of_pool"])
        for k in POOL_SIZES:
            if k > p3["max_pool"]:
                continue
            sub_data_id = p3["pool_data_id"][:k]
            sub_natoms = p3["pool_natoms"][:k]
            sub_bins = np.array([size_bin(int(n), edges) for n in sub_natoms])
            cnt = Counter(zip(sub_data_id.tolist(), sub_bins.tolist()))
            for (d, b), c in sorted(cnt.items()):
                w.writerow([k, d, b, c, c / k])

    # DATASET_AUDIT.md
    lines = []
    lines.append("# eSEN Irrep OOD Population — Dataset Audit\n")
    lines.append(f"Generated by `scan_omol_ood.py`. Seed = {SEED}.\n")
    lines.append("## 1. Neutral training element support\n")
    lines.append(f"- Source: `{p1['src']}`\n")
    lines.append(f"- Shards scanned: {p1['n_shards']} (full scan, {p1['scan_seconds']:.0f}s)\n")
    lines.append(f"- Elements observed ({len(p1['support'])}): {sorted(p1['support'])}\n")
    lines.append("\n## 2. Val eligibility scan\n")
    lines.append(f"- Source: `{VAL_SRC}`\n")
    lines.append(f"- Shards scanned: {p2['n_shards']} (full scan, {p2['scan_seconds']:.0f}s)\n")
    lines.append(f"- Total Val structures: {p2['total_val']:,}\n")
    lines.append(f"- charge==0: {p2['n_charge0']:,}\n")
    lines.append(f"- spin==1: {p2['n_spin1']:,}\n")
    lines.append(f"- charge==0 AND spin==1: {p2['n_charge0_spin1']:,}\n")
    lines.append(f"- + all Z in Neutral support -> **eligible: {p2['n_eligible']:,}** ({100*p2['fraction_eligible']:.3f}% of Val)\n")
    lines.append("\n### Eligible counts by data_id\n")
    for d, c in sorted(p2["data_id_counts_eligible"].items(), key=lambda x: -x[1]):
        lines.append(f"- {d}: {c:,}\n")
    lines.append("\n### Eligible atom-count distribution\n")
    lines.append(f"- min={eligible_summary['eligible_natoms_stats']['min']}, "
                  f"median={eligible_summary['eligible_natoms_stats']['median']}, "
                  f"mean={eligible_summary['eligible_natoms_stats']['mean']:.1f}, "
                  f"p90={eligible_summary['eligible_natoms_stats']['p90']}, "
                  f"max={eligible_summary['eligible_natoms_stats']['max']}\n")
    lines.append("\n## 3. Frozen OOD pool\n")
    lines.append(f"- Nested prefixes: {p3['pool_sizes']}\n")
    lines.append(f"- Stratification: data_id x natoms-quartile-bin, sequential proportional (largest-remainder) interleaving, per-stratum shuffle seed={SEED}\n")
    lines.append(f"- quartile edges (natoms, from eligible pop): {p3['size_bin_edges_natoms_quartiles']}\n")
    lines.append("\nOutputs in this directory: `neutral_element_support.json`, `eligible_summary.json`, "
                  "`eligible_full_index.npz` (full eligible table), `ood_pool_manifest.json`, `ood_pool_indices.npy`, `subset_summary.csv`.\n")
    with open(os.path.join(OUT_DIR, "DATASET_AUDIT.md"), "w") as f:
        f.writelines(lines)

    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
