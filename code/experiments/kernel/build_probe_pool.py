#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/build_large_m_probe_pool.py
# source revision withheld for anonymous review, original SHA256 e1e6b4356a11a262343f950a4e8034bcd541903564944535199da2ccfa915553
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/build_large_m_probe_pool.py
# source revision withheld for anonymous review, original SHA256 e1e6b4356a11a262343f950a4e8034bcd541903564944535199da2ccfa915553
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Extend the frozen 256-atom force-NTK probe pool to 1024 atoms using the identical sampling
procedure, so the first 256 rows stay bit-identical and every nested-pool comparison remains
valid.

Validates the nesting property explicitly (prefix hash match against the existing
probe_indices.npz) before writing anything.

Usage:
  python pipeline/kernel/build_probe_pool.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from pipeline.kernel.force_ntk import (
    OUTPUT_DIR as STATIC_OUTPUT_DIR,
    VAL_PATH,
    TRAIN_RMSD,
    PROBE_SEED,
    M as M_ORIG,
    canonical_probe_hash,
    load_dataset,
    sha256_file,
)

OUT_DIR = REPO_ROOT / "analysis_outputs/large_m_convergence_2026_08_19"
M_EXT = 1024


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_npz = OUT_DIR / "probe_indices_ext.npz"
    out_json = OUT_DIR / "probe_manifest_ext.json"
    if out_npz.exists() or out_json.exists():
        raise RuntimeError("extended probe outputs already exist; refusing to overwrite")

    dataset = load_dataset()
    if len(dataset) < M_EXT:
        raise RuntimeError(f"validation set has only {len(dataset)} configurations")

    # Identical procedure to freeze_probe() in kernel/force_ntk.py,
    # with M replaced by M_EXT. config_rng.permutation(len(dataset))[:M] is a PREFIX of
    # config_rng.permutation(len(dataset))[:M_EXT] for any M <= M_EXT (same seed, same
    # permutation call) -- so the first M_ORIG rows are guaranteed identical by construction,
    # not just by luck. Same argument for the atom_rng sequential draws below.
    config_rng = np.random.RandomState(PROBE_SEED)
    config_indices = config_rng.permutation(len(dataset))[:M_EXT].astype(np.int64)
    atom_rng = np.random.RandomState((PROBE_SEED * 1_000_003 + 7) % (2**31 - 1))
    atom_indices = np.empty(M_EXT, dtype=np.int64)
    natoms = np.empty(M_EXT, dtype=np.int64)
    atomic_numbers = np.empty(M_EXT, dtype=np.int64)
    raw_forces = np.empty((M_EXT, 3), dtype=np.float64)
    t0 = time.time()
    for row, g in enumerate(config_indices):
        item = dataset[int(g)]
        n = int(len(item.atomic_numbers))
        i = int(atom_rng.randint(0, n))
        atom_indices[row] = i
        natoms[row] = n
        atomic_numbers[row] = int(item.atomic_numbers[i])
        raw_forces[row] = np.asarray(item.forces[i].detach().cpu(), dtype=np.float64)
        if (row + 1) % 128 == 0:
            print(f"  probe rows {row+1}/{M_EXT}, elapsed={time.time()-t0:.1f}s", flush=True)
    y = (raw_forces / TRAIN_RMSD).reshape(-1)
    probe_hash = canonical_probe_hash(config_indices, atom_indices, y)

    # --- Mandatory nesting validation against the existing frozen M=256 probe ---
    orig = np.load(STATIC_OUTPUT_DIR / "probe_indices.npz", allow_pickle=False)
    orig_config = orig["config_indices"].astype(np.int64)
    orig_atom = orig["atom_indices"].astype(np.int64)
    orig_y = orig["y"].astype(np.float64)
    prefix_config_ok = bool(np.array_equal(config_indices[:M_ORIG], orig_config))
    prefix_atom_ok = bool(np.array_equal(atom_indices[:M_ORIG], orig_atom))
    prefix_y_ok = bool(np.array_equal(y.reshape(-1, 3)[:M_ORIG].reshape(-1), orig_y))
    prefix_hash = canonical_probe_hash(config_indices[:M_ORIG], atom_indices[:M_ORIG],
                                        y.reshape(-1, 3)[:M_ORIG].reshape(-1))
    orig_hash = str(orig["probe_hash"])
    prefix_hash_ok = prefix_hash == orig_hash
    if not (prefix_config_ok and prefix_atom_ok and prefix_y_ok and prefix_hash_ok):
        raise AssertionError(
            f"NESTING VALIDATION FAILED: config={prefix_config_ok} atom={prefix_atom_ok} "
            f"y={prefix_y_ok} hash={prefix_hash_ok} (prefix_hash={prefix_hash}, orig={orig_hash})")
    print(f"Nesting validation PASSED: first {M_ORIG} rows of the M={M_EXT} pool are bit-identical "
          f"to the frozen M={M_ORIG} probe (probe_hash={prefix_hash}).", flush=True)

    np.savez(
        out_npz,
        config_indices=config_indices,
        atom_indices=atom_indices,
        natoms=natoms,
        atomic_numbers=atomic_numbers,
        raw_forces=raw_forces,
        y=y,
        m=np.int64(M_EXT),
        probe_seed=np.int64(PROBE_SEED),
        train_rmsd=np.float64(TRAIN_RMSD),
        probe_hash=np.array(probe_hash),
    )
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "large-m convergence check of GemNet-OC vs eSEN raw force-NTK ordering",
        "sampling_measure": "configuration uniform, then one atom uniform within configuration "
                             "(identical procedure/seeds to kernel/force_ntk.py freeze_probe)",
        "m": M_EXT,
        "m_orig_prefix": M_ORIG,
        "probe_seed": PROBE_SEED,
        "dataset_path": str(VAL_PATH),
        "dataset_length": int(len(dataset)),
        "train_rmsd": TRAIN_RMSD,
        "target_definition": "marked physical force / train_rmsd, atom-major xyz order",
        "probe_hash": probe_hash,
        "nesting_validation": {
            "prefix_config_indices_match": prefix_config_ok,
            "prefix_atom_indices_match": prefix_atom_ok,
            "prefix_y_match": prefix_y_ok,
            "prefix_probe_hash_match": prefix_hash_ok,
            "orig_probe_hash": orig_hash,
        },
        "probe_indices_npz_sha256": sha256_file(out_npz),
    }
    out_json.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
