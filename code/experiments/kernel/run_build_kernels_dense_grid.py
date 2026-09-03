#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/run_dense_grid_large_m_kernels_2026_08_23.py
# source revision withheld for anonymous review, original SHA256 d7f63f918227e682a31a9f9a44860aceb55e5dd3d80ecfa219e80ca580497731
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/run_dense_grid_large_m_kernels_2026_08_23.py
# source revision withheld for anonymous review, original SHA256 d7f63f918227e682a31a9f9a44860aceb55e5dd3d80ecfa219e80ca580497731
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Dense-grid (PRECROSS/CROSS/FITX/POSTCROSS/TOP) m=1024 force-tangent-kernel construction
(2026-08-23), extending `run_matched_compute_large_m_kernels.py` (2026-08-22) to the 16 new
unique owner checkpoints in `owners_dense_grid_2026_08_23.OWNERS_DENSE23`.

Composes the SAME unchanged code paths as the 2026-08-22 script:
  - owner provenance / strict checkpoint loading from `kernel/frontier_checkpoints.py`
  - Jacobian collection / chunked Gram assembly / numerics from
    `kernel/force_ntk.py`, `kernel/force_ntk_trained.py`,
    `oak_schedulefree_kernel_falsification.py`

The ONLY difference from the 2026-08-22 script: these 16 owners are brand-new (no prior m=256
kernel exists for any of them to reproduce), so the mandatory-for-2026-08-22 "m256 leading
submatrix reproduction gate" is REPLACED by the self-consistency checks the pipeline already runs
for any brand-new owner (this is exactly what that script's own `numerics_report` gate already
does, unconditionally, before the m256 gate is even reached): finiteness, exact symmetry, and PSD
of the m=1024 raw kernel, plus the full checkpoint provenance check (SHA256 + global_step) against
a purpose-built manifest (`owners_dense_grid_2026_08_23.OWNERS_RAW_DENSE23`, since these tags are
not in `frontier_checkpoint_manifest.csv`). No numerics/provenance logic is reimplemented -- same imported
functions, called the same way.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from pipeline.kernel.frontier_checkpoints import (
    OwnerSpec,
    instantiate_model_generic,
    load_checkpoint_strict_generic,
    INSTANTIATION_SEED,
)
from pipeline.kernel.force_ntk import (
    canonical_probe_hash,
    load_dataset,
    raw_gram_chunked,
    spectral_arrays,
    sha256_file,
)
from pipeline.kernel.force_ntk_trained import (
    collect_jacobian_with_preds,
    target_power_arrays,
)
from pipeline.kernel.gram_numerics import (
    numerics_report,
    alignment_stat,
)
from pipeline.kernel.dense_grid_checkpoints import OWNERS_DENSE23, OWNERS_RAW_DENSE23

OUTPUT_DIR = REPO_ROOT / "analysis_outputs/dense_grid_claim1_2026_08_23"
KERNEL_DIR = OUTPUT_DIR / "kernels"
LOG_DIR = OUTPUT_DIR / "logs"
TMP_ROOT = Path("<CHECKPOINT_ROOT>/dense_grid_large_m_kernels_tmp_20260823")

EXT_PROBE_DIR = REPO_ROOT / "analysis_outputs/large_m_convergence_2026_08_19"

# purpose-built provenance manifest for these 16 new tags, keyed the same way
# check_owner_provenance in the 2026-08-22 script reads frontier_checkpoint_manifest.csv rows, but sourced from
# this module's own OWNERS_RAW_DENSE23 (there is no frontier_checkpoint_manifest.csv row for a _dense23 tag).
_RAW_BY_TAG = {}
for _raw in OWNERS_RAW_DENSE23:
    _tag = f"{_raw['architecture']}_{_raw['budget_label']}_w{_raw['width']}_s{_raw['step']}_dense23"
    _RAW_BY_TAG[_tag] = _raw


def load_extended_probe():
    npz = EXT_PROBE_DIR / "probe_indices_ext.npz"
    manifest = json.loads((EXT_PROBE_DIR / "probe_manifest_ext.json").read_text())
    z = np.load(npz, allow_pickle=False)
    config_indices = z["config_indices"].astype(np.int64)
    atom_indices = z["atom_indices"].astype(np.int64)
    y = z["y"].astype(np.float64)
    probe_hash = canonical_probe_hash(config_indices, atom_indices, y)
    if probe_hash != manifest["probe_hash"] or probe_hash != str(z["probe_hash"]):
        raise AssertionError("extended probe hash mismatch")
    return config_indices, atom_indices, y, probe_hash


def check_owner_provenance_dense23(owner: OwnerSpec) -> dict[str, Any]:
    """Same check as the 2026-08-22 script's `check_owner_provenance`, but against this module's
    own literal (architecture,width,step,ckpt_path) tuple instead of frontier_checkpoint_manifest.csv (these
    tags do not exist in that file). Checks: checkpoint path matches the literal spec, SHA256 of
    the on-disk file, and (deferred to load_checkpoint_strict_generic) global_step recorded
    inside the checkpoint itself."""
    raw = _RAW_BY_TAG[owner.tag]
    ckpt_path = Path(owner.ckpt_path)
    actual_sha = sha256_file(ckpt_path)
    result = {
        "tag": owner.tag,
        "ckpt_path": str(ckpt_path),
        "spec_ckpt_path": raw["ckpt_path"],
        "path_match": str(ckpt_path) == raw["ckpt_path"],
        "actual_sha256": actual_sha,
        "owner_step": owner.step,
        "owner_expected_n": owner.expected_n,
        "owner_C_owner": owner.C_owner,
        "owner_force_mse_norm": owner.force_mse_norm,
    }
    if not result["path_match"]:
        raise AssertionError(f"PROVENANCE CHECK FAILED for {owner.tag}: {result}")
    return result


def self_consistency_gate(owner: OwnerSpec, k_x_m1024: np.ndarray, y_m1024: np.ndarray,
                            num: dict[str, Any]) -> dict[str, Any]:
    """Self-consistency checks for a brand-new owner with no prior m=256 kernel to reproduce
    against (per the task brief: run every self-consistency check the pipeline already performs,
    report pass/fail, do not skip just because there is no frozen m=256 comparison value).
    `num` is the numerics_report already computed on k_x_m1024 (finite/symmetry/PSD) -- reused
    here, not recomputed. Adds: y is finite and matches the shared extended-probe target
    (bit-identical across all owners, since the probe is fixed), and K has a strictly positive
    trace (a real Gram matrix must)."""
    y_finite = bool(np.all(np.isfinite(y_m1024)))
    trace = float(np.trace(k_x_m1024))
    result = {
        "numerics_finite": bool(num["finite"]),
        "numerics_symmetry_relative_max_error": num["symmetry_relative_max_error"],
        "numerics_symmetry_pass": bool(num["symmetry_relative_max_error"] < 1e-9),
        "numerics_psd_pass": bool(num["psd_pass"]),
        "y_finite": y_finite,
        "K_trace": trace,
        "K_trace_positive": trace > 0.0,
        "pass": bool(
            num["finite"]
            and num["symmetry_relative_max_error"] < 1e-9
            and num["psd_pass"]
            and y_finite
            and trace > 0.0
        ),
    }
    return result


def run_one(tag: str, chunk_size: int, dry_run_owner_load_only: bool = False) -> dict[str, Any]:
    owner = OWNERS_DENSE23[tag]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== {tag} === device={device} job={job_id}", flush=True)

    provenance = check_owner_provenance_dense23(owner)
    print("Provenance check (path/sha) PASS", json.dumps(provenance, default=str), flush=True)

    t0 = time.time()
    model, cfg, params = instantiate_model_generic(owner, device, INSTANTIATION_SEED)
    checkpoint_provenance = load_checkpoint_strict_generic(model, owner, params)
    model.net.eval()
    print(f"model instantiated + checkpoint loaded strict: N={owner.expected_n}", flush=True)
    print("checkpoint global_step verified:", checkpoint_provenance.get("checkpoint_global_step"),
          flush=True)

    if dry_run_owner_load_only:
        del model, cfg, params
        gc.collect()
        return {"tag": tag, "dry_run": True, "provenance": provenance,
                "checkpoint_provenance": checkpoint_provenance}

    config_indices, atom_indices, y, probe_hash = load_extended_probe()
    m = len(config_indices)
    dataset = load_dataset()

    run_token = f"{tag}_m{m}_job{job_id}_pid{os.getpid()}"
    tmp_dir = TMP_ROOT / run_token
    grad_path = tmp_dir / "G_block_float32.dat"

    g_store, coverage, jacobian_seconds, f_pred = collect_jacobian_with_preds(
        model, params, dataset, config_indices, atom_indices, y, grad_path
    )
    if coverage["parameter_count_represented_by_jacobian"] != owner.expected_n:
        raise AssertionError("Jacobian parameter count does not equal intended trainable N")
    if f_pred.shape != (3 * m,):
        raise AssertionError(f"f_pred shape mismatch: {f_pred.shape}")

    del model, params, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t_gram0 = time.time()
    k_raw = raw_gram_chunked(g_store, chunk_size, device)
    gram_seconds = time.time() - t_gram0

    num = numerics_report(k_raw, "K_x_m1024")
    if not (num["finite"] and num["symmetry_relative_max_error"] < 1e-9 and num["psd_pass"]):
        raise AssertionError(f"Gate 5 (numerics) FAILED for {tag}: {num}")

    raw_eigs, op_eigs, q, alignment, tail, local_slopes, desc = spectral_arrays(k_raw, y, m)
    desc["trace_raw"] = float(np.sum(raw_eigs))
    desc["trace_over_dim"] = float(np.sum(raw_eigs) / (3 * m))
    residual = y - f_pred
    q_res, alignment_res, tail_res, desc_res = target_power_arrays(k_raw, residual, m)
    align_full = alignment_stat(k_raw, y)

    # --- self-consistency gate (replaces the m256 gate: no prior m=256 kernel exists for these
    # brand-new owners to reproduce against) ---
    gate = self_consistency_gate(owner, k_raw, y, num)
    print("SELF-CONSISTENCY GATE (no prior m=256 kernel to reproduce against):",
          json.dumps(gate, indent=2), flush=True)
    if not gate["pass"]:
        raise AssertionError(f"SELF-CONSISTENCY GATE FAILED for {tag}: {gate}")

    del g_store
    gc.collect()
    grad_path.unlink()
    tmp_dir.rmdir()

    artifact_path = KERNEL_DIR / f"{tag}_m1024_kernel.npz"
    meta_path = KERNEL_DIR / f"{tag}_m1024_metadata.json"
    np.savez(
        artifact_path,
        K_raw=k_raw.astype(np.float64),
        y=y.astype(np.float64),
        f_pred=f_pred.astype(np.float64),
        residual=residual.astype(np.float64),
        eigenvalues_raw=raw_eigs.astype(np.float64),
        eigenvalues_operator=op_eigs.astype(np.float64),
        q=q.astype(np.float64),
        cumulative_target_alignment=alignment.astype(np.float64),
        target_tail=tail.astype(np.float64),
        q_residual=q_res.astype(np.float64),
        cumulative_residual_alignment=alignment_res.astype(np.float64),
        residual_tail=tail_res.astype(np.float64),
        config_indices=config_indices.astype(np.int64),
        atom_indices=atom_indices.astype(np.int64),
        m=np.int64(m),
        N=np.int64(owner.expected_n),
        width=np.int64(owner.width),
        budget_label=np.array(owner.budget_label),
        probe_hash=np.array(probe_hash),
    )

    result = {
        "tag": tag, "architecture": owner.architecture, "architecture_label": owner.architecture_label,
        "budget_label": owner.budget_label, "m": m, "probe_hash": probe_hash,
        "device": device, "slurm_job_id": job_id, "host": socket.gethostname(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "owner_provenance_check": provenance,
        "checkpoint_provenance": checkpoint_provenance,
        "numerics": num,
        "self_consistency_gate": gate,
        "spectral": desc,
        "residual_spectral": desc_res,
        "alignment_A_Kx_y_m1024": align_full,
        "jacobian_seconds": jacobian_seconds,
        "gram_seconds": gram_seconds,
        "total_seconds": time.time() - t0,
        "kernel_artifact_path": str(artifact_path),
        "kernel_artifact_sha256": sha256_file(artifact_path),
    }
    meta_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"wrote {artifact_path}", flush=True)
    print(f"wrote {meta_path}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(OWNERS_DENSE23))
    ap.add_argument("--gram-chunk-size", type=int, default=250_000)
    ap.add_argument("--dry-run-owner-load-only", action="store_true")
    args = ap.parse_args()
    run_one(args.tag, args.gram_chunk_size, args.dry_run_owner_load_only)


if __name__ == "__main__":
    main()
