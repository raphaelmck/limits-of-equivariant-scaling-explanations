#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/four_arch_trained_force_kernel_pilot.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 ee2ea9e0a54b6ac4d27bb980f1ba6a366980db8a555fc3802216b8e0be0a5df2
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/four_arch_trained_force_kernel_pilot.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 ee2ea9e0a54b6ac4d27bb980f1ba6a366980db8a555fc3802216b8e0be0a5df2
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Trained force-tangent-kernel pilot (Phase 2): K0 vs K50k vs K500k full block force-NTK
for the four large-width architectures, reusing the audited static-pilot kernel machinery
without modification (model instantiation, chunked Gram formation, spectral arrays, probe
loading/hash verification are imported unchanged from four_arch_initial_force_kernel_pilot).

This script adds exactly what the static pilot did not need:
  - checkpoint loading at step=50000 / step=500000, with strict state-dict load and the same
    MC-EGNN legacy force_decoder.width_mult / absent-_mup_ref_param provenance checks used in
    the static pilot's audit_one();
  - capture of the network's own force prediction f_pred (the static pilot never saved this);
  - K0 reconstruction at seed=1 (matching every training run's cfg.seed), rather than reusing
    the static pilot's seed=0 K0 artifacts. Training seeds the RNG before instantiating the
    datamodule and then the model (train_omol.py: L.seed_everything -> datamodule -> model),
    so matching cfg.seed=1 is necessary but not sufficient for a bitwise-identical initial
    state; this K0 is therefore SAME CONFIG/SEED, NOT PROVEN BITWISE TRAJECTORY, never claimed
    as an exact reconstruction of the training run's initialization.

Single mode: `kernel --tag <large_tag> --time K0|K50k|K500k`.
"""
from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

from pipeline.kernel.force_ntk import (
    SPECS as ALL_SPECS,
    OUTPUT_DIR as STATIC_OUTPUT_DIR,
    TRAIN_RMSD,
    sha256_file,
    instantiate_model,
    load_dataset,
    load_probe,
    raw_gram_chunked,
    spectral_arrays,
)

LARGE_TAGS = ["mpnn_large_hc2048", "mcegnn_large_hc512",
              "gemnet_oc_large_width256", "esen_large_sphere64"]
SPECS = {t: ALL_SPECS[t] for t in LARGE_TAGS}

OUTPUT_DIR = REPO_ROOT / "analysis_outputs/four_arch_trained_force_kernel_pilot_2026_08_15"
K0_SEED = 1  # matches cfg.seed=1 in every training run's .hydra/config.yaml
STATIC_K0_SEED = 0  # the seed the static (initial-kernel) pilot used
TRAINING_CFG_SEED = 1
RCOND = 1e-10

TIMES = ["K0", "K50k", "K500k"]
STEP_OF = {"K50k": 50_000, "K500k": 500_000}


def checkpoint_glob_for(spec, step: int) -> str:
    return spec.checkpoint_glob.replace("step=500000", f"step={step}")


def resolve_checkpoint_at(spec, step: int) -> Path:
    matches = sorted(Path(x) for x in glob.glob(checkpoint_glob_for(spec, step)))
    if len(matches) != 1:
        raise RuntimeError(f"{spec.tag}: expected exactly one checkpoint at step={step}, got {matches}")
    return matches[0]


def load_checkpoint_strict(model, spec, ckpt_path: Path, params) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    parameter_state_keys = {f"net.{name}" for name, _ in params}
    missing = sorted(parameter_state_keys - set(state))
    if missing:
        raise AssertionError(f"{spec.tag}: checkpoint lacks net parameters {missing}")
    state_net_n = int(sum(state[k].numel() for k in parameter_state_keys))
    if state_net_n != spec.expected_n:
        raise AssertionError(
            f"{spec.tag}: checkpoint net state count={state_net_n}, expected={spec.expected_n}")

    legacy = None
    if spec.architecture == "mcegnn":
        decoder = model.net.force_decoder
        expected_mult = math.isqrt(spec.width) / math.isqrt(102)
        actual_mult = float(decoder.width_mult)
        leaked = [name for name, module in model.net.named_modules()
                  if hasattr(module, "_mup_ref_param")]
        if leaked:
            raise AssertionError(f"{spec.tag}: leaked _mup_ref_param attributes: {leaked}")
        if not math.isclose(actual_mult, expected_mult, rel_tol=0.0, abs_tol=1e-15):
            raise AssertionError(
                f"{spec.tag}: force_decoder.width_mult={actual_mult}, legacy expected={expected_mult}")
        legacy = {
            "force_decoder_width_mult": actual_mult,
            "legacy_formula": "isqrt(hidden_channels)/isqrt(102)",
            "mup_ref_param_attributes": leaked,
        }

    global_step = int(ckpt["global_step"]) if "global_step" in ckpt else None
    return {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_size_bytes": ckpt_path.stat().st_size,
        "checkpoint_global_step": global_step,
        "checkpoint_strict_load": True,
        "checkpoint_net_state_parameter_count": state_net_n,
        "legacy_mcegnn_checkpoint_audit": legacy,
    }


def collect_jacobian_with_preds(model, params, dataset, config_indices, atom_indices,
                                y_expected, grad_path: Path, chunk_probe_check: bool = True):
    """Identical to the audited collect_jacobian in kernel/force_ntk.py,
    plus capture of the network's own (already train_rmsd-normalized) force prediction on
    every marked atom -- the one thing the static pilot never saved."""
    m = len(config_indices)
    names = [name for name, _ in params]
    numels = np.array([p.numel() for _, p in params], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(numels)])
    p_full = int(offsets[-1])
    grad_path.parent.mkdir(parents=True, exist_ok=False)
    g_store = np.memmap(grad_path, dtype=np.float32, mode="w+", shape=(3 * m, p_full))
    ever_nonzero: set[str] = set()
    ever_has_grad: set[str] = set()
    observed_y = np.empty(3 * m, dtype=np.float64)
    f_pred = np.empty(3 * m, dtype=np.float64)
    t0 = time.time()
    device = next(model.net.parameters()).device

    for row, (g_idx, atom_idx) in enumerate(zip(config_indices, atom_indices)):
        item = dataset[int(g_idx)]
        batch = atomicdata_list_to_batch([item]).to(device)
        if not hasattr(batch, "z"):
            batch.z = batch.atomic_numbers.long()
        preds = model.net(batch)
        force_pred = preds["forces"]
        force_gt = batch.forces
        observed_y[3 * row:3 * row + 3] = (
            force_gt[int(atom_idx)].detach().cpu().numpy().astype(np.float64) / TRAIN_RMSD
        )
        # force_pred is model.net's direct output, trained against force_gt/train_rmsd
        # (src/model/omol_module.py: f_loss_fn(f_pred, f_gt / self.train_rmsd, ...)) -- already
        # in the same normalized units as y, no further scaling needed.
        f_pred[3 * row:3 * row + 3] = (
            force_pred[int(atom_idx)].detach().cpu().numpy().astype(np.float64)
        )
        for c in range(3):
            model.net.zero_grad(set_to_none=True)
            force_pred[int(atom_idx), c].backward(retain_graph=(c < 2))
            out_row = 3 * row + c
            for j, (name, p) in enumerate(params):
                lo, hi = int(offsets[j]), int(offsets[j + 1])
                if p.grad is None:
                    g_store[out_row, lo:hi] = 0.0
                    continue
                ever_has_grad.add(name)
                block = p.grad.detach().reshape(-1).float().cpu().numpy()
                g_store[out_row, lo:hi] = block
                if np.any(block != 0.0):
                    ever_nonzero.add(name)
        model.net.zero_grad(set_to_none=True)
        del preds, force_pred, force_gt, batch
        if (row + 1) % max(1, m // 16) == 0 or row + 1 == m:
            print(f"  Jacobian atoms {row + 1}/{m}, elapsed={time.time()-t0:.1f}s", flush=True)
    g_store.flush()
    if chunk_probe_check and not np.array_equal(observed_y, y_expected):
        max_err = float(np.max(np.abs(observed_y - y_expected)))
        if not np.allclose(observed_y, y_expected, rtol=0.0, atol=1e-12):
            raise AssertionError(f"target differs from frozen probe, max_abs={max_err}")
    if not np.all(np.isfinite(f_pred)):
        raise AssertionError("f_pred contains non-finite values")
    coverage = {
        "all_trainable_parameter_names": names,
        "parameter_numels": {n: int(v) for n, v in zip(names, numels)},
        "parameter_count_represented_by_jacobian": p_full,
        "parameter_tensors_with_some_autograd_tensor": sorted(ever_has_grad),
        "parameter_tensors_with_some_nonzero_entry": sorted(ever_nonzero),
        "systematically_grad_none_parameter_tensors": sorted(set(names) - ever_has_grad),
        "systematically_zero_or_none_parameter_tensors": sorted(set(names) - ever_nonzero),
    }
    return g_store, coverage, time.time() - t0, f_pred


def target_power_arrays(k_raw: np.ndarray, target: np.ndarray, m: int):
    """Same construction as spectral_arrays' q/alignment/tail block, applied to an arbitrary
    target vector (used here for the residual r_t, in addition to y)."""
    k_sym = 0.5 * (k_raw + k_raw.T)
    evals, evecs = np.linalg.eigh(k_sym)
    order = np.argsort(evals)[::-1]
    u = evecs[:, order]
    q = (u.T @ target) ** 2 / m
    total_q = float(np.dot(target, target) / m)
    alignment = np.cumsum(q) / total_q if total_q > 0 else np.full_like(q, np.nan)
    tail = total_q - np.cumsum(q)
    tail[np.abs(tail) < 1e-14 * max(total_q, 1.0)] = 0.0
    quantiles = {}
    for pct in [50, 80, 90, 95]:
        quantiles[f"k{pct}"] = int(np.searchsorted(alignment, pct / 100.0) + 1)
    desc = {
        "parseval_sum_q": float(q.sum()),
        "parseval_target_norm_over_m": total_q,
        "parseval_abs_error": float(abs(q.sum() - total_q)),
        **quantiles,
    }
    return q, alignment, tail, desc


def execute_kernel(spec, time_tag: str, tmp_root: Path, chunk_size: int) -> dict[str, Any]:
    if time_tag not in TIMES:
        raise ValueError(time_tag)
    static_audit_path = STATIC_OUTPUT_DIR / "model_reconstruction_audit.json"
    if not static_audit_path.exists():
        raise RuntimeError("static-pilot model reconstruction audit is missing; production is gated on it")
    audit = json.loads(static_audit_path.read_text())
    if audit.get("status") != "PASS":
        raise RuntimeError("static-pilot model reconstruction audit did not pass")

    config_indices, atom_indices, y, probe_hash = load_probe(None)
    m = len(config_indices)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_token = f"{spec.tag}_{time_tag}_m{m}_job{job_id}_pid{os.getpid()}"
    tmp_dir = tmp_root / run_token
    grad_path = tmp_dir / "G_block_float32.dat"
    artifact_dir = OUTPUT_DIR / "kernels"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{spec.tag}_{time_tag}_m{m}_kernel.npz"
    meta_path = artifact_dir / f"{spec.tag}_{time_tag}_m{m}_metadata.json"
    if artifact_path.exists() or meta_path.exists():
        raise RuntimeError(f"refusing to overwrite existing result for {run_token}")

    t0 = time.time()
    # For K50k/K500k the instantiation seed has no effect on the final kernel: every
    # parameter tensor is overwritten by the strict checkpoint load immediately below. It is
    # recorded purely for provenance symmetry with the K0 case.
    init_seed = K0_SEED if time_tag == "K0" else K0_SEED
    model, cfg, params = instantiate_model(spec, device, seed=init_seed)

    checkpoint_provenance = None
    if time_tag != "K0":
        step = STEP_OF[time_tag]
        ckpt_path = resolve_checkpoint_at(spec, step)
        checkpoint_provenance = load_checkpoint_strict(model, spec, ckpt_path, params)
        checkpoint_provenance["checkpoint_requested_step"] = step
        model.net.eval()

    if spec.architecture == "mcegnn":
        expected_mult = math.isqrt(spec.width) / 10
        assert not hasattr(model.net.force_decoder, "_mup_ref_param")
        assert math.isclose(float(model.net.force_decoder.width_mult), expected_mult,
                            rel_tol=0.0, abs_tol=1e-15)

    dataset = load_dataset()
    g_store, coverage, jacobian_seconds, f_pred = collect_jacobian_with_preds(
        model, params, dataset, config_indices, atom_indices, y, grad_path
    )
    if coverage["parameter_count_represented_by_jacobian"] != spec.expected_n:
        raise AssertionError("Jacobian parameter count does not equal intended trainable N")
    if f_pred.shape != (3 * m,):
        raise AssertionError(f"f_pred shape mismatch: {f_pred.shape}")

    del model, params, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    k_raw = raw_gram_chunked(g_store, chunk_size, device)

    symmetry_abs = float(np.max(np.abs(k_raw - k_raw.T)))
    symmetry_rel = symmetry_abs / max(float(np.max(np.abs(k_raw))), 1e-300)
    raw_eigs, op_eigs, q, alignment, tail, local_slopes, desc = spectral_arrays(k_raw, y, m)
    psd_tol = 1e-10 * max(float(raw_eigs[0]), 0.0)
    psd_pass = float(raw_eigs[-1]) >= -psd_tol
    if symmetry_rel > 1e-12 or not psd_pass or desc["parseval_abs_error"] > 1e-10:
        raise AssertionError(
            f"numerical validation failed: symmetry_rel={symmetry_rel}, "
            f"min_eig={raw_eigs[-1]}, psd_tol={psd_tol}, parseval={desc['parseval_abs_error']}"
        )

    residual = y - f_pred
    f_pred_reshaped = f_pred.reshape(m, 3)
    if f_pred_reshaped.shape != (m, 3) or not np.all(np.isfinite(f_pred_reshaped)):
        raise AssertionError("f_pred failed shape/finite assertion before write")
    f_pred_sha256 = hashlib.sha256(np.ascontiguousarray(f_pred, dtype=np.float64).tobytes()).hexdigest()

    q_res, alignment_res, tail_res, desc_res = target_power_arrays(k_raw, residual, m)

    trajectory_classification = (
        "SAME_CONFIG_SEED_NOT_PROVEN_BITWISE_TRAJECTORY" if time_tag == "K0" else
        "CHECKPOINT_EXACT_TRAINED_STATE"
    )

    metadata = {
        **asdict(spec),
        "time_tag": time_tag,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                             text=True).strip(),
        "host": socket.gethostname(),
        "slurm_job_id": job_id,
        "init_seed": init_seed,
        "k0_reconstruction_seed": K0_SEED,
        "static_pilot_k0_seed_used_originally": STATIC_K0_SEED,
        "training_run_cfg_seed": TRAINING_CFG_SEED,
        "trajectory_classification": trajectory_classification,
        "trajectory_classification_reason": (
            "train_omol.py calls L.seed_everything(cfg.seed=1) before instantiating the "
            "datamodule and then the model (datamodule first); this pilot's instantiate_model "
            "calls set_seed(seed) immediately before instantiating only the model, with no "
            "datamodule in between. cfg.seed matches (1) but RNG-stream consumption order is "
            "not proven identical, so K0 is not claimed bitwise-identical to the training run's "
            "initialization."
            if time_tag == "K0" else
            "This kernel is computed from the exact strict-loaded training checkpoint state "
            "dict at the requested step; it is the literal trained parameter state, not a "
            "reconstruction."
        ),
        "checkpoint_provenance": checkpoint_provenance,
        "m": m,
        "probe_hash": probe_hash,
        "train_rmsd": float(cfg.force_field_module.train_rmsd),
        "operator_normalization": "K_op = K_raw / m",
        "target_power_normalization": "q_j = (u_j^T target)^2 / m",
        "row_order": "atom-major interleaved xyz",
        "parameter_set": "all trainable parameters of model.net; zero/unused columns retained",
        "gram_accumulation_dtype": "float64",
        "jacobian_storage_dtype": "float32",
        "gram_chunk_size_parameters": chunk_size,
        "temp_run_path": str(tmp_dir),
        "jacobian_seconds": jacobian_seconds,
        "total_seconds": time.time() - t0,
        "symmetry_max_abs_error": symmetry_abs,
        "symmetry_relative_max_error": symmetry_rel,
        "psd_tolerance": psd_tol,
        "psd_pass": psd_pass,
        "spectral": desc,
        "residual_spectral": desc_res,
        "parameter_coverage": coverage,
        "residual_norm_sq": float(np.dot(residual, residual)),
        "y_norm_sq": float(np.dot(y, y)),
        "f_pred_sha256": f_pred_sha256,
        "f_pred_shape": list(f_pred_reshaped.shape),
    }
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
        local_spectral_log_slopes=local_slopes.astype(np.float64),
        config_indices=config_indices.astype(np.int64),
        atom_indices=atom_indices.astype(np.int64),
        m=np.int64(m),
        N=np.int64(spec.expected_n),
        width=np.int64(spec.width),
        init_seed=np.int64(init_seed),
        time_tag=np.array(time_tag),
        probe_hash=np.array(probe_hash),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    metadata["kernel_artifact_sha256"] = sha256_file(artifact_path)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {artifact_path}", flush=True)
    print(f"wrote {meta_path}", flush=True)

    g_store.flush()
    del g_store
    gc.collect()
    grad_path.unlink()
    tmp_dir.rmdir()
    return metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["kernel"])
    ap.add_argument("--tag", required=True, choices=sorted(SPECS))
    ap.add_argument("--time", required=True, choices=TIMES)
    ap.add_argument("--tmp-root", type=Path,
                    default=Path("<CHECKPOINT_ROOT>/"
                                 "four_arch_trained_force_kernel_pilot_tmp_20260815"))
    ap.add_argument("--gram-chunk-size", type=int, default=250_000)
    args = ap.parse_args()
    spec = SPECS[args.tag]
    execute_kernel(spec, args.time, args.tmp_root, args.gram_chunk_size)


if __name__ == "__main__":
    main()
