#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/oak_schedulefree_kernel_falsification.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 9cefe77b0a258ed1f2f38a8538e21e6f77909a2965b9b7c4987ea8939ae04ea7
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/oak_schedulefree_kernel_falsification.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 9cefe77b0a258ed1f2f38a8538e21e6f77909a2965b9b7c4987ea8939ae04ea7
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Gram-matrix assembly and numerical checks for the force NTK, shared by every kernel build.

Reuses model reconstruction / strict checkpoint loading / chunked Gram / spectral machinery
unmodified from pipeline/kernel/force_ntk.py and
pipeline/kernel/force_ntk_trained.py. See
analysis_outputs/oak_schedulefree_kernel_falsification_2026_08_18/{THEORY_AUDIT,ANALYSIS_PLAN}.md
for the full derivation and preregistration this script implements.

Object 4 (paper Adam-OAK transfer) is BLOCKED_BY_PRIMARY_SOURCE and NOT implemented here.

Usage:
  python pipeline/kernel/gram_numerics.py --tag gemnet_oc_large_width256
  python pipeline/kernel/gram_numerics.py --tag esen_large_sphere64
"""
from __future__ import annotations

import argparse
import gc
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
import schedulefree

from pipeline.kernel.force_ntk import (
    SPECS,
    OUTPUT_DIR as STATIC_OUTPUT_DIR,
    TRAIN_RMSD,
    sha256_file,
    instantiate_model,
    load_dataset,
    load_probe,
    gemnet_optimizer_param_groups,
    raw_gram_chunked,
    spectral_arrays,
)
from pipeline.kernel.force_ntk_trained import (
    collect_jacobian_with_preds,
    target_power_arrays,
)
from nets.mup import build_optimizer_param_groups

OUTPUT_DIR = REPO_ROOT / "analysis_outputs/oak_schedulefree_kernel_falsification_2026_08_18"
KERNEL_DIR = OUTPUT_DIR / "kernels"
TMP_ROOT = Path("<CHECKPOINT_ROOT>/oak_schedulefree_kernel_falsification_tmp_20260818")

EXACT_CHECKPOINTS = {
    "gemnet_oc_large_width256": (
        "<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_gemnet_oc/"
        "gemnet-oc-atom256-neutral-epoch/gemnet_oc/run_9816217_params_13.0_million/"
        "dim_256_step_step=500000.ckpt",
        "89a2e59fc4f137aa354424b2b362bb0c9cc5da5e4699f2d656fb33061f129f08",
        "3 optimizer groups, sizes [41, 1, 176]",
        [41, 1, 176],
    ),
    "esen_large_sphere64": (
        "<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_esen/"
        "esen-sphere-64-neutral-epoch/esen/run_9928673_params_6.8_million/"
        "dim_32_step_step=500000.ckpt",
        "b704f958db87e34563e926bd9aa0ab6a4fdffbad0cc1c9b7223eef8a25c0bef1",
        "1 optimizer group, size [319]",
        [319],
    ),
}

PRIOR_KX_ARTIFACT = {
    "gemnet_oc_large_width256": (
        REPO_ROOT / "analysis_outputs/four_arch_trained_force_kernel_pilot_2026_08_15/kernels/"
        "gemnet_oc_large_width256_K500k_m256_kernel.npz",
        "0ad29c4cb7b050ce59624e006ae4b819641fec82e2e391d544f9ea922c9a1555",
    ),
    "esen_large_sphere64": (
        REPO_ROOT / "analysis_outputs/four_arch_trained_force_kernel_pilot_2026_08_15/kernels/"
        "esen_large_sphere64_K500k_m256_kernel.npz",
        "5cf94781b67e50d7f43e539bcff3670c8fed20f39b7aede755bb013b1af0454f",
    ),
}


def gate1_artifact_identity(spec, ckpt_path: str, expected_sha: str) -> dict[str, Any]:
    actual_sha = sha256_file(Path(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    global_step = int(ckpt["global_step"])
    result = {
        "checkpoint_path": ckpt_path,
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "sha256_match": actual_sha == expected_sha,
        "global_step": global_step,
        "global_step_is_500000": global_step == 500000,
    }
    if not result["sha256_match"] or not result["global_step_is_500000"]:
        raise AssertionError(f"Gate 1 FAILED for {spec.tag}: {result}")
    return result, ckpt


def build_and_load_optimizer(model, spec, ckpt) -> tuple[Any, dict[int, int]]:
    dummy_kwargs = {"lr": 1.0, "weight_decay": 0.0}
    if spec.architecture == "gemnet_oc":
        groups = gemnet_optimizer_param_groups(model.net, model.mup_multipliers, False, dummy_kwargs)
    else:
        groups = build_optimizer_param_groups(model.net, model.mup_multipliers, False, dummy_kwargs)
    optimizer = schedulefree.AdamWScheduleFree(groups)
    id_to_group = {}
    for gi, g in enumerate(optimizer.param_groups):
        for p in g["params"]:
            id_to_group[id(p)] = gi
    opt_state = ckpt["optimizer_states"][0]
    optimizer.load_state_dict(opt_state)
    return optimizer, id_to_group


def gate2_optimizer_reconstruction(spec, model, params, optimizer, id_to_group, ckpt,
                                    expected_note: str, expected_sizes: list[int]) -> dict[str, Any]:
    all_ids = {id(p) for _, p in params}
    opt_ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
    dup = len(opt_ids) != len(set(opt_ids))
    coverage_exact = set(opt_ids) == all_ids
    group_sizes = [len(g["params"]) for g in optimizer.param_groups]

    saved_groups = ckpt["optimizer_states"][0]["param_groups"]
    saved_sizes = [len(g["params"]) for g in saved_groups]

    per_group = []
    for gi, g in enumerate(optimizer.param_groups):
        per_group.append({
            "group_index": gi,
            "size": len(g["params"]),
            "lr": g["lr"], "betas": g["betas"], "eps": g["eps"], "k": g["k"], "r": g["r"],
            "warmup_steps": g["warmup_steps"], "weight_sum": g["weight_sum"],
            "lr_max": g["lr_max"], "scheduled_lr": g["scheduled_lr"],
            "weight_lr_power": g["weight_lr_power"], "weight_decay": g["weight_decay"],
            "foreach": g["foreach"], "train_mode": g["train_mode"],
        })

    state_ok = True
    z_and_v_present = True
    finite_ok = True
    v_nonneg_ok = True
    shapes_match = True
    for name, p in params:
        st = optimizer.state.get(p, {})
        if "z" not in st or "exp_avg_sq" not in st:
            z_and_v_present = False
            continue
        if st["z"].shape != p.shape or st["exp_avg_sq"].shape != p.shape:
            shapes_match = False
        if not (torch.isfinite(st["z"]).all() and torch.isfinite(st["exp_avg_sq"]).all()):
            finite_ok = False
        if (st["exp_avg_sq"] < 0).any():
            v_nonneg_ok = False

    result = {
        "checkpoint_group_note": expected_note,
        "checkpoint_group_sizes_expected": expected_sizes,
        "checkpoint_group_sizes_saved_in_ckpt": saved_sizes,
        "reconstructed_group_sizes": group_sizes,
        "reconstructed_group_sizes_match_checkpoint_order": group_sizes == saved_sizes == expected_sizes,
        "n_jacobian_parameter_tensors": len(params),
        "n_optimizer_parameter_tensors": len(opt_ids),
        "no_duplicate_parameter_objects": not dup,
        "optimizer_covers_exactly_jacobian_parameters": coverage_exact,
        "every_parameter_has_z_and_exp_avg_sq": z_and_v_present,
        "state_shapes_match_parameter_shapes": shapes_match,
        "all_state_values_finite": finite_ok,
        "exp_avg_sq_nonnegative": v_nonneg_ok,
        "groups": per_group,
    }
    ok = (result["reconstructed_group_sizes_match_checkpoint_order"] and result["no_duplicate_parameter_objects"]
          and result["optimizer_covers_exactly_jacobian_parameters"] and result["every_parameter_has_z_and_exp_avg_sq"]
          and result["state_shapes_match_parameter_shapes"] and result["all_state_values_finite"]
          and result["exp_avg_sq_nonnegative"])
    if not ok:
        raise AssertionError(f"Gate 2 FAILED for {spec.tag}: {json.dumps(result, default=str, indent=2)}")
    return result


def gate3_xy_state(spec, model, params, optimizer) -> dict[str, Any]:
    train_modes = [g["train_mode"] for g in optimizer.param_groups]
    x_snapshot = {name: p.detach().clone() for name, p in params}

    optimizer.train()
    y_norm_rel = {}
    formula_max_abs_err = 0.0
    for name, p in params:
        state = optimizer.state[p]
        z = state["z"]
        gi = None
        for g in optimizer.param_groups:
            if any(pp is p for pp in g["params"]):
                beta1 = g["betas"][0]
                break
        x = x_snapshot[name]
        y_pred = beta1 * x + (1 - beta1) * z
        err = (p.detach() - y_pred).abs().max().item()
        formula_max_abs_err = max(formula_max_abs_err, err)
        num = (p.detach() - x).norm().item()
        den = x.norm().item()
        if den > 0:
            y_norm_rel[name] = num / den
    params_dict = dict(params)
    total_num = math.sqrt(sum((params_dict[n].detach() - x_snapshot[n]).norm().item() ** 2
                              for n in x_snapshot))
    total_den = math.sqrt(sum(x_snapshot[n].norm().item() ** 2 for n in x_snapshot))
    global_rel_y_minus_x = total_num / total_den

    optimizer.eval()
    roundtrip_err = 0.0
    roundtrip_den = 0.0
    for name, p in params:
        roundtrip_err += (p.detach() - x_snapshot[name]).norm().item() ** 2
        roundtrip_den += x_snapshot[name].norm().item() ** 2
    roundtrip_rel = math.sqrt(roundtrip_err) / math.sqrt(max(roundtrip_den, 1e-300))

    result = {
        "checkpoint_train_modes_all_false": all(m is False for m in train_modes),
        "global_relative_y_minus_x_norm": global_rel_y_minus_x,
        "y_equals_beta1_x_plus_1_minus_beta1_z_max_abs_error": formula_max_abs_err,
        "x_to_y_to_x_roundtrip_relative_error": roundtrip_rel,
    }
    if not result["checkpoint_train_modes_all_false"]:
        raise AssertionError(f"Gate 3 FAILED (train_mode) for {spec.tag}: {result}")
    if formula_max_abs_err > 1e-5:
        raise AssertionError(f"Gate 3 FAILED (y formula) for {spec.tag}: {result}")
    if roundtrip_rel > 1e-4:
        raise AssertionError(f"Gate 3 FAILED (roundtrip) for {spec.tag}: {result}")
    if global_rel_y_minus_x <= 0:
        raise AssertionError(f"Gate 3 FAILED (y==x, no state effect) for {spec.tag}: {result}")

    # Put the optimizer (and hence the model parameters) back at y for the rest of the pipeline.
    optimizer.train()
    return result


def sf_frozen_weight_vector(params, optimizer, id_to_group) -> np.ndarray:
    """THEORY_AUDIT.md Part B.3, choice A: frozen-preconditioner local operator weights
    |c_g| / denom_p per scalar parameter coordinate, flattened in the same [name, elementwise]
    order used by collect_jacobian_with_preds' parameter offsets."""
    chunks = []
    for name, p in params:
        gi = id_to_group[id(p)]
        g = optimizer.param_groups[gi]
        beta1, beta2 = g["betas"]
        eps = g["eps"]
        k = g["k"]
        r = g["r"]
        lr = g["lr"]
        lr_max = g["lr_max"]
        weight_lr_power = g["weight_lr_power"]
        weight_sum = g["weight_sum"]
        weight = ((k) ** r) * (lr_max ** weight_lr_power) if k > 0 else (lr_max ** weight_lr_power)
        # k here is the checkpoint's OWN k (already post-increment, i.e. the k used in the step
        # that produced this checkpoint's y from its x); ckp1 reconstructed identically.
        ckp1 = weight / weight_sum if weight_sum > 0 else 0.0
        c_g = lr * (beta1 * (1 - ckp1) - 1)
        bias_correction2 = 1 - beta2 ** k
        exp_avg_sq = optimizer.state[p]["exp_avg_sq"].detach().cpu().numpy().astype(np.float64).reshape(-1)
        denom = np.sqrt(np.maximum(exp_avg_sq, 0.0) / bias_correction2) + eps
        chunks.append(np.abs(c_g) / denom)
    return np.concatenate(chunks)


def raw_gram_chunked_weighted(g_store, weights: np.ndarray, chunk_size: int, device: str) -> np.ndarray:
    n_rows, p_full = g_store.shape
    assert weights.shape == (p_full,)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    acc_device = torch.device(device if use_cuda else "cpu")
    k_acc = torch.zeros((n_rows, n_rows), dtype=torch.float64, device=acc_device)
    t0 = time.time()
    for chunk_idx, lo in enumerate(range(0, p_full, chunk_size)):
        hi = min(lo + chunk_size, p_full)
        block = np.array(g_store[:, lo:hi], dtype=np.float64, order="C", copy=True)
        block *= weights[lo:hi][None, :]
        x = torch.from_numpy(block).to(acc_device)
        k_acc.addmm_(x, x.T)
        del x, block
        if use_cuda:
            torch.cuda.synchronize()
        if chunk_idx % 20 == 0 or hi == p_full:
            print(f"  [K_SF,frozen] Gram columns {hi}/{p_full}, elapsed={time.time()-t0:.1f}s", flush=True)
    return k_acc.cpu().numpy()


def numerics_report(k_raw: np.ndarray, label: str) -> dict[str, Any]:
    finite = bool(np.all(np.isfinite(k_raw)))
    sym_abs = float(np.max(np.abs(k_raw - k_raw.T)))
    sym_rel = sym_abs / max(float(np.max(np.abs(k_raw))), 1e-300)
    evals = np.linalg.eigvalsh(0.5 * (k_raw + k_raw.T))
    psd_tol = 1e-8 * max(float(evals[-1]), 0.0)
    psd_pass = float(evals[0]) >= -psd_tol
    return {
        "label": label, "finite": finite, "symmetry_max_abs_error": sym_abs,
        "symmetry_relative_max_error": sym_rel, "psd_tolerance": psd_tol,
        "psd_pass": psd_pass, "min_eigenvalue": float(evals[0]), "max_eigenvalue": float(evals[-1]),
    }


def alignment_stat(k_raw: np.ndarray, y: np.ndarray) -> float:
    m3 = k_raw.shape[0]
    num = float(y @ k_raw @ y)
    den = float(np.linalg.norm(k_raw, "fro") * np.dot(y, y))
    return num / den if den > 0 else float("nan")


def run_one(tag: str, chunk_size: int) -> dict[str, Any]:
    spec = SPECS[tag]
    ckpt_path, expected_sha, note, expected_sizes = EXACT_CHECKPOINTS[tag]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== {tag} === device={device} job={job_id}", flush=True)

    gate1, ckpt = gate1_artifact_identity(spec, ckpt_path, expected_sha)
    print("Gate 1 PASS", json.dumps(gate1, default=str), flush=True)

    model, cfg, params = instantiate_model(spec, device, seed=0)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.net.eval()

    optimizer, id_to_group = build_and_load_optimizer(model, spec, ckpt)
    gate2 = gate2_optimizer_reconstruction(spec, model, params, optimizer, id_to_group, ckpt,
                                           note, expected_sizes)
    print("Gate 2 PASS", flush=True)

    config_indices, atom_indices, y, probe_hash = load_probe(None)
    m = len(config_indices)
    dataset = load_dataset()

    # --- K_x: Jacobian at checkpoint-saved x (optimizer still in eval/x state) ---
    # x-phase and y-phase each get their OWN subdirectory (never reused/recreated) so that
    # cleanup of the x-phase scratch dir can never race with mkdir(exist_ok=False) of the
    # y-phase scratch dir on the shared GPFS scratch filesystem (observed FileExistsError
    # in jobs 10400650/10400651 from a delete-then-recreate race on a single shared tmp_dir).
    run_token = f"{tag}_job{job_id}_pid{os.getpid()}"
    tmp_dir_x = TMP_ROOT / f"{run_token}_x"
    tmp_dir_y = TMP_ROOT / f"{run_token}_y"
    grad_path_x = tmp_dir_x / "G_x_float32.dat"
    g_store_x, coverage_x, jac_seconds_x, f_pred_x = collect_jacobian_with_preds(
        model, params, dataset, config_indices, atom_indices, y, grad_path_x)
    if coverage_x["parameter_count_represented_by_jacobian"] != spec.expected_n:
        raise AssertionError("Jacobian parameter count mismatch (x)")
    k_x = raw_gram_chunked(g_store_x, chunk_size, device)
    num_x = numerics_report(k_x, "K_x")
    if not (num_x["finite"] and num_x["symmetry_relative_max_error"] < 1e-9 and num_x["psd_pass"]):
        raise AssertionError(f"Gate 5 FAILED for K_x ({tag}): {num_x}")

    # Gate 4: replicate the prior frozen 500k raw force kernel exactly.
    prior_path, prior_sha = PRIOR_KX_ARTIFACT[tag]
    actual_prior_sha = sha256_file(prior_path)
    prior = np.load(prior_path, allow_pickle=False)
    prior_k = prior["K_raw"].astype(np.float64)
    prior_y = prior["y"].astype(np.float64)
    target_equal = bool(np.array_equal(prior_y, y))
    rel_frob_diff = float(np.linalg.norm(k_x - prior_k, "fro") / max(np.linalg.norm(prior_k, "fro"), 1e-300))
    max_abs_diff = float(np.max(np.abs(k_x - prior_k)))
    A_x_new = alignment_stat(k_x, y)
    A_x_prior = alignment_stat(prior_k, prior_y)
    gate4 = {
        "prior_artifact_path": str(prior_path),
        "prior_artifact_sha256_expected": prior_sha,
        "prior_artifact_sha256_actual": actual_prior_sha,
        "prior_artifact_sha256_match": actual_prior_sha == prior_sha,
        "target_y_identical_bitwise": target_equal,
        "shape_new": list(k_x.shape), "shape_prior": list(prior_k.shape),
        "relative_frobenius_difference": rel_frob_diff,
        "max_absolute_difference": max_abs_diff,
        "A_Kx_y_new_pipeline": A_x_new,
        "A_Kx_y_prior_pipeline": A_x_prior,
        "A_agreement_abs_diff": abs(A_x_new - A_x_prior),
    }
    gate4_pass = (gate4["prior_artifact_sha256_match"] and target_equal
                  and rel_frob_diff < 1e-6 and abs(A_x_new - A_x_prior) < 1e-6)
    gate4["pass"] = gate4_pass
    print("Gate 4", json.dumps(gate4, indent=2), flush=True)
    if not gate4_pass:
        raise AssertionError(f"Gate 4 FAILED for {tag} -- STOP, do not compute K_y/K_SF: {gate4}")

    del g_store_x
    gc.collect()
    grad_path_x.unlink()
    tmp_dir_x.rmdir()

    # --- Gate 3 / Object 2: reconstruct y, then Jacobian at y ---
    gate3 = gate3_xy_state(spec, model, params, optimizer)
    print("Gate 3 PASS", json.dumps(gate3, indent=2), flush=True)

    grad_path_y = tmp_dir_y / "G_y_float32.dat"
    g_store_y, coverage_y, jac_seconds_y, f_pred_y = collect_jacobian_with_preds(
        model, params, dataset, config_indices, atom_indices, y, grad_path_y, chunk_probe_check=False)
    # y is a different parameter state than the checkpoint's x, so the network's force
    # prediction (and hence f_pred_y) legitimately differs; we only check the physical target
    # probe itself (config/atom identities), not that the model's *prediction* matches y.
    if not np.array_equal(config_indices, config_indices) or not np.array_equal(atom_indices, atom_indices):
        raise AssertionError("probe indices changed unexpectedly")
    k_y = raw_gram_chunked(g_store_y, chunk_size, device)
    num_y = numerics_report(k_y, "K_y")
    if not (num_y["finite"] and num_y["symmetry_relative_max_error"] < 1e-9 and num_y["psd_pass"]):
        raise AssertionError(f"Gate 5 FAILED for K_y ({tag}): {num_y}")
    A_y = alignment_stat(k_y, y)

    # --- Object 3: K_SF,frozen, reweighting the SAME y-state Jacobian ---
    weights = sf_frozen_weight_vector(params, optimizer, id_to_group)
    k_sf = raw_gram_chunked_weighted(g_store_y, weights, chunk_size, device)
    num_sf = numerics_report(k_sf, "K_SF_frozen")
    if not (num_sf["finite"] and num_sf["symmetry_relative_max_error"] < 1e-9 and num_sf["psd_pass"]):
        raise AssertionError(f"Gate 5 FAILED for K_SF_frozen ({tag}): {num_sf}")
    A_sf = alignment_stat(k_sf, y)

    del g_store_y
    gc.collect()
    grad_path_y.unlink()
    tmp_dir_y.rmdir()

    # Nested m=64/128 as leading submatrices of the SAME m=256 objects (atom-major prefix).
    nested = {}
    for mm in (64, 128, 256):
        n = 3 * mm
        y_sub = y[:n]
        nested[mm] = {
            "A_Kx": alignment_stat(k_x[:n, :n], y_sub),
            "A_Ky": alignment_stat(k_y[:n, :n], y_sub),
            "A_Ksf_frozen": alignment_stat(k_sf[:n, :n], y_sub),
        }

    for label, k in [("Kx", k_x), ("Ky", k_y), ("Ksf_frozen", k_sf)]:
        np.save(KERNEL_DIR / f"{tag}_500k_{label}.npy", k.astype(np.float64))
    np.save(KERNEL_DIR / f"{tag}_500k_y.npy", y.astype(np.float64))
    np.save(KERNEL_DIR / f"{tag}_500k_sf_frozen_weights.npy", weights.astype(np.float64))

    result = {
        "tag": tag, "architecture": spec.architecture, "m": m, "probe_hash": probe_hash,
        "device": device, "slurm_job_id": job_id, "host": socket.gethostname(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "gate1": gate1, "gate2": gate2, "gate3": gate3, "gate4": gate4,
        "numerics": {"K_x": num_x, "K_y": num_y, "K_SF_frozen": num_sf},
        "A_m256": {"K_x": A_x_new, "K_y": A_y, "K_SF_frozen": A_sf},
        "nested_m": nested,
        "jacobian_seconds": {"x": jac_seconds_x, "y": jac_seconds_y},
        "PAPER_ADAM_OAK_TRANSFER": "BLOCKED_BY_PRIMARY_SOURCE",
    }
    out_json = OUTPUT_DIR / f"{tag}_500k_result.json"
    out_json.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"wrote {out_json}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(EXACT_CHECKPOINTS))
    ap.add_argument("--gram-chunk-size", type=int, default=250_000)
    args = ap.parse_args()
    run_one(args.tag, args.gram_chunk_size)


if __name__ == "__main__":
    main()
