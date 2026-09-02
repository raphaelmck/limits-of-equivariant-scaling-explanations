#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/four_arch_initial_force_kernel_pilot.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 da79a628e4d6cc8f6aa51d4aeda553f498fb5722621ed13bb9af379ef534802a
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/four_arch_initial_force_kernel_pilot.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 da79a628e4d6cc8f6aa51d4aeda553f498fb5722621ed13bb9af379ef534802a
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Frozen-probe, full block force-NTK production for the four-architecture pilot.

This script intentionally has three gated modes:
  freeze-probe  -- writes the common configuration-balanced marked-atom probe first;
  audit         -- reconstructs all eight models and checks checkpoint/count/provenance;
  validate      -- compares chunked and unchunked raw Gram formation on m=8;
  kernel        -- computes one seed-0, m=256 full-network 3m x 3m force kernel.

The kernel is raw G G^T in atom-major coordinate order.  Operator eigenvalues are
raw eigenvalues divided by the number of marked atoms m (never by 3m).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import numpy as np
import torch
from fairchem.core.datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
from omegaconf import OmegaConf

from nets.mup import MuReadout, SO3_MuReadout, build_optimizer_param_groups


OUTPUT_DIR = REPO_ROOT / "analysis_outputs/four_arch_initial_force_kernel_pilot_2026_08_14"
CONFIG_DIR = OUTPUT_DIR / "inputs/configs"
VAL_PATH = Path("<DATA_ROOT>/open_mol/neutral_val")
TRAIN_RMSD = 6.383
M = 256
PROBE_SEED = 20_260_814
INIT_SEED = 0
RCOND = 1e-10


@dataclass(frozen=True)
class Spec:
    tag: str
    architecture: str
    architecture_label: str
    width_label: str
    width: int
    size: str
    config_name: str
    expected_n: int
    checkpoint_glob: str


SPECS = {
    s.tag: s
    for s in [
        Spec("mpnn_medium_hc1024", "mpnn", "MPNN", "hc", 1024, "medium",
             "mpnn_medium_hc1024.yaml", 12_793_348,
             "<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_corrected_stats/"
             "dim-1024-neutral-epoch-corrected-stats/mpnn/*/dim_1024_step_step=500000.ckpt"),
        Spec("mpnn_large_hc2048", "mpnn", "MPNN", "hc", 2048, "large",
             "mpnn_large_hc2048.yaml", 50_752_516,
             "<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_corrected_stats/"
             "dim-2048-neutral-epoch-corrected-stats/mpnn/*/dim_2048_step_step=500000.ckpt"),
        Spec("mcegnn_medium_hc320", "mcegnn", "MC-EGNN", "hc", 320, "medium",
             "mcegnn_medium_hc320_legacy.yaml", 10_372_498,
             "<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_mc_egnn/"
             "mc-egnn-hc-320-neutral-epoch/mc_egnn/*/dim_320_step_step=500000.ckpt"),
        Spec("mcegnn_large_hc512", "mcegnn", "MC-EGNN", "hc", 512, "large",
             "mcegnn_large_hc512_legacy.yaml", 26_080_791,
             "<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_mc_egnn/"
             "mc-egnn-hc-512-neutral-epoch/mc_egnn/*/dim_512_step_step=500000.ckpt"),
        Spec("gemnet_oc_medium_width128", "gemnet_oc", "GemNet-OC", "width", 128, "medium",
             "gemnet_oc_medium_width128.yaml", 3_377_792,
             "<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_gemnet_oc/"
             "gemnet-oc-atom128-neutral-epoch/gemnet_oc/*/dim_128_step_step=500000.ckpt"),
        Spec("gemnet_oc_large_width256", "gemnet_oc", "GemNet-OC", "width", 256, "large",
             "gemnet_oc_large_width256.yaml", 13_026_560,
             "<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_gemnet_oc/"
             "gemnet-oc-atom256-neutral-epoch/gemnet_oc/*/dim_256_step_step=500000.ckpt"),
        Spec("esen_medium_sphere32", "esen", "eSEN", "sphere_channels", 32, "medium",
             "esen_medium_sphere32.yaml", 3_409_906,
             "<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_esen/"
             "esen-sphere-32-neutral-epoch/esen/*/dim_32_step_step=500000.ckpt"),
        Spec("esen_large_sphere64", "esen", "eSEN", "sphere_channels", 64, "large",
             "esen_large_sphere64.yaml", 6_785_202,
             "<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_esen/"
             "esen-sphere-64-neutral-epoch/esen/*/dim_32_step_step=500000.ckpt"),
    ]
}

ORIGINAL_CONFIG_SOURCES = {
    "mpnn_medium_hc1024": "<PROJECT_ROOT>/upstream-training/outputs/2026-05-26/14-18-16/.hydra/config.yaml",
    "mpnn_large_hc2048": "<PROJECT_ROOT>/upstream-training/outputs/2026-05-26/20-43-03/.hydra/config.yaml",
    "mcegnn_medium_hc320": "<PROJECT_ROOT>/upstream-training/outputs/2026-06-05/22-20-01/.hydra/config.yaml",
    "mcegnn_large_hc512": "<PROJECT_ROOT>/upstream-training/outputs/2026-06-06/02-36-22/.hydra/config.yaml",
    "gemnet_oc_medium_width128": "<PROJECT_ROOT>/upstream-training/outputs/2026-06-12/15-48-33/.hydra/config.yaml",
    "gemnet_oc_large_width256": "<PROJECT_ROOT>/upstream-training/outputs/2026-06-12/16-15-52/.hydra/config.yaml",
    "esen_medium_sphere32": "<PROJECT_ROOT>/upstream-training/outputs/2026-06-04/14-49-41/.hydra/config.yaml",
    "esen_large_sphere64": "<PROJECT_ROOT>/upstream-training/outputs/2026-06-26/14-46-11/.hydra/config.yaml",
}


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_probe_hash(config_indices: np.ndarray, atom_indices: np.ndarray,
                         y: np.ndarray) -> str:
    h = hashlib.sha256()
    for name, arr, dtype in [
        ("config_indices", config_indices, "<i8"),
        ("atom_indices", atom_indices, "<i8"),
        ("y", y, "<f8"),
    ]:
        a = np.ascontiguousarray(arr, dtype=np.dtype(dtype))
        h.update(name.encode() + b"\0")
        h.update(np.asarray(a.shape, dtype="<i8").tobytes())
        h.update(a.tobytes())
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# GemNet-OC was trained after (and only became constructible after) the June-10 MuP
# name-handling fix documented in MUP_BLAST_RADIUS.md.  Keep that fix local to GemNet;
# MC-EGNN continues to use the pinned commit's legacy implementation and has no
# _mup_ref_param attribute.  These functions reproduce the non-collapsing name behavior of
# the June-10 helper.  The later _mup_ref_param capability is irrelevant to GemNet (no module
# has that attribute) and is deliberately not implemented here.
def gemnet_get_mup_multipliers(base_model, main_model):
    base_shapes = {name: p.shape for name, p in base_model.named_parameters()}
    model_shapes = {name: p.shape for name, p in main_model.named_parameters()}
    if set(base_shapes) != set(model_shapes):
        raise AssertionError("GemNet base/main parameter names differ")
    multipliers = {}
    for name, base_dims in base_shapes.items():
        dims = model_shapes[name]
        num_inf_dims = 0
        multiplier = 1.0
        for base_dim, dim in zip(base_dims, dims):
            if int(base_dim) != int(dim):
                num_inf_dims += 1
                multiplier = float(dim) / float(base_dim)
        multipliers[name] = (multiplier, num_inf_dims > 1)
    return multipliers


def gemnet_mup_init(model, multipliers):
    for name, module in model.named_modules():
        if isinstance(module, MuReadout):
            module.width_mult = multipliers[f"{name}.weight"][0]
            module._rescale_parameters()
        if isinstance(module, SO3_MuReadout):
            module.width_mult = multipliers[f"{name}.weight"][0]
            module._rescale_parameters()
    for name, param in model.named_parameters():
        if "bias" in name and name in multipliers:
            param.data *= multipliers[name][0] ** 0.5


def gemnet_optimizer_param_groups(model, multipliers, decoupled_wd, optimizer_kwargs):
    def new_group():
        group = dict(optimizer_kwargs)
        group["params"] = []
        return group
    groups = defaultdict(new_group)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "_fsdp_wrapped_module." in name:
            name = name.split("_fsdp_wrapped_module.")[-1]
        elif name.startswith("module."):
            name = name.split("module.")[-1]
        multiplier, is_matrix_like = multipliers[name]
        groups[multiplier if is_matrix_like else 1.0]["params"].append(param)
    for width_mult, group in groups.items():
        group["lr"] /= width_mult
        if not decoupled_wd:
            group["weight_decay"] *= width_mult
    return list(groups.values())


def load_dataset() -> AseDBDataset:
    return AseDBDataset({"src": str(VAL_PATH),
                         "a2g_args": dict(r_energy=True, r_forces=True)})


def freeze_probe() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_npz = OUTPUT_DIR / "probe_indices.npz"
    out_json = OUTPUT_DIR / "probe_manifest.json"
    if out_npz.exists() or out_json.exists():
        raise RuntimeError("probe outputs already exist; refusing to overwrite the frozen probe")

    dataset = load_dataset()
    if len(dataset) < M:
        raise RuntimeError(f"validation set has only {len(dataset)} configurations")
    config_rng = np.random.RandomState(PROBE_SEED)
    config_indices = config_rng.permutation(len(dataset))[:M].astype(np.int64)
    atom_rng = np.random.RandomState((PROBE_SEED * 1_000_003 + 7) % (2**31 - 1))
    atom_indices = np.empty(M, dtype=np.int64)
    natoms = np.empty(M, dtype=np.int64)
    atomic_numbers = np.empty(M, dtype=np.int64)
    raw_forces = np.empty((M, 3), dtype=np.float64)
    for row, g in enumerate(config_indices):
        item = dataset[int(g)]
        n = int(len(item.atomic_numbers))
        i = int(atom_rng.randint(0, n))
        atom_indices[row] = i
        natoms[row] = n
        atomic_numbers[row] = int(item.atomic_numbers[i])
        raw_forces[row] = np.asarray(item.forces[i].detach().cpu(), dtype=np.float64)
    y = (raw_forces / TRAIN_RMSD).reshape(-1)
    probe_hash = canonical_probe_hash(config_indices, atom_indices, y)
    np.savez(
        out_npz,
        config_indices=config_indices,
        atom_indices=atom_indices,
        natoms=natoms,
        atomic_numbers=atomic_numbers,
        raw_forces=raw_forces,
        y=y,
        m=np.int64(M),
        probe_seed=np.int64(PROBE_SEED),
        train_rmsd=np.float64(TRAIN_RMSD),
        probe_hash=np.array(probe_hash),
    )
    data_files = sorted(VAL_PATH.glob("*.aselmdb"))
    identity_h = hashlib.sha256()
    total_bytes = 0
    for p in data_files:
        total_bytes += p.stat().st_size
        identity_h.update(p.name.encode() + b"\0")
        identity_h.update(str(p.stat().st_size).encode() + b"\0")
        identity_h.update(bytes.fromhex(sha256_file(p)))
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sampling_measure": "configuration uniform, then one atom uniform within configuration",
        "m": M,
        "probe_seed": PROBE_SEED,
        "configuration_rng": "numpy.random.RandomState(seed).permutation",
        "atom_rng_seed": int((PROBE_SEED * 1_000_003 + 7) % (2**31 - 1)),
        "distinct_configurations": int(len(np.unique(config_indices))),
        "dataset_path": str(VAL_PATH),
        "dataset_length": int(len(dataset)),
        "dataset_data_file_count": len(data_files),
        "dataset_data_total_bytes": total_bytes,
        "dataset_content_identity_sha256": identity_h.hexdigest(),
        "train_rmsd": TRAIN_RMSD,
        "target_definition": "marked physical force / train_rmsd, atom-major xyz order",
        "probe_hash": probe_hash,
        "probe_indices_npz_sha256": sha256_file(out_npz),
    }
    out_json.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def resolve_checkpoint(spec: Spec) -> Path:
    import glob
    matches = sorted(Path(x) for x in glob.glob(spec.checkpoint_glob))
    if len(matches) != 1:
        raise RuntimeError(f"{spec.tag}: expected exactly one checkpoint, got {matches}")
    return matches[0]


def instantiate_model(spec: Spec, device: str, seed: int = INIT_SEED):
    cfg_path = CONFIG_DIR / spec.config_name
    cfg = OmegaConf.load(cfg_path)
    rmsd = float(cfg.force_field_module.train_rmsd)
    if rmsd != TRAIN_RMSD:
        raise AssertionError(f"{spec.tag}: config train_rmsd={rmsd}, expected {TRAIN_RMSD}")
    set_seed(seed)
    if spec.architecture == "gemnet_oc":
        # GraphModel imported these helpers into its own module namespace, so patch exactly
        # that construction-time namespace and restore it immediately after instantiation.
        import src.model.omol_module as omol_module
        old_get, old_init = omol_module.get_mup_multipliers, omol_module.mup_init
        omol_module.get_mup_multipliers = gemnet_get_mup_multipliers
        omol_module.mup_init = gemnet_mup_init
        try:
            model = hydra.utils.instantiate(cfg.force_field_module)
        finally:
            omol_module.get_mup_multipliers, omol_module.mup_init = old_get, old_init
    else:
        model = hydra.utils.instantiate(cfg.force_field_module)
    model.free_scheduler = False
    model.net.eval()
    model = model.to(device)
    params = [(name, p) for name, p in model.net.named_parameters() if p.requires_grad]
    n = int(sum(p.numel() for _, p in params))
    if n != spec.expected_n or int(model.total_params) != spec.expected_n:
        raise AssertionError(
            f"{spec.tag}: trainable count={n}, model.total_params={model.total_params}, "
            f"expected={spec.expected_n}"
        )
    return model, cfg, params


def audit_one(spec: Spec, device: str = "cpu") -> dict[str, Any]:
    model, cfg, params = instantiate_model(spec, device)
    all_ids = {id(p) for _, p in params}
    group_builder = (gemnet_optimizer_param_groups if spec.architecture == "gemnet_oc"
                     else build_optimizer_param_groups)
    groups = group_builder(
        model.net, model.mup_multipliers,
        cfg.force_field_module.optimizer.name == "adamw",
        {"lr": float(cfg.force_field_module.optimizer.lr),
         "weight_decay": float(cfg.force_field_module.optimizer.weight_decay)},
    )
    opt_params = [p for g in groups for p in g["params"]]
    opt_ids = [id(p) for p in opt_params]
    if len(opt_ids) != len(set(opt_ids)) or set(opt_ids) != all_ids:
        raise AssertionError(f"{spec.tag}: optimizer parameter groups do not exactly cover model.net")

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
                f"{spec.tag}: force_decoder.width_mult={actual_mult}, "
                f"legacy expected={expected_mult}"
            )
        legacy = {
            "force_decoder_width_mult": actual_mult,
            "legacy_formula": "isqrt(hidden_channels)/isqrt(102)",
            "mup_ref_param_attributes": leaked,
        }

    ckpt_path = resolve_checkpoint(spec)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # Strict loading checks the complete GraphModel state, including net/base_net shapes/names.
    model.load_state_dict(state, strict=True)
    parameter_state_keys = {f"net.{name}" for name, _ in params}
    missing_parameter_keys = sorted(parameter_state_keys - set(state))
    if missing_parameter_keys:
        raise AssertionError(f"{spec.tag}: checkpoint lacks net parameters {missing_parameter_keys}")
    state_net_n = int(sum(state[k].numel() for k in parameter_state_keys))
    if state_net_n != spec.expected_n:
        raise AssertionError(f"{spec.tag}: checkpoint net state count={state_net_n}")
    original_config_path = Path(ORIGINAL_CONFIG_SOURCES[spec.tag])
    if sha256_file(original_config_path) != sha256_file(CONFIG_DIR / spec.config_name):
        raise AssertionError(f"{spec.tag}: isolated config snapshot differs from original artifact")
    result = {
        **asdict(spec),
        "config_path": str(CONFIG_DIR / spec.config_name),
        "config_sha256": sha256_file(CONFIG_DIR / spec.config_name),
        "original_config_path": str(original_config_path),
        "original_config_sha256": sha256_file(original_config_path),
        "train_rmsd": float(cfg.force_field_module.train_rmsd),
        "trainable_network_parameter_count": spec.expected_n,
        "checkpoint_net_state_parameter_count": state_net_n,
        "optimizer_parameter_count": int(sum(p.numel() for p in opt_params)),
        "optimizer_parameter_tensor_count": len(opt_params),
        "trainable_parameter_tensor_count": len(params),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_size_bytes": ckpt_path.stat().st_size,
        "checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_strict_load": True,
        "legacy_mcegnn_audit": legacy,
        "mup_construction_variant": (
            "gemnet_june10_exact_parameter_names" if spec.architecture == "gemnet_oc"
            else "pinned_commit_legacy"
        ),
    }
    del model, ckpt, state
    gc.collect()
    return result


def audit_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in SPECS.values():
        print(f"AUDIT {spec.tag}", flush=True)
        results.append(audit_one(spec))
    gemnet_files = sorted((REPO_ROOT / "nets/gemnet_oc").rglob("*.py"))
    gemnet_files += sorted((REPO_ROOT / "nets/gemnet").rglob("*.py"))
    out = {
        "status": "PASS",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                             text=True).strip(),
        "host": socket.gethostname(),
        "models": results,
        "gemnet_source_snapshot": {
            "provenance_limit": (
                "Vendored GemNet/GemNet-OC source was untracked at training time; the June-10 "
                "common helpers, June-12 GemNet-OC layers, and June-16 model.py snapshot are "
                "accepted only because both canonical checkpoints strict-load and reproduce the "
                "published exact N. The post-training model.py change exposes hidden_state and "
                "does not add parameters."
            ),
            "files": {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in gemnet_files},
            "training_mup_snapshot_path": str(
                OUTPUT_DIR / "inputs/gemnet_training_mup_snapshot.py"),
            "training_mup_snapshot_sha256": sha256_file(
                OUTPUT_DIR / "inputs/gemnet_training_mup_snapshot.py"),
            "mup_blast_radius_report_sha256": "f11bf8556daa35ae9b4bab53745b1615dae95ada03ca04508e9c1e8f22ac5377",
        },
    }
    path = OUTPUT_DIR / "model_reconstruction_audit.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


def load_probe(limit: int | None = None):
    probe_path = OUTPUT_DIR / "probe_indices.npz"
    manifest_path = OUTPUT_DIR / "probe_manifest.json"
    if not probe_path.exists() or not manifest_path.exists():
        raise RuntimeError("frozen probe is missing; run freeze-probe before any kernel")
    z = np.load(probe_path, allow_pickle=False)
    config_indices = z["config_indices"].astype(np.int64)
    atom_indices = z["atom_indices"].astype(np.int64)
    y = z["y"].astype(np.float64)
    probe_hash = canonical_probe_hash(config_indices, atom_indices, y)
    manifest = json.loads(manifest_path.read_text())
    if probe_hash != manifest["probe_hash"] or probe_hash != str(z["probe_hash"]):
        raise AssertionError("frozen probe hash mismatch")
    if limit is not None:
        config_indices = config_indices[:limit]
        atom_indices = atom_indices[:limit]
        y = y.reshape(-1, 3)[:limit].reshape(-1)
    return config_indices, atom_indices, y, probe_hash


def make_grad_store(n_rows: int, n_params: int, path: Path, in_memory: bool):
    if in_memory:
        return np.zeros((n_rows, n_params), dtype=np.float32), False
    path.parent.mkdir(parents=True, exist_ok=False)
    return np.memmap(path, dtype=np.float32, mode="w+", shape=(n_rows, n_params)), True


def collect_jacobian(model, params, dataset, config_indices, atom_indices, y_expected,
                     grad_path: Path, in_memory: bool):
    m = len(config_indices)
    names = [name for name, _ in params]
    numels = np.array([p.numel() for _, p in params], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(numels)])
    p_full = int(offsets[-1])
    g_store, is_memmap = make_grad_store(3 * m, p_full, grad_path, in_memory)
    ever_nonzero: set[str] = set()
    ever_has_grad: set[str] = set()
    observed_y = np.empty(3 * m, dtype=np.float64)
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
    if isinstance(g_store, np.memmap):
        g_store.flush()
    if not np.array_equal(observed_y, y_expected):
        max_err = float(np.max(np.abs(observed_y - y_expected)))
        if not np.allclose(observed_y, y_expected, rtol=0.0, atol=1e-12):
            raise AssertionError(f"target differs from frozen probe, max_abs={max_err}")
    coverage = {
        "all_trainable_parameter_names": names,
        "parameter_numels": {n: int(v) for n, v in zip(names, numels)},
        "parameter_count_represented_by_jacobian": p_full,
        "parameter_tensors_with_some_autograd_tensor": sorted(ever_has_grad),
        "parameter_tensors_with_some_nonzero_entry": sorted(ever_nonzero),
        "systematically_grad_none_parameter_tensors": sorted(set(names) - ever_has_grad),
        "systematically_zero_or_none_parameter_tensors": sorted(set(names) - ever_nonzero),
    }
    return g_store, coverage, time.time() - t0


def raw_gram_chunked(g_store, chunk_size: int, device: str) -> np.ndarray:
    n_rows, p_full = g_store.shape
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    acc_device = torch.device(device if use_cuda else "cpu")
    k_acc = torch.zeros((n_rows, n_rows), dtype=torch.float64, device=acc_device)
    t0 = time.time()
    for chunk_idx, lo in enumerate(range(0, p_full, chunk_size)):
        hi = min(lo + chunk_size, p_full)
        block = np.array(g_store[:, lo:hi], dtype=np.float64, order="C", copy=True)
        x = torch.from_numpy(block).to(acc_device)
        k_acc.addmm_(x, x.T)
        del x, block
        if use_cuda:
            torch.cuda.synchronize()
        if chunk_idx % 20 == 0 or hi == p_full:
            print(f"  Gram columns {hi}/{p_full}, elapsed={time.time()-t0:.1f}s", flush=True)
    return k_acc.cpu().numpy()


def raw_gram_unchunked(g_store, device: str) -> np.ndarray:
    block = np.array(g_store, dtype=np.float64, order="C", copy=True)
    acc_device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available()
                              else "cpu")
    x = torch.from_numpy(block).to(acc_device)
    k = x @ x.T
    return k.cpu().numpy()


def spectral_arrays(k_raw: np.ndarray, y: np.ndarray, m: int):
    k_sym = 0.5 * (k_raw + k_raw.T)
    evals, evecs = np.linalg.eigh(k_sym)
    order = np.argsort(evals)[::-1]
    raw = evals[order]
    u = evecs[:, order]
    op = raw / m
    q = (u.T @ y) ** 2 / m
    total_q = float(np.dot(y, y) / m)
    alignment = np.cumsum(q) / total_q if total_q > 0 else np.full_like(q, np.nan)
    tail = total_q - np.cumsum(q)
    tail[np.abs(tail) < 1e-14 * max(total_q, 1.0)] = 0.0
    threshold = max(float(raw[0]), 0.0) * RCOND
    rank = int(np.count_nonzero(raw > threshold))
    positive = np.clip(op[:rank], 0.0, None)
    if positive.sum() > 0:
        weights = positive / positive.sum()
        effective_rank = float(np.exp(-np.sum(weights * np.log(weights))))
    else:
        effective_rank = 0.0
    local_slopes = np.full_like(op, np.nan)
    if rank >= 3:
        ranks = np.arange(1, rank + 1, dtype=np.float64)
        local_slopes[:rank] = -np.gradient(np.log(positive), np.log(ranks))
    quantiles = {}
    for pct in [50, 80, 90, 95]:
        quantiles[f"k{pct}"] = int(np.searchsorted(alignment, pct / 100.0) + 1)
    desc = {
        "numerical_rank": rank,
        "rank_fraction": rank / len(raw),
        "rcond": RCOND,
        "effective_rank": effective_rank,
        "effective_rank_definition": "exp(-sum_j p_j log p_j), p_j=lambda_j/sum(lambda>rcond)",
        "min_raw_eigenvalue": float(raw[-1]),
        "max_raw_eigenvalue": float(raw[0]),
        "parseval_sum_q": float(q.sum()),
        "parseval_target_norm_over_m": total_q,
        "parseval_abs_error": float(abs(q.sum() - total_q)),
        **quantiles,
    }
    return raw, op, q, alignment, tail, local_slopes, desc


def execute_kernel(spec: Spec, tmp_root: Path, chunk_size: int, m_limit: int | None,
                   validate_unchunked: bool) -> dict[str, Any]:
    if not (OUTPUT_DIR / "model_reconstruction_audit.json").exists():
        raise RuntimeError("model reconstruction audit is missing; production is gated on it")
    audit = json.loads((OUTPUT_DIR / "model_reconstruction_audit.json").read_text())
    if audit.get("status") != "PASS":
        raise RuntimeError("model reconstruction audit did not pass")

    config_indices, atom_indices, y, probe_hash = load_probe(m_limit)
    m = len(config_indices)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_token = f"{spec.tag}_seed{INIT_SEED}_m{m}_job{job_id}_pid{os.getpid()}"
    tmp_dir = tmp_root / run_token
    grad_path = tmp_dir / "G_block_float32.dat"
    artifact_dir = OUTPUT_DIR / "kernels"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{spec.tag}_seed0_m{m}_kernel.npz"
    meta_path = artifact_dir / f"{spec.tag}_seed0_m{m}_metadata.json"
    if artifact_path.exists() or meta_path.exists():
        raise RuntimeError(f"refusing to overwrite existing result for {run_token}")

    t0 = time.time()
    model, cfg, params = instantiate_model(spec, device)
    if spec.architecture == "mcegnn":
        expected_mult = math.isqrt(spec.width) / 10
        assert not hasattr(model.net.force_decoder, "_mup_ref_param")
        assert math.isclose(float(model.net.force_decoder.width_mult), expected_mult,
                            rel_tol=0.0, abs_tol=1e-15)
    dataset = load_dataset()
    in_memory = bool(validate_unchunked)
    g_store, coverage, jacobian_seconds = collect_jacobian(
        model, params, dataset, config_indices, atom_indices, y, grad_path, in_memory
    )
    if coverage["parameter_count_represented_by_jacobian"] != spec.expected_n:
        raise AssertionError("Jacobian parameter count does not equal intended trainable N")

    # Release the network before the Gram pass; this reduces peak GPU and host memory.
    del model, params, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    k_raw = raw_gram_chunked(g_store, chunk_size, device)
    validation = None
    if validate_unchunked:
        k_unchunked = raw_gram_unchunked(g_store, device)
        denom = max(float(np.linalg.norm(k_unchunked, "fro")), 1e-300)
        gram_rel = float(np.linalg.norm(k_raw - k_unchunked, "fro") / denom)
        eig_chunk = np.linalg.eigvalsh(0.5 * (k_raw + k_raw.T))
        eig_full = np.linalg.eigvalsh(0.5 * (k_unchunked + k_unchunked.T))
        eig_rel = float(np.linalg.norm(eig_chunk - eig_full) /
                        max(float(np.linalg.norm(eig_full)), 1e-300))
        validation = {"gram_relative_frobenius_error": gram_rel,
                      "eigenspectrum_relative_l2_error": eig_rel,
                      "pass": gram_rel < 1e-10 and eig_rel < 1e-10}
        if not validation["pass"]:
            raise AssertionError(f"chunked-vs-unchunked validation failed: {validation}")

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

    metadata = {
        **asdict(spec),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                             text=True).strip(),
        "host": socket.gethostname(),
        "slurm_job_id": job_id,
        "init_seed": INIT_SEED,
        "m": m,
        "probe_hash": probe_hash,
        "train_rmsd": float(cfg.force_field_module.train_rmsd),
        "operator_normalization": "K_op = K_raw / m",
        "target_power_normalization": "q_j = (u_j^T y)^2 / m",
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
        "parameter_coverage": coverage,
        "chunked_vs_unchunked": validation,
    }
    np.savez(
        artifact_path,
        K_raw=k_raw.astype(np.float64),
        y=y.astype(np.float64),
        eigenvalues_raw=raw_eigs.astype(np.float64),
        eigenvalues_operator=op_eigs.astype(np.float64),
        q=q.astype(np.float64),
        cumulative_target_alignment=alignment.astype(np.float64),
        target_tail=tail.astype(np.float64),
        local_spectral_log_slopes=local_slopes.astype(np.float64),
        config_indices=config_indices.astype(np.int64),
        atom_indices=atom_indices.astype(np.int64),
        m=np.int64(m),
        N=np.int64(spec.expected_n),
        width=np.int64(spec.width),
        init_seed=np.int64(INIT_SEED),
        probe_hash=np.array(probe_hash),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    metadata["kernel_artifact_sha256"] = sha256_file(artifact_path)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {artifact_path}", flush=True)
    print(f"wrote {meta_path}", flush=True)

    if isinstance(g_store, np.memmap):
        g_store.flush()
        del g_store
        gc.collect()
        grad_path.unlink()
        tmp_dir.rmdir()
    return metadata


def validate(spec: Spec, tmp_root: Path, chunk_size: int) -> None:
    result = execute_kernel(spec, tmp_root, chunk_size, m_limit=8, validate_unchunked=True)
    path = OUTPUT_DIR / "chunked_vs_unchunked_validation.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["freeze-probe", "audit", "validate", "kernel"])
    ap.add_argument("--tag", choices=sorted(SPECS))
    ap.add_argument("--tmp-root", type=Path,
                    default=Path("<CHECKPOINT_ROOT>/"
                                 "four_arch_initial_force_kernel_pilot_tmp_20260814"))
    ap.add_argument("--gram-chunk-size", type=int, default=250_000)
    args = ap.parse_args()
    if args.mode == "freeze-probe":
        freeze_probe()
    elif args.mode == "audit":
        audit_all()
    else:
        if args.tag is None:
            ap.error("--tag is required for validate/kernel")
        spec = SPECS[args.tag]
        if args.mode == "validate":
            validate(spec, args.tmp_root, args.gram_chunk_size)
        else:
            execute_kernel(spec, args.tmp_root, args.gram_chunk_size,
                           m_limit=None, validate_unchunked=False)


if __name__ == "__main__":
    main()
