#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/four_arch_matched_compute_frontier_kernel.py
# source revision withheld for anonymous review, original SHA256 15e80029589cd7a837fac9615a3a265fef1f3464e1336d51096ea1dc36a07870
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/four_arch_matched_compute_frontier_kernel.py
# source revision withheld for anonymous review, original SHA256 15e80029589cd7a837fac9615a3a265fef1f3464e1336d51096ea1dc36a07870
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Matched-compute frontier-owner force-tangent-kernel pilot (2026-08-15).

At three predeclared common-compute budgets (LOW/MID/HIGH, 20/50/80% of the log-C
span [8.49e15, 1.224e18] shared by all four architectures on force_mse_norm), take the
OBSERVED force-frontier-owner checkpoint for each architecture (argmin force_mse_norm among
checkpoints with C <= C_budget, from the audited
`four_arch_force_scaling_repaired_2026_08_14/four_arch_force_frontier_owners.csv`) and compute
its full-block force tangent kernel with the exact same audited machinery as the static and
K0/K50k/K500k trained-kernel pilots:

  - collect_jacobian_with_preds, raw_gram_chunked, spectral_arrays, target_power_arrays are
    imported UNCHANGED from kernel/force_ntk.py /
    kernel/force_ntk_trained.py;
  - load_checkpoint_strict's legacy-MC-EGNN / no-_mup_ref_param check logic is reproduced
    verbatim (parameterized on an OwnerSpec rather than the fixed-width Spec);
  - instantiate_model's logic (seed -> hydra.utils.instantiate(cfg.force_field_module),
    GemNet-OC mup-namespace patch, trainable-parameter collection/count assertion) is
    reproduced verbatim, generalized to load an arbitrary run's isolated `.hydra/config.yaml`
    snapshot instead of the fixed eight-tag CONFIG_DIR lookup.

All 12 selected owners are literal trained checkpoints (no budget's owner is a K0/random-init
model), so there is no K0-reconstruction-seed caveat anywhere in this pilot.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from pipeline.kernel.force_ntk import (
    TRAIN_RMSD,
    sha256_file,
    canonical_probe_hash,
    load_dataset,
    load_probe as load_probe_static,
    raw_gram_chunked,
    spectral_arrays,
    gemnet_get_mup_multipliers,
    gemnet_mup_init,
    OUTPUT_DIR as STATIC_OUTPUT_DIR,
)
from pipeline.kernel.force_ntk_trained import (
    collect_jacobian_with_preds,
    target_power_arrays,
)

MAIN_WORKTREE = Path("<PROJECT_ROOT>/upstream-training")
OUTPUT_DIR = REPO_ROOT / "analysis_outputs/four_arch_matched_compute_frontier_kernel_2026_08_15"
CONFIG_DIR = OUTPUT_DIR / "inputs/configs"
RCOND = 1e-10
TRAINING_CFG_SEED = 1
INSTANTIATION_SEED = 1  # matches every training run's cfg.seed=1; irrelevant for these
                         # checkpoint-loaded kernels since every parameter is overwritten by
                         # the strict state-dict load immediately after instantiation.


@dataclass(frozen=True)
class OwnerSpec:
    tag: str
    architecture: str
    architecture_label: str
    budget_label: str
    width_label: str
    width: int
    step: int
    expected_n: int
    D_atom_tokens: int
    C_owner: float
    C_budget: float
    utilization: float
    force_mse_norm: float
    ckpt_path: str
    original_config_path: str
    config_name: str


ARCH_LABEL = {"mpnn": "MPNN", "egnn": "MC-EGNN", "gemnet_oc": "GemNet-OC", "esen": "eSEN"}
WIDTH_LABEL = {"mpnn": "hc", "egnn": "hc", "gemnet_oc": "width", "esen": "sphere_channels"}

# Frozen 2026-08-15, from selected_frontier_owners.csv (see that file / REPORT.md section 0
# for the full derivation from four_arch_force_frontier_owners.csv). Each entry is a literal
# argmin-force_mse_norm checkpoint at C <= C_budget; none coincide across budgets, so all 12
# are unique new kernels.
OWNERS_RAW = [
    dict(budget_label="LOW", architecture="mpnn", width=607, step=400000, expected_n=4546197,
         D_atom_tokens=693100729, C_owner=2.202529745959452e+16, C_budget=2.2946971642918196e+16,
         force_mse_norm=0.0157542140583355,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_corrected_stats/dim-607-neutral-epoch-corrected-stats/mpnn/run_9782261_params_4.5_million/dim_607_step_step=400000.ckpt",
         original_config_path=str(MAIN_WORKTREE / "outputs/2026-06-08/11-47-15/.hydra/config.yaml")),
    dict(budget_label="MID", architecture="mpnn", width=1150, step=500000, expected_n=16106329,
         D_atom_tokens=866390858, C_owner=9.754108964876658e+16, C_budget=1.0195623578119141e+17,
         force_mse_norm=0.0109093944632446,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_corrected_stats/dim-1150-neutral-epoch-corrected-stats/mpnn/run_10325226_params_16.1_million/dim_1150_step_step=500000.ckpt",
         original_config_path=str(MAIN_WORKTREE / "outputs/2026-08-09/17-10-47/.hydra/config.yaml")),
    dict(budget_label="HIGH", architecture="mpnn", width=2557, step=450000, expected_n=78984522,
         D_atom_tokens=779777582, C_owner=4.305166134682947e+17, C_budget=4.5300417747619904e+17,
         force_mse_norm=0.0080524201064766,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_corrected_stats/dim-2557-neutral-epoch-corrected-stats/mpnn/run_10325229_params_79.0_million/dim_2557_step_step=450000.ckpt",
         original_config_path=str(MAIN_WORKTREE / "outputs/2026-08-09/17-35-54/.hydra/config.yaml")),

    dict(budget_label="LOW", architecture="egnn", width=96, step=100000, expected_n=1055050,
         D_atom_tokens=173280589, C_owner=1.5406214890718402e+16, C_budget=2.2946971642918196e+16,
         force_mse_norm=0.0080014705545307,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_mc_egnn/mc-egnn-hc-96-neutral-epoch/mc_egnn/run_9745361_params_1.1_million/dim_96_step_step=100000.ckpt",
         original_config_path=str(MAIN_WORKTREE / "outputs/2026-06-05/19-49-19/.hydra/config.yaml")),
    dict(budget_label="MID", architecture="egnn", width=96, step=450000, expected_n=1055050,
         D_atom_tokens=779691165, C_owner=6.9321611299373976e+16, C_budget=1.0195623578119141e+17,
         force_mse_norm=0.0048411353709021,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_mc_egnn/mc-egnn-hc-96-neutral-epoch/mc_egnn/run_9745361_params_1.1_million/dim_96_step_step=450000.ckpt",
         original_config_path=str(MAIN_WORKTREE / "outputs/2026-06-05/19-49-19/.hydra/config.yaml")),
    dict(budget_label="HIGH", architecture="egnn", width=320, step=250000, expected_n=10372498,
         D_atom_tokens=433116378, C_owner=3.78582870890503e+17, C_budget=4.5300417747619904e+17,
         force_mse_norm=0.0034946079534482,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_mc_egnn/mc-egnn-hc-320-neutral-epoch/mc_egnn/run_9745363_params_10.4_million/dim_320_step_step=250000.ckpt",
         original_config_path=str(MAIN_WORKTREE / "outputs/2026-06-05/22-20-01/.hydra/config.yaml")),

    dict(budget_label="LOW", architecture="gemnet_oc", width=48, step=200000, expected_n=575472,
         D_atom_tokens=346647268, C_owner=2.1053730974271748e+16, C_budget=2.2946971642918196e+16,
         force_mse_norm=0.003287852691115,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_gemnet_oc/gemnet-oc-atom48-neutral-epoch/gemnet_oc/run_9816221_params_575.5_thousand/dim_48_step_step=200000.ckpt",
         original_config_path=str(MAIN_WORKTREE / "outputs/2026-06-12/16-37-08/.hydra/config.yaml")),
    dict(budget_label="MID", architecture="gemnet_oc", width=64, step=500000, expected_n=937280,
         D_atom_tokens=866390858, C_owner=8.570384390018376e+16, C_budget=1.0195623578119141e+17,
         force_mse_norm=0.001844648895683,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_gemnet_oc/gemnet-oc-atom64-neutral-epoch/gemnet_oc/run_9816214_params_937.3_thousand/dim_64_step_step=500000.ckpt",
         original_config_path=str(MAIN_WORKTREE / "outputs/2026-06-12/15-12-18/.hydra/config.yaml")),
    dict(budget_label="HIGH", architecture="gemnet_oc", width=144, step=500000, expected_n=4236240,
         D_atom_tokens=866390858, C_owner=3.873570882593403e+17, C_budget=4.5300417747619904e+17,
         force_mse_norm=0.0012216940107775,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_gemnet_oc/gemnet-oc-atom144-neutral-epoch/gemnet_oc/run_9816222_params_4.2_million/dim_144_step_step=500000.ckpt",
         original_config_path=str(MAIN_WORKTREE / "outputs/2026-06-12/17-10-33/.hydra/config.yaml")),

    dict(budget_label="LOW", architecture="esen", width=10, step=50000, expected_n=1089390,
         D_atom_tokens=86650769, C_owner=2.10579670352222e+16, C_budget=2.2946971642918196e+16,
         force_mse_norm=0.0060544097484207,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_esen/esen-sphere-10-neutral-epoch/esen/run_9741255_params_1.1_million/dim_32_step_step=50000.ckpt",
         original_config_path=str(MAIN_WORKTREE / "outputs/2026-06-05/12-34-36/.hydra/config.yaml")),
    dict(budget_label="MID", architecture="esen", width=16, step=150000, expected_n=1722258,
         D_atom_tokens=259964327, C_owner=9.987863619290285e+16, C_budget=1.0195623578119141e+17,
         force_mse_norm=0.001978949702442,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_esen/esen-sphere-16-neutral-epoch/esen/run_9721383_params_1.7_million/dim_32_step_step=150000.ckpt",
         original_config_path=str(MAIN_WORKTREE / "outputs/2026-06-05/12-55-53/.hydra/config.yaml")),
    dict(budget_label="HIGH", architecture="esen", width=40, step=275000, expected_n=4253730,
         D_atom_tokens=476605759, C_owner=4.522617321737471e+17, C_budget=4.5300417747619904e+17,
         force_mse_norm=0.0008361147460075,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_esen/esen-sphere-40-neutral-epoch/esen/run_9928671_params_4.3_million/dim_32_step_step=275000.ckpt",
         original_config_path=str(MAIN_WORKTREE / "outputs/2026-06-27/13-40-54/.hydra/config.yaml")),
]


def _build_owners() -> dict[str, OwnerSpec]:
    owners = {}
    for raw in OWNERS_RAW:
        arch = raw["architecture"]
        tag = f"{arch}_{raw['budget_label']}_w{raw['width']}_s{raw['step']}"
        owners[tag] = OwnerSpec(
            tag=tag,
            architecture=arch,
            architecture_label=ARCH_LABEL[arch],
            budget_label=raw["budget_label"],
            width_label=WIDTH_LABEL[arch],
            width=raw["width"],
            step=raw["step"],
            expected_n=raw["expected_n"],
            D_atom_tokens=raw["D_atom_tokens"],
            C_owner=raw["C_owner"],
            C_budget=raw["C_budget"],
            utilization=raw["C_owner"] / raw["C_budget"],
            force_mse_norm=raw["force_mse_norm"],
            ckpt_path=raw["ckpt_path"],
            original_config_path=raw["original_config_path"],
            config_name=f"{tag}.yaml",
        )
    return owners


OWNERS = _build_owners()
assert len(OWNERS) == 12, f"expected 12 unique owners, got {len(OWNERS)}"


def snapshot_config(owner: OwnerSpec) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    dst = CONFIG_DIR / owner.config_name
    src = Path(owner.original_config_path)
    if not dst.exists():
        dst.write_bytes(src.read_bytes())
    if sha256_file(dst) != sha256_file(src):
        raise AssertionError(f"{owner.tag}: config snapshot diverged from original {src}")
    return dst


def instantiate_model_generic(owner: OwnerSpec, device: str, seed: int):
    """Reproduces four_arch_initial_force_kernel_pilot.instantiate_model's logic exactly,
    generalized to load an arbitrary run's isolated config snapshot instead of the fixed
    eight-tag CONFIG_DIR lookup."""
    import random
    cfg_path = snapshot_config(owner)
    cfg = OmegaConf.load(cfg_path)
    rmsd = float(cfg.force_field_module.train_rmsd)
    if rmsd != TRAIN_RMSD:
        raise AssertionError(f"{owner.tag}: config train_rmsd={rmsd}, expected {TRAIN_RMSD}")
    cfg_seed = int(cfg.seed)
    if cfg_seed != TRAINING_CFG_SEED:
        raise AssertionError(f"{owner.tag}: cfg.seed={cfg_seed}, expected {TRAINING_CFG_SEED}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if owner.architecture == "gemnet_oc":
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
    if n != owner.expected_n or int(model.total_params) != owner.expected_n:
        raise AssertionError(
            f"{owner.tag}: trainable count={n}, model.total_params={model.total_params}, "
            f"expected={owner.expected_n}"
        )
    return model, cfg, params


def load_checkpoint_strict_generic(model, owner: OwnerSpec, params) -> dict[str, Any]:
    """Reproduces four_arch_trained_force_kernel_pilot.load_checkpoint_strict exactly,
    parameterized on OwnerSpec instead of the fixed-width Spec."""
    ckpt_path = Path(owner.ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    parameter_state_keys = {f"net.{name}" for name, _ in params}
    missing = sorted(parameter_state_keys - set(state))
    if missing:
        raise AssertionError(f"{owner.tag}: checkpoint lacks net parameters {missing}")
    state_net_n = int(sum(state[k].numel() for k in parameter_state_keys))
    if state_net_n != owner.expected_n:
        raise AssertionError(
            f"{owner.tag}: checkpoint net state count={state_net_n}, expected={owner.expected_n}")

    legacy = None
    if owner.architecture == "egnn":
        decoder = model.net.force_decoder
        expected_mult = math.isqrt(owner.width) / math.isqrt(102)
        actual_mult = float(decoder.width_mult)
        leaked = [name for name, module in model.net.named_modules()
                  if hasattr(module, "_mup_ref_param")]
        if leaked:
            raise AssertionError(f"{owner.tag}: leaked _mup_ref_param attributes: {leaked}")
        if not math.isclose(actual_mult, expected_mult, rel_tol=0.0, abs_tol=1e-15):
            raise AssertionError(
                f"{owner.tag}: force_decoder.width_mult={actual_mult}, legacy expected={expected_mult}")
        legacy = {
            "force_decoder_width_mult": actual_mult,
            "legacy_formula": "isqrt(hidden_channels)/isqrt(102)",
            "mup_ref_param_attributes": leaked,
        }

    global_step = int(ckpt["global_step"]) if "global_step" in ckpt else None
    if global_step != owner.step:
        raise AssertionError(
            f"{owner.tag}: checkpoint_global_step={global_step}, requested step={owner.step}")
    return {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_size_bytes": ckpt_path.stat().st_size,
        "checkpoint_global_step": global_step,
        "checkpoint_strict_load": True,
        "checkpoint_net_state_parameter_count": state_net_n,
        "legacy_mcegnn_checkpoint_audit": legacy,
    }


def audit_one(owner: OwnerSpec, device: str = "cpu") -> dict[str, Any]:
    model, cfg, params = instantiate_model_generic(owner, device, INSTANTIATION_SEED)
    cp = load_checkpoint_strict_generic(model, owner, params)
    model.net.eval()
    if owner.architecture == "egnn":
        expected_mult = math.isqrt(owner.width) / 10
        assert not hasattr(model.net.force_decoder, "_mup_ref_param")
        assert math.isclose(float(model.net.force_decoder.width_mult), expected_mult,
                            rel_tol=0.0, abs_tol=1e-15)
    result = {**asdict(owner), "checkpoint_provenance": cp,
              "config_snapshot_sha256": sha256_file(CONFIG_DIR / owner.config_name),
              "original_config_sha256": sha256_file(Path(owner.original_config_path))}
    del model, cfg, params
    gc.collect()
    return result


def audit_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for tag, owner in OWNERS.items():
        print(f"AUDIT {tag}", flush=True)
        results.append(audit_one(owner))
    out = {
        "status": "PASS",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                             text=True).strip(),
        "host": socket.gethostname(),
        "owners": results,
    }
    path = OUTPUT_DIR / "owner_reconstruction_audit.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


def execute_kernel(owner: OwnerSpec, tmp_root: Path, chunk_size: int) -> dict[str, Any]:
    audit_path = OUTPUT_DIR / "owner_reconstruction_audit.json"
    if not audit_path.exists():
        raise RuntimeError("owner reconstruction audit is missing; production is gated on it")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS":
        raise RuntimeError("owner reconstruction audit did not pass")

    config_indices, atom_indices, y, probe_hash = load_probe_static(None)
    m = len(config_indices)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_token = f"{owner.tag}_m{m}_job{job_id}_pid{os.getpid()}"
    tmp_dir = tmp_root / run_token
    grad_path = tmp_dir / "G_block_float32.dat"
    artifact_dir = OUTPUT_DIR / "kernels"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{owner.tag}_m{m}_kernel.npz"
    meta_path = artifact_dir / f"{owner.tag}_m{m}_metadata.json"
    if artifact_path.exists() or meta_path.exists():
        raise RuntimeError(f"refusing to overwrite existing result for {run_token}")

    t0 = time.time()
    model, cfg, params = instantiate_model_generic(owner, device, INSTANTIATION_SEED)
    checkpoint_provenance = load_checkpoint_strict_generic(model, owner, params)
    model.net.eval()

    if owner.architecture == "egnn":
        expected_mult = math.isqrt(owner.width) / 10
        assert not hasattr(model.net.force_decoder, "_mup_ref_param")
        assert math.isclose(float(model.net.force_decoder.width_mult), expected_mult,
                            rel_tol=0.0, abs_tol=1e-15)

    dataset = load_dataset()
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

    metadata = {
        **asdict(owner),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                             text=True).strip(),
        "host": socket.gethostname(),
        "slurm_job_id": job_id,
        "init_seed": INSTANTIATION_SEED,
        "training_run_cfg_seed": TRAINING_CFG_SEED,
        "trajectory_classification": "CHECKPOINT_EXACT_TRAINED_STATE",
        "trajectory_classification_reason": (
            "This kernel is computed from the exact strict-loaded training checkpoint state "
            "dict at the requested step (the observed force-frontier owner for this budget); "
            "it is the literal trained parameter state, not a reconstruction. No K0 kernel is "
            "part of this pilot -- every selected owner is a trained checkpoint."
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
        N=np.int64(owner.expected_n),
        width=np.int64(owner.width),
        init_seed=np.int64(INSTANTIATION_SEED),
        budget_label=np.array(owner.budget_label),
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
    ap.add_argument("mode", choices=["audit", "kernel"])
    ap.add_argument("--tag", choices=sorted(OWNERS))
    ap.add_argument("--tmp-root", type=Path,
                    default=Path("<CHECKPOINT_ROOT>/"
                                 "four_arch_matched_compute_frontier_kernel_tmp_20260815"))
    ap.add_argument("--gram-chunk-size", type=int, default=250_000)
    args = ap.parse_args()
    if args.mode == "audit":
        audit_all()
    else:
        if args.tag is None:
            ap.error("--tag is required for kernel mode")
        execute_kernel(OWNERS[args.tag], args.tmp_root, args.gram_chunk_size)


if __name__ == "__main__":
    main()
