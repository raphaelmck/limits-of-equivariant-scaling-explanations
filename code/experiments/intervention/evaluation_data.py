# Adapted for anonymous release from analysis_scripts/repr_probe/stage3a_data.py
# source revision withheld for anonymous review, original SHA256 768ec5333ec66c7b76116fc657b3965cdfe840f5e44e0dab80bbe9d8ef53a862
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Shared data and metric utilities for the interventions: dataset paths, the training-set force
scale, and single- and multi-molecule batch construction.
"""
from __future__ import annotations

import numpy as np
import torch
from fairchem.core.datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

TRAIN_RMSD = 6.383
VAL_PATH = "<DATA_ROOT>/open_mol/neutral_val"


def load_dataset() -> AseDBDataset:
    return AseDBDataset({"src": VAL_PATH, "a2g_args": dict(r_energy=True, r_forces=True)})


def build_batch(dataset: AseDBDataset, config_indices, device: str):
    items = [dataset[int(ci)] for ci in config_indices]
    batch = atomicdata_list_to_batch(items).to(device)
    if not hasattr(batch, "z"):
        batch.z = batch.atomic_numbers.long()
    return batch


def per_atom_force_error(forces_pred: torch.Tensor, batch, train_rmsd: float = TRAIN_RMSD) -> torch.Tensor:
    """||f_pred - f_gt/train_rmsd||_2 per atom -- exactly the per-atom term
    fairchem.core.modules.loss.L2NormLoss computes, before DDPLoss's flat mean-over-atoms
    reduction (src/model/omol_module.py:252, f_loss_fn=DDPLoss('l2norm','mean'))."""
    target = batch.forces / train_rmsd
    return torch.linalg.vector_norm(forces_pred - target, ord=2, dim=-1)


def per_config_mean_error(per_atom_err: torch.Tensor, batch) -> np.ndarray:
    """Mean per-atom error within each configuration in the batch (batch.batch gives the
    config index per atom, batch.natoms the atom count per config, both in dataset config-index
    order as passed into build_batch)."""
    batch_idx = batch.batch
    n_configs = int(batch.natoms.numel())
    out = np.zeros(n_configs, dtype=np.float64)
    err_np = per_atom_err.detach().cpu().numpy().astype(np.float64)
    idx_np = batch_idx.detach().cpu().numpy()
    counts = np.zeros(n_configs, dtype=np.int64)
    for i, e in zip(idx_np, err_np):
        out[i] += e
        counts[i] += 1
    out = out / np.maximum(counts, 1)
    return out


def flat_atom_mean(per_atom_err_list) -> float:
    """Flat mean over ALL atoms across ALL provided per-atom-error arrays/tensors -- matches
    DDPLoss(reduction='mean')'s global (not per-config-averaged) atom weighting exactly."""
    total = 0.0
    n = 0
    for arr in per_atom_err_list:
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        total += float(np.sum(arr))
        n += len(arr)
    return total / max(n, 1)
