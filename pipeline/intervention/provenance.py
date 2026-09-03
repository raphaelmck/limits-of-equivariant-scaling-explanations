# Adapted for anonymous release from analysis_scripts/repr_probe/provenance.py
# source revision withheld for anonymous review, original SHA256 b7c7267c14501ab6d49d44aac781bbfc2963af8e7d627c400ff9a014b18c1aac
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Provenance schema attached to every result row: checkpoint identity, configuration, code
state, and evaluation pool.

Every saved probe result must carry all fields listed in the task spec. This module builds
that record as a plain dict (JSON-serializable) from the pieces already available elsewhere
in this package (OwnerSpec/checkpoint provenance, RepresentationBundle metadata, RidgeFitResult).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def build_record(
    *,
    owner,  # OwnerSpec
    checkpoint_provenance: dict,
    bundle,  # RepresentationBundle
    probe_class: str,  # "native" | "common"
    allowed_geometric_inputs: list[str],
    fit_result,  # ridge.RidgeFitResult
    n_fit: int,
    n_val: int,
    n_test: int,
    target_normalization: str,
    split_hashes: dict[str, str],
    random_seed: int,
) -> dict[str, Any]:
    symmetry_type = (
        "vector_equivariant" if bundle.vector_features is not None
        else "irrep_equivariant" if bundle.irreps is not None
        else ("invariant" if bundle.metadata.get("scalar_is_invariant") else "frame_dependent")
    )
    rep_dim = (
        bundle.vector_features.shape[-1] if bundle.vector_features is not None
        else bundle.scalar_features.shape[-1] if bundle.scalar_features is not None
        else {l: t.shape[-1] for l, t in bundle.irreps.items()}
    )
    return {
        "git_sha": git_sha(),
        "checkpoint_path": checkpoint_provenance["checkpoint_path"],
        "checkpoint_sha256": checkpoint_provenance["checkpoint_sha256"],
        "checkpoint_global_step": checkpoint_provenance["checkpoint_global_step"],
        "config_snapshot_path": owner.original_config_path,
        "dataset_split_hashes": split_hashes,
        "architecture": bundle.architecture,
        "architecture_label": owner.architecture_label,
        "width": owner.width,
        "width_label": owner.width_label,
        "training_step": owner.step,
        "measured_compute_flops": owner.C_owner,
        "neural_force_mse_norm": owner.force_mse_norm,
        "representation_name": f"{bundle.architecture}.{bundle.layer}",
        "representation_layer": bundle.layer,
        "representation_dimensions": rep_dim,
        "symmetry_type": symmetry_type,
        "entity_type": bundle.entity_type,
        "probe_class": probe_class,
        "allowed_geometric_inputs": allowed_geometric_inputs,
        "bias_on": bundle.metadata.get("bias_on"),
        "width_mult": bundle.metadata["width_mult"],
        "selected_lambda": fit_result.selected_lambda,
        "selected_lambda_grid_index": fit_result.selected_lambda_grid_index,
        "lambda_grid": fit_result.grid.tolist(),
        "n_probe_fit": n_fit,
        "n_probe_val": n_val,
        "n_probe_test": n_test,
        "target_normalization": target_normalization,
        "E_probe_val": fit_result.e_probe_val,
        "E_probe_test": fit_result.e_probe_test,
        "random_seed": random_seed,
        "dtype": bundle.metadata.get("dtype"),
        "fit_kind": fit_result.kind,
    }
