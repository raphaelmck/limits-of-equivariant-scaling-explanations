# Adapted for anonymous release from analysis_scripts/repr_probe/checkpoint_loading.py
# source revision withheld for anonymous review, original SHA256 086f943f68121cfdeaa9fe6ea61c59045139e37d9edf116b76c5d06ba2da297f
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Checkpoint reconstruction: instantiate a model from its frozen configuration snapshot and
load the checkpoint state dict strictly.

Reuses (imports, does not duplicate) the already-audited, PASS-status reconstruction
machinery from pipeline/kernel/frontier_checkpoints.py: OwnerSpec,
instantiate_model_generic, load_checkpoint_strict_generic. That machinery was independently
verified against all 12 owners (owner_reconstruction_audit.json, status PASS) before this
pilot existed -- see the checkpoint reconstruction test: this module's job is only
to build the 12 OwnerSpec instances from THIS pilot's frozen checkpoint_manifest.yaml (same
12 checkpoints as the sibling kernel pilot, restated here so provenance traces to this
pilot's own frozen manifest) and re-run that verified reconstruction path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pipeline.kernel.frontier_checkpoints import (
    ARCH_LABEL,
    WIDTH_LABEL,
    INSTANTIATION_SEED,
    OwnerSpec,
    instantiate_model_generic,
    load_checkpoint_strict_generic,
)

REPR_PROBE_PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPR_PROBE_PKG_DIR.parents[1]
DESIGN_DIR = (
    REPO_ROOT
    / "analysis_outputs"
    / "four_arch_representation_probe_frontier_DESIGN_2026_08_15"
)
CHECKPOINT_MANIFEST_PATH = DESIGN_DIR / "checkpoint_manifest.yaml"

# design-freeze manifest calls the architecture "mcegnn"; the model code / OwnerSpec /
# ARCH_LABEL / WIDTH_LABEL convention (inherited from the sibling kernel pilots) calls it "egnn".
ARCH_ALIAS = {"mcegnn": "egnn"}


def _arch_code(design_arch: str) -> str:
    return ARCH_ALIAS.get(design_arch, design_arch)


def load_owner_specs() -> dict[str, OwnerSpec]:
    raw = yaml.safe_load(CHECKPOINT_MANIFEST_PATH.read_text())
    owners: dict[str, OwnerSpec] = {}
    for o in raw["owners"]:
        arch = _arch_code(o["architecture"])
        tag = f"{arch}_{o['budget']}_w{o['width']}_s{o['step']}"
        owners[tag] = OwnerSpec(
            tag=tag,
            architecture=arch,
            architecture_label=ARCH_LABEL[arch],
            budget_label=o["budget"],
            width_label=WIDTH_LABEL[arch],
            width=o["width"],
            step=o["step"],
            expected_n=o["param_count"],
            D_atom_tokens=o["D_atom_tokens"],
            C_owner=o["C_owner_flops"],
            C_budget=o["C_budget_flops"],
            utilization=o["utilization"],
            force_mse_norm=o["validation_force_mse_norm"],
            ckpt_path=o["checkpoint_path"],
            original_config_path=o["original_config_path"],
            config_name=f"{tag}.yaml",
        )
    assert len(owners) == 12, f"expected 12 unique owners, got {len(owners)}"
    return owners


OWNERS = load_owner_specs()


def reconstruct(tag: str, device: str = "cpu") -> tuple[Any, Any, Any, dict]:
    """Instantiate + strict-load one owner's checkpoint.

    Returns (model, cfg, params, checkpoint_provenance). Raises AssertionError on ANY
    reconstruction mismatch (trainable-parameter count, checkpoint_global_step, MC-EGNN
    legacy width_mult formula, etc.) -- reconstruction failures must block downstream work,
    not warn and continue.
    """
    owner = OWNERS[tag]
    model, cfg, params = instantiate_model_generic(owner, device, INSTANTIATION_SEED)
    provenance = load_checkpoint_strict_generic(model, owner, params)
    model.net.eval()
    return model, cfg, params, provenance
