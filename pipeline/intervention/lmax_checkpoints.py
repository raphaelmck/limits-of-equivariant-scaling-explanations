# Adapted for anonymous release from analysis_scripts/repr_probe/stage3b_owners.py
# source revision withheld for anonymous review, original SHA256 d45db14807ba00b8d95b887028d18b7655d4e8a1b425b387ca61f9494734add6
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Checkpoint specifications for the ell_max=2 versus ell_max=4 comparison: three ell_max=4
frontier checkpoints and the three ell_max=2 checkpoints matched to them in compute, built
from the study's own checkpoint manifest rather than from the four-architecture registry.
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.kernel.frontier_checkpoints import (
    ARCH_LABEL,
    WIDTH_LABEL,
    INSTANTIATION_SEED,
    OwnerSpec,
    instantiate_model_generic,
    load_checkpoint_strict_generic,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "analysis_outputs" / "stage3b_lmax2_vs_lmax4_2026_08_16" / "stage3b_checkpoint_manifest.json"


def load_owner_specs() -> dict[str, OwnerSpec]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    owners: dict[str, OwnerSpec] = {}
    for lmax_key, cfg_path_key in (("lmax4", "original_config_path"), ("lmax2", "config_snapshot_path")):
        for tier, o in manifest["owners"][lmax_key].items():
            tag = f"esen_{lmax_key}_{tier}_w{o['width']}_s{o['step']}"
            owners[tag] = OwnerSpec(
                tag=tag,
                architecture="esen",
                architecture_label=ARCH_LABEL["esen"],
                budget_label=tier,
                width_label=WIDTH_LABEL["esen"],
                width=o["width"],
                step=o["step"],
                expected_n=o["N_net_params"],
                D_atom_tokens=o["D_atom_tokens"],
                C_owner=o["C_flops"],
                C_budget=manifest["C_targets_corrected"][tier],
                utilization=o["utilization"],
                force_mse_norm=None,  # not used; the lmax=2 vs lmax=4 study recomputes force metrics fresh (task Section 8)
                ckpt_path=o["checkpoint_path"],
                original_config_path=o[cfg_path_key],
                config_name=f"{tag}.yaml",
            )
    assert len(owners) == 6, f"expected 6 the lmax=2 vs lmax=4 study owners, got {len(owners)}"
    return owners


OWNERS = load_owner_specs()


def reconstruct(tag: str, device: str = "cpu"):
    owner = OWNERS[tag]
    model, cfg, params = instantiate_model_generic(owner, device, INSTANTIATION_SEED)
    provenance = load_checkpoint_strict_generic(model, owner, params)
    model.net.eval()
    return model, cfg, params, provenance
