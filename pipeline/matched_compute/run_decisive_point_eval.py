#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/stage3b1_decisive_point_eval.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 638aad93c3865403063120a33e1f5ea916e2a41bb9644f59cfa3fe7d3737d1cc
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/stage3b1_decisive_point_eval.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 638aad93c3865403063120a33e1f5ea916e2a41bb9644f59cfa3fe7d3737d1cc
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""the matched-frontier study: full-val evaluation (with per-graph residuals retained, for paired bootstrap)
of the single decisive lmax4 checkpoint at C_common_max: sphere=40, step=200000 (same run,
run_9928671, as the lmax=2 vs lmax=4 study's own frozen lmax4 HIGH owner at step=275000 -- reuses that owner's
exact config path unchanged, just swaps ckpt_path/step/N/D). This is the true lmax4 frontier
owner at C_common_max=3.724e17 per stage3b1_frontier_lmax4.json, needed to compute a paired
bootstrap CI for H_force at the decisive near-C_common_max comparison point (l2: w64/s525000,
already has stored per-graph residuals in the lmax=2 vs lmax=4 study's own force_performance_results_full_val.jsonl).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from pipeline.intervention import evaluation_data
from pipeline.intervention.esen_intervention import ESenInterventionForward
from pipeline.kernel.frontier_checkpoints import (
    ARCH_LABEL, WIDTH_LABEL, INSTANTIATION_SEED, OwnerSpec,
    instantiate_model_generic, load_checkpoint_strict_generic,
)

OUT_DIR = REPO_ROOT / "analysis_outputs" / "stage3b1_true_lmax_frontiers_2026_08_16"
TRAIN_RMSD = 6.383
CHUNK = 32

owner = OwnerSpec(
    tag="esen_lmax4_DECISIVE_w40_s200000", architecture="esen", architecture_label=ARCH_LABEL["esen"],
    budget_label="DECISIVE", width_label=WIDTH_LABEL["esen"], width=40, step=200000,
    expected_n=4253730, D_atom_tokens=346647268, C_owner=3.700463062270997e17, C_budget=3.700463062270997e17,
    utilization=1.0, force_mse_norm=None,
    ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_esen/esen-sphere-40-neutral-epoch/esen/run_9928671_params_4.3_million/dim_32_step_step=200000.ckpt",
    original_config_path="<PROJECT_ROOT>/upstream-training/outputs/2026-06-27/13-40-54/.hydra/config.yaml",
    config_name="esen_lmax4_HIGH_reused_for_decisive.yaml",
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    dataset = evaluation_data.load_dataset()
    config_indices = list(range(len(dataset)))
    print(f"population=full_val n_configs={len(config_indices)}", flush=True)
    chunks = [evaluation_data.build_batch(dataset, config_indices[i:i + CHUNK], device)
              for i in range(0, len(config_indices), CHUNK)]

    model, cfg, params = instantiate_model_generic(owner, device, INSTANTIATION_SEED)
    cp = load_checkpoint_strict_generic(model, owner, params)
    model.net.eval()
    fwd = ESenInterventionForward(model)

    per_graph_sq_means = []
    t0 = time.time()
    for b in chunks:
        out, _ = fwd(b, intervention=None)
        target = b.forces / TRAIN_RMSD
        residual = out["forces"] - target
        sq_per_atom = residual.pow(2).sum(dim=-1)
        natoms = b.natoms.to(device)
        ng = int(natoms.shape[0])
        sq_graph = torch.zeros(ng, device=device).index_add_(0, b.batch, sq_per_atom) / natoms.clamp_min(1)
        per_graph_sq_means.extend(sq_graph.detach().cpu().numpy().tolist())

    L = float(np.mean(per_graph_sq_means))
    row = {
        "tag": owner.tag, "ell_max": 4, "width": owner.width, "step": owner.step,
        "N_net_params": owner.expected_n, "D_atom_tokens": owner.D_atom_tokens,
        "C_flops": owner.C_owner, "checkpoint_sha256": cp["checkpoint_sha256"],
        "population": "full_val", "n_configs": len(config_indices),
        "L_primary_force_mse_norm_graph_uniform_squared": L,
        "per_graph_sq_means": per_graph_sq_means,
        "wall_seconds": time.time() - t0,
    }
    (OUT_DIR / "stage3b1_decisive_point_lmax4_w40_s200000.json").write_text(json.dumps(row, default=str))
    print(f"L_primary={L:.6f} wall={row['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
