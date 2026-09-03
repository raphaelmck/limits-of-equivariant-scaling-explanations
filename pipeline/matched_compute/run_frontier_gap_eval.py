#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/stage3b1_gap_eval.py
# source revision withheld for anonymous review, original SHA256 40322e3af078be89b02b6728bdb3b380172eb7c2ff33a20f5b5e4758d8846f77
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/stage3b1_gap_eval.py
# source revision withheld for anonymous review, original SHA256 40322e3af078be89b02b6728bdb3b380172eb7c2ff33a20f5b5e4758d8846f77
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""the matched-frontier study Phase C gap-fill: full-neutral_val force_mse_norm evaluation for a small,
pre-specified shortlist of ell_max=2 checkpoints whose compute falls in the region between
the 2026-08-13 empirical_lmax_adversarial_audit's last-observed lmax2 frontier point
(sphere=32, step=525000, C=1.888e17) and the current (2026-08-16) lmax2 checkpoint-store ceiling
(sphere=64, step=525000, C~3.71-3.72e17, already evaluated by Stage 3B as its HIGH owner,
L_primary=0.000950). This gap did not exist at audit time because widths 40/48/64/etc. had not
yet been trained past ~step 200-400k; by the lmax=2 vs lmax=4 study's run (2026-08-16) the entire width x 525000
grid is populated (see stage3b_lmax2_inventory.json).

Shortlist (frozen before inspecting any of these losses):
  w=40, step=400000   (C~1.786e17, just below the old audit ceiling)
  w=40, step=525000   (C~2.344e17)
  w=64, step=300000   (C~2.121e17)
  w=64, step=400000   (C~2.828e17)

Reuses w=40's and w=64's existing the lmax=2 vs lmax=4 study config snapshots (config is step-independent),
swapping in the ckpt_path/step/N/D for each new candidate. Same reconstruction path
(instantiate_model_generic/load_checkpoint_strict_generic) and same metric implementation
(graph-uniform-weighted squared force_mse_norm) as run_stage3b_force_performance.py --full-val,
so results are directly comparable to the lmax=2 vs lmax=4 study and 2026-08-13-audit numbers without
re-deriving the formula.
"""
from __future__ import annotations

import json
import sys
import time
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

LMAX_STUDY_DIR = REPO_ROOT / "analysis_outputs" / "stage3b_lmax2_vs_lmax4_2026_08_16"
OUT_DIR = REPO_ROOT / "analysis_outputs" / "stage3b1_true_lmax_frontiers_2026_08_16"
LMAX2_INV = json.loads((LMAX_STUDY_DIR / "stage3b_lmax2_inventory.json").read_text())["records"]
TRAIN_RMSD = 6.383
CHUNK = 32
KAPPA_L2 = 217.2151

SHORTLIST = [
    dict(width=40, step=400000, config_snapshot="esen_lmax2_MID_w40_s250000.yaml"),
    dict(width=40, step=525000, config_snapshot="esen_lmax2_MID_w40_s250000.yaml"),
    dict(width=64, step=300000, config_snapshot="esen_lmax2_HIGH_w64_s525000.yaml"),
    dict(width=64, step=400000, config_snapshot="esen_lmax2_HIGH_w64_s525000.yaml"),
]


def find_record(width, step):
    for r in LMAX2_INV:
        if r.get("sphere_channels_from_state_dict") == width and r.get("global_step") == step:
            return r
    raise KeyError(f"no lmax2 record for width={width} step={step}")


def build_owner(item) -> OwnerSpec:
    r = find_record(item["width"], item["step"])
    N = r["net_trainable_param_count_from_state_dict"]
    D = r["token_processed"]
    C = KAPPA_L2 * N * D
    tag = f"esen_lmax2_GAP_w{item['width']}_s{item['step']}"
    return OwnerSpec(
        tag=tag, architecture="esen", architecture_label=ARCH_LABEL["esen"],
        budget_label="GAP", width_label=WIDTH_LABEL["esen"], width=item["width"], step=item["step"],
        expected_n=N, D_atom_tokens=D, C_owner=C, C_budget=C, utilization=1.0,
        force_mse_norm=None, ckpt_path=r["checkpoint_path"],
        original_config_path=str(LMAX_STUDY_DIR / "config_snapshots" / item["config_snapshot"]),
        config_name=item["config_snapshot"],
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    dataset = evaluation_data.load_dataset()
    config_indices = list(range(len(dataset)))
    print(f"population=full_val n_configs={len(config_indices)}", flush=True)
    chunks = [evaluation_data.build_batch(dataset, config_indices[i:i + CHUNK], device)
              for i in range(0, len(config_indices), CHUNK)]

    out_path = OUT_DIR / "stage3b1_gap_eval_results.jsonl"
    rows = []
    with open(out_path, "w") as fh:
        for item in SHORTLIST:
            owner = build_owner(item)
            t0 = time.time()
            model, cfg, params = instantiate_model_generic(owner, device, INSTANTIATION_SEED)
            provenance = load_checkpoint_strict_generic(model, owner, params)
            model.net.eval()
            fwd = ESenInterventionForward(model)

            per_graph_sq_means = []
            for b in chunks:
                out, _ = fwd(b, intervention=None)
                target = b.forces / TRAIN_RMSD
                residual = out["forces"] - target
                sq_per_atom = residual.pow(2).sum(dim=-1)
                natoms = b.natoms.to(device)
                ng = int(natoms.shape[0])
                sq_graph = torch.zeros(ng, device=device).index_add_(0, b.batch, sq_per_atom) / natoms.clamp_min(1)
                per_graph_sq_means.extend(sq_graph.detach().cpu().numpy().tolist())

            L_primary = float(np.mean(per_graph_sq_means))
            row = {
                "tag": owner.tag, "ell_max": 2, "width": owner.width, "step": owner.step,
                "N_net_params": owner.expected_n, "D_atom_tokens": owner.D_atom_tokens,
                "kappa_used": KAPPA_L2, "C_flops": owner.C_owner,
                "checkpoint_path": owner.ckpt_path, "checkpoint_sha256": provenance["checkpoint_sha256"],
                "population": "full_val", "n_configs": len(config_indices),
                "L_primary_force_mse_norm_graph_uniform_squared": L_primary,
                "wall_seconds": time.time() - t0,
            }
            rows.append(row)
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            print(f"[{owner.tag}] C={owner.C_owner:.4e} L_primary={L_primary:.6f} wall={row['wall_seconds']:.1f}s", flush=True)

    (OUT_DIR / "stage3b1_gap_eval_results.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
