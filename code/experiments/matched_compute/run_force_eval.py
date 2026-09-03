#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/run_stage3b_force_performance.py
# source revision withheld for anonymous review, original SHA256 74abc60d2a05ab7d04d5761af8a94657c38f48b0d35378718bb028b78f7909d8
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/run_stage3b_force_performance.py
# source revision withheld for anonymous review, original SHA256 74abc60d2a05ab7d04d5761af8a94657c38f48b0d35378718bb028b78f7909d8
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""the lmax=2 vs lmax=4 study Part A: neural force performance for all 6 frozen owners (3 ell_max=4 unchanged
from the initial pilot/3A, 3 ell_max=2 from this stage's own matched-compute search), on the SAME
configurations under the SAME two metrics (task Section 8/9):

  primary   = force_mse_norm: graph-uniform-weighted, SQUARED (reconcile_baseline_metric_exact.py's
              formula; cross-validated ratio=1.000 vs the historically recorded training-time
              metric on the full val set in the fixed-normalization control)
  secondary = the intervention study intervention convention: flat atom-weighted, UNSQUARED L2 norm
              (pipeline/intervention/evaluation_data.py, unchanged)

Runs on the frozen M=1024 nested test population (development population, per task Section 8);
a separate --full-val flag runs the identical computation on the full 27,697-config neutral_val
split for headline confirmation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from pipeline.intervention import lmax_checkpoints, evaluation_data
from pipeline.intervention.esen_intervention import ESenInterventionForward

OUT_DIR = REPO_ROOT / "analysis_outputs" / "stage3b_lmax2_vs_lmax4_2026_08_16"
EXPANDED_DIR = REPO_ROOT / "analysis_outputs" / "four_arch_representation_probe_pilot_2026_08_15" / "expanded_pool"
TRAIN_RMSD = 6.383
CHUNK = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-val", action="store_true", help="run on the full 27,697-config neutral_val instead of M=1024")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    dataset = evaluation_data.load_dataset()

    if args.full_val:
        config_indices = list(range(len(dataset)))
        pop_label = "full_val"
    else:
        expanded = np.load(EXPANDED_DIR / "test_layer_M1024.npz")
        config_indices = [int(c) for c in expanded["config_indices"]]
        pop_label = "M1024"
    print(f"population={pop_label} n_configs={len(config_indices)}", flush=True)

    chunks = [evaluation_data.build_batch(dataset, config_indices[i:i + CHUNK], device)
              for i in range(0, len(config_indices), CHUNK)]

    out_path = OUT_DIR / f"force_performance_results_{pop_label}.jsonl"
    rows = []
    with open(out_path, "w") as fh:
        for tag in sorted(lmax_checkpoints.OWNERS):
            owner = lmax_checkpoints.OWNERS[tag]
            t0 = time.time()
            model, cfg, params, cp = lmax_checkpoints.reconstruct(tag, device=device)
            model.net.eval()
            fwd = ESenInterventionForward(model)

            per_graph_sq_means, per_graph_l2_means = [], []
            per_config_l2_sum, per_config_l2_count = [], []  # for the secondary flat-atom metric's config-level bootstrap
            for b in chunks:
                out, _ = fwd(b, intervention=None)
                target = b.forces / TRAIN_RMSD
                residual = out["forces"] - target
                sq_per_atom = residual.pow(2).sum(dim=-1)
                l2_per_atom = residual.norm(dim=-1)
                natoms = b.natoms.to(device)
                ng = int(natoms.shape[0])
                sq_graph = torch.zeros(ng, device=device).index_add_(0, b.batch, sq_per_atom) / natoms.clamp_min(1)
                l2_graph = torch.zeros(ng, device=device).index_add_(0, b.batch, l2_per_atom) / natoms.clamp_min(1)
                per_graph_sq_means.extend(sq_graph.detach().cpu().numpy().tolist())
                per_graph_l2_means.extend(l2_graph.detach().cpu().numpy().tolist())
                s_arr = torch.zeros(ng, device=device).index_add_(0, b.batch, l2_per_atom)
                per_config_l2_sum.extend(s_arr.detach().cpu().numpy().tolist())
                per_config_l2_count.extend(natoms.detach().cpu().numpy().tolist())

            L_primary_force_mse_norm = float(np.mean(per_graph_sq_means))
            L_graph_uniform_l2 = float(np.mean(per_graph_l2_means))
            L_secondary_flat_atom_l2 = float(np.sum(per_config_l2_sum) / np.sum(per_config_l2_count))

            row = {
                "tag": tag, "ell_max": 4 if "lmax4" in tag else 2,
                "budget": owner.budget_label, "width": owner.width, "step": owner.step,
                "checkpoint_sha256": cp["checkpoint_sha256"], "n_params": owner.expected_n,
                "C_flops": owner.C_owner, "utilization": owner.utilization,
                "population": pop_label, "n_configs": len(config_indices),
                "L_primary_force_mse_norm_graph_uniform_squared": L_primary_force_mse_norm,
                "L_graph_uniform_unsquared": L_graph_uniform_l2,
                "L_secondary_flat_atom_unsquared": L_secondary_flat_atom_l2,
                "per_graph_sq_means": per_graph_sq_means,
                "per_graph_l2_means": per_graph_l2_means,
                "per_config_l2_sum": per_config_l2_sum,
                "per_config_l2_count": per_config_l2_count,
                "config_indices": config_indices,
                "wall_seconds": time.time() - t0,
            }
            rows.append(row)
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            print(f"[{tag}] L_primary={L_primary_force_mse_norm:.6f} L_secondary={L_secondary_flat_atom_l2:.6f} "
                  f"wall={row['wall_seconds']:.1f}s", flush=True)

    (OUT_DIR / f"force_performance_results_{pop_label}.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
