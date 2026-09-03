#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/run_fvs_controlA_block9_stats.py
# source revision withheld for anonymous review, original SHA256 fe13f75d3df0de38aba568e9b256643fe787401229224926212c75c8d28ec032
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/run_fvs_controlA_block9_stats.py
# source revision withheld for anonymous review, original SHA256 fe13f75d3df0de38aba568e9b256643fe787401229224926212c75c8d28ec032
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Block-9 baseline representation-magnitude statistics: the per-degree activation power that
the degree-balanced perturbation magnitude is normalized by.

The existing block-9 alpha dose-response run (analysis_outputs/stage3_block9_dose_response_2026_08_19)
already stores, per (tier, ell, alpha) cell, the RATIO

    q_power = sum_chunks ||h_int - h_base||_F^2 / sum_chunks ||h_base||_F^2      (block-9 output)

from which P_total = sqrt(q_power) and P_sector = 1 - alpha follow exactly, with NO new forward
passes. What it does NOT store is the ABSOLUTE scale, i.e. RMS(h_base) over the complete [N,25,C]
block-9 tensor and RMS(h_base^(ell)) over each degree-ell sector. This script computes ONLY those
baseline (intervention-free) statistics, on exactly the same frozen owners, population, chunking
and dtype as the dose-response run. No intervention is applied anywhere in this script.

Conventions frozen and reused verbatim:
  owners      : analysis_outputs/stage3a_esen_irrep_intervention_2026_08_16/stage3a_manifest.json
                (LOW=esen_LOW_w10_s50000, MID=esen_MID_w16_s150000, HIGH=esen_HIGH_w40_s275000)
  population  : four_arch_representation_probe_pilot_2026_08_15/expanded_pool/test_layer_M1024.npz,
                first M config_indices
  chunking    : CHUNK=32 (identical to run_stage3_block9_dose_response.py)
  block       : LAYER=9 output (node_embedding_per_block[9]), i.e. the intervention SITE
  metric      : per_atom_force_error / flat atom mean (only used to re-verify L_original)
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

from pipeline.intervention import checkpoint_loading, evaluation_data
from pipeline.intervention.esen_intervention import ESenInterventionForward, IRREP_SLICES
from pipeline.intervention.run_norm_control import git_state, per_config_accum

OUT_DIR = REPO_ROOT / "analysis_outputs" / "final_validation_sprint_2026_08_20" / "control_A_perturbation_magnitude"
EXPANDED_DIR = REPO_ROOT / "analysis_outputs" / "four_arch_representation_probe_pilot_2026_08_15" / "expanded_pool"
INTERVENTION_MANIFEST = REPO_ROOT / "analysis_outputs" / "stage3a_esen_irrep_intervention_2026_08_16" / "stage3a_manifest.json"
DOSE_JSONL = REPO_ROOT / "analysis_outputs" / "stage3_block9_dose_response_2026_08_19" / "dose_response_M1024.jsonl"

CHUNK = 32
LAYER = 9

# frozen dose-response baselines that this script MUST reproduce before its stats are trusted
FROZEN_L_ORIGINAL = {"LOW": 0.023626, "MID": 0.011179, "HIGH": 0.006124}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(INTERVENTION_MANIFEST.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    dataset = evaluation_data.load_dataset()
    expanded = np.load(EXPANDED_DIR / "test_layer_M1024.npz")
    m = 8 if args.smoke_test else args.m
    config_indices = [int(c) for c in expanded["config_indices"][:m]]
    chunks = [evaluation_data.build_batch(dataset, config_indices[i:i + CHUNK], device)
              for i in range(0, len(config_indices), CHUNK)]
    print(f"{len(chunks)} chunks, {len(config_indices)} molecules", flush=True)

    gstate = git_state()
    out = {
        "purpose": "Control A: baseline block-9 representation magnitude (no intervention applied)",
        "layer": LAYER, "m_eval": m, "n_configs": len(config_indices),
        "config_indices_sha_first8": [int(c) for c in config_indices[:8]],
        "git_head": gstate["head"], "git_n_changed_files": gstate["n_changed"],
        "tiers": {},
    }

    for budget, owner in manifest["esen_owners"].items():
        tag = owner["tag"]
        print(f"=== {budget} {tag} ===", flush=True)
        t0 = time.time()
        model, cfg, params, cp = checkpoint_loading.reconstruct(tag, device=device)
        model.net.eval()
        fwd = ESenInterventionForward(model)

        ss_total = 0.0
        n_total = 0
        ss_ell = {e: 0.0 for e in IRREP_SLICES}
        n_ell = {e: 0 for e in IRREP_SLICES}
        sums_all, counts_all = [], []
        for b in chunks:
            out_b, _ = fwd(b, intervention=None)
            h9 = out_b["node_embedding_per_block"][LAYER]
            ss_total += float((h9.double() ** 2).sum().item())
            n_total += int(h9.numel())
            for e, (s, en) in IRREP_SLICES.items():
                sec = h9[:, s:en, :]
                ss_ell[e] += float((sec.double() ** 2).sum().item())
                n_ell[e] += int(sec.numel())
            pa = evaluation_data.per_atom_force_error(out_b["forces"], b)
            sa, ca = per_config_accum(pa, b)
            sums_all += sa
            counts_all += ca
        L_original = float(np.sum(sums_all) / max(np.sum(counts_all), 1))

        rms_total = float(np.sqrt(ss_total / n_total))
        H = {str(e): float(np.sqrt(ss_ell[e] / max(n_ell[e], 1))) for e in IRREP_SLICES}
        frac = {str(e): float(ss_ell[e] / ss_total) for e in IRREP_SLICES}
        rel = FROZEN_L_ORIGINAL.get(budget)
        rel_err = abs(L_original - rel) / rel if rel else None

        out["tiers"][budget] = {
            "tag": tag, "checkpoint_path": owner["checkpoint_path"],
            "width": owner["width"], "step": owner["step"],
            "expected_param_count": owner["expected_param_count"],
            "sph_feature_size": int(model.net.backbone.sph_feature_size),
            "sphere_channels": int(model.net.backbone.sphere_channels),
            "L_original": L_original,
            "frozen_L_original_dose_response": rel,
            "L_original_rel_err_vs_frozen": rel_err,
            "L_original_reproduction_passed": bool(rel_err is not None and rel_err < 5e-3),
            "ss_total_block9": ss_total, "n_elements_block9": n_total,
            "rms_total_block9": rms_total,
            "ss_by_ell": {str(e): ss_ell[e] for e in IRREP_SLICES},
            "n_elements_by_ell": {str(e): n_ell[e] for e in IRREP_SLICES},
            "H_ell_sector_rms": H,
            "power_fraction_by_ell": frac,
            "wall_seconds": time.time() - t0,
        }
        print(json.dumps(out["tiers"][budget], indent=1), flush=True)
        del model, fwd
        if device == "cuda":
            torch.cuda.empty_cache()

    path = OUT_DIR / (f"block9_baseline_stats_smoke_M{m}.json" if args.smoke_test
                      else f"block9_baseline_stats_M{m}.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"DONE -> {path}", flush=True)


if __name__ == "__main__":
    main()
