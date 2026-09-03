#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/run_frontier_ell4_centered_ss.py
# source revision withheld for anonymous review, original SHA256 3b36d48454105e701ad4d087b150b7744341582212e627d3dbdd65c1a3179aa6
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/run_frontier_ell4_centered_ss.py
# source revision withheld for anonymous review, original SHA256 3b36d48454105e701ad4d087b150b7744341582212e627d3dbdd65c1a3179aa6
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Forward-pass measurement of the per-degree activation power with degree-0 centering, which
closes the centering caveat in the degree-balanced perturbation magnitude.

Baseline-ONLY forward pass (intervention=None throughout -- no intervention rerun, no training)
for the same 4 audited ell_max=4 owners (w10/50k, w16/150k, w20/200k, w24/300k) on the SAME frozen
M=1024 molecule pool, computing the block-9 (LAYER=9, node_embedding_per_block[9], the
intervention site itself) per-degree sum-of-squares -- but this time applying the SAME l=0
per-atom, per-channel centering step `rms_norm_sh` itself applies before computing its
data-dependent scale (`esen_norm_control.compute_centered_and_rms`, steps 1-2: subtract the
per-atom mean over the channel axis of the single l=0 coefficient, leave every other degree
untouched). Degrees 1-4 are byte-identical to the existing (uncentered) `ss_by_ell` -- centering
in this norm module only ever touches the l=0 slot -- so only `ss_by_ell["0"]` changes here; it is
still recomputed from scratch (not derived) as the cheapest correct way to get it exactly right.

This produces the exact SAME QUADRATIC FORM eSEN's rms_norm_sh uses for its degree-balanced scale
statistic (component normalization, std_balance_degrees=True, l=0 centered) -- it is NOT a rerun
of the full normalization operation (no affine weight/bias, no eps, no final -1/2 power is applied
here; those cancel in the P_bal ratio exactly as documented in the prior REPORT.md and are not
needed to get SS_0(centered) right).
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

from pipeline.intervention import evaluation_data, checkpoint_loading
from pipeline.intervention.lmax_frontier_checkpoints import build_extra_owners
from pipeline.kernel.frontier_checkpoints import (
    INSTANTIATION_SEED, instantiate_model_generic, load_checkpoint_strict_generic,
)
from pipeline.intervention.esen_intervention import ESenInterventionForward, IRREP_SLICES
from pipeline.intervention.run_norm_control import git_state

OUT_DIR = REPO_ROOT / "analysis_outputs" / "frontier_ell4_degree_balanced_perturbation_2026_08_21"
EXPANDED_DIR = REPO_ROOT / "analysis_outputs" / "four_arch_representation_probe_pilot_2026_08_15" / "expanded_pool"

CHUNK = 32
LAYER = 9

OWNERS = [
    {"key": "w10/50k",  "kind": "checkpoint_loading", "tag": "esen_LOW_w10_s50000",
     "width": 10, "step": 50000, "C_flops": 2.399e16,
     "frozen_L_baseline": 0.023626211951686613},
    {"key": "w16/150k", "kind": "checkpoint_loading", "tag": "esen_MID_w16_s150000",
     "width": 16, "step": 150000, "C_flops": 1.127e17,
     "frozen_L_baseline": None},  # checked against dose_response instead, see below
    {"key": "w20/200k", "kind": "extra_owner", "tag": "esen_lmax4_UPPERMID_w20_s200000",
     "width": 20, "step": 200000, "C_flops": 1.865e17,
     "frozen_L_baseline": 0.008749231413080481},
    {"key": "w24/300k", "kind": "extra_owner", "tag": "esen_lmax4_NEARMAX_w24_s300000",
     "width": 24, "step": 300000, "C_flops": 3.347e17,
     "frozen_L_baseline": 0.007132378582695413},
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1024, choices=[8, 1024])
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "logs").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    dataset = evaluation_data.load_dataset()
    expanded = np.load(EXPANDED_DIR / "test_layer_M1024.npz")
    m = 8 if args.smoke_test else args.m
    config_indices = [int(c) for c in expanded["config_indices"][:m]]
    chunks = [evaluation_data.build_batch(dataset, config_indices[i:i + CHUNK], device)
              for i in range(0, len(config_indices), CHUNK)]
    print(f"{len(chunks)} chunks, {len(config_indices)} molecules", flush=True)

    EXTRA = build_extra_owners()
    gstate = git_state()
    out_path = OUT_DIR / (f"centered_ss_smoke_M{m}.json" if args.smoke_test else f"centered_ss_M{m}.json")

    dose_path = REPO_ROOT / "analysis_outputs" / "stage3_block9_dose_response_2026_08_19" / "dose_response_M1024.jsonl"
    dose_baselines = {}
    for line in dose_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["cell_id"] == "baseline":
                dose_baselines[r["budget"]] = r["L_flat"]

    out = {"purpose": "l=0-centered block-9 per-degree SS for the exact-quadratic-form P_bal "
                       "denominator (closes the centering caveat).", "layer": LAYER, "m_eval": m,
           "git_head": gstate["head"], "git_n_changed_files": gstate["n_changed"], "owners": {}}

    for spec in OWNERS:
        tag = spec["tag"]
        key = spec["key"]
        print(f"=== {key} ({tag}) ===", flush=True)
        t0 = time.time()
        if spec["kind"] == "checkpoint_loading":
            model, cfg, params, cp = checkpoint_loading.reconstruct(tag, device=device)
        else:
            owner = EXTRA[tag]
            assert owner.width == spec["width"] and owner.step == spec["step"]
            model, cfg, params = instantiate_model_generic(owner, device, INSTANTIATION_SEED)
            cp = load_checkpoint_strict_generic(model, owner, params)
        model.net.eval()
        fwd = ESenInterventionForward(model)
        assert int(model.net.backbone.sph_feature_size) == 25

        sums_all, counts_all = [], []
        ss_total, n_total = 0.0, 0
        ss_ell_raw = {e: 0.0 for e in IRREP_SLICES}
        ss_ell_centered0 = 0.0
        n_ell = {e: 0 for e in IRREP_SLICES}
        for b in chunks:
            out_b, _ = fwd(b, intervention=None)
            pa = evaluation_data.per_atom_force_error(out_b["forces"], b)
            from pipeline.intervention.run_norm_control import per_config_accum
            s, c = per_config_accum(pa, b)
            sums_all += s
            counts_all += c
            h9 = out_b["node_embedding_per_block"][LAYER]

            ss_total += float((h9.double() ** 2).sum().item())
            n_total += int(h9.numel())
            for e, (s0, e0) in IRREP_SLICES.items():
                sec = h9[:, s0:e0, :].double()
                ss_ell_raw[e] += float((sec ** 2).sum().item())
                n_ell[e] += int(sec.numel())

            # ---- l=0 centering, EXACT match to esen_norm_control.compute_centered_and_rms ----
            l0 = h9.narrow(1, 0, 1).double()               # [N,1,C]
            l0_mean = l0.mean(dim=2, keepdim=True)          # [N,1,1] -- per-atom mean over channels
            l0_centered = l0 - l0_mean
            ss_ell_centered0 += float((l0_centered ** 2).sum().item())

        L_base = float(np.sum(sums_all) / max(np.sum(counts_all), 1))

        # ---- validation: reproduce frozen baseline L for this owner ----
        if spec["frozen_L_baseline"] is not None:
            frozen = spec["frozen_L_baseline"]
        else:
            frozen = dose_baselines["MID"]
        rel = abs(L_base - frozen) / frozen
        print(f"  L_base={L_base:.8f} frozen={frozen:.8f} rel_err={rel:.2e} "
              f"({time.time()-t0:.1f}s)", flush=True)

        out["owners"][key] = {
            "tag": tag, "width": spec["width"], "step": spec["step"], "C_flops": spec["C_flops"],
            "L_baseline": L_base, "frozen_L_baseline": frozen,
            "baseline_reproduction_rel_err": rel, "baseline_reproduction_passed": bool(rel < 5e-3),
            "ss_total_block9_raw": ss_total, "n_elements_block9": n_total,
            "ss_by_ell_raw": {str(e): ss_ell_raw[e] for e in IRREP_SLICES},
            "n_elements_by_ell": {str(e): n_ell[e] for e in IRREP_SLICES},
            "ss_ell0_centered": ss_ell_centered0,
            "ss_ell0_centered_vs_raw_ratio": ss_ell_centered0 / ss_ell_raw[0] if ss_ell_raw[0] else None,
        }

    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"DONE -> {out_path}", flush=True)
    for k, v in out["owners"].items():
        print(f"  {k}: SS0_raw={v['ss_by_ell_raw']['0']:.4e}  SS0_centered={v['ss_ell0_centered']:.4e}  "
              f"ratio={v['ss_ell0_centered_vs_raw_ratio']:.6f}  baseline_ok={v['baseline_reproduction_passed']}")


if __name__ == "__main__":
    main()
