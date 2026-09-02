#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/run_depth_localization_block3_block6.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 20cd5ab347a97dffeddd8c3c9a8a0abd56339ea7fa702369963ce21f0b8a410c
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/run_depth_localization_block3_block6.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 20cd5ab347a97dffeddd8c3c9a8a0abd56339ea7fa702369963ce21f0b8a410c
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Depth-localization follow-up to the block-9 result: the same ell=4 intervention applied at
blocks 3 and 6.

Adds block k=3 and k=6 (the two remaining the irrep-intervention study `depths_block_index` entries not yet swept
across all four canonical ell_max=4 frontier owners) for the SAME four owners already used at
block 9: w10/50k, w16/150k, w20/200k, w24/300k. Reuses, verbatim, the exact validated
intervention/control/metric/population stack -- no new intervention logic, no retraining:

  intervention : pipeline/intervention/esen_intervention.py (InterventionSpec, layer=k
                 is already a free parameter of the existing implementation)
  norm control : pipeline/intervention/esen_norm_control.py (NORM_FIXED: the ablated
                 block-k signal is propagated through the network UNCHANGED to block 11, where
                 the shared final-norm RMS statistic is substituted with the cached BASELINE
                 statistic -- exactly run_stage3a1_norm_control.py's / run_frontier_ell4_
                 sensitivity.py's own convention, generalized over k only)
  population   : four_arch_representation_probe_pilot_2026_08_15/expanded_pool/
                 test_layer_M1024.npz, same frozen M=1024 molecule pool, same config_indices
  metric       : evaluation_data.per_atom_force_error, flat atom-weighted mean (same as every
                 sibling stage in this line of work)
  owners       : w10/50k, w16/150k loaded via checkpoint_loading.reconstruct (tags
                 esen_LOW_w10_s50000 / esen_MID_w16_s150000, same as stage3a/stage3a1/dose
                 response); w20/200k, w24/300k loaded via build_extra_owners() +
                 instantiate_model_generic (same as run_frontier_ell4_sensitivity.py)
  ell          : 4 only (task scope: do not expand to other irreps)
  alphas       : full frozen 5-point grid [1.00, 0.75, 0.50, 0.25, 0.00] (runtime negligible --
                 4 owners x 2 blocks x 5 alphas x M=1024, ~minutes on one A100)

Also records intervention-free block-k magnitude stats (raw ss_by_ell AND the exact l=0-centered
SS_0, matching run_frontier_ell4_centered_ss.py's centering step) so the same degree-balanced
P_bal metric used in the final block-9 four-owner analysis can be computed at k=3/6 without any
extra forward pass.

Validation gates (all must pass before numbers are used), same convention as
run_frontier_ell4_sensitivity.py:
  1. alpha=1.0 identity at the block-k site
  2. sector isolation at alpha=0.5 (only the ell=4 slice changes, scaled by exactly 0.5)
  3. baseline L reproduces each owner's own frozen baseline (block-independent -- intervention=
     None is identical regardless of which block a later intervention will target)
  4. rotation equivariance at alpha=0.5, reported against the model's own un-intervened
     equivariance noise floor
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

from pipeline.intervention import evaluation_data, rotation, checkpoint_loading
from pipeline.intervention.lmax_frontier_checkpoints import build_extra_owners
from pipeline.kernel.frontier_checkpoints import (
    INSTANTIATION_SEED, instantiate_model_generic, load_checkpoint_strict_generic,
)
from pipeline.intervention.esen_intervention import (
    ESenInterventionForward, InterventionSpec, IRREP_SLICES,
)
from pipeline.intervention import esen_norm_control as nc
from pipeline.intervention.run_norm_control import git_state, per_config_accum

OUT_DIR = REPO_ROOT / "analysis_outputs" / "depth_localization_block3_block6_2026_08_22"
EXPANDED_DIR = REPO_ROOT / "analysis_outputs" / "four_arch_representation_probe_pilot_2026_08_15" / "expanded_pool"

CHUNK = 32
LAYERS = [3, 6]
NUM_LAYERS = 12
ELL = 4
ALPHAS = [1.00, 0.75, 0.50, 0.25, 0.00]

# Same 4 canonical ell_max=4 frontier owners as frontier_ell4_sensitivity_2026_08_20 /
# frontier_ell4_degree_balanced_perturbation_2026_08_21, same frozen baseline-L targets (these
# are intervention-free, so block-independent -- identical whichever block a later intervention
# targets).
OWNERS = [
    {"key": "w10/50k", "kind": "checkpoint_loading", "tag": "esen_LOW_w10_s50000",
     "width": 10, "step": 50000, "C_flops": 2.399e16,
     "frozen_L_baseline": 0.023626211951686613},
    {"key": "w16/150k", "kind": "checkpoint_loading", "tag": "esen_MID_w16_s150000",
     "width": 16, "step": 150000, "C_flops": 1.127e17,
     "frozen_L_baseline": None},  # validated against dose-response MID baseline below instead
    {"key": "w20/200k", "kind": "extra_owner", "tag": "esen_lmax4_UPPERMID_w20_s200000",
     "width": 20, "step": 200000, "C_flops": 1.865e17,
     "frozen_L_baseline": 0.008749231251962742},
    {"key": "w24/300k", "kind": "extra_owner", "tag": "esen_lmax4_NEARMAX_w24_s300000",
     "width": 24, "step": 300000, "C_flops": 3.347e17,
     "frozen_L_baseline": 0.007132378582695413},
]

DOSE_PATH = REPO_ROOT / "analysis_outputs" / "stage3_block9_dose_response_2026_08_19" / "dose_response_M1024.jsonl"


def main():
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

    dose_baselines = {}
    for line in DOSE_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["cell_id"] == "baseline":
                dose_baselines[r["budget"]] = r["L_flat"]

    EXTRA = build_extra_owners()
    gstate = git_state()
    out_path = OUT_DIR / (f"depth_localization_smoke_M{m}.jsonl" if args.smoke_test
                          else f"depth_localization_M{m}.jsonl")
    stats_path = OUT_DIR / (f"block_stats_smoke_M{m}.json" if args.smoke_test
                            else f"block_stats_M{m}.json")
    validation = {"baseline_reproduction": {}, "identity_alpha1": {}, "sector_isolation": {},
                  "rotation_equivariance": {}, "checkpoint_metadata": {},
                  "same_config_indices": {"first8": config_indices[:8], "n": len(config_indices)}}
    stats_out = {"purpose": "intervention-free per-block magnitude stats (raw + l=0-centered "
                 "ss_by_ell) for the 4 canonical owners at blocks 3 and 6.",
                 "layers": LAYERS, "m_eval": m, "owners": {}}

    with open(out_path, "w") as fh:
        for spec in OWNERS:
            key, tag = spec["key"], spec["tag"]
            print(f"=== {key} ({tag}) ===", flush=True)
            if spec["kind"] == "checkpoint_loading":
                model, cfg, params, cp = checkpoint_loading.reconstruct(tag, device=device)
            else:
                owner = EXTRA[tag]
                assert owner.width == spec["width"] and owner.step == spec["step"]
                model, cfg, params = instantiate_model_generic(owner, device, INSTANTIATION_SEED)
                cp = load_checkpoint_strict_generic(model, owner, params)
            model.net.eval()
            fwd = ESenInterventionForward(model)
            norm_module = model.net.backbone.norm
            assert int(model.net.backbone.sph_feature_size) == 25

            validation["checkpoint_metadata"][key] = {
                "checkpoint_sha256": cp.get("checkpoint_sha256"), "width": spec["width"],
                "step": spec["step"], "C_flops_frozen": spec["C_flops"]}

            # ---- baseline pass (block-independent): L, cached prenorm block-11 AND both
            #      target blocks, per-block raw + l=0-centered ss_by_ell ----
            t0 = time.time()
            sums_all, counts_all = [], []
            base11 = []
            base_k = {k: [] for k in LAYERS}
            ss_total = {k: 0.0 for k in LAYERS}
            n_total = {k: 0 for k in LAYERS}
            ss_ell = {k: {e: 0.0 for e in IRREP_SLICES} for k in LAYERS}
            ss_ell0_centered = {k: 0.0 for k in LAYERS}
            n_ell = {k: {e: 0 for e in IRREP_SLICES} for k in LAYERS}
            base_first = {}
            for i, b in enumerate(chunks):
                out_b, _ = fwd(b, intervention=None)
                pa = evaluation_data.per_atom_force_error(out_b["forces"], b)
                s, c = per_config_accum(pa, b)
                sums_all += s
                counts_all += c
                base11.append(out_b["node_embedding_per_block"][NUM_LAYERS - 1])
                for k in LAYERS:
                    hk = out_b["node_embedding_per_block"][k]
                    base_k[k].append(hk)
                    if i == 0:
                        base_first[k] = hk
                    ss_total[k] += float((hk.double() ** 2).sum().item())
                    n_total[k] += int(hk.numel())
                    for e, (s0, e0) in IRREP_SLICES.items():
                        sec = hk[:, s0:e0, :].double()
                        ss_ell[k][e] += float((sec ** 2).sum().item())
                        n_ell[k][e] += int(sec.numel())
                    l0 = hk.narrow(1, 0, 1).double()
                    l0_centered = l0 - l0.mean(dim=2, keepdim=True)
                    ss_ell0_centered[k] += float((l0_centered ** 2).sum().item())
            L_base = float(np.sum(sums_all) / max(np.sum(counts_all), 1))
            print(f"  baseline L={L_base:.8f}  ({time.time()-t0:.1f}s)", flush=True)

            stats_out["owners"][key] = {
                "tag": tag, "width": spec["width"], "step": spec["step"],
                "C_flops": spec["C_flops"], "L_baseline": L_base,
                "by_layer": {
                    str(k): {
                        "ss_total": ss_total[k], "n_elements": n_total[k],
                        "ss_by_ell_raw": {str(e): ss_ell[k][e] for e in IRREP_SLICES},
                        "n_elements_by_ell": {str(e): n_ell[k][e] for e in IRREP_SLICES},
                        "ss_ell0_centered": ss_ell0_centered[k],
                        "ss_ell0_centered_vs_raw_ratio": (
                            ss_ell0_centered[k] / ss_ell[k][0] if ss_ell[k][0] else None),
                    } for k in LAYERS
                },
            }

            # ---- baseline reproduction ----
            frozen = spec["frozen_L_baseline"] if spec["frozen_L_baseline"] is not None else dose_baselines["MID"]
            rel = abs(L_base - frozen) / frozen
            validation["baseline_reproduction"][key] = {
                "L_baseline": L_base, "frozen": frozen, "rel_err": rel, "passed": bool(rel < 5e-3)}
            print(f"  repro baseline rel_err={rel:.2e}", flush=True)

            validation["identity_alpha1"][key] = {}
            validation["sector_isolation"][key] = {}
            validation["rotation_equivariance"][key] = {}

            for k in LAYERS:
                # ---- identity at alpha=1 ----
                out1, _ = fwd(chunks[0], intervention=InterventionSpec(layer=k, ell=ELL, alpha=1.0))
                err = (out1["node_embedding_per_block"][k] - base_first[k]).abs().max().item()
                scale = base_first[k].abs().max().clamp_min(1e-8).item()
                validation["identity_alpha1"][key][str(k)] = {"rel": err / scale, "passed": bool(err / scale < 1e-4)}

                # ---- sector isolation at alpha=0.5 ----
                out_iso, _ = fwd(chunks[0], intervention=InterventionSpec(layer=k, ell=ELL, alpha=0.5))
                h_iso = out_iso["node_embedding_per_block"][k]
                s4, e4 = IRREP_SLICES[ELL]
                other = (torch.cat([h_iso[:, :s4, :], h_iso[:, e4:, :]], dim=1) -
                         torch.cat([base_first[k][:, :s4, :], base_first[k][:, e4:, :]], dim=1)).abs().max().item()
                other_scale = torch.cat([base_first[k][:, :s4, :], base_first[k][:, e4:, :]],
                                        dim=1).abs().max().clamp_min(1e-8).item()
                tgt = (h_iso[:, s4:e4, :] - 0.5 * base_first[k][:, s4:e4, :]).abs().max().item()
                tgt_scale = base_first[k][:, s4:e4, :].abs().max().clamp_min(1e-8).item()
                validation["sector_isolation"][key][str(k)] = {
                    "other_sectors_rel": other / other_scale, "target_scaled_rel": tgt / tgt_scale,
                    "passed": bool(other / other_scale < 1e-4 and tgt / tgt_scale < 1e-4)}
                print(f"  [k={k}] identity={validation['identity_alpha1'][key][str(k)]} "
                      f"isolation={validation['sector_isolation'][key][str(k)]}", flush=True)

                # ---- sweep ----
                for alpha in ALPHAS:
                    spec_iv = InterventionSpec(layer=k, ell=ELL, alpha=alpha)
                    t1 = time.time()
                    sums_i, counts_i = [], []
                    q_num, q_den = 0.0, 0.0
                    for b, hb11 in zip(chunks, base11):
                        out_a, diag = fwd(b, intervention=spec_iv)
                        q_num += diag["power_removed"]
                        q_den += diag["power_total_pre"]
                        h_nf = nc.norm_fixed_output(norm_module, out_a["node_embedding_per_block"][NUM_LAYERS - 1], hb11)
                        out_nf = nc.run_heads(model.net, b, h_nf)
                        pa = evaluation_data.per_atom_force_error(out_nf["forces"], b)
                        s, c = per_config_accum(pa, b)
                        sums_i += s
                        counts_i += c
                    L_int = float(np.sum(sums_i) / max(np.sum(counts_i), 1))
                    delta = float(np.log(L_int) - np.log(L_base))
                    q_power = (q_num / q_den) if q_den > 0 else None
                    fh.write(json.dumps({
                        "key": key, "tag": tag, "width": spec["width"], "step": spec["step"],
                        "C_flops": spec["C_flops"], "layer": k, "ell": ELL, "alpha": alpha,
                        "kind": "normfixed", "m_eval": m, "n_configs": len(config_indices),
                        "L_baseline": L_base, "L_int": L_int, "delta": delta, "q_power": q_power,
                        "per_config_sum": sums_i, "per_config_natoms": counts_i,
                        "baseline_per_config_sum": sums_all, "baseline_per_config_natoms": counts_all,
                        "checkpoint_sha256": cp.get("checkpoint_sha256"),
                        "dtype": "float32", "git_head": gstate["head"],
                        "git_n_changed_files": gstate["n_changed"], "wall_seconds": time.time() - t1,
                    }, default=str) + "\n")
                    fh.flush()
                    print(f"  [k={k} alpha={alpha:.2f}] L={L_int:.8f} delta={delta:+.5f}", flush=True)

                # ---- rotation equivariance at alpha=0.5 vs own noise floor ----
                single = evaluation_data.build_batch(dataset, config_indices[:1], device)
                rot_tol = 3e-3

                def normfixed_forward(fwd=fwd, norm_module=norm_module, model=model, k=k):
                    def _f(b):
                        out_b, _ = fwd(b, intervention=None)
                        hb = out_b["node_embedding_per_block"][NUM_LAYERS - 1]
                        out_a, _ = fwd(b, intervention=InterventionSpec(layer=k, ell=ELL, alpha=0.5))
                        ha = out_a["node_embedding_per_block"][NUM_LAYERS - 1]
                        return nc.run_heads(model.net, b, nc.norm_fixed_output(norm_module, ha, hb))
                    return _f

                def baseline_forward(fwd=fwd):
                    def _f(b):
                        out_b, _ = fwd(b, intervention=None)
                        return {"forces": out_b["forces"]}
                    return _f

                seed_results = {}
                for seed in [999, 4242, 77]:
                    R = rotation.random_rotation(seed=seed, dtype=single.pos.dtype).to(device)
                    chk = rotation.check_force_equivariance(normfixed_forward(), single, R, atol=rot_tol)
                    chk0 = rotation.check_force_equivariance(baseline_forward(), single, R, atol=rot_tol)
                    seed_results[str(seed)] = {
                        "max_rel_error": chk.max_rel_error, "passed": bool(chk.passed),
                        "baseline_noise_floor_max_rel_error": chk0.max_rel_error,
                        "excess_over_noise_floor": chk.max_rel_error - chk0.max_rel_error}
                validation["rotation_equivariance"][key][str(k)] = {
                    "tol": rot_tol, "by_seed": seed_results,
                    "passed_all_seeds": all(v["passed"] for v in seed_results.values()),
                    "max_excess_over_noise_floor": max(v["excess_over_noise_floor"] for v in seed_results.values())}
                print(f"  [k={k}] rot: {seed_results}", flush=True)

            del model, fwd
            if device == "cuda":
                torch.cuda.empty_cache()

    stats_path.write_text(json.dumps(stats_out, indent=2) + "\n")
    val_path = OUT_DIR / (f"validation_smoke_M{m}.json" if args.smoke_test else f"validation_M{m}.json")
    val_path.write_text(json.dumps(validation, indent=2, default=str) + "\n")
    print(f"DONE -> {out_path}\n     -> {stats_path}\n     -> {val_path}", flush=True)


if __name__ == "__main__":
    main()
