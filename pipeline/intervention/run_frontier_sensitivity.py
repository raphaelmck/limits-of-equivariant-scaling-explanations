#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/run_frontier_ell4_sensitivity.py
# source revision withheld for anonymous review, original SHA256 a299baea1b25b3659c2028bb7800ac971458aa0f720c28db0ec9dd68a1be7cdf
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/run_frontier_ell4_sensitivity.py
# source revision withheld for anonymous review, original SHA256 a299baea1b25b3659c2028bb7800ac971458aa0f720c28db0ec9dd68a1be7cdf
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Block-9 ell=4 alpha sweep for the two frontier checkpoints that do not already have a frozen
five-point sweep: width 20 at step 200000, and width 24 at step 300000.

w10/s50000 and w16/s150000 are NOT re-run -- their full 5-alpha ell=4 sweep already exists,
with per-configuration sums and q_power, in
`stage3_block9_dose_response_2026_08_19/dose_response_M1024.jsonl` (as tiers LOW and MID; the
checkpoints are byte-identical owners).

Everything is the validated, frozen convention, reused verbatim:
  intervention : pipeline/intervention/esen_intervention.py (InterventionSpec / apply_intervention)
  norm control : pipeline/intervention/esen_norm_control.py (NORM_FIXED)
  population   : four_arch_representation_probe_pilot_2026_08_15/expanded_pool/test_layer_M1024.npz, first M
  metric       : evaluation_data.per_atom_force_error, flat atom-weighted mean
  chunking     : CHUNK=32, float32, eval()
  block        : LAYER = 9
  alphas       : [1.00, 0.75, 0.50, 0.25, 0.00]  (identical to the frozen dose response)

This script ALSO records the intervention-free block-9 magnitude statistics (total and per-degree
sums of squares) for these two owners, which is what turns the recorded q_power into an absolute
P_total and lets q_power be cross-validated against (1-alpha)^2 * SS_ell/SS_total.

Validation gates (all must pass before the numbers are used):
  1. baseline L reproduces the frozen gap-erasure L4 for that owner
  2. alpha in {0.5, 0.0} reproduces the frozen gap-erasure intervention_damage for that owner
  3. alpha = 1.0 identity at the block-9 site
  4. sector isolation at alpha = 0.5 (only the ell=4 slice changes, and it scales by exactly 0.5)
  5. rotation equivariance at alpha = 0.5, reported against the model's own un-intervened
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

from pipeline.intervention import evaluation_data, rotation
from pipeline.intervention.lmax_frontier_checkpoints import build_extra_owners
from pipeline.kernel.frontier_checkpoints import (
    INSTANTIATION_SEED, instantiate_model_generic, load_checkpoint_strict_generic,
)
from pipeline.intervention.esen_intervention import (
    ESenInterventionForward, InterventionSpec, IRREP_SLICES,
)
from pipeline.intervention import esen_norm_control as nc
from pipeline.intervention.run_norm_control import git_state, per_config_accum

OUT_DIR = REPO_ROOT / "analysis_outputs" / "frontier_ell4_sensitivity_2026_08_20"
EXPANDED_DIR = REPO_ROOT / "analysis_outputs" / "four_arch_representation_probe_pilot_2026_08_15" / "expanded_pool"

CHUNK = 32
LAYER = 9
NUM_LAYERS = 12
ELL = 4
ALPHAS = [1.00, 0.75, 0.50, 0.25, 0.00]

# The two owners missing a 5-point sweep, with their frozen the matched-frontier study identity and the
# gap-erasure values they must reproduce.
OWNERS = {
    "esen_lmax4_UPPERMID_w20_s200000": {
        "label": "w20/200k", "width": 20, "step": 200000, "C_flops": 1.865e17,
        "frozen_L_baseline": 0.008749231251962742,
        "frozen_damage": {0.50: 0.45723088339869733, 0.00: 0.9065906139002413},
    },
    "esen_lmax4_NEARMAX_w24_s300000": {
        "label": "w24/300k", "width": 24, "step": 300000, "C_flops": 3.347e17,
        "frozen_L_baseline": 0.007132378582695413,
        "frozen_damage": {0.50: 0.5260657076595541, 0.00: 0.998497759179148},
    },
}

EXTRA = build_extra_owners()


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

    gstate = git_state()
    out_path = OUT_DIR / (f"frontier_ell4_smoke_M{m}.jsonl" if args.smoke_test
                          else f"frontier_ell4_M{m}.jsonl")
    stats_path = OUT_DIR / (f"block9_stats_smoke_M{m}.json" if args.smoke_test
                            else f"block9_stats_M{m}.json")
    validation = {"baseline_reproduction": {}, "frozen_damage_reproduction": {},
                  "identity_alpha1": {}, "sector_isolation": {}, "rotation_equivariance": {},
                  "checkpoint_metadata": {},
                  "same_config_indices": {"first8": config_indices[:8], "n": len(config_indices)}}
    stats_out = {"purpose": "intervention-free block-9 magnitude stats for the two owners run here",
                 "layer": LAYER, "m_eval": m, "tiers": {}}

    with open(out_path, "w") as fh:
        for tag, meta in OWNERS.items():
            print(f"=== {tag} ({meta['label']}) ===", flush=True)
            owner = EXTRA[tag]
            assert owner.width == meta["width"] and owner.step == meta["step"], (owner.width, owner.step)
            model, cfg, params = instantiate_model_generic(owner, device, INSTANTIATION_SEED)
            cp = load_checkpoint_strict_generic(model, owner, params)
            model.net.eval()
            fwd = ESenInterventionForward(model)
            norm_module = model.net.backbone.norm
            assert int(model.net.backbone.sph_feature_size) == 25

            validation["checkpoint_metadata"][tag] = {
                k: cp.get(k) for k in ("checkpoint_path", "checkpoint_sha256",
                                       "checkpoint_global_step", "checkpoint_net_state_parameter_count")}
            validation["checkpoint_metadata"][tag].update(
                {"width": owner.width, "step": owner.step, "C_flops_frozen": meta["C_flops"],
                 "expected_n": owner.expected_n, "D_atom_tokens": owner.D_atom_tokens})

            # ---- baseline pass: L, per-config sums, cached block-11 prenorm, block-9 magnitudes ----
            t0 = time.time()
            sums_all, counts_all = [], []
            base11 = []
            ss_total, n_total = 0.0, 0
            ss_ell = {e: 0.0 for e in IRREP_SLICES}
            n_ell = {e: 0 for e in IRREP_SLICES}
            base9_first = None
            for i, b in enumerate(chunks):
                out_b, _ = fwd(b, intervention=None)
                pa = evaluation_data.per_atom_force_error(out_b["forces"], b)
                s, c = per_config_accum(pa, b)
                sums_all += s
                counts_all += c
                base11.append(out_b["node_embedding_per_block"][NUM_LAYERS - 1])
                h9 = out_b["node_embedding_per_block"][LAYER]
                if i == 0:
                    base9_first = h9
                ss_total += float((h9.double() ** 2).sum().item())
                n_total += int(h9.numel())
                for e, (s0, e0) in IRREP_SLICES.items():
                    sec = h9[:, s0:e0, :]
                    ss_ell[e] += float((sec.double() ** 2).sum().item())
                    n_ell[e] += int(sec.numel())
            L_base = float(np.sum(sums_all) / max(np.sum(counts_all), 1))
            print(f"  baseline L={L_base:.8f}  ({time.time()-t0:.1f}s)", flush=True)

            stats_out["tiers"][tag] = {
                "label": meta["label"], "width": owner.width, "step": owner.step,
                "C_flops": meta["C_flops"], "L_original": L_base,
                "ss_total_block9": ss_total, "n_elements_block9": n_total,
                "rms_total_block9": float(np.sqrt(ss_total / n_total)),
                "ss_by_ell": {str(e): ss_ell[e] for e in IRREP_SLICES},
                "n_elements_by_ell": {str(e): n_ell[e] for e in IRREP_SLICES},
                "H_ell_sector_rms": {str(e): float(np.sqrt(ss_ell[e] / max(n_ell[e], 1))) for e in IRREP_SLICES},
                "power_fraction_by_ell": {str(e): float(ss_ell[e] / ss_total) for e in IRREP_SLICES},
            }

            # ---- V1 baseline reproduction ----
            rel = abs(L_base - meta["frozen_L_baseline"]) / meta["frozen_L_baseline"]
            validation["baseline_reproduction"][tag] = {
                "L_baseline": L_base, "frozen_L4": meta["frozen_L_baseline"],
                "rel_err": rel, "passed": bool(rel < 5e-3)}
            print(f"  repro baseline rel_err={rel:.2e}", flush=True)

            # ---- V3 alpha=1 identity ----
            out1, _ = fwd(chunks[0], intervention=InterventionSpec(layer=LAYER, ell=ELL, alpha=1.0))
            err = (out1["node_embedding_per_block"][LAYER] - base9_first).abs().max().item()
            scale = base9_first.abs().max().clamp_min(1e-8).item()
            validation["identity_alpha1"][tag] = {"rel": err / scale, "passed": bool(err / scale < 1e-4)}

            # ---- V4 sector isolation at alpha=0.5 ----
            out_iso, _ = fwd(chunks[0], intervention=InterventionSpec(layer=LAYER, ell=ELL, alpha=0.5))
            h_iso = out_iso["node_embedding_per_block"][LAYER]
            s4, e4 = IRREP_SLICES[ELL]
            other = (torch.cat([h_iso[:, :s4, :], h_iso[:, e4:, :]], dim=1) -
                     torch.cat([base9_first[:, :s4, :], base9_first[:, e4:, :]], dim=1)).abs().max().item()
            other_scale = torch.cat([base9_first[:, :s4, :], base9_first[:, e4:, :]],
                                    dim=1).abs().max().clamp_min(1e-8).item()
            tgt = (h_iso[:, s4:e4, :] - 0.5 * base9_first[:, s4:e4, :]).abs().max().item()
            tgt_scale = base9_first[:, s4:e4, :].abs().max().clamp_min(1e-8).item()
            validation["sector_isolation"][tag] = {
                "other_sectors_rel": other / other_scale, "target_scaled_rel": tgt / tgt_scale,
                "passed": bool(other / other_scale < 1e-4 and tgt / tgt_scale < 1e-4)}
            print(f"  identity={validation['identity_alpha1'][tag]} isolation={validation['sector_isolation'][tag]}",
                  flush=True)

            # ---- the sweep ----
            damage_seen = {}
            for alpha in ALPHAS:
                spec = InterventionSpec(layer=LAYER, ell=ELL, alpha=alpha)
                t1 = time.time()
                sums_i, counts_i = [], []
                q_num, q_den = 0.0, 0.0
                for b, hb11 in zip(chunks, base11):
                    out_a, diag = fwd(b, intervention=spec)
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
                damage_seen[alpha] = delta
                fh.write(json.dumps({
                    "tag": tag, "label": meta["label"], "width": owner.width, "step": owner.step,
                    "C_flops": meta["C_flops"], "layer": LAYER, "ell": ELL, "alpha": alpha,
                    "kind": "normfixed", "m_eval": m, "n_configs": len(config_indices),
                    "L_baseline": L_base, "L_int": L_int, "delta": delta, "q_power": q_power,
                    "per_config_sum": sums_i, "per_config_natoms": counts_i,
                    "baseline_per_config_sum": sums_all, "baseline_per_config_natoms": counts_all,
                    "checkpoint_sha256": cp.get("checkpoint_sha256"),
                    "dtype": "float32", "git_head": gstate["head"],
                    "git_n_changed_files": gstate["n_changed"], "wall_seconds": time.time() - t1,
                }, default=str) + "\n")
                fh.flush()
                print(f"  [alpha={alpha:.2f}] L={L_int:.8f} delta={delta:+.5f} "
                      f"P_total={np.sqrt(q_power) if q_power else 0.0:.5f}", flush=True)

            # ---- V2 frozen damage reproduction ----
            repro = {}
            for a, target in meta["frozen_damage"].items():
                got = damage_seen[a]
                r = abs(got - target) / abs(target)
                repro[f"alpha{a:.2f}"] = {"delta": got, "frozen": target, "rel_err": r,
                                          "passed": bool(r < 5e-3)}
            validation["frozen_damage_reproduction"][tag] = repro
            print(f"  repro frozen damage: {repro}", flush=True)

            # ---- V5 rotation equivariance vs its own noise floor ----
            single = evaluation_data.build_batch(dataset, config_indices[:1], device)
            rot_tol = 3e-3

            def normfixed_forward(fwd=fwd, norm_module=norm_module, model=model):
                def _f(b):
                    out_b, _ = fwd(b, intervention=None)
                    hb = out_b["node_embedding_per_block"][NUM_LAYERS - 1]
                    out_a, _ = fwd(b, intervention=InterventionSpec(layer=LAYER, ell=ELL, alpha=0.5))
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
            validation["rotation_equivariance"][tag] = {
                "tol": rot_tol, "by_seed": seed_results,
                "passed_all_seeds": all(v["passed"] for v in seed_results.values()),
                "max_excess_over_noise_floor": max(v["excess_over_noise_floor"] for v in seed_results.values())}
            print(f"  rot: {seed_results}", flush=True)

            del model, fwd
            if device == "cuda":
                torch.cuda.empty_cache()

    stats_path.write_text(json.dumps(stats_out, indent=2) + "\n")
    val_path = OUT_DIR / (f"validation_smoke_M{m}.json" if args.smoke_test else f"validation_M{m}.json")
    val_path.write_text(json.dumps(validation, indent=2, default=str) + "\n")
    print(f"DONE -> {out_path}\n     -> {stats_path}\n     -> {val_path}", flush=True)


if __name__ == "__main__":
    main()
