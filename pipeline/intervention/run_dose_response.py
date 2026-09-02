#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/run_stage3_block9_dose_response.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 17e0d3ea95b5b05e0b3a4d3f686fa0ecc0d940f3a7cda48da227bf6f2a00e4df
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/run_stage3_block9_dose_response.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 17e0d3ea95b5b05e0b3a4d3f686fa0ecc0d940f3a7cda48da227bf6f2a00e4df
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Block-9 alpha dose-response for the ell=4 intervention.

Tests whether the validated block-9 NORM_FIXED high-order irrep dependence is a genuine graded
functional dependence rather than an artifact of the extreme alpha=0 ablation. Reuses, verbatim,
`InterventionSpec`/`apply_intervention` alpha-scaling convention
(pipeline/intervention/esen_intervention.py) and the fixed-normalization counterfactual
(pipeline/intervention/esen_norm_control.py) -- no new intervention logic, only a sweep
over alpha at fixed layer=9 using the SAME frozen LOW/MID/HIGH eSEN owners and M<=1024
population as the fixed-normalization control's run_stage3a1_norm_control.py.

alpha=1.0 must be identity (no masking); alpha=0.0 must exactly reproduce
norm_control_M1024.jsonl's normfixed_L9_ell{2,3,4} cells.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from pipeline.intervention import checkpoint_loading, rotation, evaluation_data
from pipeline.intervention.esen_intervention import ESenInterventionForward, InterventionSpec
from pipeline.intervention import esen_norm_control as nc
from pipeline.intervention.run_norm_control import git_state, per_config_accum

OUT_DIR = REPO_ROOT / "analysis_outputs" / "stage3_block9_dose_response_2026_08_19"
EXPANDED_DIR = REPO_ROOT / "analysis_outputs" / "four_arch_representation_probe_pilot_2026_08_15" / "expanded_pool"
INTERVENTION_MANIFEST = REPO_ROOT / "analysis_outputs" / "stage3a_esen_irrep_intervention_2026_08_16" / "stage3a_manifest.json"
NORM_CONTROL_M1024 = REPO_ROOT / "analysis_outputs" / "stage3a1_norm_control_temporal_2026_08_16" / "norm_control_M1024.jsonl"

CHUNK = 32
LAYER = 9
ELLS = [2, 3, 4]
ALPHAS = [1.00, 0.75, 0.50, 0.25, 0.00]
NUM_LAYERS = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1024, choices=[8, 1024])
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
    print(f"{len(chunks)} chunks of <= {CHUNK} molecules, {len(config_indices)} total", flush=True)

    out_path = OUT_DIR / (f"dose_response_smoke_M{m}.jsonl" if args.smoke_test else f"dose_response_M{m}.jsonl")
    existing: dict[tuple[str, str], dict] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing[(r["budget"], r["cell_id"])] = r
        print(f"resuming: {len(existing)} already done", flush=True)

    gstate = git_state()

    def write_row(fh, budget, owner, cell_id, layer, ell, alpha, sums, counts, L_original, wall,
                  disp_sum, disp_count, extra=None):
        L = float(np.sum(sums) / max(np.sum(counts), 1))
        ratio = L / L_original if L_original else float("nan")
        log_ratio = float(np.log(ratio)) if ratio > 0 else float("nan")
        force_displacement = float(disp_sum / max(disp_count, 1))
        row = {
            "budget": budget, "tag": owner["tag"], "architecture": "esen",
            "checkpoint_path": owner["checkpoint_path"], "width": owner["width"], "step": owner["step"],
            "cell_id": cell_id, "kind": "normfixed_dose_response", "layer": layer,
            "depth_z_corrected": None if layer is None else round((layer + 1) / NUM_LAYERS, 4),
            "ell": ell, "alpha": alpha, "m_eval": m, "n_configs": len(config_indices),
            "L_flat": L, "L_original": L_original, "abs_diff": L - L_original, "ratio": ratio,
            "log_ratio": log_ratio, "per_config_sum": sums, "per_config_natoms": counts,
            "force_displacement_mean": force_displacement,
            "dtype": "float32", "git_head": gstate["head"], "git_n_changed_files": gstate["n_changed"],
            "wall_seconds": wall,
        }
        if extra:
            row.update(extra)
        fh.write(json.dumps(row, default=str) + "\n")
        fh.flush()
        print(f"  [{budget} {cell_id}] L={L:.6f} ratio={ratio:.4f} log_ratio={log_ratio:+.4f} "
              f"disp={force_displacement:.6f} wall={wall:.1f}s", flush=True)

    validation = {"identity_alpha1": {}, "sector_isolation": {}, "rotation_equivariance_alpha0.5": {}}

    with open(out_path, "a") as fh:
        for budget, owner in manifest["esen_owners"].items():
            tag = owner["tag"]
            print(f"=== {budget} {tag} ===", flush=True)
            model, cfg, params, cp = checkpoint_loading.reconstruct(tag, device=device)
            model.net.eval()
            fwd = ESenInterventionForward(model)
            norm_module = model.net.backbone.norm

            # ---- baseline (single pass: L_original + cached prenorm/forces per chunk) ----
            t0 = time.time()
            sums_all, counts_all = [], []
            base_prenorm_chunks = []
            base_forces_chunks = []
            for b in chunks:
                out_b, _ = fwd(b, intervention=None)
                pa = evaluation_data.per_atom_force_error(out_b["forces"], b)
                s, c = per_config_accum(pa, b)
                sums_all += s
                counts_all += c
                base_prenorm_chunks.append(out_b["node_embedding_per_block"][11])
                base_forces_chunks.append(out_b["forces"])
            L_original = float(np.sum(sums_all) / max(np.sum(counts_all), 1))
            key = (budget, "baseline")
            if key not in existing:
                write_row(fh, budget, owner, "baseline", None, None, None,
                          sums_all, counts_all, L_original, time.time() - t0, 0.0, 1)
                existing[key] = {"L_flat": L_original}
            else:
                L_original = existing[key]["L_flat"]

            # ---- identity check: alpha=1.0 must leave block-9 prenorm identical (relative tol,
            # matching the fixed-normalization control check 5's precedent that alpha=1.0 forward passes accumulate
            # small fp32 scatter/reduce nondeterminism through blocks 10-11, not exact bit-identity) ----
            spec1 = InterventionSpec(layer=LAYER, ell=4, alpha=1.0)
            out1, _ = fwd(chunks[0], intervention=spec1)
            ref = base_prenorm_chunks[0]
            id_err = (out1["node_embedding_per_block"][11] - ref).abs().max().item()
            id_scale = ref.abs().max().clamp_min(1e-8).item()
            validation["identity_alpha1"][budget] = {
                "max_abs_diff_prenorm11": id_err, "max_rel_diff_prenorm11": id_err / id_scale,
                "passed": (id_err / id_scale) < 1e-4,
            }

            # ---- sector isolation check at an intermediate alpha (0.5): only targeted ell sector
            # at the intervention SITE (block 9's own output) changes; other sectors untouched ----
            spec_iso = InterventionSpec(layer=LAYER, ell=4, alpha=0.5)
            out_iso, _ = fwd(chunks[0], intervention=spec_iso)
            h9_iso = out_iso["node_embedding_per_block"][LAYER]
            out_none, _ = fwd(chunks[0], intervention=None)
            h9_none = out_none["node_embedding_per_block"][LAYER]
            from pipeline.intervention.esen_intervention import IRREP_SLICES
            s4, e4 = IRREP_SLICES[4]
            other_err = (torch.cat([h9_iso[:, :s4, :], h9_iso[:, e4:, :]], dim=1) -
                         torch.cat([h9_none[:, :s4, :], h9_none[:, e4:, :]], dim=1)).abs().max().item()
            other_scale = torch.cat([h9_none[:, :s4, :], h9_none[:, e4:, :]], dim=1).abs().max().clamp_min(1e-8).item()
            sector_abs_err = (h9_iso[:, s4:e4, :] - 0.5 * h9_none[:, s4:e4, :]).abs().max().item()
            sector_scale = h9_none[:, s4:e4, :].abs().max().clamp_min(1e-8).item()
            sector_scaled_ok = (sector_abs_err / sector_scale) < 1e-4
            validation["sector_isolation"][budget] = {
                "other_sectors_max_abs_diff": other_err, "other_sectors_rel": other_err / other_scale,
                "target_sector_scaled_correctly": sector_scaled_ok,
                "passed": (other_err / other_scale) < 1e-4 and sector_scaled_ok,
            }

            # ---- rotation-equivariance preserved at alpha=0.5 (NORM_FIXED), reusing the exact
            # the fixed-normalization control check-9 pattern (validate_stage3a1.py normfixed_forward) and its tolerance ----
            single = evaluation_data.build_batch(dataset, config_indices[:1], device)

            def normfixed_forward_alpha(k, ell, alpha):
                def _f(b):
                    out_b, _ = fwd(b, intervention=None)
                    hb = out_b["node_embedding_per_block"][11]
                    spec = InterventionSpec(layer=k, ell=ell, alpha=alpha)
                    out_a, _ = fwd(b, intervention=spec)
                    ha = out_a["node_embedding_per_block"][11]
                    h_nf = nc.norm_fixed_output(norm_module, ha, hb)
                    return nc.run_heads(model.net, b, h_nf)
                return _f

            # 3 seeds, matching the fixed-normalization control validate_stage3a1.py check 9's precedent (that check
            # was only ever run on the MID tier; here we additionally cover LOW/HIGH, so report
            # every seed rather than silently taking the best one).
            rot_tol = 3e-3
            seed_results = {}
            for seed in [999, 4242, 77]:
                R = rotation.random_rotation(seed=seed, dtype=single.pos.dtype).to(device)
                chk = rotation.check_force_equivariance(normfixed_forward_alpha(LAYER, 4, 0.5), single, R, atol=rot_tol)
                seed_results[str(seed)] = {"max_rel_error": chk.max_rel_error, "passed": bool(chk.passed)}
            validation["rotation_equivariance_alpha0.5"][budget] = {
                "tol": rot_tol, "by_seed": seed_results,
                "passed_all_seeds": all(v["passed"] for v in seed_results.values()),
                "passed_median_seed": bool(sorted(v["max_rel_error"] for v in seed_results.values())[1] < rot_tol),
            }

            for ell in ELLS:
                for alpha in ALPHAS:
                    cell_id = f"normfixed_L{LAYER}_ell{ell}_alpha{alpha:.2f}"
                    if (budget, cell_id) in existing:
                        continue
                    spec = InterventionSpec(layer=LAYER, ell=ell, alpha=alpha)
                    t0 = time.time()
                    sums_nf, counts_nf = [], []
                    disp_sum, disp_count = 0.0, 0
                    q_power_total, power_total_total = 0.0, 0.0
                    for b, h_base_pn, f_base in zip(chunks, base_prenorm_chunks, base_forces_chunks):
                        out_a, diag = fwd(b, intervention=spec)
                        q_power_total += diag["power_removed"]
                        power_total_total += diag["power_total_pre"]
                        h_ablated_prenorm = out_a["node_embedding_per_block"][11]
                        h_nf = nc.norm_fixed_output(norm_module, h_ablated_prenorm, h_base_pn)
                        out_nf = nc.run_heads(model.net, b, h_nf)
                        pa_nf = evaluation_data.per_atom_force_error(out_nf["forces"], b)
                        s2, c2 = per_config_accum(pa_nf, b)
                        sums_nf += s2
                        counts_nf += c2
                        disp = torch.linalg.vector_norm(out_nf["forces"] - f_base, ord=2, dim=-1)
                        disp_sum += float(disp.sum().item())
                        disp_count += disp.numel()
                    wall = time.time() - t0
                    q_power = q_power_total / power_total_total if power_total_total > 0 else None
                    write_row(fh, budget, owner, cell_id, LAYER, ell, alpha,
                              sums_nf, counts_nf, L_original, wall, disp_sum, disp_count,
                              extra={"q_power": q_power})

    val_path = OUT_DIR / (f"validation_smoke_M{m}.json" if args.smoke_test else f"validation_M{m}.json")
    val_path.write_text(json.dumps(validation, indent=2, default=str) + "\n")
    print(f"validation -> {val_path}", flush=True)
    print(f"DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
