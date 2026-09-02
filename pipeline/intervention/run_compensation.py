#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/run_stage3_compensation.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 715ff25595acfd5c68c2f29ee05af4c9b369351fe1199bef84080644847bb8de
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/run_stage3_compensation.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 715ff25595acfd5c68c2f29ee05af4c9b369351fe1199bef84080644847bb8de
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Compensation follow-up: whether compute-matched ell_max=2 and ell_max=4 models reach
comparable performance through different causal reliance on their shared degrees (0, 1, 2)
at block 9, under the fixed-normalization control.

Reuses verbatim: ESenInterventionForward/InterventionSpec/IRREP_SLICES
(pipeline/intervention/esen_intervention.py), the NORM_FIXED convention
(pipeline/intervention/esen_norm_control.py), the M=1024 population + per-atom metric
(pipeline/intervention/evaluation_data.py), and the exact checkpoint-reconstruction paths /
matched-pair owner tags already frozen by the immediately preceding stage
(the gap-erasure runner's PAIRS dict / reconstruct_owner) -- no new
intervention or normalization logic, only: (a) running the SAME block-9 NORM_FIXED intervention
on BOTH architectures instead of only lmax4, restricted to the shared ell in {0,1,2} (individually
and jointly), and (b) per-config-level bookkeeping needed for paired bootstrap C_ell CIs.

IRREP_SLICES[0]=(0,1), [1]=(1,4), [2]=(4,9) do not depend on lmax (verified against the actual
lmax=2 checkpoint's backbone.sph_feature_size==9 in main() before any intervention is run) -- so
the existing IRREP_SLICES dict is reused unmodified for both architectures.
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

from pipeline.intervention import checkpoint_loading, lmax_checkpoints, evaluation_data, rotation
from pipeline.intervention.lmax_frontier_checkpoints import build_extra_owners
from pipeline.kernel.frontier_checkpoints import (
    INSTANTIATION_SEED, instantiate_model_generic, load_checkpoint_strict_generic,
)
from pipeline.intervention.esen_intervention import ESenInterventionForward, InterventionSpec, IRREP_SLICES
from pipeline.intervention import esen_norm_control as nc
from pipeline.intervention.run_norm_control import git_state, per_config_accum

OUT_DIR = REPO_ROOT / "analysis_outputs" / "stage3_lmax2_lmax4_compensation_2026_08_20"
EXPANDED_DIR = REPO_ROOT / "analysis_outputs" / "four_arch_representation_probe_pilot_2026_08_15" / "expanded_pool"
LAYER = 9
NUM_LAYERS = 12
CHUNK = 32
ALPHAS = [0.5, 0.0]
SHARED_ELLS = {"ell0": 0, "ell1": 1, "ell2": 2, "ell012": [0, 1, 2]}

EXTRA_OWNERS = build_extra_owners()

# Matched pairs -- identical tags/provenance to run_stage3_gap_erasure.py's PAIRS (reused verbatim).
PAIRS = {
    "LOW": {
        "l4": ("checkpoint_loading", "esen_LOW_w10_s50000"),
        "l2": ("extra", "esen_lmax2_LOWtrue_w16_s125000"),
        "C4": 2.399e16, "C2": 2.326e16,
        "frozen_L4": 0.02362621203799249,
        "frozen_L2": 0.019261466912076988,
        "frozen_alpha0": {  # ell -> frozen Delta (log-ratio), from irrep_intervention_frontier.json
            ("l4", 0): 0.8634043757279725, ("l4", 1): 2.1211739886848564, ("l4", 2): 0.33985808550399516,
            ("l2", 0): 1.7563846923149669, ("l2", 1): 2.2736880179368715, ("l2", 2): 1.025490956908693,
        },
    },
    "MID": {
        "l4": ("checkpoint_loading", "esen_MID_w16_s150000"),
        "l2": ("extra", "esen_lmax2_MIDtrue_w32_s300000"),
        "C4": 1.127e17, "C2": 1.079e17,
        "frozen_L4": 0.01117863637670748,
        "frozen_L2": 0.01011390566556508,
        "frozen_alpha0": {
            ("l4", 0): 1.8542858505004594, ("l4", 1): 2.704783400296174, ("l4", 2): 1.354413913649843,
            ("l2", 0): 2.885177456533132, ("l2", 1): 2.898473780536333, ("l2", 2): 2.052136867628433,
        },
    },
    "UPPERMID": {
        "l4": ("extra", "esen_lmax4_UPPERMID_w20_s200000"),
        "l2": ("extra", "esen_lmax2_UPPERMID_w32_s525000"),
        "C4": 1.865e17, "C2": 1.888e17,
        "frozen_L4": 0.00874923130113212,
        "frozen_L2": 0.008400005322621189,
        "frozen_alpha0": {
            ("l4", 0): 2.4680491740069477, ("l4", 1): 2.9324236549855724, ("l4", 2): 1.3710308336343084,
            ("l2", 0): 3.226422518602976, ("l2", 1): 3.0727564684715936, ("l2", 2): 2.2185497373721805,
        },
    },
    "HIGH": {
        "l4": ("extra", "esen_lmax4_NEARMAX_w24_s300000"),
        "l2": ("lmax_checkpoints", "esen_lmax2_HIGH_w64_s525000"),
        "C4": 3.347e17, "C2": 3.724e17,
        "frozen_L4": 0.007132378738350824,
        "frozen_L2": 0.006675305061219506,
        "frozen_alpha0": {
            ("l4", 0): 2.2733978146159064, ("l4", 1): 3.1009963385480432, ("l4", 2): 1.7516152768342004,
            ("l2", 0): 3.6549008766957494, ("l2", 1): 3.3222101102192987, ("l2", 2): 2.0614983098845454,
        },
    },
}


def reconstruct_owner(spec, device):
    kind, tag = spec
    if kind == "checkpoint_loading":
        model, cfg, params, cp = checkpoint_loading.reconstruct(tag, device=device)
    elif kind == "lmax_checkpoints":
        model, cfg, params, cp = lmax_checkpoints.reconstruct(tag, device=device)
    elif kind == "extra":
        owner = EXTRA_OWNERS[tag]
        model, cfg, params = instantiate_model_generic(owner, device, INSTANTIATION_SEED)
        cp = load_checkpoint_strict_generic(model, owner, params)
        model.net.eval()
    else:
        raise ValueError(kind)
    return model, cfg, params, cp


def eval_baseline(fwd, chunks):
    sums, counts = [], []
    base9, base11 = [], []
    for b in chunks:
        out, _ = fwd(b, intervention=None)
        pa = evaluation_data.per_atom_force_error(out["forces"], b)
        s, c = per_config_accum(pa, b)
        sums += s
        counts += c
        base9.append(out["node_embedding_per_block"][LAYER])
        base11.append(out["node_embedding_per_block"][NUM_LAYERS - 1])
    L = float(np.sum(sums) / max(np.sum(counts), 1))
    return L, sums, counts, base9, base11


def eval_intervention_normfixed(fwd, norm_module, model, chunks, base11, ell, alpha):
    spec = InterventionSpec(layer=LAYER, ell=ell, alpha=alpha)
    sums, counts = [], []
    for b, hb11 in zip(chunks, base11):
        out_a, diag = fwd(b, intervention=spec)
        h_abl_11 = out_a["node_embedding_per_block"][NUM_LAYERS - 1]
        h_nf = nc.norm_fixed_output(norm_module, h_abl_11, hb11)
        out_nf = nc.run_heads(model.net, b, h_nf)
        pa = evaluation_data.per_atom_force_error(out_nf["forces"], b)
        s, c = per_config_accum(pa, b)
        sums += s
        counts += c
    L = float(np.sum(sums) / max(np.sum(counts), 1))
    return L, sums, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1024, choices=[8, 1024])
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--pairs", nargs="*", default=list(PAIRS.keys()))
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
    print(f"{len(chunks)} chunks, {len(config_indices)} molecules, pairs={args.pairs}", flush=True)

    out_path = OUT_DIR / (f"compensation_smoke_M{m}.jsonl" if args.smoke_test else f"compensation_M{m}.jsonl")
    existing = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing.add((r["pair"], r["cell_id"]))
        print(f"resuming: {len(existing)} rows already done", flush=True)

    gstate = git_state()
    validation = {
        "reproduction_baseline": {}, "reproduction_alpha0": {},
        "identity_alpha1": {}, "sector_isolation": {}, "rotation_equivariance": {},
        "lmax2_coeff_axis_check": {},
    }

    with open(out_path, "a") as fh:
        for pair_name in args.pairs:
            spec = PAIRS[pair_name]
            print(f"=== pair {pair_name} ===", flush=True)

            models = {}
            for arch, ownerspec_key in (("l4", "l4"), ("l2", "l2")):
                model, cfg, params, cp = reconstruct_owner(spec[ownerspec_key], device)
                fwd = ESenInterventionForward(model)
                norm = model.net.backbone.norm
                lmax = model.net.backbone.lmax if hasattr(model.net.backbone, "lmax") else None
                sph = model.net.backbone.sph_feature_size
                L, sums, counts, base9, base11 = eval_baseline(fwd, chunks)
                models[arch] = dict(model=model, fwd=fwd, norm=norm, L=L, sums=sums, counts=counts,
                                     base9=base9, base11=base11, sph=sph, lmax=lmax)
                print(f"  {arch}: L={L:.6f} sph_feature_size={sph} lmax={lmax}", flush=True)

            # ---- lmax2 coefficient-axis check ----
            sph2 = models["l2"]["sph"]
            validation["lmax2_coeff_axis_check"][pair_name] = {
                "sph_feature_size": sph2, "expected_9_for_lmax2": sph2 == 9, "passed": sph2 == 9,
            }
            assert sph2 == 9, f"lmax2 owner {spec['l2']} has sph_feature_size={sph2}, expected 9 -- IRREP_SLICES assumption invalid"
            assert models["l4"]["sph"] == 25, f"lmax4 owner has sph_feature_size={models['l4']['sph']}, expected 25"

            L4, L2 = models["l4"]["L"], models["l2"]["L"]
            gap = float(np.log(L2) - np.log(L4))
            print(f"  gap=log(L2)-log(L4)={gap:+.4f}", flush=True)

            # ---- validation: baseline reproduction ----
            if m == 1024:
                rel4 = abs(L4 - spec["frozen_L4"]) / spec["frozen_L4"]
                rel2 = abs(L2 - spec["frozen_L2"]) / spec["frozen_L2"]
                validation["reproduction_baseline"][pair_name] = {
                    "L4": L4, "frozen_L4": spec["frozen_L4"], "rel_err_L4": rel4,
                    "L2": L2, "frozen_L2": spec["frozen_L2"], "rel_err_L2": rel2,
                    "passed": rel4 < 5e-3 and rel2 < 5e-3,
                }
                print(f"  repro baseline: L4 rel_err={rel4:.2e} L2 rel_err={rel2:.2e}", flush=True)

            # ---- validation: alpha=1 identity, both archs ----
            id_results = {}
            for arch in ("l4", "l2"):
                spec1 = InterventionSpec(layer=LAYER, ell=2, alpha=1.0)
                out1, _ = models[arch]["fwd"](chunks[0], intervention=spec1)
                ref = models[arch]["base9"][0]
                id_err = (out1["node_embedding_per_block"][LAYER] - ref).abs().max().item()
                id_scale = ref.abs().max().clamp_min(1e-8).item()
                id_results[arch] = {"rel": id_err / id_scale, "passed": (id_err / id_scale) < 1e-4}
            validation["identity_alpha1"][pair_name] = id_results
            print(f"  identity_alpha1: {id_results}", flush=True)

            # ---- validation: sector isolation, both archs, ell=2 alpha=0.5 ----
            iso_results = {}
            for arch in ("l4", "l2"):
                spec_iso = InterventionSpec(layer=LAYER, ell=2, alpha=0.5)
                out_iso, _ = models[arch]["fwd"](chunks[0], intervention=spec_iso)
                h9_iso = out_iso["node_embedding_per_block"][LAYER]
                h9_none = models[arch]["base9"][0]
                s2, e2 = IRREP_SLICES[2]
                other = (torch.cat([h9_iso[:, :s2, :], h9_iso[:, e2:, :]], dim=1) -
                         torch.cat([h9_none[:, :s2, :], h9_none[:, e2:, :]], dim=1)).abs().max().item()
                other_scale = torch.cat([h9_none[:, :s2, :], h9_none[:, e2:, :]], dim=1).abs().max().clamp_min(1e-8).item()
                target = (h9_iso[:, s2:e2, :] - 0.5 * h9_none[:, s2:e2, :]).abs().max().item()
                target_scale = h9_none[:, s2:e2, :].abs().max().clamp_min(1e-8).item()
                iso_results[arch] = {
                    "other_sectors_rel": other / other_scale, "target_scaled_rel": target / target_scale,
                    "passed": (other / other_scale) < 1e-4 and (target / target_scale) < 1e-4,
                }
            validation["sector_isolation"][pair_name] = iso_results
            print(f"  sector_isolation: {iso_results}", flush=True)

            def write_row(arch, cell_id, kind, ell, alpha, L, sums=None, counts=None, extra=None):
                if (pair_name, f"{arch}_{cell_id}") in existing:
                    return
                base_L = models[arch]["L"]
                delta = float(np.log(L) - np.log(base_L)) if kind != "baseline" else None
                row = {
                    "pair": pair_name, "arch": arch, "cell_id": f"{arch}_{cell_id}", "kind": kind,
                    "layer": LAYER, "ell": ell, "alpha": alpha, "m_eval": m, "n_configs": len(config_indices),
                    "L_baseline": base_L, "L_int": L if kind != "baseline" else None, "delta": delta,
                    "C4_flops": spec["C4"], "C2_flops": spec["C2"],
                    "per_config_sum": sums, "per_config_natoms": counts,
                    "dtype": "float32", "git_head": gstate["head"], "git_n_changed_files": gstate["n_changed"],
                }
                if extra:
                    row.update(extra)
                fh.write(json.dumps(row, default=str) + "\n")
                fh.flush()
                print(f"  [{arch} {cell_id}] L={L:.6f} delta={delta}", flush=True)

            for arch in ("l4", "l2"):
                write_row(arch, "baseline", "baseline", None, None, models[arch]["L"],
                          models[arch]["sums"], models[arch]["counts"])
                for key, ell in SHARED_ELLS.items():
                    for alpha in ALPHAS:
                        cell_id = f"{key}_alpha{alpha:.2f}"
                        L, sums, counts = eval_intervention_normfixed(
                            models[arch]["fwd"], models[arch]["norm"], models[arch]["model"],
                            chunks, models[arch]["base11"], ell, alpha,
                        )
                        write_row(arch, cell_id, "normfixed", ell, alpha, L, sums, counts)

            # ---- validation: reproduce frozen alpha=0 Delta for ell=0,1,2 (from irrep_intervention_frontier.json) ----
            if m == 1024:
                repro_a0 = {}
                for arch in ("l4", "l2"):
                    for ell in (0, 1, 2):
                        key = (arch, ell)
                        if key not in spec["frozen_alpha0"]:
                            continue
                        L, _, _ = eval_intervention_normfixed(
                            models[arch]["fwd"], models[arch]["norm"], models[arch]["model"],
                            chunks, models[arch]["base11"], ell, 0.0,
                        )
                        delta = float(np.log(L) - np.log(models[arch]["L"]))
                        target = spec["frozen_alpha0"][key]
                        rel = abs(delta - target) / abs(target)
                        repro_a0[f"{arch}_ell{ell}"] = {"delta": delta, "frozen": target, "rel_err": rel, "passed": rel < 5e-3}
                validation["reproduction_alpha0"][pair_name] = repro_a0
                print(f"  repro alpha0: {repro_a0}", flush=True)

            # ---- rotation equivariance: required 3 conditions ----
            rot_conditions = []
            if pair_name == "LOW":
                rot_conditions.append(("l2", 2, 0.5))
            if pair_name == "HIGH":
                rot_conditions.append(("l2", 2, 0.5))
                rot_conditions.append(("l4", 2, 0.5))
            if rot_conditions:
                single = evaluation_data.build_batch(dataset, config_indices[:1], device)
                rot_tol = 3e-3
                for arch, ell, alpha in rot_conditions:
                    fwd_arch = models[arch]["fwd"]
                    norm_arch = models[arch]["norm"]
                    model_arch = models[arch]["model"]

                    def normfixed_forward(k=LAYER, ell=ell, alpha=alpha, fwd_arch=fwd_arch,
                                           norm_arch=norm_arch, model_arch=model_arch):
                        def _f(b):
                            out_b, _ = fwd_arch(b, intervention=None)
                            hb = out_b["node_embedding_per_block"][NUM_LAYERS - 1]
                            sp = InterventionSpec(layer=k, ell=ell, alpha=alpha)
                            out_a, _ = fwd_arch(b, intervention=sp)
                            ha = out_a["node_embedding_per_block"][NUM_LAYERS - 1]
                            h_nf = nc.norm_fixed_output(norm_arch, ha, hb)
                            return nc.run_heads(model_arch.net, b, h_nf)
                        return _f

                    seed_results = {}
                    for seed in [999, 4242, 77]:
                        R = rotation.random_rotation(seed=seed, dtype=single.pos.dtype).to(device)
                        chk = rotation.check_force_equivariance(normfixed_forward(), single, R, atol=rot_tol)
                        seed_results[str(seed)] = {"max_rel_error": chk.max_rel_error, "passed": bool(chk.passed)}
                    key = f"{pair_name}_{arch}_ell{ell}_alpha{alpha}"
                    validation["rotation_equivariance"][key] = {
                        "tol": rot_tol, "by_seed": seed_results,
                        "passed_all_seeds": all(v["passed"] for v in seed_results.values()),
                    }
                    print(f"  rotation-equivariance {key}: {seed_results}", flush=True)

            for arch in ("l4", "l2"):
                del models[arch]["model"], models[arch]["fwd"]
            torch.cuda.empty_cache() if device == "cuda" else None

    val_path = OUT_DIR / (f"validation_smoke_M{m}.json" if args.smoke_test else f"validation_M{m}.json")
    val_path.write_text(json.dumps(validation, indent=2, default=str) + "\n")
    print(f"validation -> {val_path}", flush=True)
    print(f"DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
