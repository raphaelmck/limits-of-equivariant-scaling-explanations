#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/run_fvs_controlB_same_width.py
# source revision withheld for anonymous review, original SHA256 d1515a5e65e270379c7f50234dc2e4bfdcff2df4adb48044605d7b3d3e7b176a
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/run_fvs_controlB_same_width.py
# source revision withheld for anonymous review, original SHA256 d1515a5e65e270379c7f50234dc2e4bfdcff2df4adb48044605d7b3d3e7b176a
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Same-width control: the ell_max=2 versus ell_max=4 intervention comparison at equal channel
width rather than at equal compute.

Question: are the shared-irrep (ell in {0,1,2}) block-9 causal-reliance differences seen under
MATCHED COMPUTE (analysis_outputs/stage3_lmax2_lmax4_compensation_2026_08_20) still present when
NOMINAL WIDTH is held fixed?

This is a straight re-run of `pipeline/intervention/run_compensation.py`'s experiment with the
ONLY change being which ell_max=2 checkpoint each ell_max=4 owner is paired against: instead of
the compute-matched (and therefore much wider) lmax2 owner, the lmax2 checkpoint with the
IDENTICAL sphere_channels AND the IDENTICAL global_step. Everything else -- intervention code,
NORM_FIXED convention, block k=9, ell grouping, alpha grid, population, per-atom force metric,
chunking, per-config bookkeeping, validation checks -- is imported/copied verbatim.

PRIMARY MATCHING CONVENTION (declared before any intervention result was inspected; see
INVENTORY.md Section 5 for the alternatives considered and why they were rejected):
    same sphere_channels AND same global_step  =>  same width, same optimizer steps, and
    (verified) same number of atom tokens processed to within 0.1%. The only thing that differs
    between the two models of a pair is ell_max. Compute is deliberately NOT matched (lmax2 is
    ~3.9-4.2x cheaper at the same width/step); that is the point of the control -- compute
    matching is the other, already-completed comparison.

The four lmax4 owners are exactly the four the matched-compute compensation stage used, so every
lmax4 number produced here is a hard reproduction check against that stage's frozen output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from pipeline.intervention import checkpoint_loading, evaluation_data, rotation
from pipeline.intervention.lmax_frontier_checkpoints import build_extra_owners, KAPPA_L2
from pipeline.kernel.frontier_checkpoints import (
    ARCH_LABEL, WIDTH_LABEL, INSTANTIATION_SEED, OwnerSpec,
    instantiate_model_generic, load_checkpoint_strict_generic,
)
from pipeline.intervention.esen_intervention import ESenInterventionForward, InterventionSpec, IRREP_SLICES
from pipeline.intervention import esen_norm_control as nc
from pipeline.intervention.run_norm_control import git_state, per_config_accum
from pipeline.intervention.run_compensation import (
    LAYER, NUM_LAYERS, CHUNK, ALPHAS, SHARED_ELLS, eval_baseline, eval_intervention_normfixed,
)

SPRINT_DIR = REPO_ROOT / "analysis_outputs" / "final_validation_sprint_2026_08_20"
OUT_DIR = SPRINT_DIR / "control_B_same_width"
CFG_DIR = SPRINT_DIR / "config_snapshots"
EXPANDED_DIR = REPO_ROOT / "analysis_outputs" / "four_arch_representation_probe_pilot_2026_08_15" / "expanded_pool"
LMAX_STUDY_DIR = REPO_ROOT / "analysis_outputs" / "stage3b_lmax2_vs_lmax4_2026_08_16"
LMAX2_INV = json.loads((LMAX_STUDY_DIR / "stage3b_lmax2_inventory.json").read_text())["records"]

EXTRA_OWNERS = build_extra_owners()


def _find_l2(width, step):
    hits = [r for r in LMAX2_INV
            if r.get("sphere_channels_from_state_dict") == width
            and r.get("global_step") == step
            and "dense0to50k" not in r["width_dir_name"]
            and "flopctrl" not in r["width_dir_name"]]
    if len(hits) != 1:
        raise KeyError(f"lmax2 w={width} s={step}: expected exactly 1 main-ladder record, got {len(hits)}")
    return hits[0]


def l2_same_width_owner(width, step):
    r = _find_l2(width, step)
    if r["lmax_from_state_dict_Jd_count"] != 2:
        raise AssertionError(f"w{width} s{step}: state-dict lmax={r['lmax_from_state_dict_Jd_count']}, expected 2")
    N = r["net_trainable_param_count_from_state_dict"]
    D = r["token_processed"]
    tag = f"esen_lmax2_SAMEW_w{width}_s{step}"
    return OwnerSpec(
        tag=tag, architecture="esen", architecture_label=ARCH_LABEL["esen"],
        budget_label=f"SAMEW_w{width}", width_label=WIDTH_LABEL["esen"], width=width, step=step,
        expected_n=N, D_atom_tokens=D, C_owner=KAPPA_L2 * N * D, C_budget=KAPPA_L2 * N * D,
        utilization=1.0, force_mse_norm=None, ckpt_path=r["checkpoint_path"],
        original_config_path=str(CFG_DIR / f"esen_lmax2_w{width}_generated.yaml"),
        config_name=f"esen_lmax2_w{width}_generated.yaml",
    ), r


# ---- the four SAME-WIDTH pairs. lmax4 side is byte-identical to the matched-compute stage's
# ---- lmax4 owners (so its numbers double as a frozen-value reproduction check).
PAIRS = {
    "SW_w10_s50000": {
        "D4_atom_tokens": 86650769, "N4_params": 1089390, "C4_flops_kappa_l4": 2.3689e+16,
        "l4": ("checkpoint_loading", "esen_LOW_w10_s50000"), "width": 10, "step": 50000,
        "matched_compute_pair": "LOW",
        "frozen_L4": 0.02362621203799249,
        "frozen_l4_alpha0": {0: 0.8634043757279725, 1: 2.1211739886848564, 2: 0.33985808550399516},
    },
    "SW_w16_s150000": {
        "D4_atom_tokens": 259964327, "N4_params": 1722258, "C4_flops_kappa_l4": 1.1236e+17,
        "l4": ("checkpoint_loading", "esen_MID_w16_s150000"), "width": 16, "step": 150000,
        "matched_compute_pair": "MID",
        "frozen_L4": 0.01117863637670748,
        "frozen_l4_alpha0": {0: 1.8542858505004594, 1: 2.704783400296174, 2: 1.354413913649843},
    },
    "SW_w20_s200000": {
        "D4_atom_tokens": 346575759, "N4_params": 2144170, "C4_flops_kappa_l4": 1.8649e+17,
        "l4": ("extra", "esen_lmax4_UPPERMID_w20_s200000"), "width": 20, "step": 200000,
        "matched_compute_pair": "UPPERMID",
        "frozen_L4": 0.00874923130113212,
        "frozen_l4_alpha0": {0: 2.4680491740069477, 1: 2.9324236549855724, 2: 1.3710308336343084},
    },
    "SW_w24_s300000": {
        "D4_atom_tokens": 519711759, "N4_params": 2566082, "C4_flops_kappa_l4": 3.3468e+17,
        "l4": ("extra", "esen_lmax4_NEARMAX_w24_s300000"), "width": 24, "step": 300000,
        "matched_compute_pair": "HIGH",
        "frozen_L4": 0.007132378738350824,
        "frozen_l4_alpha0": {0: 2.2733978146159064, 1: 3.1009963385480432, 2: 1.7516152768342004},
    },
}


def reconstruct_owner(spec, device):
    kind, tag = spec
    if kind == "checkpoint_loading":
        return checkpoint_loading.reconstruct(tag, device=device)
    if kind == "extra":
        owner = EXTRA_OWNERS[tag]
        model, cfg, params = instantiate_model_generic(owner, device, INSTANTIATION_SEED)
        cp = load_checkpoint_strict_generic(model, owner, params)
        model.net.eval()
        return model, cfg, params, cp
    raise ValueError(kind)


def reconstruct_l2(owner, device):
    model, cfg, params = instantiate_model_generic(owner, device, INSTANTIATION_SEED)
    cp = load_checkpoint_strict_generic(model, owner, params)
    model.net.eval()
    return model, cfg, params, cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1024, choices=[8, 1024])
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--pairs", nargs="*", default=list(PAIRS.keys()))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (SPRINT_DIR / "logs").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    dataset = evaluation_data.load_dataset()
    expanded = np.load(EXPANDED_DIR / "test_layer_M1024.npz")
    m = 8 if args.smoke_test else args.m
    config_indices = [int(c) for c in expanded["config_indices"][:m]]
    chunks = [evaluation_data.build_batch(dataset, config_indices[i:i + CHUNK], device)
              for i in range(0, len(config_indices), CHUNK)]
    print(f"{len(chunks)} chunks, {len(config_indices)} molecules, pairs={args.pairs}", flush=True)

    out_path = OUT_DIR / (f"same_width_smoke_M{m}.jsonl" if args.smoke_test else f"same_width_M{m}.jsonl")
    existing = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing.add((r["pair"], r["cell_id"]))
        print(f"resuming: {len(existing)} rows already done", flush=True)

    gstate = git_state()
    validation = {
        "reproduction_l4_baseline": {}, "reproduction_l4_alpha0": {},
        "identity_alpha1": {}, "sector_isolation": {}, "normfixed_used": True,
        "same_config_indices": {"first8": config_indices[:8], "n": len(config_indices)},
        "rotation_equivariance": {}, "checkpoint_metadata": {},
        "lmax_coeff_axis_check": {},
    }

    with open(out_path, "a") as fh:
        for pair_name in args.pairs:
            spec = PAIRS[pair_name]
            width, step = spec["width"], spec["step"]
            print(f"=== pair {pair_name} (width={width}, step={step}) ===", flush=True)

            l2_owner, l2_rec = l2_same_width_owner(width, step)
            models = {}
            for arch in ("l4", "l2"):
                if arch == "l4":
                    model, cfg, params, cp = reconstruct_owner(spec["l4"], device)
                else:
                    model, cfg, params, cp = reconstruct_l2(l2_owner, device)
                fwd = ESenInterventionForward(model)
                L, sums, counts, base9, base11 = eval_baseline(fwd, chunks)
                sph = int(model.net.backbone.sph_feature_size)
                lmax_attr = getattr(model.net.backbone, "lmax", None)
                lmax = int(lmax_attr) if lmax_attr is not None else int(round(sph ** 0.5) - 1)
                models[arch] = dict(model=model, fwd=fwd, norm=model.net.backbone.norm,
                                    L=L, sums=sums, counts=counts, base9=base9, base11=base11,
                                    sph=sph, lmax=lmax, cp=cp,
                                    C_flops=(None if arch == "l4" else l2_owner.C_owner))
                print(f"  {arch}: L={L:.6f} sph={models[arch]['sph']} lmax={models[arch]['lmax']}", flush=True)

            # ---- V: coefficient-axis / lmax check ----
            assert models["l2"]["sph"] == 9 and models["l2"]["lmax"] == 2, models["l2"]
            assert models["l4"]["sph"] == 25 and models["l4"]["lmax"] == 4, models["l4"]
            validation["lmax_coeff_axis_check"][pair_name] = {
                "l2_sph_feature_size": models["l2"]["sph"], "l2_lmax": models["l2"]["lmax"],
                "l4_sph_feature_size": models["l4"]["sph"], "l4_lmax": models["l4"]["lmax"],
                "passed": True,
            }

            # ---- V: exact checkpoint metadata ----
            validation["checkpoint_metadata"][pair_name] = {
                "l4": {k: models["l4"]["cp"].get(k) for k in
                       ("checkpoint_path", "checkpoint_sha256", "checkpoint_global_step",
                        "checkpoint_net_state_parameter_count")},
                "l2": {k: models["l2"]["cp"].get(k) for k in
                       ("checkpoint_path", "checkpoint_sha256", "checkpoint_global_step",
                        "checkpoint_net_state_parameter_count")},
                "l2_owner": {"tag": l2_owner.tag, "width": l2_owner.width, "step": l2_owner.step,
                             "expected_n": l2_owner.expected_n, "D_atom_tokens": l2_owner.D_atom_tokens,
                             "C_flops_kappa_l2": l2_owner.C_owner,
                             "config_snapshot": l2_owner.original_config_path},
                "D4_atom_tokens": spec["D4_atom_tokens"], "N4_params": spec["N4_params"],
                "C4_flops_kappa_l4": spec["C4_flops_kappa_l4"],
                "token_ratio_l2_over_l4": l2_owner.D_atom_tokens / spec["D4_atom_tokens"],
                "compute_ratio_l2_over_l4": l2_owner.C_owner / spec["C4_flops_kappa_l4"],
            }

            # ---- V: baseline reproduction of the frozen matched-compute lmax4 value ----
            if m == 1024:
                rel4 = abs(models["l4"]["L"] - spec["frozen_L4"]) / spec["frozen_L4"]
                validation["reproduction_l4_baseline"][pair_name] = {
                    "L4": models["l4"]["L"], "frozen_L4": spec["frozen_L4"],
                    "rel_err": rel4, "passed": rel4 < 5e-3,
                }
                print(f"  repro l4 baseline rel_err={rel4:.2e}", flush=True)

            # ---- V: alpha=1 identity ----
            id_results = {}
            for arch in ("l4", "l2"):
                out1, _ = models[arch]["fwd"](chunks[0], intervention=InterventionSpec(layer=LAYER, ell=2, alpha=1.0))
                ref = models[arch]["base9"][0]
                err = (out1["node_embedding_per_block"][LAYER] - ref).abs().max().item()
                scale = ref.abs().max().clamp_min(1e-8).item()
                id_results[arch] = {"rel": err / scale, "passed": (err / scale) < 1e-4}
            validation["identity_alpha1"][pair_name] = id_results
            print(f"  identity_alpha1: {id_results}", flush=True)

            # ---- V: sector isolation (ell=2, alpha=0.5) ----
            iso = {}
            for arch in ("l4", "l2"):
                out_iso, _ = models[arch]["fwd"](chunks[0], intervention=InterventionSpec(layer=LAYER, ell=2, alpha=0.5))
                h_iso = out_iso["node_embedding_per_block"][LAYER]
                h_ref = models[arch]["base9"][0]
                s2, e2 = IRREP_SLICES[2]
                other = (torch.cat([h_iso[:, :s2, :], h_iso[:, e2:, :]], dim=1) -
                         torch.cat([h_ref[:, :s2, :], h_ref[:, e2:, :]], dim=1)).abs().max().item()
                other_scale = torch.cat([h_ref[:, :s2, :], h_ref[:, e2:, :]], dim=1).abs().max().clamp_min(1e-8).item()
                tgt = (h_iso[:, s2:e2, :] - 0.5 * h_ref[:, s2:e2, :]).abs().max().item()
                tgt_scale = h_ref[:, s2:e2, :].abs().max().clamp_min(1e-8).item()
                iso[arch] = {"other_sectors_rel": other / other_scale, "target_scaled_rel": tgt / tgt_scale,
                             "passed": (other / other_scale) < 1e-4 and (tgt / tgt_scale) < 1e-4}
            validation["sector_isolation"][pair_name] = iso
            print(f"  sector_isolation: {iso}", flush=True)

            def write_row(arch, cell_id, kind, ell, alpha, L, sums=None, counts=None, extra=None):
                if (pair_name, f"{arch}_{cell_id}") in existing:
                    return
                base_L = models[arch]["L"]
                delta = float(np.log(L) - np.log(base_L)) if kind != "baseline" else None
                row = {
                    "pair": pair_name, "arch": arch, "cell_id": f"{arch}_{cell_id}", "kind": kind,
                    "width": width, "step": step, "layer": LAYER, "ell": ell, "alpha": alpha,
                    "m_eval": m, "n_configs": len(config_indices),
                    "L_baseline": base_L, "L_int": L if kind != "baseline" else None, "delta": delta,
                    "lmax": models[arch]["lmax"],
                    "C_flops": (models["l4"] if arch == "l4" else models["l2"]).get("C_flops"),
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
                        L, sums, counts = eval_intervention_normfixed(
                            models[arch]["fwd"], models[arch]["norm"], models[arch]["model"],
                            chunks, models[arch]["base11"], ell, alpha,
                        )
                        write_row(arch, f"{key}_alpha{alpha:.2f}", "normfixed", ell, alpha, L, sums, counts)

            # ---- V: reproduce frozen lmax4 alpha=0 Deltas ----
            if m == 1024:
                repro = {}
                for ell, target in spec["frozen_l4_alpha0"].items():
                    L, _, _ = eval_intervention_normfixed(
                        models["l4"]["fwd"], models["l4"]["norm"], models["l4"]["model"],
                        chunks, models["l4"]["base11"], ell, 0.0)
                    d = float(np.log(L) - np.log(models["l4"]["L"]))
                    rel = abs(d - target) / abs(target)
                    repro[f"l4_ell{ell}"] = {"delta": d, "frozen": target, "rel_err": rel, "passed": rel < 5e-3}
                validation["reproduction_l4_alpha0"][pair_name] = repro
                print(f"  repro l4 alpha0: {repro}", flush=True)

            # ---- V: rotation equivariance, one ell=2 alpha=0.5 cell in EACH architecture, every pair ----
            single = evaluation_data.build_batch(dataset, config_indices[:1], device)
            rot_tol = 3e-3
            for arch in ("l4", "l2"):
                fwd_a, norm_a, model_a = models[arch]["fwd"], models[arch]["norm"], models[arch]["model"]

                def normfixed_forward(fwd_a=fwd_a, norm_a=norm_a, model_a=model_a):
                    def _f(b):
                        out_b, _ = fwd_a(b, intervention=None)
                        hb = out_b["node_embedding_per_block"][NUM_LAYERS - 1]
                        out_a, _ = fwd_a(b, intervention=InterventionSpec(layer=LAYER, ell=2, alpha=0.5))
                        ha = out_a["node_embedding_per_block"][NUM_LAYERS - 1]
                        return nc.run_heads(model_a.net, b, nc.norm_fixed_output(norm_a, ha, hb))
                    return _f

                # NOISE FLOOR: the SAME check on the completely un-intervened forward pass.
                # Any equivariance error already present here is the model's own fp32 forward
                # noise on this molecule, not something the intervention introduced.
                def baseline_forward(fwd_a=fwd_a):
                    def _f(b):
                        out_b, _ = fwd_a(b, intervention=None)
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
                        "excess_over_noise_floor": chk.max_rel_error - chk0.max_rel_error,
                    }
                key = f"{pair_name}_{arch}_ell2_alpha0.5"
                validation["rotation_equivariance"][key] = {
                    "tol": rot_tol, "by_seed": seed_results,
                    "passed_all_seeds": all(v["passed"] for v in seed_results.values()),
                    "passed_median_seed": bool(sorted(v["max_rel_error"] for v in seed_results.values())[1] < rot_tol),
                    "max_excess_over_noise_floor": max(v["excess_over_noise_floor"] for v in seed_results.values()),
                    "passed_vs_noise_floor": all(
                        v["max_rel_error"] <= max(rot_tol, 1.5 * v["baseline_noise_floor_max_rel_error"])
                        for v in seed_results.values()),
                }
                print(f"  rot {key}: {seed_results}", flush=True)

            for arch in ("l4", "l2"):
                del models[arch]["model"], models[arch]["fwd"]
            if device == "cuda":
                torch.cuda.empty_cache()

    val_path = OUT_DIR / (f"validation_smoke_M{m}.json" if args.smoke_test else f"validation_M{m}.json")
    val_path.write_text(json.dumps(validation, indent=2, default=str) + "\n")
    print(f"validation -> {val_path}", flush=True)
    print(f"DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
