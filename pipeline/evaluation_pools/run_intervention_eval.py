#!/usr/bin/env python3
# Adapted for anonymous release from esen_irrep_ood_2026_08_22/eval/run_gpu_eval.py
# source revision unversioned at port time, original SHA256 9916e3315357bbab1ddf88cc513e230128efbbb0d098f195e778a09d20d4f032
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from esen_irrep_ood_2026_08_22/eval/run_gpu_eval.py
# source revision unversioned at port time, original SHA256 9916e3315357bbab1ddf88cc513e230128efbbb0d098f195e778a09d20d4f032
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Forward-pass evaluation of the block-9 ell=4 intervention on both evaluation populations.
For each of the four ell_max=4 frontier checkpoints: a baseline pass with the block-9 and
block-11 pre-normalization hidden states cached, then the intervened pass at each alpha under
the fixed-normalization control, writing per-configuration force-error sums and atom counts.

Reuses, unmodified, the existing validated intervention machinery from
the analysis repository (checkpoint loading, intervention,
NORM_FIXED, loss convention, owner registry) -- every reused function is imported, not reimplemented.

For the four audited ell_max=4 frontier owners (w10/50k, w16/150k, w20/200k, w24/300k):
  - baseline forward pass (block-9 and block-11 pre-norm hidden states cached)
  - block-9, ell=4, alpha in {1,.75,.5,.25,0}, NORM_FIXED intervention forward pass
  - degree-balanced (rms_norm_sh-quadratic-form) block-9 sum-of-squares by degree,
    l=0 EXACTLY centered (same convention as
    upstream-analysis/the degree-balanced aggregation)
  - run on BOTH the frozen ID (Neutral val) pool and the frozen OOD (OMol25 val) pool,
    nested M up to 16384 (all M in {1024,...,16384} are prefixes of the same M=16384 pass,
    so nested-M is obtained downstream by slicing per-config arrays, not by rerunning).

For the four compute-matched ell_max=2 partners: baseline-only forward pass (forces only),
same two domains, same M=16384 population -- no intervention (matched-compute BASELINE
force-performance comparison only, per the task).

No new intervention/loading logic is added: ESenInterventionForward, InterventionSpec,
esen_norm_control.{compute_centered_and_rms,norm_fixed_output,run_heads},
instantiate_model_generic/load_checkpoint_strict_generic, evaluation_data.{build_batch,
per_atom_force_error}, and the OwnerSpec registries (lmax_checkpoints.OWNERS,
lmax_frontier_checkpoints.build_extra_owners) are imported verbatim from the kernel-pilot-clean
repo and called exactly as the existing frontier-sensitivity script calls them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

KPC_ROOT = Path("<PIPELINE_ROOT>")
sys.path.insert(0, str(KPC_ROOT))

import numpy as np
import torch
from fairchem.core.datasets import AseDBDataset

from pipeline.intervention import evaluation_data
from pipeline.intervention.lmax_checkpoints import OWNERS as S3B_OWNERS
from pipeline.intervention.lmax_frontier_checkpoints import build_extra_owners
from pipeline.kernel.frontier_checkpoints import (
    INSTANTIATION_SEED, instantiate_model_generic, load_checkpoint_strict_generic,
)
from pipeline.intervention.esen_intervention import (
    ESenInterventionForward, InterventionSpec, IRREP_SLICES,
)
from pipeline.intervention import esen_norm_control as nc
from pipeline.intervention.run_norm_control import git_state, per_config_accum

OOD_ROOT = Path("<PROJECT_ROOT>/analysis_outputs/esen_irrep_ood_2026_08_22")
OUT_DIR = OOD_ROOT / "eval" / "raw"
ID_POOL_PATH = OOD_ROOT / "id_pool" / "id_pool_indices.npy"
OOD_POOL_PATH = OOD_ROOT / "dataset" / "ood_pool_indices.npy"
OOD_VAL_SRC = "<CHECKPOINT_ROOT>/omol25/val"

CHUNK = 32
LAYER = 9
NUM_LAYERS = 12
ELL = 4
ALPHAS = [1.00, 0.75, 0.50, 0.25, 0.00]
M_MAX = 16384
DEGREE_WEIGHTS = {0: 1.0, 1: 1.0 / 3, 2: 1.0 / 5, 3: 1.0 / 7, 4: 1.0 / 9}

EXTRA = build_extra_owners()
LMAX4_OWNERS = {
    "w10/50k": S3B_OWNERS["esen_lmax4_LOW_w10_s50000"],
    "w16/150k": S3B_OWNERS["esen_lmax4_MID_w16_s150000"],
    "w20/200k": EXTRA["esen_lmax4_UPPERMID_w20_s200000"],
    "w24/300k": EXTRA["esen_lmax4_NEARMAX_w24_s300000"],
}
# Existing compute-matched ell_max=2 partners (the matched-frontier study true-frontier pairing --
# see lmax_frontier_checkpoints.py module docstring for the exact derivation).
LMAX2_MATCHED = {
    "w10/50k": EXTRA["esen_lmax2_LOWtrue_w16_s125000"],
    "w16/150k": EXTRA["esen_lmax2_MIDtrue_w32_s300000"],
    "w20/200k": EXTRA["esen_lmax2_UPPERMID_w32_s525000"],
    "w24/300k": S3B_OWNERS["esen_lmax2_HIGH_w64_s525000"],
}


def get_datasets():
    id_ds = AseDBDataset({"src": evaluation_data.VAL_PATH, "a2g_args": dict(r_energy=True, r_forces=True)})
    ood_ds = AseDBDataset({"src": OOD_VAL_SRC, "a2g_args": dict(r_energy=True, r_forces=True)})
    return {"ID": id_ds, "OOD": ood_ds}


def load_pool_indices(m_max=M_MAX):
    id_idx = np.load(ID_POOL_PATH)[:m_max].tolist()
    ood_idx = np.load(OOD_POOL_PATH)[:m_max].tolist()
    assert len(id_idx) == m_max, f"ID pool has only {len(id_idx)} < {m_max}"
    assert len(ood_idx) == m_max, f"OOD pool has only {len(ood_idx)} < {m_max}"
    return {"ID": id_idx, "OOD": ood_idx}


def make_chunks(dataset, indices, device):
    return [evaluation_data.build_batch(dataset, indices[i:i + CHUNK], device) for i in range(0, len(indices), CHUNK)]


def sanity_checks(fwd, chunk0):
    """Cheap structural sanity checks reused from run_frontier_ell4_sensitivity.py's
    V3 (identity at alpha=1) and V4 (sector isolation at alpha=0.5), on one chunk only."""
    out0, _ = fwd(chunk0, intervention=None)
    base9 = out0["node_embedding_per_block"][LAYER]

    out1, _ = fwd(chunk0, intervention=InterventionSpec(layer=LAYER, ell=ELL, alpha=1.0))
    err = (out1["node_embedding_per_block"][LAYER] - base9).abs().max().item()
    scale = base9.abs().max().clamp_min(1e-8).item()
    identity_ok = (err / scale) < 1e-4

    out_iso, _ = fwd(chunk0, intervention=InterventionSpec(layer=LAYER, ell=ELL, alpha=0.5))
    h_iso = out_iso["node_embedding_per_block"][LAYER]
    s4, e4 = IRREP_SLICES[ELL]
    other = (torch.cat([h_iso[:, :s4, :], h_iso[:, e4:, :]], dim=1) -
             torch.cat([base9[:, :s4, :], base9[:, e4:, :]], dim=1)).abs().max().item()
    other_scale = torch.cat([base9[:, :s4, :], base9[:, e4:, :]], dim=1).abs().max().clamp_min(1e-8).item()
    tgt = (h_iso[:, s4:e4, :] - 0.5 * base9[:, s4:e4, :]).abs().max().item()
    tgt_scale = base9[:, s4:e4, :].abs().max().clamp_min(1e-8).item()
    isolation_ok = (other / other_scale) < 1e-4 and (tgt / tgt_scale) < 1e-4
    return {"identity_alpha1_rel": err / scale, "identity_ok": bool(identity_ok),
            "sector_isolation_other_rel": other / other_scale, "sector_isolation_target_rel": tgt / tgt_scale,
            "sector_isolation_ok": bool(isolation_ok)}


def run_lmax4_owner(tag, owner, datasets, pool_idx, device, out_fh, meta_out):
    print(f"=== LMAX4 owner {tag} (ckpt width={owner.width} step={owner.step}) ===", flush=True)
    model, cfg, params = instantiate_model_generic(owner, device, INSTANTIATION_SEED)
    cp = load_checkpoint_strict_generic(model, owner, params)
    model.net.eval()
    fwd = ESenInterventionForward(model)
    norm_module = model.net.backbone.norm
    assert int(model.net.backbone.sph_feature_size) == 25

    owner_meta = {
        "tag": tag, "width": owner.width, "step": owner.step, "C_flops": owner.C_owner,
        "ckpt_path": cp.get("checkpoint_path"), "checkpoint_sha256": cp.get("checkpoint_sha256"),
        "checkpoint_global_step": cp.get("checkpoint_global_step"),
    }

    for domain in ("ID", "OOD"):
        t_domain0 = time.time()
        chunks = make_chunks(datasets[domain], pool_idx[domain], device)
        print(f"  [{domain}] {len(chunks)} chunks, {len(pool_idx[domain])} molecules", flush=True)

        sanity = sanity_checks(fwd, chunks[0])
        print(f"  [{domain}] sanity: {sanity}", flush=True)

        # ---- baseline pass: L, per-config sums, cached block-11 prenorm, block-9 ss_by_ell (raw + centered) ----
        t0 = time.time()
        sums_all, counts_all = [], []
        base11 = []
        ss_ell_raw = {e: 0.0 for e in IRREP_SLICES}
        ss_ell_centered = {e: 0.0 for e in IRREP_SLICES}
        n_ell = {e: 0 for e in IRREP_SLICES}
        for b in chunks:
            out_b, _ = fwd(b, intervention=None)
            pa = evaluation_data.per_atom_force_error(out_b["forces"], b)
            s, c = per_config_accum(pa, b)
            sums_all += s
            counts_all += c
            base11.append(out_b["node_embedding_per_block"][NUM_LAYERS - 1].detach())
            h9 = out_b["node_embedding_per_block"][LAYER]
            feature_centered, _ = nc.compute_centered_and_rms(norm_module, h9)
            for e, (s0, en) in IRREP_SLICES.items():
                sec_raw = h9[:, s0:en, :]
                sec_c = feature_centered[:, s0:en, :]
                ss_ell_raw[e] += float((sec_raw.double() ** 2).sum().item())
                ss_ell_centered[e] += float((sec_c.double() ** 2).sum().item())
                n_ell[e] += int(sec_raw.numel())
        L_base = float(np.sum(sums_all) / max(np.sum(counts_all), 1))
        print(f"  [{domain}] baseline L={L_base:.8f}  ({time.time()-t0:.1f}s)", flush=True)

        # ---- alpha sweep, NORM_FIXED ----
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
            out_fh.write(json.dumps({
                "tag": tag, "domain": domain, "width": owner.width, "step": owner.step,
                "C_flops": owner.C_owner, "layer": LAYER, "ell": ELL, "alpha": alpha,
                "kind": "normfixed", "m_eval": len(pool_idx[domain]),
                "L_baseline": L_base, "L_int": L_int, "delta": delta, "q_power": q_power,
                "per_config_sum": sums_i, "per_config_natoms": counts_i,
                "baseline_per_config_sum": sums_all, "baseline_per_config_natoms": counts_all,
                "pool_indices_prefix8": pool_idx[domain][:8],
                "checkpoint_sha256": cp.get("checkpoint_sha256"), "dtype": "float32",
                "wall_seconds": time.time() - t1,
            }, default=str) + "\n")
            out_fh.flush()
            print(f"  [{domain}] [alpha={alpha:.2f}] L={L_int:.8f} delta={delta:+.5f} "
                  f"q_power={q_power:.6f} ({time.time()-t1:.1f}s)", flush=True)

        meta_out[f"{tag}::{domain}"] = {
            **owner_meta, "domain": domain, "L_baseline": L_base,
            "ss_by_ell_raw": ss_ell_raw, "ss_by_ell_centered": ss_ell_centered,
            "n_elements_by_ell": n_ell,
            "sanity": sanity, "wall_seconds_domain": time.time() - t_domain0,
        }
        del chunks, base11
        if device == "cuda":
            torch.cuda.empty_cache()

    del model, fwd
    if device == "cuda":
        torch.cuda.empty_cache()


def run_lmax2_baseline(tag, owner, datasets, pool_idx, device, out_fh, meta_out):
    print(f"=== LMAX2 matched-compute partner for {tag} (width={owner.width} step={owner.step}) ===", flush=True)
    model, cfg, params = instantiate_model_generic(owner, device, INSTANTIATION_SEED)
    cp = load_checkpoint_strict_generic(model, owner, params)
    model.net.eval()
    fwd = ESenInterventionForward(model)

    owner_meta = {
        "tag": owner.tag, "matched_to": tag, "width": owner.width, "step": owner.step,
        "C_flops": owner.C_owner, "ckpt_path": cp.get("checkpoint_path"),
        "checkpoint_sha256": cp.get("checkpoint_sha256"),
        "checkpoint_global_step": cp.get("checkpoint_global_step"),
    }

    for domain in ("ID", "OOD"):
        t0 = time.time()
        chunks = make_chunks(datasets[domain], pool_idx[domain], device)
        sums_all, counts_all = [], []
        for b in chunks:
            out_b, _ = fwd(b, intervention=None)
            pa = evaluation_data.per_atom_force_error(out_b["forces"], b)
            s, c = per_config_accum(pa, b)
            sums_all += s
            counts_all += c
        L_base = float(np.sum(sums_all) / max(np.sum(counts_all), 1))
        print(f"  [{domain}] lmax2 baseline L={L_base:.8f}  ({time.time()-t0:.1f}s)", flush=True)
        out_fh.write(json.dumps({
            "tag": owner.tag, "matched_to": tag, "domain": domain, "width": owner.width,
            "step": owner.step, "C_flops": owner.C_owner, "kind": "lmax2_baseline",
            "m_eval": len(pool_idx[domain]), "L_baseline": L_base,
            "per_config_sum": sums_all, "per_config_natoms": counts_all,
            "checkpoint_sha256": cp.get("checkpoint_sha256"), "dtype": "float32",
            "wall_seconds": time.time() - t0,
        }, default=str) + "\n")
        out_fh.flush()
        meta_out[f"{owner.tag}::{domain}"] = {**owner_meta, "domain": domain, "L_baseline": L_base}
        del chunks
        if device == "cuda":
            torch.cuda.empty_cache()

    del model, fwd
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true", help="tiny M, all owners, fast pipeline check")
    args = ap.parse_args()
    m_max = 16 if args.smoke_test else M_MAX

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} smoke_test={args.smoke_test} m_max={m_max}", flush=True)

    datasets = get_datasets()
    pool_idx = load_pool_indices(m_max)
    print(f"ID pool: {len(pool_idx['ID'])} molecules, OOD pool: {len(pool_idx['OOD'])} molecules", flush=True)

    meta_out = {"git": git_state(), "layer": LAYER, "ell": ELL, "alphas": ALPHAS,
                "m_max": m_max, "degree_weights": DEGREE_WEIGHTS, "owners": {}}

    suffix = "_smoke" if args.smoke_test else ""
    lmax4_path = OUT_DIR / f"lmax4_block9_intervention{suffix}.jsonl"
    lmax2_path = OUT_DIR / f"lmax2_baseline{suffix}.jsonl"

    with open(lmax4_path, "w") as fh4:
        for tag, owner in LMAX4_OWNERS.items():
            run_lmax4_owner(tag, owner, datasets, pool_idx, device, fh4, meta_out["owners"])

    with open(lmax2_path, "w") as fh2:
        for tag, owner in LMAX2_MATCHED.items():
            run_lmax2_baseline(tag, owner, datasets, pool_idx, device, fh2, meta_out["owners"])

    (OUT_DIR / f"run_meta{suffix}.json").write_text(json.dumps(meta_out, indent=2, default=str) + "\n")
    print(f"DONE -> {lmax4_path}\n     -> {lmax2_path}\n     -> {OUT_DIR / 'run_meta.json'}", flush=True)


if __name__ == "__main__":
    main()
