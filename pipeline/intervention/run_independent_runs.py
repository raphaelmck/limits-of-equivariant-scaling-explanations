#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/run_seed_replication_ell4_sensitivity.py
# source revision withheld for anonymous review, original SHA256 671cda8099f2196fa741cc2fbbb34a164ab491fae0de41490d75004ae2eddcc7
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/run_seed_replication_ell4_sensitivity.py
# source revision withheld for anonymous review, original SHA256 671cda8099f2196fa741cc2fbbb34a164ab491fae0de41490d75004ae2eddcc7
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Block-9 ell=4 alpha sweep for the four checkpoints of the two additional independent training
runs, at the low-compute (width 10, step 50000) and high-compute (width 24, step 300000)
endpoints.

This is a GENERALIZATION of `run_frontier_ell4_sensitivity.py` to an arbitrary checkpoint path
+ width + step + (seed, data_seed) label, replacing that script's hardcoded seed-1-only
`lmax_frontier_checkpoints.build_extra_owners()` registry. The intervention math itself is imported
and reused UNCHANGED from `pipeline.intervention.esen_intervention` / `pipeline.intervention.esen_norm_control` /
`pipeline.intervention.evaluation_data` / `pipeline.intervention.rotation` / `pipeline.intervention.run_norm_control.per_config_accum`
-- nothing about the ell=4/block-9/NORM_FIXED intervention itself is reimplemented here.

The only genuinely new code is checkpoint/config loading, because
`pipeline.kernel.frontier_checkpoints.instantiate_model_generic` hardcodes an assertion
`cfg.seed == 1` (`TRAINING_CFG_SEED = 1`), which is true for every owner it was written for but
false by design for these 4 new checkpoints (cfg.seed in {2, 3}, since varying the seed is the
point of this run). `instantiate_model_seedrepl` / `load_checkpoint_strict_seedrepl` below reproduce that
function's logic verbatim except for parameterizing the expected seed/data_seed instead of
hardcoding 1, and dropping the GemNet-OC/MC-EGNN branches (irrelevant -- eSEN only here).

Since these are genuinely new checkpoints, there is no frozen L_baseline / frozen_damage to
reproduce (unlike run_frontier_ell4_sensitivity.py's gates 1/2). Gates 3 (alpha=1 identity), 4
(sector isolation), and 5 (rotation equivariance vs. noise floor) are still run and reported --
a checkpoint failing any of these should be flagged, not silently used.

alphas=[1.00,0.75,0.50,0.25,0.00], block k=9, ell=4, NORM_FIXED, M=1024 frozen molecule pool
(same `four_arch_representation_probe_pilot_2026_08_15/expanded_pool/test_layer_M1024.npz` file
used everywhere else in this line of work), `evaluation_data.per_atom_force_error` metric, CHUNK=32.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from pipeline.intervention import evaluation_data, rotation
from pipeline.kernel.frontier_checkpoints import INSTANTIATION_SEED
from pipeline.kernel.force_ntk import TRAIN_RMSD, sha256_file
from pipeline.intervention.esen_intervention import (
    ESenInterventionForward, InterventionSpec, IRREP_SLICES,
)
from pipeline.intervention import esen_norm_control as nc
from pipeline.intervention.run_norm_control import git_state, per_config_accum

OUT_DIR = REPO_ROOT / "analysis_outputs" / "seed_replication_2026_08_20" / "results"
CONFIG_SNAP_DIR = OUT_DIR / "config_snapshots"
EXPANDED_DIR = REPO_ROOT / "analysis_outputs" / "four_arch_representation_probe_pilot_2026_08_15" / "expanded_pool"
TRAINING_REPO = Path("<PROJECT_ROOT>/upstream-training")

CHUNK = 32
LAYER = 9
NUM_LAYERS = 12
ELL = 4
ALPHAS = [1.00, 0.75, 0.50, 0.25, 0.00]


@dataclass(frozen=True)
class SeedOwner:
    tag: str
    role_tier: str
    width: int
    step: int
    seed: int
    data_seed: int
    ckpt_path: str
    original_config_path: str
    expected_n: int
    D_atom_tokens: int
    C_flops: float
    override_seed_fields: bool = False


# expected_n / D_atom_tokens / C_flops are taken from the seed=1 architecture-identical owners
# (frontier_checkpoints.OWNERS_RAW's esen LOW w10/s50000 row; and
# lmax_frontier_checkpoints._LMAX4_EXACT[(24, 300000)]) -- these are properties of the ARCHITECTURE
# (lmax=4, sphere_channels, num_layers, ...), fixed by the training config, and are
# seed-independent: only weight VALUES differ across seeds, not shapes/counts. This is verified,
# not merely assumed: `load_checkpoint_strict_seedrepl` below asserts the checkpoint's own
# state-dict parameter count for exactly this set of `net.*` keys equals `expected_n`, so a
# mismatch would raise rather than silently pass.
#
# original_config_path: resolved `.hydra/config.yaml` for each job, found under
# upstream-training/outputs/2026-08-2{0,1}/*/ by matching exp_name/seed/sphere_channels
# against run_manifest.json's job table. No resolved config exists on disk specifically for
# w10/seed3 (only a w10/seed2 dump was found) -- per the run manifest/3, the w10
# net config is identical across seeds except the `seed` and `data.datamodule.data.seed` fields,
# so the w10/seed2 config file is reused for the seed3 owner too, with the *expected* cfg.seed
# passed as 3 (instantiate_model_seedrepl asserts the loaded config's cfg.seed equals the owner's
# declared seed, so this would fail loudly if that assumption were ever wrong for this file).
OWNERS: dict[str, SeedOwner] = {
    "seedrepl_LOW_w10_seed2": SeedOwner(
        tag="seedrepl_LOW_w10_seed2", role_tier="LOW", width=10, step=50000, seed=2, data_seed=2,
        ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_esen/esen-sphere-10-neutral-epoch-seed2/esen/run_10429387_params_1.1_million/dim_32_step_step=50000.ckpt",
        original_config_path=str(TRAINING_REPO / "outputs/2026-08-20/16-54-28/.hydra/config.yaml"),
        expected_n=1089390, D_atom_tokens=86650769, C_flops=2.10579670352222e16,
    ),
    "seedrepl_LOW_w10_seed3": SeedOwner(
        tag="seedrepl_LOW_w10_seed3", role_tier="LOW", width=10, step=50000, seed=3, data_seed=3,
        ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_esen/esen-sphere-10-neutral-epoch-seed3/esen/run_10429388_params_1.1_million/dim_32_step_step=50000.ckpt",
        original_config_path=str(TRAINING_REPO / "outputs/2026-08-20/16-54-28/.hydra/config.yaml"),
        expected_n=1089390, D_atom_tokens=86650769, C_flops=2.10579670352222e16,
        # No resolved .hydra/config.yaml exists on disk anywhere for a w10/seed3 job (searched;
        # only a w10/seed2 dump was found -- see run_manifest.json / module docstring). The
        # w10/seed2 file's own `seed`/`data.datamodule.data.seed` fields literally read 2, not a
        # template, so the strict cfg.seed==owner.seed check cannot be satisfied by this file for
        # the seed3 owner. override_seed_fields=True tells instantiate_model_seedrepl to
        # explicitly OVERWRITE cfg.seed/cfg.data.datamodule.data.seed to 3 in memory before
        # instantiation, rather than only asserting equality -- legitimate because (a) the net
        # architecture (shapes/param count) does not depend on this field at all, and (b) the
        # instantiation-time weight VALUES this field would seed are immediately and completely
        # overwritten by the strict checkpoint state-dict load right after, so the override has
        # zero effect on the loaded model; the load_checkpoint_strict_seedrepl step-count/param
        # assertions below still independently verify the *checkpoint itself* is the intended
        # step=50000/seed=3 artifact.
        override_seed_fields=True,
    ),
    "seedrepl_HIGH_w24_seed2": SeedOwner(
        tag="seedrepl_HIGH_w24_seed2", role_tier="HIGH", width=24, step=300000, seed=2, data_seed=2,
        ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_esen/esen-sphere-24-neutral-epoch-seed2/esen/run_10429389_params_2.6_million/dim_32_step_step=300000.ckpt",
        original_config_path=str(TRAINING_REPO / "outputs/2026-08-20/17-03-36/.hydra/config.yaml"),
        expected_n=2566082, D_atom_tokens=519711759, C_flops=3.346812983455683e17,
    ),
    "seedrepl_HIGH_w24_seed3": SeedOwner(
        tag="seedrepl_HIGH_w24_seed3", role_tier="HIGH", width=24, step=300000, seed=3, data_seed=3,
        ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_esen/esen-sphere-24-neutral-epoch-seed3/esen/run_10429390_params_2.6_million/dim_32_step_step=300000.ckpt",
        original_config_path=str(TRAINING_REPO / "outputs/2026-08-20/17-22-12/.hydra/config.yaml"),
        expected_n=2566082, D_atom_tokens=519711759, C_flops=3.346812983455683e17,
    ),
}


def snapshot_config(owner: SeedOwner) -> Path:
    CONFIG_SNAP_DIR.mkdir(parents=True, exist_ok=True)
    dst = CONFIG_SNAP_DIR / f"{owner.tag}.yaml"
    src = Path(owner.original_config_path)
    if not dst.exists():
        dst.write_bytes(src.read_bytes())
    if sha256_file(dst) != sha256_file(src):
        raise AssertionError(f"{owner.tag}: config snapshot diverged from original {src}")
    return dst


def instantiate_model_seedrepl(owner: SeedOwner, device: str, init_seed: int):
    """Generalization of pipeline.kernel.frontier_checkpoints.instantiate_model_generic:
    identical logic (train_rmsd check, RNG seeding before instantiate, trainable-param-count
    assertion), except the hardcoded `cfg.seed == 1` check is replaced by `cfg.seed ==
    owner.seed`, plus an added check that `data.datamodule.data.seed == owner.data_seed` and
    `net.sphere_channels == owner.width` (both were implicitly true-by-construction for the
    seed-1-only registry this generalizes; here they are explicit, checked assertions since
    that is no longer guaranteed by which registry the owner came from)."""
    cfg_path = snapshot_config(owner)
    cfg = OmegaConf.load(cfg_path)
    rmsd = float(cfg.force_field_module.train_rmsd)
    if rmsd != TRAIN_RMSD:
        raise AssertionError(f"{owner.tag}: config train_rmsd={rmsd}, expected {TRAIN_RMSD}")
    if owner.override_seed_fields:
        # See SeedOwner docstring note at seedrepl_LOW_w10_seed3: no genuine per-seed resolved
        # config exists for this owner, so the seed/data_seed fields are explicitly overwritten
        # (architecture-irrelevant, and immediately superseded by the strict checkpoint load).
        cfg.seed = owner.seed
        cfg.data.datamodule.data.seed = owner.data_seed
    cfg_seed = int(cfg.seed)
    if cfg_seed != owner.seed:
        raise AssertionError(f"{owner.tag}: cfg.seed={cfg_seed}, expected {owner.seed}")
    cfg_data_seed = int(cfg.data.datamodule.data.seed)
    if cfg_data_seed != owner.data_seed:
        raise AssertionError(
            f"{owner.tag}: data.datamodule.data.seed={cfg_data_seed}, expected {owner.data_seed}")
    cfg_sphere = int(cfg.force_field_module.net.sphere_channels)
    if cfg_sphere != owner.width:
        raise AssertionError(f"{owner.tag}: net.sphere_channels={cfg_sphere}, expected {owner.width}")

    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)

    model = hydra.utils.instantiate(cfg.force_field_module)
    model.free_scheduler = False
    model.net.eval()
    model = model.to(device)
    params = [(name, p) for name, p in model.net.named_parameters() if p.requires_grad]
    n = int(sum(p.numel() for _, p in params))
    if n != owner.expected_n or int(model.total_params) != owner.expected_n:
        raise AssertionError(
            f"{owner.tag}: trainable count={n}, model.total_params={model.total_params}, "
            f"expected={owner.expected_n}")
    return model, cfg, params


def load_checkpoint_strict_seedrepl(model, owner: SeedOwner, params) -> dict:
    """Identical to frontier_checkpoints.load_checkpoint_strict_generic's
    eSEN branch (no MC-EGNN width_mult / no GemNet-OC mup logic needed -- eSEN only here),
    parameterized on SeedOwner instead of OwnerSpec."""
    ckpt_path = Path(owner.ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    parameter_state_keys = {f"net.{name}" for name, _ in params}
    missing = sorted(parameter_state_keys - set(state))
    if missing:
        raise AssertionError(f"{owner.tag}: checkpoint lacks net parameters {missing}")
    state_net_n = int(sum(state[k].numel() for k in parameter_state_keys))
    if state_net_n != owner.expected_n:
        raise AssertionError(
            f"{owner.tag}: checkpoint net state count={state_net_n}, expected={owner.expected_n}")
    global_step = int(ckpt["global_step"]) if "global_step" in ckpt else None
    if global_step != owner.step:
        raise AssertionError(
            f"{owner.tag}: checkpoint_global_step={global_step}, requested step={owner.step}")
    return {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_size_bytes": ckpt_path.stat().st_size,
        "checkpoint_global_step": global_step,
        "checkpoint_strict_load": True,
        "checkpoint_net_state_parameter_count": state_net_n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1024, choices=[8, 1024])
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--only", type=str, default=None,
                     help="comma-separated subset of OWNERS keys to run (default: all 4)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    owners_to_run = OWNERS
    if args.only:
        keys = args.only.split(",")
        owners_to_run = {k: OWNERS[k] for k in keys}

    dataset = evaluation_data.load_dataset()
    expanded = np.load(EXPANDED_DIR / "test_layer_M1024.npz")
    m = 8 if args.smoke_test else args.m
    config_indices = [int(c) for c in expanded["config_indices"][:m]]
    chunks = [evaluation_data.build_batch(dataset, config_indices[i:i + CHUNK], device)
              for i in range(0, len(config_indices), CHUNK)]
    print(f"{len(chunks)} chunks, {len(config_indices)} molecules", flush=True)

    gstate = git_state()
    suffix = f"smoke_M{m}" if args.smoke_test else f"M{m}"
    out_path = OUT_DIR / f"seed_replication_ell4_{suffix}.jsonl"
    stats_path = OUT_DIR / f"block9_stats_{suffix}.json"
    val_path = OUT_DIR / f"validation_{suffix}.json"

    validation = {"identity_alpha1": {}, "sector_isolation": {}, "rotation_equivariance": {},
                  "checkpoint_metadata": {}, "instantiation_checks": {},
                  "same_config_indices": {"first8": config_indices[:8], "n": len(config_indices)}}
    stats_out = {"purpose": "intervention-free block-9 magnitude stats for the 4 new seed-replication owners",
                 "layer": LAYER, "m_eval": m, "owners": {}}

    with open(out_path, "a") as fh:
        for tag, owner in owners_to_run.items():
            print(f"=== {tag} (w{owner.width}/s{owner.step}/seed={owner.seed}) ===", flush=True)
            model, cfg, params = instantiate_model_seedrepl(owner, device, INSTANTIATION_SEED)
            cp = load_checkpoint_strict_seedrepl(model, owner, params)
            model.net.eval()
            fwd = ESenInterventionForward(model)
            norm_module = model.net.backbone.norm
            assert int(model.net.backbone.sph_feature_size) == 25

            validation["checkpoint_metadata"][tag] = {
                k: cp.get(k) for k in ("checkpoint_path", "checkpoint_sha256",
                                       "checkpoint_global_step", "checkpoint_net_state_parameter_count")}
            validation["checkpoint_metadata"][tag].update(
                {"width": owner.width, "step": owner.step, "seed": owner.seed,
                 "data_seed": owner.data_seed, "role_tier": owner.role_tier,
                 "C_flops": owner.C_flops, "expected_n": owner.expected_n,
                 "D_atom_tokens": owner.D_atom_tokens,
                 "original_config_path": owner.original_config_path})
            validation["instantiation_checks"][tag] = {
                "override_seed_fields_used": owner.override_seed_fields,
                "cfg_seed_matches_owner_seed": True, "cfg_data_seed_matches_owner_data_seed": True,
                "cfg_sphere_channels_matches_owner_width": True,
                "trainable_param_count_matches_expected_n": True,
                "checkpoint_state_dict_param_count_matches_expected_n": True,
                "note": "all True by construction -- instantiate_model_seedrepl/"
                        "load_checkpoint_strict_seedrepl raise AssertionError before this point "
                        "if any check fails, so reaching here means all passed",
            }

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

            stats_out["owners"][tag] = {
                "role_tier": owner.role_tier, "width": owner.width, "step": owner.step,
                "seed": owner.seed, "data_seed": owner.data_seed, "C_flops": owner.C_flops,
                "L_original": L_base,
                "ss_total_block9": ss_total, "n_elements_block9": n_total,
                "rms_total_block9": float(np.sqrt(ss_total / n_total)),
                "ss_by_ell": {str(e): ss_ell[e] for e in IRREP_SLICES},
                "n_elements_by_ell": {str(e): n_ell[e] for e in IRREP_SLICES},
                "H_ell_sector_rms": {str(e): float(np.sqrt(ss_ell[e] / max(n_ell[e], 1))) for e in IRREP_SLICES},
                "power_fraction_by_ell": {str(e): float(ss_ell[e] / ss_total) for e in IRREP_SLICES},
            }

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
                p_total = float(np.sqrt(q_power)) if q_power is not None else None
                fh.write(json.dumps({
                    "tag": tag, "role_tier": owner.role_tier, "width": owner.width, "step": owner.step,
                    "seed": owner.seed, "data_seed": owner.data_seed,
                    "C_flops": owner.C_flops, "layer": LAYER, "ell": ELL, "alpha": alpha,
                    "kind": "normfixed", "m_eval": m, "n_configs": len(config_indices),
                    "L_baseline": L_base, "L_int": L_int, "delta": delta, "q_power": q_power,
                    "P_total": p_total,
                    "per_config_sum": sums_i, "per_config_natoms": counts_i,
                    "baseline_per_config_sum": sums_all, "baseline_per_config_natoms": counts_all,
                    "checkpoint_path": cp.get("checkpoint_path"),
                    "checkpoint_sha256": cp.get("checkpoint_sha256"),
                    "dtype": "float32", "git_head": gstate["head"],
                    "git_n_changed_files": gstate["n_changed"], "wall_seconds": time.time() - t1,
                }, default=str) + "\n")
                fh.flush()
                print(f"  [alpha={alpha:.2f}] L={L_int:.8f} delta={delta:+.5f} P_total={p_total}", flush=True)

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
            for rseed in [999, 4242, 77]:
                R = rotation.random_rotation(seed=rseed, dtype=single.pos.dtype).to(device)
                chk = rotation.check_force_equivariance(normfixed_forward(), single, R, atol=rot_tol)
                chk0 = rotation.check_force_equivariance(baseline_forward(), single, R, atol=rot_tol)
                seed_results[str(rseed)] = {
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

    # merge stats/validation with any pre-existing partial file (so --only subsets accumulate)
    def _merge(path, new, keylevel_dicts):
        if path.exists():
            old = json.loads(path.read_text())
            for k in keylevel_dicts:
                if k in old:
                    old[k].update(new.get(k, {}))
            for k, v in new.items():
                if k not in keylevel_dicts:
                    old[k] = v
            return old
        return new

    stats_out = _merge(stats_path, stats_out, ["owners"])
    validation = _merge(val_path, validation, ["identity_alpha1", "sector_isolation",
                                                "rotation_equivariance", "checkpoint_metadata",
                                                "instantiation_checks"])
    stats_path.write_text(json.dumps(stats_out, indent=2) + "\n")
    val_path.write_text(json.dumps(validation, indent=2, default=str) + "\n")
    print(f"DONE -> {out_path}\n     -> {stats_path}\n     -> {val_path}", flush=True)


if __name__ == "__main__":
    main()
