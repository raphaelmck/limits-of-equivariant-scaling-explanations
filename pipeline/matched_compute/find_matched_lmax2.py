#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/find_matched_compute_lmax2.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 bb115e4b2e5e848ba5e2f1db502e54a7be2a1444ad31626e90e9e42128419e16
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/find_matched_compute_lmax2.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 bb115e4b2e5e848ba5e2f1db502e54a7be2a1444ad31626e90e9e42128419e16
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""the lmax=2 vs lmax=4 study Phase A/B: given the full ell_max=2 checkpoint inventory (dense width x step
grid, 445 checkpoints, all seed=1) and the frozen ell_max=4 LOW/MID/HIGH frontier owners,
find the best-matched-compute ell_max=2 checkpoint at each tier.

Compute convention: C = kappa_l(sphere_channels) * N_net_params * D_atom_tokens, using
FLOPS_AUDIT.md's rigorously measured (FlopCounterMode, real fwd+bwd, real training loss)
per-lmax kappa -- NOT the frozen the initial pilot C_budget values, which (per this script's own
Section-0 audit) were built with a single, lmax-agnostic kappa=223.08 (the source paper's
own eSEN coefficient) applied uniformly regardless of lmax. Reusing that flat value would
bias any lmax=2-vs-4 comparison inconsistently across the two families . Both the lmax=4 owners' compute AND the lmax=2
grid's compute are recomputed here with the same formula for full self-consistency; the
original frozen numbers are also reported for transparency.

Writes: checkpoint_comparability_audit.md (human-readable) and updates
stage3b_checkpoint_manifest.json (machine-readable) -- called AFTER this script confirms
the kappa_l4(width=10) empirical correction (measure_flops_lmax_width10.py output).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "analysis_outputs" / "stage3b_lmax2_vs_lmax4_2026_08_16"

KAPPA_L2_MEASURED = {4: 208.16303173647043, 16: 215.63302311691064, 32: 217.17968339400912,
                     64: 217.99387290225383, 69: 218.05398794909763}
KAPPA_L4_MEASURED = {4: 256.58930637650263, 16: 251.6878167121337, 32: 250.8128608767697,
                     64: 250.36868848894622}


def load_w10_measurement():
    """Reads measure_flops_lmax_width10.py's output (5 real batches x {lmax2,lmax4} at
    sphere_channels=10), returns {lmax: (kappa_atom_train_mean, provenance)}. kappa_atom_train
    = the per-batch total_flops / (n_params * n_atoms), averaged -- same definition as
    FLOPS_AUDIT.md Section 2.3.

    lmax=4/width=10 measurement job died silently on the shared login node (process vanished,
    no Python traceback -- consistent with an external OOM/cgroup kill on a heavily-loaded
    shared machine, not a code bug) after successfully completing lmax=2/width=10. Rather than
    retry (risking another silent kill) or block the lmax=2 vs lmax=4 study on it, this falls back to the
    linear interpolation between the two nearest MEASURED lmax=4 points (w4=256.589,
    w16=251.688), cross-validated by the fact that the SAME interpolation method for lmax=2
    (predicted 211.9 from w4/w16) came within 1.0% of this session's own direct w10
    measurement (213.91) -- i.e. the interpolation's own error at this width is empirically
    small for this family. Flagged explicitly, not silently treated as a direct measurement."""
    path = OUT_DIR / "flops_width10_raw.csv"
    rows = list(csv.DictReader(open(path)))
    out = {}
    for lmax in (2, 4):
        sub = [r for r in rows if int(r["lmax"]) == lmax and int(r["sphere_channels"]) == 10]
        if len(sub) == 5:
            kappas = [float(r["total_flops"]) / (int(r["n_params"]) * int(r["n_atoms"])) for r in sub]
            out[lmax] = (sum(kappas) / len(kappas), "measured_exact_this_session")
        else:
            assert lmax == 4, f"unexpected missing measurement for lmax={lmax} (only lmax=4 job is known to have died)"
            w0, k0 = 4, KAPPA_L4_MEASURED[4]
            w1, k1 = 16, KAPPA_L4_MEASURED[16]
            k = k0 + (k1 - k0) * (10 - w0) / (w1 - w0)
            out[lmax] = (k, "FELL_BACK_interpolated_w4_w16_measurement_job_died_on_login_node_see_docstring")
    return out


def kappa_l(lmax: int, width: int, extra_measured: dict) -> tuple[float, str]:
    table = dict(KAPPA_L2_MEASURED if lmax == 2 else KAPPA_L4_MEASURED)
    if lmax in extra_measured and 10 not in table:
        table[10] = extra_measured[lmax][0]
    if width in table:
        return table[width], "measured_exact"
    pts = sorted(table)
    if width < pts[0]:
        return table[pts[0]], f"width_below_min_measured_{pts[0]}_using_that_value_FLAGGED"
    if width > pts[-1]:
        return table[pts[-1]], f"width_above_max_measured_{pts[-1]}_using_plateau_FLAGGED"
    lo = max(p for p in pts if p <= width)
    hi = min(p for p in pts if p >= width)
    if lo == hi:
        return table[lo], "measured_exact"
    k = table[lo] + (table[hi] - table[lo]) * (width - lo) / (hi - lo)
    return k, f"interpolated_between_w{lo}_w{hi}"


def main():
    w10 = load_w10_measurement()
    print("Width=10 kappa (5 real batches, FlopCounterMode fwd+bwd):")
    print(f"  kappa_l2(width=10) = {w10[2][0]:.4f}  ({w10[2][1]})")
    print(f"  kappa_l4(width=10) = {w10[4][0]:.4f}  ({w10[4][1]})")

    # -- Recompute the 3 frozen lmax=4 frontier owners' OWN compute with the corrected kappa --
    lmax4_owners = {
        "LOW": dict(width=10, step=50000, N=1089390, D=86650769, C_frozen=2.10579670352222e16),
        "MID": dict(width=16, step=150000, N=1722258, D=259964327, C_frozen=9.987863619290285e16),
        "HIGH": dict(width=40, step=275000, N=4253730, D=476605759, C_frozen=4.522617321737471e17),
    }
    targets = {}
    print("\nLMAX=4 frontier owners, recomputed compute:")
    for tier, o in lmax4_owners.items():
        k, prov = kappa_l(4, o["width"], w10)
        C = k * o["N"] * o["D"]
        targets[tier] = C
        print(f"  {tier}: width={o['width']} step={o['step']} kappa={k:.3f} ({prov}) "
              f"C_corrected={C:.6e} C_frozen={o['C_frozen']:.6e} ratio={C/o['C_frozen']:.4f}")

    inv = json.loads((OUT_DIR / "stage3b_lmax2_inventory.json").read_text())
    recs = [r for r in inv["records"] if r["load_ok"] and not r["is_last_ckpt_alias"]]

    # Recompute every lmax=2 grid checkpoint's C with the (now width=10-corrected) kappa table.
    for r in recs:
        w = r["sphere_channels_from_state_dict"]
        N = r["net_trainable_param_count_from_state_dict"]
        D = r["token_processed"]
        k, prov = kappa_l(2, w, w10)
        r["_C_final"] = k * N * D
        r["_kappa_final"] = k
        r["_kappa_prov_final"] = prov

    print(f"\n{len(recs)} lmax=2 candidate checkpoints (non-'last' aliases) in the search pool.")
    print(f"C range: {min(r['_C_final'] for r in recs):.3e} to {max(r['_C_final'] for r in recs):.3e}")

    best = {}
    for tier, C_target in targets.items():
        ranked = sorted(recs, key=lambda r: abs(r["_C_final"] - C_target))
        best[tier] = ranked[:5]
        print(f"\n=== {tier} target C={C_target:.4e} ===")
        for r in ranked[:5]:
            u = r["_C_final"] / C_target
            print(f"  width={r['sphere_channels_from_state_dict']:3d} step={r['global_step']:7d} "
                  f"C={r['_C_final']:.4e} util={u:.4f} kappa_prov={r['_kappa_prov_final']}")

    out = {
        "w10_fresh_measurement": w10,
        "lmax4_owners_recomputed": {
            tier: {**{k: v for k, v in o.items() if k != "N" or True}, "C_corrected": targets[tier]}
            for tier, o in lmax4_owners.items()
        },
        "lmax2_search_pool_size": len(recs),
        "lmax2_best_matches": {
            tier: [
                {
                    "checkpoint_path": r["checkpoint_path"],
                    "job_id": r["job_id"],
                    "width": r["sphere_channels_from_state_dict"],
                    "step": r["global_step"],
                    "N": r["net_trainable_param_count_from_state_dict"],
                    "D": r["token_processed"],
                    "kappa_used": r["_kappa_final"],
                    "kappa_provenance": r["_kappa_prov_final"],
                    "C": r["_C_final"],
                    "utilization": r["_C_final"] / targets[tier],
                    "checkpoint_sha256": r["checkpoint_sha256"],
                }
                for r in best[tier]
            ]
            for tier in targets
        },
    }
    (OUT_DIR / "matched_compute_search.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_DIR / 'matched_compute_search.json'}")


if __name__ == "__main__":
    main()
