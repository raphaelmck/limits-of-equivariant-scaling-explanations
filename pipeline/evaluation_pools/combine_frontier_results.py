#!/usr/bin/env python3
# Adapted for anonymous release from esen_irrep_ood_2026_08_22/eval/combine_full_frontier_results.py
# source revision unversioned at port time, original SHA256 86e18191c5e0853d0787fb5f715bbeee648afbe9701f319cab2f3a47f537bd42
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from esen_irrep_ood_2026_08_22/eval/combine_full_frontier_results.py
# source revision unversioned at port time, original SHA256 86e18191c5e0853d0787fb5f715bbeee648afbe9701f319cab2f3a47f537bd42
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Combine the per-shard evaluation outputs into the matched-frontier comparison: one row per
checkpoint and population, and for every valid matched pair a paired-bootstrap log ratio with
its 95% confidence interval on both populations.
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "<PROJECT_ROOT>/analysis_outputs/esen_irrep_ood_2026_08_22/eval")
from analyze_and_report import boot_L, N_BOOT, BOOT_SEED  # noqa: E402

ROOT = Path("<PROJECT_ROOT>/analysis_outputs/esen_irrep_ood_2026_08_22")
RAW = ROOT / "eval" / "raw"
OUT = ROOT / "eval"
PAPER_REPRO = Path("<PROJECT_ROOT>/paper_repro")

M_EVAL = 16384


def load_all_shards():
    recs = {}  # (lmax, width, step, domain) -> record
    for path in sorted(glob.glob(str(RAW / "full_frontier_baseline_shard*.jsonl"))):
        if "_smoke" in path:
            continue
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["lmax"], r["width"], r["step"], r["domain"])
            recs[key] = r
    return recs


def load_canonical_4():
    """The 4 pre-existing canonical pairs' baseline data, from run_gpu_eval.py's outputs
    (lmax4_block9_intervention.jsonl alpha=1.0 rows serve as the lmax4 baseline; alpha=1.0 with
    NORM_FIXED intervention *is* the unmodified baseline forward pass numerically -- but to avoid
    any doubt we instead reuse the already-published H_force/CI straight from
    ood_lmax2_vs_lmax4_baseline.csv / lmax2_vs_lmax4_baseline.csv, unchanged.)"""
    rows = list(csv.DictReader(open(PAPER_REPRO / "data" / "claim2" / "ood_lmax2_vs_lmax4_baseline.csv")))
    return [r for r in rows if r["M"] == "16384"]


def load_canonical_raw_into(recs):
    """Merge the 4 canonical checkpoints' own per_config_sum/natoms baseline records (lmax4 from
    lmax4_block9_intervention.jsonl's alpha=1.0 rows -- an unmodified NORM_FIXED forward pass is
    numerically the baseline; lmax2 from lmax2_baseline.jsonl) into the same (lmax,width,step,
    domain)->record dict used for the new full-frontier shards. Both sources use the SAME frozen
    ID/OOD pool index files (id_pool_indices.npy / ood_pool_indices.npy) and the same run order as
    run_full_frontier_baseline.py, so per-config arrays are index-aligned across both sources --
    this is what makes mixing a canonical checkpoint with a newly-evaluated partner in the SAME
    paired bootstrap valid. Without this, any pair where exactly ONE side is one of the 4 canonical
    checkpoints (e.g. new lmax4 owner matched to a canonical lmax2 owner) is wrongly reported
    MISSING even though both halves' raw data already exist on disk -- no new GPU eval is needed
    for those pairs, only this join."""
    n_added = 0
    for line in open(RAW / "lmax2_baseline.jsonl"):
        r = json.loads(line)
        key = (2, r["width"], r["step"], r["domain"])
        if key not in recs:
            recs[key] = r
            n_added += 1
    seen_l4 = set()
    for line in open(RAW / "lmax4_block9_intervention.jsonl"):
        r = json.loads(line)
        if r.get("alpha") != 1.0:
            continue
        key = (4, r["width"], r["step"], r["domain"])
        if key in seen_l4:
            continue
        seen_l4.add(key)
        if key not in recs:
            recs[key] = {
                "per_config_sum": r["baseline_per_config_sum"],
                "per_config_natoms": r["baseline_per_config_natoms"],
            }
            n_added += 1
    return n_added


def main():
    recs = load_all_shards()
    print(f"loaded {len(recs)} (lmax,width,step,domain) baseline records from shards")
    n_canon = load_canonical_raw_into(recs)
    print(f"merged {n_canon} additional (lmax,width,step,domain) records from the 4 canonical "
          f"checkpoints' own raw baseline data ({len(recs)} total)")

    matches = list(csv.DictReader(open(PAPER_REPRO / "analysis_out" / "lmax24_all_frontier_matches.csv")))
    valid = [m for m in matches if m["valid_match"] == "True"]

    out_rows = []
    n_ok, n_missing = 0, 0
    for m in valid:
        for domain, jdomain in (("Neutral_val", "ID"), ("Val_Comp_support_matched", "OOD")):
            r4 = recs.get((4, int(m["l4_width"]), int(m["l4_step"]), jdomain))
            r2 = recs.get((2, int(m["l2_width"]), int(m["l2_step"]), jdomain))
            if r4 is None or r2 is None:
                n_missing += 1
                out_rows.append({"l4_width": m["l4_width"], "l4_step": m["l4_step"],
                                  "l2_width": m["l2_width"], "l2_step": m["l2_step"],
                                  "domain": domain, "status": "MISSING_eval_not_yet_run"})
                continue
            # Same BOOT_SEED for both L4 and L2 (matches analyze_and_report.py's h_rows convention
            # exactly): configs are index-aligned across checkpoints (same pool order), so using
            # the identical resample-index draw for both makes this a genuinely PAIRED bootstrap,
            # not two independent ones.
            boot4, L4 = boot_L(r4["per_config_sum"], r4["per_config_natoms"], M_EVAL, BOOT_SEED)
            boot2, L2 = boot_L(r2["per_config_sum"], r2["per_config_natoms"], M_EVAL, BOOT_SEED)
            import numpy as np
            H = float(np.log(L2) - np.log(L4))
            H_boot = np.log(boot2) - np.log(boot4)
            ci_lo, ci_hi = float(np.percentile(H_boot, 2.5)), float(np.percentile(H_boot, 97.5))
            out_rows.append({"l4_width": m["l4_width"], "l4_step": m["l4_step"],
                              "l2_width": m["l2_width"], "l2_step": m["l2_step"],
                              "domain": domain, "status": "ok",
                              "H_force": H, "ci_low": ci_lo, "ci_high": ci_hi,
                              "L4": L4, "L2": L2, "n_boot": N_BOOT})
            n_ok += 1

    # Cross-check: the 4 canonical pairs are now computed through the exact same code path as
    # every other pair (via the recs merge above), not hardcoded -- confirm this reproduces the
    # already-published H_force/CI from ood_lmax2_vs_lmax4_baseline.csv (M=16384) to sanity-check
    # the merge and the paired-bootstrap procedure end-to-end.
    canon_rows = load_canonical_4()
    canon_by_key = {(r["lmax4_tag"], r["domain"]): r for r in canon_rows}
    tag_by_l4 = {("10", "50000"): "w10/50k", ("16", "150000"): "w16/150k",
                 ("20", "200000"): "w20/200k", ("24", "300000"): "w24/300k"}
    domain_map = {"Neutral_val": "ID", "Val_Comp_support_matched": "OOD"}
    n_cross_ok, n_cross_bad = 0, 0
    for row in out_rows:
        if row["status"] != "ok":
            continue
        tag = tag_by_l4.get((row["l4_width"], row["l4_step"]))
        if tag is None:
            continue
        canon = canon_by_key.get((tag, domain_map[row["domain"]]))
        if canon is None:
            continue
        d = abs(row["H_force"] - float(canon["H_force"]))
        if d < 1e-3:
            n_cross_ok += 1
        else:
            n_cross_bad += 1
            print(f"CROSS-CHECK MISMATCH: {tag} {row['domain']} l2={row['l2_width']}/{row['l2_step']} "
                  f"recomputed H_force={row['H_force']:.6f} vs canonical={float(canon['H_force']):.6f} "
                  f"(only exact for identical lmax2 partner: canonical l2 tag={canon['lmax2_tag']})")
    print(f"canonical cross-check: {n_cross_ok} matched to <1e-3, {n_cross_bad} did not "
          f"(expected mismatches for pairs where the lmax2 partner differs from the historical "
          f"canonical partner, e.g. the corrected MID w32/s325000 vs historical w32/s300000)")

    with open(OUT / "lmax2_vs_lmax4_full_frontier.csv", "w", newline="") as f:
        fieldnames = sorted({k for r in out_rows for k in r})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows ({n_ok} ok, {n_missing} missing) -> lmax2_vs_lmax4_full_frontier.csv")


if __name__ == "__main__":
    main()
