#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/run_stage3a1_norm_control.py
# source revision withheld for anonymous review, original SHA256 521478c5fcead1b3eca9840711358cabc259d189a79ba55ab89b774c3176cd10
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/run_stage3a1_norm_control.py
# source revision withheld for anonymous review, original SHA256 521478c5fcead1b3eca9840711358cabc259d189a79ba55ab89b774c3176cd10
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""The natural versus fixed-normalization comparison, on the same frozen eSEN checkpoints and
evaluation pool as the main intervention. Blocks 9 and 11, degrees 2 through 4.
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

from pipeline.intervention import checkpoint_loading, evaluation_data
from pipeline.intervention.esen_intervention import ESenInterventionForward, InterventionSpec
from pipeline.intervention import esen_norm_control as nc

OUT_DIR = REPO_ROOT / "analysis_outputs" / "stage3a1_norm_control_temporal_2026_08_16"
EXPANDED_DIR = REPO_ROOT / "analysis_outputs" / "four_arch_representation_probe_pilot_2026_08_15" / "expanded_pool"
INTERVENTION_MANIFEST = REPO_ROOT / "analysis_outputs" / "stage3a_esen_irrep_intervention_2026_08_16" / "stage3a_manifest.json"

CHUNK = 32
DEPTHS = [9, 11]
ELLS = [2, 3, 4]
NUM_LAYERS = 12


def git_state() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, capture_output=True, text=True).stdout
    return {"head": head, "n_changed": len([l for l in status.splitlines() if l.strip()])}


def per_config_accum(per_atom_err: torch.Tensor, batch) -> tuple[list, list]:
    err_np = per_atom_err.detach().cpu().numpy().astype(np.float64)
    idx_np = batch.batch.detach().cpu().numpy()
    n_configs = int(batch.natoms.numel())
    sums = np.zeros(n_configs, dtype=np.float64)
    counts = np.zeros(n_configs, dtype=np.int64)
    for i, e in zip(idx_np, err_np):
        sums[i] += e
        counts[i] += 1
    return sums.tolist(), counts.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1024, choices=[8, 1024])
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

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

    out_path = OUT_DIR / (f"norm_control_smoke_M{m}.jsonl" if args.smoke_test else f"norm_control_M{m}.jsonl")
    existing: dict[tuple[str, str], dict] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing[(r["budget"], r["cell_id"])] = r
        print(f"resuming: {len(existing)} already done", flush=True)

    gstate = git_state()

    def write_row(fh, budget, owner, cell_id, kind, layer, ell, sums, counts, L_original, wall, extra=None):
        L = float(np.sum(sums) / max(np.sum(counts), 1))
        ratio = L / L_original if L_original else float("nan")
        log_ratio = float(np.log(ratio)) if ratio > 0 else float("nan")
        row = {
            "budget": budget, "tag": owner["tag"], "architecture": "esen",
            "checkpoint_path": owner["checkpoint_path"], "width": owner["width"], "step": owner["step"],
            "cell_id": cell_id, "kind": kind, "layer": layer,
            "depth_z_corrected": None if layer is None else round((layer + 1) / NUM_LAYERS, 4),
            "ell": ell, "m_eval": m, "n_configs": len(config_indices),
            "L_flat": L, "L_original": L_original, "abs_diff": L - L_original, "ratio": ratio,
            "log_ratio": log_ratio, "per_config_sum": sums, "per_config_natoms": counts,
            "dtype": "float32", "git_head": gstate["head"], "git_n_changed_files": gstate["n_changed"],
            "wall_seconds": wall,
        }
        if extra:
            row.update(extra)
        fh.write(json.dumps(row, default=str) + "\n")
        fh.flush()
        print(f"  [{budget} {cell_id}] L={L:.6f} ratio={ratio:.4f} log_ratio={log_ratio:+.4f} wall={wall:.1f}s", flush=True)

    with open(out_path, "a") as fh:
        for budget, owner in manifest["esen_owners"].items():
            tag = owner["tag"]
            print(f"=== {budget} {tag} ===", flush=True)
            model, cfg, params, cp = checkpoint_loading.reconstruct(tag, device=device)
            model.net.eval()
            fwd = ESenInterventionForward(model)
            norm_module = model.net.backbone.norm

            # ---- baseline (single pass: L_original + cached prenorm/postnorm chunks) ----
            t0 = time.time()
            sums_all, counts_all = [], []
            base_prenorm_chunks = []
            base_postnorm_chunks = []
            for b in chunks:
                out_b, _ = fwd(b, intervention=None)
                pa = evaluation_data.per_atom_force_error(out_b["forces"], b)
                s, c = per_config_accum(pa, b)
                sums_all += s
                counts_all += c
                base_prenorm_chunks.append(out_b["node_embedding_per_block"][11])
                base_postnorm_chunks.append(out_b["node_embedding"])
            L_original = float(np.sum(sums_all) / max(np.sum(counts_all), 1))
            key = (budget, "baseline")
            if key not in existing:
                write_row(fh, budget, owner, "baseline", "baseline", None, None,
                          sums_all, counts_all, L_original, time.time() - t0)
                existing[key] = {"L_flat": L_original}
            else:
                L_original = existing[key]["L_flat"]

            for k in DEPTHS:
                for ell in ELLS:
                    nat_key = (budget, f"natural_L{k}_ell{ell}")
                    nf_key = (budget, f"normfixed_L{k}_ell{ell}")
                    need_nat = nat_key not in existing
                    need_nf = nf_key not in existing
                    if not (need_nat or need_nf):
                        continue
                    spec = InterventionSpec(layer=k, ell=ell, alpha=0.0)
                    t0 = time.time()
                    sums_nat, counts_nat = [], []
                    sums_nf, counts_nf = [], []
                    q_power_total, power_total_total = 0.0, 0.0
                    for b, h_base_pn in zip(chunks, base_prenorm_chunks):
                        out_a, diag = fwd(b, intervention=spec)
                        pa_nat = evaluation_data.per_atom_force_error(out_a["forces"], b)
                        s, c = per_config_accum(pa_nat, b)
                        sums_nat += s
                        counts_nat += c
                        q_power_total += diag["power_removed"]
                        power_total_total += diag["power_total_pre"]
                        h_ablated_prenorm = out_a["node_embedding_per_block"][11]
                        h_nf = nc.norm_fixed_output(norm_module, h_ablated_prenorm, h_base_pn)
                        out_nf = nc.run_heads(model.net, b, h_nf)
                        pa_nf = evaluation_data.per_atom_force_error(out_nf["forces"], b)
                        s2, c2 = per_config_accum(pa_nf, b)
                        sums_nf += s2
                        counts_nf += c2
                    wall = time.time() - t0
                    q_power = q_power_total / power_total_total if power_total_total > 0 else None
                    if need_nat:
                        write_row(fh, budget, owner, f"natural_L{k}_ell{ell}", "natural", k, ell,
                                  sums_nat, counts_nat, L_original, wall, extra={"q_power": q_power})
                    if need_nf:
                        write_row(fh, budget, owner, f"normfixed_L{k}_ell{ell}", "norm_fixed", k, ell,
                                  sums_nf, counts_nf, L_original, wall, extra={"q_power": q_power})

            # ---- POSTNORM, block 11 only ----
            for ell in ELLS:
                pn_key = (budget, f"postnorm_L11_ell{ell}")
                if pn_key in existing:
                    continue
                t0 = time.time()
                sums_pn, counts_pn = [], []
                for b, h_base_post in zip(chunks, base_postnorm_chunks):
                    h_masked = nc.postnorm_mask(h_base_post, ell)
                    out_pm = nc.run_heads(model.net, b, h_masked)
                    pa = evaluation_data.per_atom_force_error(out_pm["forces"], b)
                    s, c = per_config_accum(pa, b)
                    sums_pn += s
                    counts_pn += c
                write_row(fh, budget, owner, f"postnorm_L11_ell{ell}", "postnorm", 11, ell,
                          sums_pn, counts_pn, L_original, time.time() - t0)

    print(f"DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
