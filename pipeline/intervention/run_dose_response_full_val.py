#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/run_stage3_block9_dose_response_fullval_subset.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 1e27e6b53930875806637bedaf31399265b556f4f0af0b2244389a424f34da1d
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/run_stage3_block9_dose_response_fullval_subset.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 1e27e6b53930875806637bedaf31399265b556f4f0af0b2244389a424f34da1d
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Full-population confirmation of the block-9 dose response: ell=4 only, the lowest- and
highest-compute checkpoints, alpha in {1.0, 0.5, 0.0}, over the full 27,697-configuration
Neutral validation population in deterministic dataset order.
"""
from __future__ import annotations

import json
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
from pipeline.intervention.run_norm_control import git_state, per_config_accum

OUT_DIR = REPO_ROOT / "analysis_outputs" / "stage3_block9_dose_response_2026_08_19"
INTERVENTION_MANIFEST = REPO_ROOT / "analysis_outputs" / "stage3a_esen_irrep_intervention_2026_08_16" / "stage3a_manifest.json"

CHUNK = 64
LAYER = 9
ELL = 4
TIERS = ["LOW", "HIGH"]
ALPHAS = [1.00, 0.50, 0.00]


def main():
    manifest = json.loads(INTERVENTION_MANIFEST.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = evaluation_data.load_dataset()
    n_full = len(dataset)
    config_indices = list(range(n_full))
    print(f"device={device} n_full_val_configs={n_full}", flush=True)

    chunks = [evaluation_data.build_batch(dataset, config_indices[i:i + CHUNK], device)
              for i in range(0, len(config_indices), CHUNK)]
    print(f"{len(chunks)} chunks of <= {CHUNK}", flush=True)

    out_path = OUT_DIR / "dose_response_fullval_subset.jsonl"
    existing: dict[tuple[str, str], dict] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing[(r["budget"], r["cell_id"])] = r
        print(f"resuming: {len(existing)} already done", flush=True)

    gstate = git_state()

    def write_row(fh, budget, owner, cell_id, layer, ell, alpha, sums, counts, L_original, wall):
        L = float(np.sum(sums) / max(np.sum(counts), 1))
        ratio = L / L_original if L_original else float("nan")
        log_ratio = float(np.log(ratio)) if ratio > 0 else float("nan")
        row = {
            "budget": budget, "tag": owner["tag"], "checkpoint_path": owner["checkpoint_path"],
            "width": owner["width"], "step": owner["step"],
            "cell_id": cell_id, "layer": layer, "ell": ell, "alpha": alpha,
            "n_configs": len(config_indices),
            "L_flat": L, "L_original": L_original, "ratio": ratio, "log_ratio": log_ratio,
            "per_config_sum": sums, "per_config_natoms": counts,
            "dtype": "float32", "git_head": gstate["head"], "wall_seconds": wall,
        }
        fh.write(json.dumps(row, default=str) + "\n")
        fh.flush()
        print(f"  [{budget} {cell_id}] L={L:.6f} log_ratio={log_ratio:+.4f} wall={wall:.1f}s", flush=True)

    with open(out_path, "a") as fh:
        for budget in TIERS:
            owner = manifest["esen_owners"][budget]
            tag = owner["tag"]
            print(f"=== {budget} {tag} ===", flush=True)
            model, cfg, params, cp = checkpoint_loading.reconstruct(tag, device=device)
            model.net.eval()
            fwd = ESenInterventionForward(model)
            norm_module = model.net.backbone.norm

            t0 = time.time()
            sums_all, counts_all = [], []
            base_prenorm_chunks = []
            for b in chunks:
                out_b, _ = fwd(b, intervention=None)
                pa = evaluation_data.per_atom_force_error(out_b["forces"], b)
                s, c = per_config_accum(pa, b)
                sums_all += s
                counts_all += c
                base_prenorm_chunks.append(out_b["node_embedding_per_block"][11])
            L_original = float(np.sum(sums_all) / max(np.sum(counts_all), 1))
            key = (budget, "baseline")
            if key not in existing:
                write_row(fh, budget, owner, "baseline", None, None, None,
                          sums_all, counts_all, L_original, time.time() - t0)
                existing[key] = {"L_flat": L_original}
            else:
                L_original = existing[key]["L_flat"]

            for alpha in ALPHAS:
                cell_id = f"normfixed_L{LAYER}_ell{ELL}_alpha{alpha:.2f}"
                if (budget, cell_id) in existing:
                    continue
                spec = InterventionSpec(layer=LAYER, ell=ELL, alpha=alpha)
                t0 = time.time()
                sums_nf, counts_nf = [], []
                for b, h_base_pn in zip(chunks, base_prenorm_chunks):
                    out_a, _ = fwd(b, intervention=spec)
                    h_ablated_prenorm = out_a["node_embedding_per_block"][11]
                    h_nf = nc.norm_fixed_output(norm_module, h_ablated_prenorm, h_base_pn)
                    out_nf = nc.run_heads(model.net, b, h_nf)
                    pa_nf = evaluation_data.per_atom_force_error(out_nf["forces"], b)
                    s2, c2 = per_config_accum(pa_nf, b)
                    sums_nf += s2
                    counts_nf += c2
                wall = time.time() - t0
                write_row(fh, budget, owner, cell_id, LAYER, ELL, alpha, sums_nf, counts_nf, L_original, wall)

    print(f"DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
