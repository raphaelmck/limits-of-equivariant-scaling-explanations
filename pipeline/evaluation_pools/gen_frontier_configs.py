#!/usr/bin/env python3
# Adapted for anonymous release from esen_irrep_ood_2026_08_22/eval/gen_full_frontier_configs.py
# source revision unversioned at port time, original SHA256 ca5238816a03c38288a311c58b0fa5138e2ba102ac2bf87b6ea71eec252842b5
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from esen_irrep_ood_2026_08_22/eval/gen_full_frontier_configs.py
# source revision unversioned at port time, original SHA256 ca5238816a03c38288a311c58b0fa5138e2ba102ac2bf87b6ea71eec252842b5
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Generate a configuration snapshot for every (angular order, width) combination the
full-frontier evaluation needs, by taking an existing frozen snapshot for the same angular
order and overriding only the spherical-channel width.
"""
import yaml
from pathlib import Path

S3B1_CFG = Path("<PIPELINE_ROOT>/"
                 "analysis_outputs/stage3b1_true_lmax_frontiers_2026_08_16/config_snapshots")
OUT = Path("<PROJECT_ROOT>/analysis_outputs/esen_irrep_ood_2026_08_22/eval/"
           "config_snapshots_full_frontier")
OUT.mkdir(parents=True, exist_ok=True)

TEMPLATES = {
    4: S3B1_CFG / "esen_lmax4_w20_generated.yaml",
    2: S3B1_CFG / "esen_lmax2_w16_generated.yaml",
}

import json
manifest = json.load(open("<PROJECT_ROOT>/analysis_outputs/esen_irrep_ood_2026_08_22/"
                           "eval/full_frontier_manifest.json"))
needed = {4: set(r["width"] for r in manifest["lmax4"]), 2: set(r["width"] for r in manifest["lmax2"])}

for lmax, widths in needed.items():
    template = yaml.safe_load(open(TEMPLATES[lmax]))
    assert template["force_field_module"]["net"]["lmax"] == lmax
    for w in sorted(widths):
        out_path = OUT / f"esen_lmax{lmax}_w{w}_generated.yaml"
        if out_path.exists():
            continue
        cfg = yaml.safe_load(open(TEMPLATES[lmax]))
        cfg["force_field_module"]["net"]["sphere_channels"] = w
        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print("wrote", out_path)

print("done")
