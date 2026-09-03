#!/usr/bin/env python3
# Adapted for anonymous release from analysis_scripts/owners_dense_grid_2026_08_23.py
# source revision withheld for anonymous review, original SHA256 34db1f7b72e66489aa473628df162e38dc245dce355157f601cfc5f96496939b
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
# Adapted for anonymous release from analysis_scripts/owners_dense_grid_2026_08_23.py
# source revision withheld for anonymous review, original SHA256 34db1f7b72e66489aa473628df162e38dc245dce355157f601cfc5f96496939b
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Checkpoint specifications for the five added evaluation budgets that bracket the GemNet-OC
and eSEN crossover.

This module does NOT reimplement anything from `kernel/frontier_checkpoints.py` --
it only builds 16 additional `OwnerSpec` instances (imported from that module unchanged) for
checkpoints that are not among the original 12 LOW/MID/HIGH owners. `original_config_path` for
each owner was resolved from `paper_repro/data/claim1/four_arch_force_checkpoint_table.csv`'s own
`run_dir` column (the audited hydra-output-directory field for that exact training run), matched
on (architecture, width, step) -- the same provenance source `four_arch_force_checkpoint_table.csv`
itself, not a new guess. N/D/C/force_mse_norm values were cross-checked against both that table and
the literal task-provided checkpoint table and agree to full precision.

Budget-to-owner mapping (dedup applied -- 20 (arch,budget) cells collapse to 16 unique
checkpoints; PRECROSS's mpnn owner is the pre-existing MID owner and is NOT included here, it is
reused verbatim from `frontier_checkpoints.OWNERS["mpnn_MID_w1150_s500000"]`):

  PRECROSS (1.230e17): mpnn=MID owner (reused, not here), egnn=160/300000, gemnet_oc=80/450000, esen=8/350000
  CROSS    (1.250e17): mpnn=1448/400000, egnn=160/300000 (dup of PRECROSS), gemnet_oc=80/450000 (dup), esen=20/150000
  FITX     (1.660e17): mpnn=1583/450000, egnn=160/400000, gemnet_oc=96/450000, esen=20/200000
  POSTCROSS(2.060e17): mpnn=1857/400000, egnn=160/400000 (dup of FITX), gemnet_oc=112/400000, esen=40/125000
  TOP      (9.550e17): mpnn=2557/500000, egnn=320/500000, gemnet_oc=256/400000, esen=64/350000

16 unique (architecture, width, step) tuples -> 16 OwnerSpec entries below, tagged with a
`_dense23` suffix distinct from the original 12 tags. Each owner's `budget_label` is set to the
FIRST (lowest-C) budget it serves; the module also exposes `OWNER_BUDGET_TAGS` mapping every one
of the 5 new budget labels (plus the reused PRECROSS/mpnn cell) to the tag(s) that serve it, so
downstream KRR/reporting code can label each budget cell without rebuilding a kernel per cell.
"""
from __future__ import annotations

from pathlib import Path

from pipeline.kernel.frontier_checkpoints import (
    ARCH_LABEL,
    WIDTH_LABEL,
    OwnerSpec,
)

MAIN_WORKTREE = Path("<PROJECT_ROOT>/upstream-training")
REPO_ROOT_FOR_REPAIR = Path(__file__).resolve().parents[1]

# KNOWN DATA ISSUE (found 2026-08-23, pre-existing, NOT introduced by this task): the historical
# hydra config snapshot at
# upstream-training/outputs/2026-06-22/16-27-11/.hydra/config.yaml (gemnet_oc width=96
# training run, job 9902521) has a corrupted trailing byte sequence -- the file ends
# "...group: debug\nug\n" where the stray extra "ug\n" line is not valid YAML continuation and
# breaks `yaml.safe_load`/`OmegaConf.load`. The file's sha256 is stable across repeated reads
# (b89630822fa82aee1ca0abde9cff75e6923843ce77e9c9a9bcb57f9f8f6289fa), so this is a genuine
# write-time corruption in the original training run's own output, not a filesystem read glitch.
# All content through and including "group: debug" (line 152) is well-formed and matches every
# other gemnet_oc config's schema exactly (seed=1, train_rmsd=6.383, exp_name consistent with
# job 9902521/width=96). Per the hard constraint "do not modify the training repository or
# any historical repo file's values," the corrupted source file itself is left untouched. Instead
# a byte-for-byte copy of the source MINUS the single stray trailing "ug" line is written once to
# this repo's own analysis_outputs/ (not the training repository), and ONLY the gemnet_oc
# width=96 owner below points its `original_config_path` at that repaired copy instead of the
# corrupted original. This is the only owner (of 16) needing this treatment.
_REPAIRED_GEMNET_OC_W96_CONFIG = (
    REPO_ROOT_FOR_REPAIR
    / "analysis_outputs/dense_grid_claim1_2026_08_23/repaired_configs/gemnet_oc_w96_s450000_config_repaired.yaml"
)

# Each entry: (architecture, width, step, expected_n, D_atom_tokens, C_owner, force_mse_norm,
#              ckpt_path, run_dir, first_budget_label, C_budget_of_first_label)
OWNERS_RAW_DENSE23 = [
    dict(architecture="mpnn", width=1448, step=400000, expected_n=25458016,
         D_atom_tokens=693029927, C_owner=1.2332573712061338e+17,
         force_mse_norm=0.0107365928842486,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_corrected_stats/dim-1448-neutral-epoch-corrected-stats/mpnn/run_9790137_params_25.5_million/dim_1448_step_step=400000.ckpt",
         run_dir="outputs/2026-06-09/syn-dim-1448", budget_label="CROSS", C_budget=1.250e17),
    dict(architecture="mpnn", width=1583, step=450000, expected_n=30396045,
         D_atom_tokens=779777582, C_owner=1.656780597625177e+17,
         force_mse_norm=0.0097973475347791,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_corrected_stats/dim-1583-neutral-epoch-corrected-stats/mpnn/run_10325227_params_30.4_million/dim_1583_step_step=450000.ckpt",
         run_dir="outputs/2026-08-10/00-34-58", budget_label="FITX", C_budget=1.660e17),
    dict(architecture="mpnn", width=1857, step=400000, expected_n=41763072,
         D_atom_tokens=693194561, C_owner=2.0236004118374925e+17,
         force_mse_norm=0.0094515709373153,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_corrected_stats/dim-1857-neutral-epoch-corrected-stats/mpnn/run_10325228_params_41.8_million/dim_1857_step_step=400000.ckpt",
         run_dir="outputs/2026-08-09/17-22-05", budget_label="POSTCROSS", C_budget=2.060e17),
    dict(architecture="mpnn", width=2557, step=500000, expected_n=78984522,
         D_atom_tokens=866390858, C_owner=4.783359598122561e+17,
         force_mse_norm=0.0077457507378873,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_corrected_stats/dim-2557-neutral-epoch-corrected-stats/mpnn/run_10325229_params_79.0_million/dim_2557_step_step=500000.ckpt",
         run_dir="outputs/2026-08-09/17-35-54", budget_label="TOP", C_budget=9.550e17),

    dict(architecture="egnn", width=160, step=300000, expected_n=2745453,
         D_atom_tokens=519819133, C_owner=1.2026500337429005e+17,
         force_mse_norm=0.0045521875302867,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_mc_egnn/mc-egnn-hc-160-neutral-epoch/mc_egnn/run_9745362_params_2.7_million/dim_160_step_step=300000.ckpt",
         run_dir="outputs/2026-06-05/21-12-51", budget_label="PRECROSS", C_budget=1.230e17),
    dict(architecture="egnn", width=160, step=400000, expected_n=2745453,
         D_atom_tokens=693092019, C_owner=1.603533012004937e+17,
         force_mse_norm=0.0038754822057255,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_mc_egnn/mc-egnn-hc-160-neutral-epoch/mc_egnn/run_9745362_params_2.7_million/dim_160_step_step=400000.ckpt",
         run_dir="outputs/2026-06-05/21-12-51", budget_label="FITX", C_budget=1.660e17),
    dict(architecture="egnn", width=320, step=500000, expected_n=10372498,
         D_atom_tokens=866325306, C_owner=7.572466388481238e+17,
         force_mse_norm=0.002714732227235,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_mc_egnn/mc-egnn-hc-320-neutral-epoch/mc_egnn/run_9745363_params_10.4_million/dim_320_step_step=500000.ckpt",
         run_dir="outputs/2026-06-05/22-20-01", budget_label="TOP", C_budget=9.550e17),

    dict(architecture="gemnet_oc", width=80, step=450000, expected_n=1398416,
         D_atom_tokens=779777582, C_owner=1.1508645680800122e+17,
         force_mse_norm=0.0017378933923385,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_gemnet_oc/gemnet-oc-atom80-neutral-epoch/gemnet_oc/run_10325230_params_1.4_million/dim_80_step_step=450000.ckpt",
         run_dir="outputs/2026-08-14/10-06-02", budget_label="PRECROSS", C_budget=1.230e17),
    dict(architecture="gemnet_oc", width=96, step=450000, expected_n=1958880,
         D_atom_tokens=779777582, C_owner=1.61211369515264e+17,
         force_mse_norm=0.0015887785634637,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_gemnet_oc/gemnet-oc-atom96-neutral-epoch/gemnet_oc/run_9902521_params_2.0_million/dim_96_step_step=450000.ckpt",
         run_dir="outputs/2026-06-22/16-27-11", budget_label="FITX", C_budget=1.660e17),
    dict(architecture="gemnet_oc", width=112, step=400000, expected_n=2618672,
         D_atom_tokens=693194561, C_owner=1.9158139924273338e+17,
         force_mse_norm=0.0014332995966137,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_gemnet_oc/gemnet-oc-atom112-neutral-epoch/gemnet_oc/run_10325231_params_2.6_million/dim_112_step_step=400000.ckpt",
         run_dir="outputs/2026-08-10/22-16-50", budget_label="POSTCROSS", C_budget=2.060e17),
    dict(architecture="gemnet_oc", width=256, step=400000, expected_n=13026560,
         D_atom_tokens=693194561, C_owner=9.530199246486084e+17,
         force_mse_norm=0.0010210147545868,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_gemnet_oc/gemnet-oc-atom256-neutral-epoch/gemnet_oc/run_9816217_params_13.0_million/dim_256_step_step=400000.ckpt",
         run_dir="outputs/2026-06-12/16-15-52", budget_label="TOP", C_budget=9.550e17),

    dict(architecture="esen", width=8, step=350000, expected_n=878434,
         D_atom_tokens=606536446, C_owner=1.1885752289735322e+17,
         force_mse_norm=0.0018798745178881,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_esen/esen-sphere-8-neutral-epoch/esen/run_9897808_params_878.4_thousand/dim_32_step_step=350000.ckpt",
         run_dir="outputs/2026-06-22/10-38-38", budget_label="PRECROSS", C_budget=1.230e17),
    dict(architecture="esen", width=20, step=150000, expected_n=2144170,
         D_atom_tokens=259982236, C_owner=1.2435507843387587e+17,
         force_mse_norm=0.0016600938222114,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_esen/esen-sphere-20-neutral-epoch/esen/run_9741258_params_2.1_million/dim_32_step_step=150000.ckpt",
         run_dir="outputs/2026-06-05/13-03-12", budget_label="CROSS", C_budget=1.250e17),
    dict(architecture="esen", width=20, step=200000, expected_n=2144170,
         D_atom_tokens=346575759, C_owner=1.657746173616457e+17,
         force_mse_norm=0.001402125713543,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_esen/esen-sphere-20-neutral-epoch/esen/run_9741258_params_2.1_million/dim_32_step_step=200000.ckpt",
         run_dir="outputs/2026-06-05/13-03-12", budget_label="FITX", C_budget=1.660e17),
    dict(architecture="esen", width=40, step=125000, expected_n=4253730,
         D_atom_tokens=216671114, C_owner=2.056040059928345e+17,
         force_mse_norm=0.0012586217585427,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_esen/esen-sphere-40-neutral-epoch/esen/run_9928671_params_4.3_million/dim_32_step_step=125000.ckpt",
         run_dir="outputs/2026-06-27/13-40-54", budget_label="POSTCROSS", C_budget=2.060e17),
    dict(architecture="esen", width=64, step=350000, expected_n=6785202,
         D_atom_tokens=606536446, C_owner=9.180795621277942e+17,
         force_mse_norm=0.0006214341292885,
         ckpt_path="<CHECKPOINT_ROOT>/geometric_scaling_checkpoints_esen/esen-sphere-64-neutral-epoch/esen/run_9928673_params_6.8_million/dim_32_step_step=350000.ckpt",
         run_dir="outputs/2026-06-26/21-33-19", budget_label="TOP", C_budget=9.550e17),
]

assert len(OWNERS_RAW_DENSE23) == 16


def _build_owners_dense23() -> dict[str, OwnerSpec]:
    owners = {}
    for raw in OWNERS_RAW_DENSE23:
        arch = raw["architecture"]
        tag = f"{arch}_{raw['budget_label']}_w{raw['width']}_s{raw['step']}_dense23"
        owners[tag] = OwnerSpec(
            tag=tag,
            architecture=arch,
            architecture_label=ARCH_LABEL[arch],
            budget_label=raw["budget_label"],
            width_label=WIDTH_LABEL[arch],
            width=raw["width"],
            step=raw["step"],
            expected_n=raw["expected_n"],
            D_atom_tokens=raw["D_atom_tokens"],
            C_owner=raw["C_owner"],
            C_budget=raw["C_budget"],
            utilization=raw["C_owner"] / raw["C_budget"],
            force_mse_norm=raw["force_mse_norm"],
            ckpt_path=raw["ckpt_path"],
            original_config_path=(
                str(_REPAIRED_GEMNET_OC_W96_CONFIG)
                if (arch == "gemnet_oc" and raw["width"] == 96 and raw["step"] == 450000)
                else str(MAIN_WORKTREE / raw["run_dir"] / ".hydra/config.yaml")
            ),
            config_name=f"{tag}.yaml",
        )
    return owners


OWNERS_DENSE23 = _build_owners_dense23()
assert len(OWNERS_DENSE23) == 16, f"expected 16 unique new owners, got {len(OWNERS_DENSE23)}"

# Which budgets each unique checkpoint tag serves (a checkpoint reused across 2 budgets appears
# twice). Budget "PRECROSS"'s mpnn cell is NOT here -- it reuses the existing MID owner tag
# `mpnn_MID_w1150_s500000` from frontier_checkpoints.OWNERS verbatim.
BUDGET_TO_TAGS = {
    "PRECROSS": [
        "mpnn_MID_w1150_s500000",  # reused existing MID owner, NOT in OWNERS_DENSE23
        "egnn_PRECROSS_w160_s300000_dense23",
        "gemnet_oc_PRECROSS_w80_s450000_dense23",
        "esen_PRECROSS_w8_s350000_dense23",
    ],
    "CROSS": [
        "mpnn_CROSS_w1448_s400000_dense23",
        "egnn_PRECROSS_w160_s300000_dense23",       # dup checkpoint, same as PRECROSS
        "gemnet_oc_PRECROSS_w80_s450000_dense23",    # dup checkpoint, same as PRECROSS
        "esen_CROSS_w20_s150000_dense23",
    ],
    "FITX": [
        "mpnn_FITX_w1583_s450000_dense23",
        "egnn_FITX_w160_s400000_dense23",
        "gemnet_oc_FITX_w96_s450000_dense23",
        "esen_FITX_w20_s200000_dense23",
    ],
    "POSTCROSS": [
        "mpnn_POSTCROSS_w1857_s400000_dense23",
        "egnn_FITX_w160_s400000_dense23",            # dup checkpoint, same as FITX
        "gemnet_oc_POSTCROSS_w112_s400000_dense23",
        "esen_POSTCROSS_w40_s125000_dense23",
    ],
    "TOP": [
        "mpnn_TOP_w2557_s500000_dense23",
        "egnn_TOP_w320_s500000_dense23",
        "gemnet_oc_TOP_w256_s400000_dense23",
        "esen_TOP_w64_s350000_dense23",
    ],
}
