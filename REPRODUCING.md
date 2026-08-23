# Reproducing — honest current state

**This pass does not make the pipeline runnable.** `scripts/` and `src/` are empty placeholders
(see their `README.md` stubs). There is no code in this directory yet that will regenerate any
number in `data/`. This document exists so nobody mistakes "the data layer is clean" for "you can
run `python scripts/claim1/run_everything.py` and get a paper."

## What full reproduction currently requires

To regenerate any value in `data/claim1/` or `data/claim2/` from scratch today, you must go back
to the two source repositories this consolidation pass read from (both are read-only inputs to
this project and were not modified):

1. **`dev-equivariant-scaling-laws/`** — the training-side repo: SLURM launch scripts, the
   original four-architecture empirical force-scaling checkpoint table and fits
   (`analysis_outputs/four_arch_force_scaling_repaired_2026_08_14/`), the eSEN-only
   source/capacity-at-init study, and `FLOPS_AUDIT.md` (the per-lmax kappa FLOPs convention used
   throughout Claim 2's compute axis).
2. **`dev-equivariant-scaling-laws-kernel-pilot-clean/`** (branch `raphael/esen-spectra-extraction`)
   — where essentially all of Claim 1's four-architecture kernel/KRR pipeline and all of Claim 2's
   Stage-3 irrep-intervention/dose-response/compensation/control-A-D machinery lives, under
   `analysis_scripts/` and `analysis_outputs/`.

Every checkpoint referenced by either claim is listed, with its exact filesystem path (where it
exists) and SHA256 (where recorded upstream), in `manifests/checkpoints.csv`. Any checkpoint
marked `status=PENDING` in that file does not exist yet — see the "Seed generality" section of
`paper/CLAIMS.md`.

## What a later pass needs to do to make this runnable

1. **Migrate/refactor analysis code** out of the two source repos'
   `analysis_scripts/{...}` trees into `scripts/claim1/` and `scripts/claim2/` here, as
   standalone scripts that take a checkpoint path (from `manifests/checkpoints.csv`) and reproduce
   one row of the corresponding `data/` file.
2. **Factor shared plumbing** (checkpoint reconstruction, the M=1024 molecule-pool loader, the
   per-atom force-error convention documented in
   `dev-equivariant-scaling-laws-kernel-pilot-clean/analysis_outputs/final_validation_sprint_2026_08_20/INVENTORY.md`
   Sec. 2-3) into `src/`.
3. **Generate figures** into `figures/`, using `paper/FIGURE_PROVENANCE.md` as the starting map of
   what already exists as a design precedent in the source repos.
4. **Add tests** in `tests/` that check a re-run of the migrated code reproduces the frozen
   numeric values already recorded in `data/` (e.g. the m=256 block-KRR reproduction gate,
   INVENTORY.md Sec. 8e; the frozen numeric tables in INVENTORY.md Sec. 8a-8d).
5. **Run the seed-replication experiment.** This is the one piece of Claim 2 that is not a
   refactor of existing results — it is an actual unrun experiment
   (`data/claim2/seed_replication_PENDING.json`, ~22-70 GPU-hours depending on which cost estimate
   is used, see `AUDIT_STAGE1.md`). No later pass can produce a non-PENDING seed-generality result
   without running it.

Until these five things happen, "reproducing" this paper means reading the frozen files in
`data/` and trusting the two source repos' own REPORT.md / summary.json audit trail, not
re-executing anything from this directory.
