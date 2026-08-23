# paper_repro

A clean, provenance-traceable data layer for a workshop paper spanning two claims about
architecture comparison in equivariant force-field scaling:

- **Claim 1** — global force-tangent-kernel (NTK/KRR/eigenlearning) diagnostics do not recover
  the comparative architecture behaviour observed empirically.
- **Claim 2** — a causal-intervention story about block-9 high-order irreps in eSEN, their
  compute-dependence, and why within-model causal importance does not imply comparative
  necessity across architectures.

This directory is intended to eventually become a standalone GitHub repository: a self-contained
record of exactly which checkpoints, kernels, interventions, and numeric results back each claim
in the paper, independent of the two large, messy research checkouts (`dev-equivariant-scaling-laws/`
and `dev-equivariant-scaling-laws-kernel-pilot-clean/`) the results were originally produced in.

## Current status: first-pass skeleton + data consolidation only

This pass is **consolidation only**: no new experiments were run, nothing was recomputed, and no
analysis code was refactored. Every number under `data/` was copied exactly from an existing,
already-validated file in one of the two source repos. See `AUDIT_STAGE1.md` for the full
file-by-file account of what was copied from where, what normalization was applied, and every
provenance ambiguity encountered.

`scripts/`, `src/`, `figures/`, and `tests/` are placeholders for a later pass (see their
individual `README.md` stubs) — they are not populated here.

## Layout

```
paper_repro/
├── README.md                  this file
├── REPRODUCING.md              honest current-state reproduction instructions
├── AUDIT_STAGE1.md             what was copied, from where, and open provenance issues
├── manifests/
│   └── checkpoints.csv         every checkpoint referenced by Claim 1 and Claim 2's final results
├── data/
│   ├── claim1/                 Claim-1 data layer (copied from paper_results/claim1/)
│   └── claim2/                 Claim-2 data layer (built from Claim-2 source analysis_outputs)
├── scripts/{claim1,claim2}/    RESERVED for a later pass (empty)
├── src/                        RESERVED for a later pass (empty)
├── figures/                    RESERVED for a later pass (empty)
├── tests/                      RESERVED for a later pass (empty)
└── paper/
    ├── CLAIMS.md                claim -> supporting-data-file mapping, incl. do-not-claim list
    └── FIGURE_PROVENANCE.md     survey of existing precedent figures for a later generation pass
```

## Where to start

- Read `paper/CLAIMS.md` first — it states each claim's exact verdict language (STRONG /
  SUPPORTED WITH QUALIFIER / PROVISIONAL) and which file in `data/` backs it.
- Read `AUDIT_STAGE1.md` for what was and wasn't done in this pass, and for every judgment call
  and unresolved discrepancy flagged during consolidation.
- Read `REPRODUCING.md` before assuming anything here is runnable end-to-end — it isn't yet.
