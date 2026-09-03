# Reproducing

There are three levels of reproduction. The first two run here; the third needs the checkpoints
and the dataset.

## 1. Reproduce the paper's numbers and figures (runs here, minutes, CPU only)

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
make all
```

- `make analysis` regenerates every derived table from `data/` into `analysis_out/`.
- `make validate` compares each regenerated table against the frozen table it reproduces and
  exits non-zero on any meaningful discrepancy.
- `make test` compares every number printed in the paper against those tables, labelled by the
  section, figure, or table it appears in.
- `make figures` renders `figures/figure1.pdf` and `figures/figure2.pdf`.

Nothing in `code/analysis/` reads a path outside this repository, so this level is self-contained
by construction. `analysis_out/` does not exist until `make analysis` creates it.

## 2. Re-derive the frozen tables from the intervention and kernel outputs

`data/` holds the per-configuration arrays the analyses aggregate: force-error sums and atom
counts per configuration for each baseline and intervened cell, per-split KRR test NMSE, and the
matched frontier pairs. From these, every damage value, log ratio, learning-curve area, and
bootstrap interval in the paper is recomputed by `make analysis` -- this is level 1, and the two
levels differ only in whether you trust the aggregation or re-run it.

What `data/` does not carry, because of size, is listed by path inside
`data/claim2/evaluation_pool_provenance.json`: the evaluation-pool index arrays and the kernel
matrices themselves. The pool definitions (population sizes, eligibility criteria, stratification
rule and seed) are recorded there in full, so the pools can be rebuilt.

## 3. Re-run the interventions and kernels from checkpoints

This needs the model checkpoints, the OMol25 Neutral and validation splits, and a GPU. Every
checkpoint referenced by any result is listed in `data/checkpoints.csv` with its architecture,
angular order, width, training step, parameter count, atom tokens processed, estimated training
compute, and, where recorded, its SHA256. Checkpoint paths in that file are written relative to a
`<CHECKPOINT_ROOT>` placeholder.

The code that did this is in `code/experiments/`, organized by what it produced (see
`code/experiments/README.md`): `kernel/` for the force NTK and its KRR curves, `intervention/` for
the degree-scaling intervention and the fixed-normalization control, `matched_compute/` for the
ell_max=2 versus ell_max=4 evaluations, and `evaluation_pools/` for the two populations. Fill in
the `<CHECKPOINT_ROOT>` and `<DATA_ROOT>` placeholders first.

Every file under `code/experiments/` was taken from a single frozen revision of the analysis
repository, committed and tagged specifically so this code is versioned rather than left in an
uncommitted working tree. That revision's identifier is withheld from this public copy for
double-blind review and is recorded privately for restoration after acceptance.

The model implementations, the training code, and the compute accounting are those of the scaling
study this work reanalyzes; see the paper's code and data availability statement.
