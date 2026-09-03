# pipeline/ — the code that produced the frozen data

Everything under `data/` was produced by the scripts in this directory: the force tangent kernels
and their KRR curves, the symmetry-preserving interventions on eSEN's irreducible
representations, the matched-compute evaluations, and the two evaluation populations.

**These scripts do not run from this repository.** They need the model checkpoints, the OMol25
splits, `torch`, `fairchem-core==2.14.0`, the model implementations from the training repository,
and a GPU. They are checked in so that the method can be read and audited — how the intervention
is applied, what the fixed-normalization control actually does, how the Jacobian and Gram matrix
are assembled, how the evaluation pools were drawn — not so that `make` can invoke them. Nothing
here is part of `make all`; the reproduction path that does run here is described in
`REPRODUCING.md`.

Every file here was taken from a single frozen revision of the analysis repository, committed and
tagged specifically so the code that produced the reported results is versioned rather than left
in an uncommitted working tree. That revision's identifier (repository, branch, commit) is
withheld from this public copy for double-blind review — see `provenance/README.md` — and is
recorded privately for restoration after acceptance. Six files under `evaluation_pools/` were
copied from a working directory that was not under version control at the time; the copies here
are now the versioned record of that code.

Output directory names in these scripts match the study names in `data/PROVENANCE.md`, so a
number in `data/` can be traced to the script that produced it.

## What produced what

### `kernel/` — Figure 1

| Script | Role |
|---|---|
| `frontier_checkpoints.py` | checkpoint registry, model instantiation, strict state-dict loading, and the force Jacobian |
| `force_ntk.py`, `force_ntk_trained.py` | Jacobian collection and force-NTK construction at a checkpoint |
| `gram_numerics.py` | chunked Gram assembly and its numerical checks |
| `build_probe_pool.py` | the 1024-atom probe pool, extending the 256-atom pool so the two stay nested |
| `run_build_kernels.py`, `run_krr.py` | kernels and KRR curves at LOW, MID, and HIGH |
| `dense_grid_checkpoints.py`, `run_build_kernels_dense_grid.py`, `run_krr_dense_grid.py` | the same at the five added budgets bracketing the crossover |
| `krr_solver.py` | the block-KRR solver: ridge grid, ridgeless pseudoinverse, inner-validation selection, scoring |

### `intervention/` — Figure 2A and the appendix

| Script | Role |
|---|---|
| `esen_intervention.py` | the intervention itself: scale one degree at one block by alpha, leaving every sub-module unmodified |
| `esen_norm_control.py` | the shared normalization, reconstructed exactly, and the fixed-normalization control |
| `checkpoint_loading.py`, `evaluation_data.py`, `rotation.py`, `provenance.py` | checkpoint reconstruction, evaluation batches and the force metric, equivariance checks, result provenance |
| `lmax_checkpoints.py`, `lmax_frontier_checkpoints.py` | the ell_max=2 and ell_max=4 checkpoint specifications |
| `run_dose_response.py`, `run_dose_response_full_val.py` | the block-9 alpha sweep, on the evaluation pool and on the full Neutral population |
| `run_activation_stats.py`, `run_degree_balanced.py` | the per-degree activation power that normalizes the degree-balanced perturbation magnitude |
| `run_frontier_sensitivity.py` | the sweep at the two highest-compute checkpoints |
| `run_depth_localization.py` | the same intervention at blocks 3 and 6 |
| `run_independent_runs.py` | the two additional independent training runs |
| `run_norm_control.py` | natural versus fixed normalization |
| `run_compensation.py`, `run_same_width_control.py` | the matched-compute and same-width controls |
| `bundle.py`, `decoder.py`, `ridge.py`, `spectra.py` | shared probe machinery used by the controls |

### `matched_compute/` — Figure 2B and the 102-pair appendix result

| Script | Role |
|---|---|
| `find_matched_lmax2.py` | the matching rule: nearest ell_max=2 frontier checkpoint in log compute |
| `run_force_eval.py` | force evaluation of a matched pair |
| `run_decisive_point_eval.py` | the highest-compute pair covered by both families, with per-configuration residuals retained for its bootstrap |
| `run_frontier_gap_eval.py` | the log ratio across the whole matched frontier |

### `evaluation_pools/` — the two populations

| Script | Role |
|---|---|
| `scan_neutral_pool.py`, `scan_support_matched_pool.py` | scan each validation split, apply the charge, spin, and elemental-support criteria, and draw the nested stratified pools |
| `run_intervention_eval.py` | the intervention forward passes on both populations, writing per-configuration force-error sums and atom counts |
| `gen_frontier_configs.py`, `run_frontier_baseline.py`, `combine_frontier_results.py` | the full-frontier baseline evaluation and its combination into the matched-pair table |

## Placeholders

Paths are written against `<CHECKPOINT_ROOT>` (the checkpoint store), `<DATA_ROOT>` (the OMol25
splits), `<PROJECT_ROOT>`, and `<PIPELINE_ROOT>`. Fill these in before running anything.
