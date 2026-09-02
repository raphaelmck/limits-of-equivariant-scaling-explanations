# data/claim1/ — kernel and scaling inputs

Frozen inputs for the kernel and scaling results. Every row carries a `source_file` column naming
the upstream analysis output it was copied from; nothing here was recomputed or adjusted during
consolidation. `make analysis` derives the paper's tables from these files, and `make validate`
checks the derivations against the frozen copies also kept here.

## Scaling

- **`four_arch_force_checkpoint_table.csv`** — the per-checkpoint (compute, normalized force MSE)
  table for all four architectures, with `metric_valid` marking the rows eligible for the fit.
  This is the raw input to the exponents and to the Figure 1A frontier.
- **`force_scaling.csv`**, **`four_arch_force_scaling_fits.csv`** — the frozen fitted exponents,
  one row per architecture, on the common compute interval [8.49e15, 1.22e18] FLOPs. Two
  additional sensitivity estimators are recorded alongside the headline exponent and are not
  refitted here.

## Checkpoint selection

- **`owner_manifest.csv`** — the 12 frontier checkpoints (4 architectures x LOW/MID/HIGH) with
  parameter count, atom tokens, target and realized compute, and utilization.
- **`dense_grid_owner_force_mse.csv`** — the same for the five added budgets, including which
  budgets reuse a neighbouring budget's checkpoint.

## Kernel and KRR

- **`existing_kernel_inventory.csv`** — every (architecture, checkpoint) cell with a constructed
  force tangent kernel and its spectral and alignment summaries. The `checkpoint_state` and `m`
  columns distinguish the probe pools; no row was created for a cell without a kernel on disk.
- **`existing_krr_inventory.csv`** — the stacked KRR table in long form: architecture, checkpoint
  state, budget, pool size, training size, ridge setting, target definition, and median and mean
  test NMSE.
- **`matched_compute_m1024_krr_results.csv`** and **`..._split_results.csv`** — the learning
  curves and the per-split test NMSE at LOW, MID, and HIGH on the 1024-atom pool. The split-level
  file is what the paired bootstrap resamples.
- **`dense_grid_krr_results.csv`**, **`dense_grid_krr_split_results.csv`**,
  **`dense_grid_kernel_inventory.csv`** — the same for the five added budgets.

## Frozen outputs kept for comparison

`ranking_comparison.csv`, `dense_grid_ranking_comparison.csv`, `dense_grid_pairwise_gap.csv`, and
`pairwise_concordance.csv` are the frozen results that `make validate` compares the regenerated
tables against.

The kernel matrices themselves are not checked in; the evaluation pool that defines them is
recorded in `data/claim2/ood_pool_provenance.json`.
