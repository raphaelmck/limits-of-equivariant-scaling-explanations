# scripts/claim1/ — kernel and scaling tables

Entry points that load `data/claim1/`, call `src/`, and write to `analysis_out/claim1/`. They read
only from this repository.

- **`build_force_scaling.py`** — the four compute-scaling exponents (paper Sec. 2).
- **`build_frontier_points.py`** — the empirical step frontier plotted in Figure 1A.
- **`build_dense_grid_ranking_comparison.py`** — neural versus KRR ranking at all eight budgets
  (Figure 1B).
- **`build_dense_grid_pairwise_gap.py`** — the signed GemNet-OC/eSEN gap, neural and KRR, at all
  eight budgets (Figure 1C, Table 1).

Run via `make analysis`, or invoke one directly with `python3 scripts/claim1/<script>.py`.
