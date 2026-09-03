# code/analysis/claim2/ — intervention and matched-compute tables

Entry points that load `data/claim2/`, call `code/analysis/common/`, and write to
`analysis_out/claim2/`. They read only from this repository.

- **`build_frontier_ell4_degree_balanced.py`** — the block-9 ell=4 dose response at the four
  frontier checkpoints (Figure 2A).
- **`build_intervention_domain_curves.py`** — intervention curves and contrasts on both evaluation
  populations, including the matched-compute log ratio behind Figure 2B.
- **`build_chemistry_domain_decomposition.py`** — the chemistry-family decomposition and the
  shared-source positive control.
- **`build_depth_interventions.py`** — blocks 3, 6, and 9, with paired contrasts between depths.
- **`build_seed_replication_result.py`** — the three independent training runs.
- **`build_lmax24_frontier.py`** — the 102 matched frontier pairs and their summary.

Run via `make analysis`, or invoke one directly with `python3 code/analysis/claim2/<script>.py`.
