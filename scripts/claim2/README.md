# scripts/claim2/ — intervention and matched-compute tables

Entry points that load `data/claim2/`, call `src/`, and write to `analysis_out/claim2/`. They read
only from this repository.

- **`build_frontier_ell4_degree_balanced.py`** — the block-9 ell=4 dose response at the four
  frontier checkpoints (Figure 2A).
- **`build_ood_curves_and_contrasts.py`** — intervention curves and contrasts on both evaluation
  populations, including the matched-compute log ratio behind Figure 2B.
- **`build_ood_domain_decomposition.py`** — the chemistry-family decomposition and the
  shared-source positive control.
- **`build_ood_absolute_effect_robustness.py`** — the same effect on an absolute rather than a
  relative scale.
- **`build_depth_localization.py`** — blocks 3, 6, and 9, with paired contrasts between depths.
- **`build_seed_replication_result.py`** — the three independent training runs.
- **`build_lmax24_frontier.py`** — the 102 matched frontier pairs and their summary.
- **`build_dose_response.py`**, **`build_compensation.py`**, **`build_same_width_control_B.py`**,
  **`build_run_replication_control_C.py`** — the supporting dose-response grid and the three
  controls (matched-compute compensation, same width, fixed-initialization replicate).

Run via `make analysis`, or invoke one directly with `python3 scripts/claim2/<script>.py`.
