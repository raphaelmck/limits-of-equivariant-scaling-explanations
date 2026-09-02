# data/claim2/ — intervention and matched-compute inputs

Frozen inputs for the intervention results. The intervention runs happen upstream on GPU and
write, per (checkpoint, degree, alpha) cell, the summed squared force error and the atom count for
each evaluation configuration. Everything in `analysis_out/claim2/` is re-aggregation over those
arrays: L = sum(per_config_sum) / sum(per_config_natoms), Delta = log(L_int / L_base), and a
paired configuration-level bootstrap.

## Per-configuration arrays

- **`dose_response.jsonl`** — the main dose-response grid, block 9, all degrees and alphas, at the
  low- and mid-compute checkpoints.
- **`frontier_ell4_sensitivity_raw.jsonl`** — the same at the two higher-compute checkpoints.
- **`depth_localization/`** — blocks 3 and 6, plus the frozen summary and validation record.
- **`seed_replication_raw.jsonl`** — the two additional independent training runs.
- **`ood_raw_per_config/`** — both evaluation populations at the 16384-configuration pool, for the
  ell_max=4 intervention and the ell_max=2 baseline, with the run metadata.
- **`compensation.jsonl`**, **`same_width_control_B.jsonl`**, **`run_replication_control_C.jsonl`**
  — the three controls.

## Pools and labels

- **`ood_pool_provenance.json`** — both evaluation populations: sizes, charge/spin/element
  eligibility criteria, the stratification rule and seed, nesting verification, and the paths of
  the index arrays that are too large to check in.
- **`ood_domain_labels.json`** — the per-configuration chemistry-family label for the
  16384-configuration pool, aligned by pool order to the arrays above.

## Matched-compute comparison

- **`matched_compute_lmax_owners.csv`** — the four matched ell_max=2 / ell_max=4 pairs that also
  have direct intervention measurements.
- **`lmax24_all_frontier_matches.csv`** — all 126 ell_max=4 frontier checkpoints and their nearest
  ell_max=2 frontier partner, with the compute ratio and whether the pair is within tolerance.
- **`lmax24_all_frontier_gaps.csv`** — log(L_2 / L_4) per valid pair on both evaluation
  populations, with intervals where a bootstrap was run.
- **`lmax4_all_frontier_owners.csv`**, **`lmax24_frontier_dependence.csv`** — the ell_max=4
  frontier checkpoints, and the training run each checkpoint belongs to, which is what the
  appendix's trajectory counts are computed from. These counts are the reason the 102 matched
  pairs must not be read as 102 independent experiments: they are dense checkpoints from 11
  ell_max=4 trajectories (nine among the matched pairs) and eight ell_max=2 trajectories.
- **`lmax4_causal_vs_comparative.csv`** — the four checkpoints that have both an intervention
  measurement and a matched-compute comparison.
- **`lmax24_repro_check.json`** — a check that the matching rule reproduces the four originally
  chosen pairs. It reproduces three of the four exactly; the mid-compute pair resolves to an
  adjacent training step of the same width.

## Frozen summaries kept for comparison

The `*_summary.json` files and `frontier_ell4_degree_balanced.csv`,
`ood_*.csv`, `seed_replication_result*` are the frozen results that `make validate` compares the
regenerated tables against.

## Additional recorded outputs

`gap_erasure*`, `pathway_rescue*`, `compensation_pathway_raw.jsonl`, and
`perturbation_control_A_*` record exploratory analyses that no result in the paper depends on.
They are kept because they are part of the record of what was tried, and they carry only
aggregated scalars, so no script here regenerates them.
