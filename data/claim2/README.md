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
- **`lmax24_frontier_dependence.csv`** — the ell_max=4 frontier checkpoints, and the training run
  each checkpoint belongs to, which is what the appendix's trajectory counts are computed from.
  These counts are the reason the 102 matched pairs must not be read as 102 independent
  experiments: they are dense checkpoints from 11 ell_max=4 trajectories (nine among the matched
  pairs) and eight ell_max=2 trajectories.

## Frozen summaries kept for comparison

`frontier_ell4_degree_balanced_summary.json`, the `ood_*.csv` files, and
`seed_replication_result_summary.json` are the frozen results that `make validate` compares the
regenerated tables against.

## Not included in this release

Several exploratory analyses from the upstream research project are not reported in the paper and
are not part of this release: a matched-compute compensation control, a same-width control, a
fixed-initialization run-replication control, a baseline activation-statistics probe, an
absolute-scale (rather than log-ratio) robustness check, and two abandoned mechanistic probes
(gap-erasure, pathway-rescue). None of the paper's reported numbers depends on them.
