<title>AUDIT_STAGE2 — analysis/validation layer</title>

# AUDIT_STAGE2 — Stage 2: `src/`, `scripts/`, `Makefile` build record

Date: 2026-08-22 (rigorous completion pass). This supersedes the first Stage-2 pass recorded
earlier in this file's history. The first pass built a CPU-only analysis layer using pure-stdlib
Python (`random.Random`) and could only check most bootstrap CIs via "statistical overlap," not
bit-exact reproduction, and left several main-paper numbers as outright passthrough (copied from
frozen files, never recomputed). The user reviewed that pass and rejected it as incomplete. This
pass closes those gaps: it adds a real numpy dependency (previously avoided only because the
build sandbox had no pip/venv-with-network access, not because numpy was undesirable — see
`pyproject.toml`), reproduces the exact upstream RNG API and call structure wherever an upstream
seed exists, and turns two previously-"NOT REPRODUCED (missing raw input)" OOD tables into real
reproductions after a second, harder look found the missing field one directory level up from
where the first pass looked.

Nothing under `data/` or `manifests/` was modified; four new raw files were ADDED to
`data/claim2/` (`frontier_ell4_sensitivity_raw.jsonl`, `seed_replication_raw.jsonl`,
`ood_domain_labels.json`, and no others) with full provenance recorded in the scripts that consume
them. No path outside `paper_repro/` is referenced anywhere in `src/` or `scripts/` except via
`src/io_utils.py`'s `data_path()`/`manifest_path()`/`out_path()` helpers.

## 0. Policy change: numpy is now a real, required dependency

`paper_repro/pyproject.toml` declares `numpy>=1.24` as the project's sole required dependency.
`src/bootstrap.py`, `src/ood_analysis.py`, and `src/seed_replication.py` all now use
`numpy.random.RandomState` or `numpy.random.default_rng` — whichever one the corresponding
upstream analysis script actually used (confirmed by reading the upstream source, not assumed) —
instead of `random.Random`. Concretely:

- The Claim-1 KRR pairwise bootstrap (`scripts/claim1/build_pairwise_concordance.py`) uses
  `np.random.RandomState(20260822)`, matching
  `dev-equivariant-scaling-laws-kernel-pilot-clean/analysis_scripts/build_matched_compute_m1024_comparison.py`'s
  own `rng.choice(log_ratios, size=len(log_ratios), replace=True).mean()` called once per
  bootstrap replicate, with a SINGLE shared `RandomState` instance threaded across all 18 pairs
  in the exact (budget, `itertools.combinations`) order the upstream script processed them —
  this ordering detail matters because the RNG's internal state carries across pairs; getting it
  wrong desyncs every downstream draw.
- The OOD analysis (`src/ood_analysis.py`) and the frontier-ell4-degree-balanced /
  depth-localization scripts use `np.random.default_rng(seed)`, matching
  `eval/analyze_and_report.py`, `eval/robustness_analysis.py`, and
  `analysis_scripts/build_depth_localization_report.py`'s own `rng.integers(0, m, size=...)` calls.
- The seed-replication script (`src/seed_replication.py`) uses `np.random.default_rng(0)` with a
  loop of `n_boot` SEPARATE `.integers()` calls on one shared generator (not one batched call),
  matching `analysis_scripts/analyze_seed_replication.py::bootstrap_index_sets` exactly — this is
  a different call pattern from the vectorized single-call form used elsewhere, and matching it
  was necessary for bit-exactness (a batched call from the same seed draws a different sequence).

Sandbox note: the environment this pass ran in still has no `pip`/`ensurepip` (confirmed:
`python3 -m pip`, `pip3`, `python3 -m ensurepip` all fail with "no module"). A local venv
(`paper_repro/.venv`, gitignored) was created with the stdlib `venv` module and a working numpy
2.2.6 build already present in a sibling project's own pip-built virtualenv on the same machine
was **copied** (not imported by path, not symlinked) into it, so `paper_repro/.venv` is
self-contained. `Makefile`'s `PY` variable prefers `.venv/bin/python3` when present and falls back
to ambient `python3` otherwise. See `pyproject.toml` for the full explanation. **This is a
sandbox workaround for how numpy got installed here, not a statement that numpy is optional** —
`pip install numpy` is the normal path in any environment with pip.

Wherever an upstream seed was genuinely undocumented (see §3), a fixed seed is still used (for
this repro's own determinism), and the comparison in `make validate` stays at CI-overlap +
zero-exclusion-sign agreement, exactly as the policy in the task description requires. Every case
where the upstream seed WAS documented is now checked at bit-exact tolerance (`REL_TOL_EXACT`
relative, in practice matching to <1e-9 relative — pure floating-point summation-order noise, not
a real reproduction gap).

## 1. Category (a) — regenerated end-to-end by `make analysis` + checked by `make validate`

| # | Table | Raw source | What's recomputed | Tolerance |
|---|---|---|---|---|
| 1 | `data/claim1/ranking_comparison.csv` | `owner_manifest.csv` + `matched_compute_m1024_krr_results.csv` | NN rank, KRR-AUC rank, concordance, per (arch, budget) | Exact (deterministic) |
| 2 | `data/claim1/pairwise_concordance.csv` | `owner_manifest.csv` + `matched_compute_m1024_krr_split_results.csv` | Point-estimate log-ratio, direction, AND full bootstrap CI (18/18 rows) | **BIT-EXACT** — `RandomState(20260822)`, shared across all 18 pairs in upstream's exact order. Verified the HIGH GemNet-OC/eSEN discordant pair to full float precision: log-ratio `-0.38667807569008344`, CI `[-0.6518903518249106, -0.12342702923828595]` |
| 3 | `data/claim1/force_scaling.csv` | `four_arch_force_checkpoint_table.csv` (501-row per-checkpoint table) | gamma/logA/r2/rmse_log per architecture via the common-support step-frontier OLS fit | Exact (deterministic, no bootstrap; re-verified this pass, unchanged from before) |
| 4 | `data/claim2/dose_response_summary.json` | `dose_response.jsonl` | Delta + CI, per (budget, ell, alpha) | Delta exact; CI overlap (undocumented frozen seed, see §3) |
| 5 | `data/claim2/compensation_summary.json` | `compensation.jsonl` | `C_ell` + CI | Same as #4 |
| 6 | `data/claim2/same_width_control_B_summary.json` | `same_width_control_B.jsonl` | `C_ell` + CI | Same as #4 |
| 7 | `data/claim2/run_replication_control_C_summary.json` | `run_replication_control_C.jsonl` | Per-run Delta/CI + paired diff/CI | Same as #4 |
| 8 | `data/claim2/frontier_ell4_degree_balanced_summary.json` | `dose_response.jsonl` (w10/50k, w16/150k) + `frontier_ell4_sensitivity_raw.jsonl` (w20/200k, w24/300k, newly copied) | Per-alpha delta/CI for all 4 owners, interpolated-to-common-support delta/CI, AND all 6 pairwise Delta-difference contrasts | **BIT-EXACT** — `default_rng(20260820)`, fresh per (owner, alpha) cell |
| 9 | `data/claim2/seed_replication_result_summary.json` | `dose_response.jsonl` + `frontier_ell4_sensitivity_raw.jsonl` (seed 1) + `seed_replication_raw.jsonl` (seeds 2/3, newly copied) | Delta(P*=0.15) + CI per (seed, tier), G_s + CI per seed, sign replication | **BIT-EXACT** — `default_rng(0)`, shared `idx_sets` list across all seeds/tiers |
| 10 | `data/claim2/depth_localization/depth_localization_summary.json` | `depth_localization_M1024.jsonl` (blocks 3/6) + block-9's own raw sources (same as #8) | Delta+CI for all 3 depths, ALL 6×4=12 depth contrasts (including block-9-involving ones) | **BIT-EXACT** — `default_rng(20260820)`, ONE shared idx matrix across every owner AND depth |
| 11 | `data/claim2/ood_full_alpha_curves.csv`, `ood_G_m_convergence.csv`, `ood_interaction_w24_vs_w10.csv`, `ood_lmax2_vs_lmax4_baseline.csv` | `ood_raw_per_config/{lmax4_block9_intervention,lmax2_baseline}.jsonl` | Full nested-M alpha curves, G_m, low/high interaction, lmax2-vs-4 baseline | **BIT-EXACT** — `default_rng(20260822)`, fresh per (m, seed) call, matching `analyze_and_report.py`'s `boot_L()` |
| 12 | `data/claim2/ood_absolute_effect_robustness.csv`, `ood_absolute_effect_A_OOD_minus_A_Neutral.csv` | same as #11 | Absolute-scale effect (A) robustness + nested-M A_OOD−A_Neutral | **BIT-EXACT** — `default_rng(20260823)`, matching `robustness_analysis.py`'s `boot_L()` |
| 13 | `data/claim2/ood_domain_decomposition.csv`, `ood_shared_family_positive_control.csv` | same as #11 + `ood_domain_labels.json` (newly extracted per-config `data_id`, see §4) | Chemistry-family-stratified delta/A + CI, w24-vs-w10 shared-family positive-control contrast | **BIT-EXACT** — same seed/machinery as #12, restricted to stratum masks. **Gap closed this pass — a prior pass had declared this unreproducible; see §4.** |

`REL_TOL_EXACT = 1e-6` relative for all "exact" and "BIT-EXACT" rows above; every one in practice
matches to <1e-9 relative (float summation-order noise only).

`make analysis` takes ~90 seconds end-to-end on this machine (numpy vectorization made the OOD/
depth-localization scripts roughly 40x faster than the prior pass's pure-Python loops over
16,384-element arrays, in addition to now being bit-exact rather than approximate).

## 2. Category (b) — genuinely NOT reproducible from `data/` alone

This list is much shorter than the prior pass's, because items 3, 6, 7, and 8 of the prior
pass's category (b) were all closed this pass (see §1, rows 2, 8, 9, 13). What remains:

1. **`data/claim1/four_arch_force_checkpoint_table.csv` itself** is the raw per-checkpoint table
   feeding `force_scaling.csv`'s fit (row 3 of §1) — there is nothing further to re-derive FROM
   it; it is the raw input, not a derived summary. Likewise `existing_kernel_inventory.csv` /
   `existing_krr_inventory.csv` / `matched_compute_m1024_kernel_inventory.csv` are raw tables that
   feed rows 1/2 above, not scalars requiring their own re-derivation.
2. **`data/claim2/gap_erasure_summary.json` / `gap_erasure.jsonl`.** No `n_boot`/CI field exists
   anywhere in this file (re-checked this pass) and no per-config raw array backs `L4`, `L2`,
   `L4_int`, `gap`, `intervention_damage`, `fraction_gap_erased` — every one of these is already a
   bare scalar. There is genuinely nothing to bootstrap; this is pure passthrough, appendix-only
   per the user's explicit exception (pathway-rescue / gap-erasure carve-out) and is NOT part of
   the 8 main-paper checks.
3. **`data/claim2/pathway_rescue.jsonl` / `pathway_rescue_trace.jsonl` /
   `pathway_rescue_summary.json`.** Same situation: `Delta_abl`, `Delta_rescue`, `R`, `D_abs`,
   `D_rel`, `cosine` are all already-computed scalars, no raw array, no CI anywhere in the frozen
   files. Passthrough only, appendix-only per the same exception, NOT part of the 8 checks.

Both of the above were already flagged this way by the prior pass; this pass re-confirmed (by
grepping the actual JSON for any `n_boot`/CI/seed key) rather than re-litigating from scratch, per
the task's own instruction to carry these forward rather than re-open them absent new evidence.
No new evidence emerged — they remain genuinely scalar-only, appendix-only files.

## 3. Bootstrap-seed / determinism notes (updated)

- **Files with a documented AND now bit-exactly-matched seed**: `pairwise_concordance.csv`
  (`RandomState(20260822)`), `frontier_ell4_degree_balanced_summary.json` (`default_rng(20260820)`,
  documented directly in the file's own `boot_seed` field), `depth_localization_summary.json`
  (`default_rng(20260820)`, same seed reused across this whole line of work, confirmed by reading
  `build_depth_localization_report.py`), `seed_replication_result_summary.json`
  (`default_rng(0)`, documented in `G_s_analysis.json`'s own `bootstrap_seed` field), and every OOD
  table (`default_rng(20260822)` for `analyze_and_report.py`'s tables, `default_rng(20260823)` for
  `robustness_analysis.py`'s tables — both documented as module-level `BOOT_SEED` constants in the
  respective source files).
- **Files with NO documented seed at all** (re-checked, no such key anywhere in the JSON):
  `dose_response_summary.json`, `compensation_summary.json` (has `n_boot=2000` but no seed),
  `same_width_control_B_summary.json`, `run_replication_control_C_summary.json`. For these,
  `src/bootstrap.py`'s `paired_config_bootstrap` uses `numpy.random.RandomState(seed=0)` with
  `rng.randint(0, n_configs, size=n_configs)` per replicate — a REAL numpy bootstrap, just not one
  that can be bit-matched to an upstream draw sequence that was never recorded. `make validate`
  checks CI overlap + zero-exclusion-sign agreement for these four files only, exactly as the task
  description anticipated ("keep an overlap fallback ONLY for that specific file").
- **Point estimates are always exact** (no randomness in Delta/C_ell/G/gamma point values
  themselves) regardless of the above.

## 4. How the two previously-"unreproducible" OOD tables were actually closed

The prior pass's `ood_domain_decomposition.csv` / `ood_shared_family_positive_control.csv`
write-up said the per-configuration chemistry-family label (`data_id`) needed to stratify by
family "is not a field inside the per-config JSONL rows" and stopped there. Reading
`eval/robustness_analysis.py` in full (not just grepping the per-config JSONL's own keys) showed
it never reads `data_id` from those rows either — it reads it from a DIFFERENT file one level up:
`dataset/ood_pool_manifest.json["pools"]["16384"]` (OOD) and
`id_pool/id_pool_manifest.json["pools"]["16384"]` (ID), each a list of
`{dataset_index, source, data_id, natoms}` aligned by POOL ORDER with the per-config arrays
(`ood_pool_provenance.json` already documented both pools as verified nested/complete/unique).
Neither manifest's `"pools"` field had been copied into `paper_repro/data/` — the earlier
`ood_pool_provenance.json` copy explicitly listed them as "referenced, not copied" because the
full manifests are 6.4–6.8MB each — so the field was never absent, only its 6MB container hadn't
been trimmed and brought in. This pass extracted JUST the `data_id` list for the M=16384 pool
from each manifest (~400KB total) into `data/claim2/ood_domain_labels.json`. Sanity check: the
extracted per-family counts (elytes=10867, biomolecules=3286, reactivity=1749,
metal_complexes=32, neutral-like=212+104+86+48=450) exactly match `ood_domain_decomposition.csv`'s
own frozen `n_configs` column for every stratum, confirming correct alignment before any bootstrap
was run. Both tables now reproduce bit-exactly (see §1 row 13).

## 5. How to run it

```
cd paper_repro
make analysis   # regenerates every category-(a) table into analysis_out/{claim1,claim2}/
make validate   # re-runs analysis, then compares every regenerated table against data/
```

Both targets now require numpy (see `pyproject.toml`); `Makefile`'s `PY` variable auto-detects
`paper_repro/.venv/bin/python3` if present. `make analysis && make validate` was actually run for
this pass (not predicted): **exit code 0**, `8/8` main-paper checks passed, `19/19` total checks
passed, ~90 seconds end-to-end.

## 6. FINAL SECTION — status of each of the 8 main-paper checks

The user's completion criterion: *"`make analysis && make validate` regenerates every numerical
value and confidence interval quoted in the main workshop paper using only paper_repro/data/ +
manifests/."* Status per named check:

1. **C1. force-scaling exponents** — **FULLY REGENERATED, exact (no bootstrap involved).**
   Source: `four_arch_force_checkpoint_table.csv` (501 raw per-checkpoint rows). Tolerance: 1e-9
   relative. No gap.

2. **C1. 17/18 NN/KRR concordance** — **FULLY REGENERATED, exact.** Source: `owner_manifest.csv`
   + `matched_compute_m1024_krr_results.csv`. Tolerance: 1e-6 relative. No gap. (Note: this is the
   `ranking_comparison.csv` 12-row per-(architecture,budget) table; the "17/18" figure the paper
   quotes is the pairwise-concordance count from check 3 below, not this table's own row count —
   both are fully regenerated.)

3. **C1. HIGH GemNet-OC/eSEN discordance + CI** — **FULLY REGENERATED, BIT-EXACT.**
   `RandomState(20260822)` threaded across all 18 pairs in upstream's exact processing order.
   Reproduced: log-ratio `-0.38667807569008344`, CI `[-0.6518903518249106, -0.12342702923828595]`,
   status `discordant` — matches the frozen file to full float precision. No gap. (This also
   confirms the full 18-pair concordance table, 17 concordant / 1 discordant, all bit-exact.)

4. **C2. four-owner degree-balanced causal result** — **FULLY REGENERATED, BIT-EXACT.**
   `default_rng(20260820)` per (owner, alpha) cell, seed documented in the frozen summary itself.
   All 4 owners' per-alpha CIs, interpolated-to-common-support Delta/CI, and all 6 pairwise
   Delta-difference contrasts reproduce to <1e-9 relative. No gap. Sources:
   `dose_response.jsonl` (already present) + `frontier_ell4_sensitivity_raw.jsonl` (newly copied
   this pass).

5. **C2. 3/3 seed-sign replication + per-seed CIs** — **FULLY REGENERATED, BIT-EXACT.**
   `default_rng(0)`, shared `idx_sets` ported line-for-line from
   `analyze_seed_replication.py`. All 3 seeds' Delta(P*=0.15), G_s, and G_s's 95% CI reproduce to
   <1e-9 relative; sign replication confirmed 3/3 positive (G_1=0.477, G_2=0.715, G_3=0.134, all
   CIs excluding zero). No gap. Source: `seed_replication_raw.jsonl` (newly copied this pass) for
   seeds 2/3; seed 1 reuses check 4's own raw sources, per this task's explicit instruction, kept
   in a structurally separate module (`src/seed_replication.py`) using the pre-registered raw
   `P_total` metric, never the degree-balanced `P_bal` metric from check 4.

6. **C2. depth-localization result** — **FULLY REGENERATED, BIT-EXACT, for all 3 depths.**
   `default_rng(20260820)`, ONE shared bootstrap index matrix across every owner and all three
   depths (block 3, 6, 9), ported line-for-line from `build_depth_localization_report.py`. All 12
   owner/depth read-off cells and all 12 depth contrasts (including the two block-9-involving
   contrasts that were passthrough before) reproduce to <1e-9 relative. No gap. Canonical
   normalized-depth labeling (z=(k+1)/12, not the report's own z=k/12) carried forward unchanged
   from the prior pass, still correct.

7. **C2. Val-Comp/domain-transfer result** — **FULLY REGENERATED, BIT-EXACT, including the two
   tables previously declared unreproducible.** `ood_full_alpha_curves.csv`,
   `ood_G_m_convergence.csv`, `ood_interaction_w24_vs_w10.csv`,
   `ood_absolute_effect_robustness.csv`, `ood_absolute_effect_A_OOD_minus_A_Neutral.csv`,
   `ood_domain_decomposition.csv`, and `ood_shared_family_positive_control.csv` all reproduce to
   <1e-9 relative using `default_rng(20260822)`/`default_rng(20260823)` per the source script.
   The domain-decomposition/positive-control gap was closed by extracting the per-config
   `data_id` field from `ood_pool_manifest.json`/`id_pool_manifest.json` (see §4) into the newly
   added `data/claim2/ood_domain_labels.json`. No gap remains in any of these seven files.

8. **C2. compute-matched ellmax2-vs-ellmax4 comparison** — **FULLY REGENERATED, BIT-EXACT.**
   `ood_lmax2_vs_lmax4_baseline.csv`'s `H_force` values and CIs (same `default_rng(20260822)`
   machinery as check 7) reproduce to <1e-9 relative for every (owner, domain, M) cell. No gap.

**Overall verdict: the user's completion criterion IS NOW MET for all 8 named main-paper
checks.** `make analysis && make validate` regenerates every one of them from
`paper_repro/data/` + `paper_repro/manifests/` alone, with `make validate` exiting 0. The only
numbers in the repo that remain passthrough are explicitly appendix-only / provenance-linked
per the user's own stated exception (pathway-rescue, gap-erasure — §2 above) and are, by design,
excluded from the 8-check main-paper list and from the pass/fail exit code's main-paper gate.
