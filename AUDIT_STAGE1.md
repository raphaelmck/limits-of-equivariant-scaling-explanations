# AUDIT_STAGE1 — Consolidation pass record

Date: 2026-08-22. This document records exactly what was copied into `paper_repro/`, where every
file came from, what normalization was performed, and every provenance ambiguity encountered.
Nothing in either source repo (`dev-equivariant-scaling-laws/`,
`dev-equivariant-scaling-laws-kernel-pilot-clean/`), `analysis_outputs/claim1_cleanup/`, or
`paper_results/claim1/` was modified, moved, or deleted.

## 1. What was copied

### `data/claim1/` — straight copy

All 12 files under `/home/mila/m/michnikr/CODE/paper_results/claim1/` were copied byte-for-byte,
unchanged, into `data/claim1/`: `README.md`, `REPORT.md`, `force_scaling.csv`,
`owner_manifest.csv`, `existing_kernel_inventory.csv`, `existing_krr_inventory.csv`,
`MISSING_CELLS.md`, `matched_compute_m1024_kernel_inventory.csv`,
`matched_compute_m1024_krr_results.csv`, `ranking_comparison.csv`, `pairwise_concordance.csv`,
`_m1024_build_status.json`. `build.py` was also copied (kept for provenance — it is the script
that produced this directory's own CSVs from the underlying `analysis_outputs/`; harmless to keep
alongside the data even though `scripts/claim1/` is reserved for a later, cleaner refactor of it).
No file was renamed; `paper_results/claim1/`'s own naming was already consistent with the
`data/claim2/` naming used here, so no rename-mapping table was needed. Every CSV in this
directory already carries its own `source_file` column (established by that directory's own
prior build process) — no additional provenance annotation was needed or added.

### `data/claim2/` — built fresh from Claim-2 source directories

No file here is a byte-for-byte copy of a single source file except where noted; JSONL/JSON files
were re-serialized with an added `source_file` field on every row (JSONL) or as a new top-level
key (JSON) pointing at the exact repo-relative source path. Row/record order and all original
fields were preserved unchanged — no aggregation, no recomputation.

| File | Source | Rows | Provenance method |
|---|---|---|---|
| `dose_response.jsonl` | `.../stage3_block9_dose_response_2026_08_19/dose_response_M1024.jsonl` | 48 | `source_file` added per row |
| `dose_response_summary.json` | `.../stage3_block9_dose_response_2026_08_19/summary.json` | — | `source_file` top-level key |
| `matched_compute_lmax_owners.csv` | `.../stage3b1_true_lmax_frontiers_2026_08_16/frontier_owner_manifest.json`, cross-referenced with `stage3b1_frontier_lmax{2,4}.json` for exact checkpoint path/N/D | 8 | new CSV, `source_file` column, see discrepancy note below |
| `compensation.jsonl` | `.../stage3_lmax2_lmax4_compensation_2026_08_20/compensation_M1024.jsonl` | 72 | `source_file` added per row |
| `compensation_summary.json` | same dir, `summary.json` | — | `source_file` top-level key |
| `compensation_pathway_raw.jsonl` | same dir, `pathway_M1024.jsonl` | — | straight copy (no per-row provenance added — flagged below as a minor gap) |
| `same_width_control_B.jsonl` | `.../final_validation_sprint_2026_08_20/control_B_same_width/same_width_M1024.jsonl` | 72 | `source_file` added per row |
| `same_width_control_B_summary.json` | same dir, `summary.json` | — | `source_file` top-level key |
| `perturbation_control_A_summary.json` | `.../control_A_perturbation_magnitude/summary.json` | — | `source_file` top-level key |
| `perturbation_control_A_baseline_stats.json` | same dir, `block9_baseline_stats_M1024.json` | — | `source_file` top-level key |
| `gap_erasure.jsonl` | `.../stage3_high_order_gap_erasure_2026_08_20/gap_erasure_M1024.jsonl` | — | `source_file` added per row |
| `gap_erasure_summary.json` | same dir, `summary.json` | — | `source_file` top-level key |
| `pathway_rescue.jsonl` | `.../stage3_block9_pathway_rescue_2026_08_19/rescue_M1024.jsonl` | — | `source_file` added per row |
| `pathway_rescue_trace.jsonl` | same dir, `trace_M1024.jsonl` | — | `source_file` added per row |
| `pathway_rescue_summary.json` | same dir, `summary.json` | — | `source_file` top-level key |
| `run_replication_control_C.jsonl` | `.../control_C_seeds/seed_replication_M1024.jsonl` | — | `source_file` + `run_replication_type` added per row |
| `run_replication_control_C_summary.json` | same dir, `summary.json` | — | `source_file` top-level key |
| `seed_replication_PENDING.json` | new file, config/planning fields only | — | authored fresh, cross-referencing `SEED_REPLICATION_REQUIRED.md` and `PRE_REGISTRATION.md` (see discrepancy note below); contains no invented result value |

**Split-level detail preserved**: `dose_response.jsonl`, `compensation.jsonl`, and
`same_width_control_B.jsonl` each retain their original `per_config_sum`/`per_config_natoms`
per-row fields, which is exactly the information needed to reconstruct the paired
configuration-level bootstrap CIs (n_boot=2000) reported in the corresponding `summary.json`
files — nothing was aggregated away.

### `manifests/checkpoints.csv`

35 rows, one authoritative table, columns: `claim, study, architecture, role_tier, lmax,
width_label, width, global_step, seed, N_trainable_params, D_atom_tokens, C_flops,
checkpoint_path, checkpoint_sha256, status, source_file`.

- 12 rows: Claim-1's LOW/MID/HIGH x {MPNN, MC-EGNN, GemNet-OC, eSEN} owners, copied from
  `paper_results/claim1/owner_manifest.csv`.
- 8 rows: Claim-2's matched-compute lmax4+lmax2 owners (LOW/MID/UPPERMID/HIGH), from
  `frontier_owner_manifest.json` cross-checked against `stage3b1_frontier_lmax{2,4}.json` and
  `final_validation_sprint_2026_08_20/INVENTORY.md` Sec. 4.
- 8 rows: Control B's same-width pairs (SW_w10/16/20/24, lmax2 and lmax4), from INVENTORY.md
  Sec. 5.
- 3 rows: the block-9 dose-response LOW/MID/HIGH owners, kept distinct from the matched-compute
  study's HIGH per the task's explicit instruction (see below). Note LOW and MID here are the
  *same checkpoints* as Claim-1's eSEN LOW/MID owners and the matched-compute-study's LOW/MID
  lmax4 owners — verified directly from `dose_response_M1024.jsonl`'s own `checkpoint_path`
  field, not assumed.
- 4 rows: PENDING seed-replication runs (LOW/HIGH x seed2/seed3), config fields only, no
  checkpoint path/sha256/N/D/C (all correctly blank, `status=PENDING`).

## 2. Schema normalization performed

- Unified column name `role_tier` used across all claim-2 checkpoint rows (source documents use
  a mix of `budget_tier`, `label`, `tag`, and bare tier names — normalized to one column per the
  task's column list).
- JSONL files: added a `source_file` key to every row (see table above) rather than a
  file-level-only pointer, since these are the primary quantitative evidence files and per-row
  provenance is unambiguous and cheap.
- JSON (dict) files: added a single top-level `source_file` key rather than annotating nested
  fields, per the task's own stated allowance for "a single top-level/header provenance record
  for the whole file."
- No unit conversions, no renaming of scientific quantities (`Delta`, `q_power`, `C_ell`, etc.
  kept exactly as named in the source files) to avoid silently changing meaning.

## 3. Provenance issues — surfaced, and their resolution status

### 3a. `frontier_owner_manifest.json`'s own HIGH-lmax4 owner is internally inconsistent with
what was actually used downstream — **RESOLVED 2026-08-22**

The raw JSON at
`stage3b1_true_lmax_frontiers_2026_08_16/frontier_owner_manifest.json`, key
`this_session_frozen_mechanism_owners.points[3]` (label "HIGH (near C_common_max)"), literally
states `lmax4_owner: {sphere: 40, step: 200000, C_flops: 3.700e17}`. But
`final_validation_sprint_2026_08_20/INVENTORY.md` Sec. 4 states the HIGH lmax4 owner used by the
gap-erasure and compensation studies is `esen_lmax4_NEARMAX_w24_s300000` (C=3.347e17) — and this
is independently confirmed by directly reading
`stage3_lmax2_lmax4_compensation_2026_08_20/summary.json` and
`stage3_high_order_gap_erasure_2026_08_20/summary.json`, both of which report `C4_flops=3.347e17`
for their "HIGH" pair, matching INVENTORY.md's table and NOT the raw JSON's own
`this_session_frozen_mechanism_owners` field. This inconsistency exists inside the source repo
itself (i.e., it predates this consolidation pass); it was surfaced, not introduced, here.

**User-confirmed resolution (2026-08-22): the executed/audited owner, w24/s300000, is the
CANONICAL final-paper Claim-2 HIGH lmax4 owner.** `manifests/checkpoints.csv`'s
`matched_compute_HIGH` row (already using w24/s300000 since the original pass) now states this
explicitly as CANONICAL in its `source_file`/note column; the stale raw-JSON value (w40/s200000)
is retained there only as a labeled historical/superseded provenance note, never used as a value
anywhere in `data/claim2/` or `manifests/checkpoints.csv`.

### 3b. The two seed-replication planning documents disagree on the HIGH endpoint —
**RESOLVED 2026-08-22**

`final_validation_sprint_2026_08_20/control_C_seeds/SEED_REPLICATION_REQUIRED.md` specifies the
HIGH endpoint for the planned seed-2/seed-3 replication as **width=40, step=275,000** (matching
the block-9 dose-response HIGH owner and Claim-1's eSEN HIGH owner). `seed_replication_2026_08_20/PRE_REGISTRATION.md`
(the more detailed, separately-written pre-registration for the *same* not-yet-run experiment)
specifies the HIGH endpoint as **width=24, step=300,000** (matching the Stage-3B.1
matched-compute-study HIGH lmax4 owner). This was the exact ambiguity the original consolidation
pass surfaced rather than silently resolved.

**User-confirmed resolution (2026-08-22): the experiment actually being run pairs LOW=w10/s50000
with HIGH=w24/s300000** (i.e. `PRE_REGISTRATION.md`'s endpoint, consistent with the Stage-3B.1
matched-compute HIGH lmax4 owner — the same checkpoint now canonical per 3a above), seeds 2 and 3.
`data/claim2/seed_replication_PENDING.json` and the 4 PENDING rows in `manifests/checkpoints.csv`
now use w24/s300000 as the canonical HIGH endpoint; `SEED_REPLICATION_REQUIRED.md`'s w40/s275000
candidate is retained in both files as an explicitly labeled superseded/historical entry, not
used as the pending endpoint. No scientific result value was added or changed — only which
not-yet-run checkpoint configuration is canonical for the still-PENDING experiment.

### 3c. Cost-estimate discrepancy following directly from 3b — **RESOLVED 2026-08-22**

Because the two documents specify different HIGH-endpoint widths, they also give different total
GPU-hour estimates for the same nominal 4-run experiment: `SEED_REPLICATION_REQUIRED.md` estimated
~22 GPU-hours for its (now-superseded) w10+w40 endpoints; `PRE_REGISTRATION.md` estimates **~70
GPU-hours** (2.5h + 32.4h per seed) for the canonical w10+w24 endpoints. Since w24/s300000 is now
the canonical HIGH endpoint (3b), **~70 GPU-hours is the canonical cost estimate** for the pending
seed-replication experiment. `seed_replication_PENDING.json` records this plus the superseded
~22-hour estimate for the historical endpoint, clearly labeled as not applicable to the canonical
experiment.

### 3d. `compensation_pathway_raw.jsonl` had no per-row `source_file` — **FIXED 2026-08-22**

Unlike every other JSONL file in `data/claim2/`, `compensation_pathway_raw.jsonl` (copied from
`stage3_lmax2_lmax4_compensation_2026_08_20/pathway_M1024.jsonl`) was originally copied without
per-row provenance annotation — a build-script ordering oversight, not a judgment call. Fixed: a
`source_file` key (`dev-equivariant-scaling-laws-kernel-pilot-clean/analysis_outputs/stage3_lmax2_lmax4_compensation_2026_08_20/pathway_M1024.jsonl`)
has been added to all 26 rows, matching the convention used by every other Claim-2 JSONL file. No
other field was changed; row order and content are otherwise identical to the original copy.

## 4. Provenance correction pass, 2026-08-22

This section records a follow-up correction pass, made before starting the code/figure refactor
and before any scientific result value existed to change. No result value in `data/claim1/` or
`data/claim2/` was altered by this pass — only (a) which of two already-existing checkpoint
identifiers is designated canonical for Claim-2's HIGH lmax4 owner and the still-PENDING
seed-replication endpoint, both previously flagged as unresolved in 3a/3b above, and (b) the
`compensation_pathway_raw.jsonl` provenance gap from 3d. Changed files: `manifests/checkpoints.csv`,
`data/claim2/seed_replication_PENDING.json`, `data/claim2/compensation_pathway_raw.jsonl`, this
file. The historical/superseded values (w40/s200000 for the matched-compute HIGH owner;
w40/s275000 for the seed-replication HIGH endpoint; the ~22-GPU-hour cost estimate) remain
recorded in all three data files as explicitly labeled provenance, per instruction — they were not
deleted, only demoted from canonical to historical.

## 5. What remains for later passes

- **Code refactoring** of the two source repos' `analysis_scripts/` into `scripts/claim1/` and
  `scripts/claim2/` (currently empty placeholders — see their `README.md` stubs and
  `REPRODUCING.md`).
- **Shared library extraction** into `src/` (checkpoint loading, M=1024 population loader,
  per-atom force-error convention) — currently duplicated logic across the two source repos, not
  touched here.
- **Figure generation** into `figures/` — `paper/FIGURE_PROVENANCE.md` surveys existing precedent
  figures as a starting map; nothing was generated in this pass.
- **Tests** in `tests/` validating that a refactored pipeline reproduces the frozen numbers in
  `data/` — none exist yet.
- **The still-unrun seed-replication experiment** (`data/claim2/seed_replication_PENDING.json`).
  This is not a refactor — it is a genuine open experiment (~70 GPU-hours for the canonical
  w10/w24 endpoints, per the resolution in 3b/3c above) that must actually be run — using the
  general `gpu:1` GRES pool, not the scarce `gpu:a100` pool, per the memory finding on Mila GPU
  scarcity — before any "seed generality" claim can move out of PROVISIONAL.
- 3a and 3b are resolved as of the 2026-08-22 correction pass (section 3 above); no further human
  decision is needed on which checkpoint is canonical before the next refactor pass.

## 6. Extension pass, 2026-08-22 (later same day) — degree-balanced closure, OOD generalization, seed-training status correction

This section records a further extension of the Stage-1 consolidation layer, done after and
independent of the section-3/section-4 correction pass above. Nothing in section 1-5 was changed
by this pass except the two heading-numbering fixes (this section was originally mis-numbered as
a second "## 4"; renumbered to "## 5"/"## 6" here, content of both untouched). As before: no
scientific result value in any pre-existing file was altered; the two sources' repos and
`analysis_outputs/claim1_cleanup/` were only read, never modified.

### 6a. Task A — exact degree-balanced-RMS closure of the block-9 ell=4 frontier-sensitivity result

Source: `dev-equivariant-scaling-laws-kernel-pilot-clean/analysis_outputs/frontier_ell4_degree_balanced_perturbation_2026_08_21/` (`REPORT.md`, `summary.json`, `degree_balanced_points.csv`,
`centered_ss_M1024.json`). This is the FINAL authoritative version of this branch — it closes the
one open caveat (uncentered `ell=0`) from an earlier, now-superseded pass; `centered_ss_smoke_M8.json`
in that directory is a smoke test, not data, and was not used.

Added:
- `data/claim2/frontier_ell4_degree_balanced.csv` — 20 rows (4 owners x 5 alphas), straight copy
  of `degree_balanced_points.csv` with a `source_file` column added per row.
- `data/claim2/frontier_ell4_degree_balanced_summary.json` — `summary.json` with a top-level
  `source_file` key, plus pointers to `centered_ss_M1024.json` and `REPORT.md` and the report's
  own "ROBUST" verdict quoted in a `verdict` field.

**Not a duplicate of `data/claim2/dose_response.jsonl`**: `dose_response.jsonl` covers a different
owner set (w10/50k, w16/150k, w40/275000 — LOW/MID/HIGH) with `ell` in {2,3,4} measured via raw
whole-tensor RMS perturbation magnitude. The new file covers a different (larger) owner set (w10,
w16, w20, w24 — the true eSEN lmax=4 frontier), `ell=4` only, measured via the exact
degree-balanced quadratic form with `ell=0` centering. Both are legitimate, non-contradictory,
non-overlapping evidence for the same underlying phenomenon (block-9 high-order causal
load-bearing across compute); this note plus `CLAIMS.md`'s new "C2 refinement" subsection make the
relationship explicit so neither file is mistaken for supplanting or duplicating the other.

### 6b. Task B — Neutral-domain vs. support-matched OOD (Val-Comp) generalization experiment

**New source-location pattern**: this task's authoritative source,
`/home/mila/m/michnikr/CODE/analysis_outputs/esen_irrep_ood_2026_08_22/`, is **not** inside either
`dev-equivariant-scaling-laws/` or `dev-equivariant-scaling-laws-kernel-pilot-clean/` — it lives
directly under the top-level `analysis_outputs/` alongside (but outside) those two repos. Every
`source_file` value for data drawn from this directory is therefore recorded relative to
`/home/mila/m/michnikr/CODE/` (e.g. `analysis_outputs/esen_irrep_ood_2026_08_22/eval/full_alpha_curves.csv`),
a visibly different path shape from the `dev-equivariant-scaling-laws-kernel-pilot-clean/analysis_outputs/...`
paths used for every other Claim-2 file so far. Future consolidation passes should check for other
top-level `analysis_outputs/*_2026_*` directories that are similarly outside both named repos.

**Process-issue disclosure (carried forward verbatim in substance from `REPORT.md` §0):** a
research-only subagent dispatched to map the existing intervention code exceeded its instructions
and, on its own initiative, wrote `id_pool/scan_id_pool.py` (ran locally) and the full
`eval/run_gpu_eval.py` / `eval/analyze_and_report.py` GPU-evaluation implementation. It did **not**
submit any GPU job and did **not** modify any tracked repository file. Before any of this was used,
every load-bearing piece was independently re-verified against its claimed source: the
degree-balanced weights against `esen_norm_control.py` and the 2026-08-21 frontier artifact; the
intervention orchestration against `nets/uma/backbone.py`; all owner tags/checkpoint paths against
`stage3b_checkpoint_manifest.json` and confirmed to exist on disk; the loss convention against the
real training `DDPLoss`; both frozen pools independently re-verified as complete/unique/nested. The
GPU run itself (job 10445341) was submitted and supervised by the primary agent, not the subagent.
This is relevant provenance history and is not omitted.

Added (all under `data/claim2/`, all with per-row/top-level `source_file` provenance):
- `ood_pool_provenance.json` — both frozen pools' (Neutral-val "ID" and Val-Comp "OOD") definitions,
  sizes, and support criteria, built from `dataset/ood_pool_manifest.json`, `dataset/DATASET_AUDIT.md`,
  `dataset/eligible_summary.json`, `id_pool/id_pool_manifest.json`, `id_pool/DATASET_AUDIT.md`,
  `robustness/neutral_val_provenance.json`. The raw `.npy`/`.npz` per-structure index arrays, and
  each manifest's own large `pools` field (6.4MB/6.8MB of index lists), are **referenced by path,
  not copied inline** — same convention as this repo's Claim-1 kernel `.npz` files.
- `ood_full_alpha_curves.csv` — 200 rows, straight copy of `eval/full_alpha_curves.csv` (+`source_file`);
  this is also where the `P_bal` common-support/matched-alignment values live (per REPORT.md §2 —
  no separate file was needed for this item).
- `ood_G_m_convergence.csv` — 20 rows, from `eval/G_m_convergence.csv`.
- `ood_interaction_w24_vs_w10.csv` — 5 rows, from `eval/interaction_w24_vs_w10.csv`.
- `ood_lmax2_vs_lmax4_baseline.csv` — 40 rows, from `eval/lmax2_vs_lmax4_baseline.csv`.
- `ood_absolute_effect_robustness.csv` (40 rows, from `robustness/baseline_effect_robustness.csv`)
  and `ood_absolute_effect_A_OOD_minus_A_Neutral.csv` (20 rows, from `robustness/A_OOD_minus_A_Neutral.csv`)
  — kept as two separate files rather than combined, since the source itself keeps them separate
  and they have different grains (per-owner/domain/M vs. paired A-difference).
- `ood_domain_decomposition.csv` — 24 rows, from `robustness/domain_decomposition_M16384.csv`.
- `ood_shared_family_positive_control.csv` — 6 rows, from `robustness/w24_vs_w10_contrast_by_stratum.csv`.
  **Ambiguity flagged and resolved by inspection**: the task's Item 9 asked which file actually
  holds "the positive control" described in `OOD_ROBUSTNESS.md`. `w24_vs_w10_contrast_by_stratum.csv`
  is confirmed to be that file — it is the only one containing both the `pooled_neutral_like_sources_in_ValComp`
  row and the `Neutral_val_pooled_(ID_reference)` row whose statistical indistinguishability *is*
  the positive-control result (§2 of `OOD_ROBUSTNESS.md`). This is distinct data from
  `domain_decomposition_M16384.csv` (which has per-stratum `L_base`/`delta`/`A` at the
  individual-owner level, not the w24-minus-w10 contrast) — kept as a separate file, not folded in.
- `ood_raw_per_config/lmax4_block9_intervention.jsonl` (40 rows, ~31MB) and
  `ood_raw_per_config/lmax2_baseline.jsonl` (8 rows, ~3MB), both copied in full from
  `eval/raw/*.jsonl` with a `source_file` key added per row, plus `ood_raw_per_config/run_meta.json`
  (top-level `source_file` key). **Size judgment**: each row already carries the full
  per-configuration `(sum, natoms)` pairs needed for the paired n_boot=2000 bootstrap (arrays up to
  16,384 elements per row) — only 48 rows total, so copying in full (rather than referencing by
  path) was judged appropriate; this is the same per-configuration-array pattern already used for
  `dose_response.jsonl`/`compensation.jsonl`/`same_width_control_B.jsonl` in section 1 above. The
  `*_smoke.jsonl` files in the source `eval/raw/` directory are smoke tests, not data, and were not
  copied.

### 6c. Task C — checkpoint manifest and seed-replication status correction (training complete, analysis pending)

**Verified independently this pass** (not merely taken on faith from the task instructions):

| tag | job_id | sacct state | sacct end | checkpoints present |
|---|---|---|---|---|
| w10_seed2 | 10429387 | COMPLETED | 2026-08-22T01:51:06 | full ladder 50k-500k + last.ckpt |
| w10_seed3 | 10429388 | COMPLETED | 2026-08-22T01:29:27 | full ladder 50k-500k + last.ckpt |
| w24_seed2 | 10429389 | COMPLETED | 2026-08-22T13:32:26 | full ladder 50k-500k + last.ckpt |
| w24_seed3 | 10429390 | COMPLETED | 2026-08-22T14:31:02 | full ladder 50k-500k + last.ckpt, last-v1.ckpt |

Source: `dev-equivariant-scaling-laws-kernel-pilot-clean/analysis_outputs/seed_replication_2026_08_20/run_manifest.json`
(job IDs, GRES `gpu:a100l:1`, config: w10/step_target=50000, w24/step_target=300000 — matches the
canonical HIGH endpoint from section 3b above). The manifest's own `status: "submitted"` field is
confirmed **stale** (written 2026-08-20 at submission time, never updated); the real status per
`sacct` is `COMPLETED` for all 4 jobs.

**Searched for a downstream analysis output and found none**: grepped
`dev-equivariant-scaling-laws-kernel-pilot-clean/analysis_outputs/` for any JSON/CSV referencing
`seed2` or `seed3` that looks like a dose-response/intervention result rather than a training
artifact. Only two hits: `seed_replication_2026_08_20/run_manifest.json` itself (the training
submission manifest, not a result) and an unrelated hit in
`stage2b_representation_target_spectra_2026_08_15/validation_report.json` (predates the seed
replication runs, unrelated). **No `G(seed)` result exists anywhere.** Per the task's own
instruction, since no analysis-with-numbers was found, the correct status is
`TRAINING_COMPLETE_ANALYSIS_PENDING`, not "RUNNING" and not a completed result — this is the
status now recorded.

Changed:
- `manifests/checkpoints.csv`'s 4 PENDING rows: `status` changed to
  `TRAINING_COMPLETE_ANALYSIS_PENDING`; `checkpoint_path` and `checkpoint_sha256` filled in from
  the actual on-disk target-step checkpoint files (w10/step=50000, w24/step=300000, one per
  seed/width combination — all four files' sha256 computed directly via `sha256sum`, each file
  15-32MB, cheap to hash). `N_trainable_params`/`D_atom_tokens`/`C_flops`/any result value left
  blank, per instruction, since no analysis output exists to source them from.
- `data/claim2/seed_replication_PENDING.json`: `status` field changed from `PENDING` to
  `TRAINING_COMPLETE_ANALYSIS_PENDING`; a new `training_status_2026_08_22` block added recording
  the sacct evidence table above and the confirmed-empty analysis-output search. File name is
  unchanged (still `seed_replication_PENDING.json`, per instruction — the *analysis* this file is
  ultimately about is still pending even though training finished). No numeric result value was
  added anywhere in the file.
- `paper/CLAIMS.md`'s seed-generality section: updated from "no result file exists yet" framing to
  explicitly distinguish training completion (done, evidenced) from analysis completion (not done);
  the claim itself remains explicitly PROVISIONAL/NOT ESTABLISHED.

### 6d. `paper/CLAIMS.md` additions (Task D)

Added a "C2 refinement" subsection under C2 for the degree-balanced closure (6a), and a new "C6"
claim section for the Neutral-domain-vs-Val-Comp OOD generalization result (6b), quoting each
source's own verdict/bottom-line language verbatim rather than softening or strengthening it (the
OOD source has no single boxed "Verdict:" tag like the degree-balanced-closure report does, so its
own "Bottom line" section language is quoted directly instead of inventing a tag not present in the
source). Updated the seed-generality section per 6c above.

### 6e. Ambiguities and judgment calls flagged in this pass

1. Whether to copy `eval/raw/*.jsonl` in full vs. reference by path (Task B item 10) — resolved in
   favor of copying in full given the small row count (48) despite the ~34MB combined size; see 6b.
2. Which file is "the positive control" (Task B item 9) — resolved as
   `w24_vs_w10_contrast_by_stratum.csv`, per the reasoning in 6b; `domain_decomposition_M16384.csv`
   is materially different data (per-owner, not the w24-vs-w10 contrast) and was kept separate.
3. Whether "matched-P_bal effects" (Task B item 3) needed its own file — resolved that
   `eval/full_alpha_curves.csv`'s own `P_bal` column already carries this; no separate file was
   built, per the task's own "check before duplicating" instruction.
4. Whether to combine the two absolute-effect-robustness source files into one output file (Task B
   item 7) — resolved to keep them separate (matching the source's own two-file split, different
   grains), documented in 6b.

## 7. Seed-replication closure, 2026-08-22 (Claim-2 seed generality — analysis run to completion)

This section records the closure of the seed-replication experiment left as
`TRAINING_COMPLETE_ANALYSIS_PENDING` at the end of section 6c. Training was already done (4 jobs,
verified via `sacct`, see 6c); this pass ran the actual GPU intervention analysis that produces
`G(seed)`, on real GPU compute against the real checkpoints, and updated every downstream file.

**What was run** (all inside `dev-equivariant-scaling-laws-kernel-pilot-clean`, NOT modifying
`dev-equivariant-scaling-laws/` or any existing frozen `analysis_outputs/` file's values — only
new files/directories were added):

- New driver `analysis_scripts/run_seed_replication_ell4_sensitivity.py`, generalizing the
  existing frozen `analysis_scripts/run_frontier_ell4_sensitivity.py` to an arbitrary
  checkpoint/width/step/seed instead of that script's hardcoded seed-1-only owner registry. The
  intervention math itself (`esen_intervention.py`, `esen_norm_control.py`, `stage3a_data.py`,
  `rotation.py`, `run_stage3a1_norm_control.per_config_accum`) is imported and reused unchanged,
  per the task's hard constraint; only checkpoint/config-loading code is new (required because
  the existing loader hardcodes `cfg.seed == 1`, false by design for 3 of the 4 new checkpoints).
- Submitted via new `scripts/run_seed_replication_ell4_sensitivity.sbatch`
  (8 CPU, 48G, `gres=gpu:a100l:1`, `--time=01:00:00`, `--partition=main`) — two jobs: 10447862
  (crashed on a config-loading edge case for the one owner with no genuine resolved `.hydra/config.yaml`
  on disk; fixed, documented in the driver's own inline comments) and its clean resubmission,
  10447863 (`COMPLETED`, wall-clock 00:01:57 for all 4 checkpoints x 5-alpha sweep at M=1024, in
  line with the ~2-minute precedent this experiment was costed on).
- Analysis/bootstrap: new `analysis_scripts/analyze_seed_replication.py`, reusing the exact
  paired-bootstrap convention (`n_boot=2000`, seed=0, per-config `(sum, natoms)` resampling)
  already established in `analyze_stage3a.py`/`analyze_stage3_block9_dose_response.py`,
  generalized to a fixed-weight linear interpolation to `P*=0.15` between the two measured alpha
  points bracketing it (no extrapolation; all 6 (seed, endpoint) cells were in-support).
- Full write-up: `analysis_outputs/seed_replication_2026_08_20/REPORT.md` (new file in the
  existing, previously-frozen `seed_replication_2026_08_20/` directory — `PRE_REGISTRATION.md`
  and `run_manifest.json` in that directory were read only, not modified).

**Result**: `G_s = Delta_HIGH,s(P*=0.15) - Delta_LOW,s(P*=0.15)`, raw `P_total` metric (per the
task's hard constraint, NOT the degree-balanced-RMS metric from section 6a):

| seed | G_s | 95% CI |
|---|---|---|
| 1 (original) | +0.477 | [0.429, 0.534] |
| 2 (new) | +0.715 | [0.658, 0.778] |
| 3 (new) | +0.134 | [0.107, 0.166] |

Sign replicates 3/3 (every seed's CI excludes zero, same sign as the original). Magnitude does
NOT tightly replicate (>5x spread across seeds). Baseline force-loss levels across the 3 seeds are
consistent at both endpoints (LOW spread 6.8%, HIGH spread 1.8%), both within the project's own
established 3-13% between-run-noise precedent (`FINAL_VALIDATION_REPORT.md` Section D item 1) —
ruling out a divergent/bad training run. All 4 new checkpoints passed the identity and
sector-isolation self-consistency gates cleanly; rotation-equivariance passed for 3 of 4 owners
outright and for the 4th (`seedrepl_LOW_w10_seed2`) with one benign, noise-floor-level flag at a
single rotation seed (documented in `REPORT.md` Section 1 — the same known small-model fp32-noise
pattern `FINAL_VALIDATION_REPORT.md` Section D item 6 already documents elsewhere in this
project).

**Verdict written into `paper/CLAIMS.md`**: seed-generality upgraded from
PROVISIONAL/NOT-ESTABLISHED to **ESTABLISHED FOR SIGN** (3/3 seeds, all seed-level CIs exclude
zero) with an explicit, un-softened caveat that magnitude is NOT tightly replicated (>5x spread).
The C2 headline verdict itself is upgraded from "STRONG (within-model, N=1 initialisation)" to
"STRONG" with the N=1 caveat narrowed to apply only to magnitude, not sign. "Claims that must NOT
appear" item 5 was narrowed to match: sign-generalization statements are now permitted;
magnitude-generalization statements are not.

**Files changed in this pass**:
- `data/claim2/seed_replication_result.csv` (new) — tidy per-(seed, tier, alpha) table plus the
  P*=0.15 interpolated row per (seed, tier), with bootstrap CIs.
- `data/claim2/seed_replication_result_summary.json` (new) — canonical G_s/CI/sign-replication/
  baseline-comparison/verdict summary; this is the file to cite going forward.
- `data/claim2/seed_replication_PENDING.json` — NOT deleted; `status` changed to
  `SUPERSEDED_BY_ANALYSIS_COMPLETE` and a note added pointing to the two new canonical files above;
  its own body (the pre-analysis planning content) left otherwise unmodified as a historical
  record.
- `manifests/checkpoints.csv` — the 4 seed-replication rows: `status` changed from
  `TRAINING_COMPLETE_ANALYSIS_PENDING` to `ANALYSIS_COMPLETE`; `N_trainable_params`/
  `D_atom_tokens`/`C_flops` filled in (these are architecture properties, seed-independent,
  verified against each checkpoint's own strictly-loaded state-dict parameter count — identical
  to the corresponding seed=1 owner at that width, as expected); `source_file` column appended
  with a pointer to this analysis.
- `paper/CLAIMS.md` — seed-generality section rewritten (see above); C2 headline verdict line
  updated; "claims that must NOT appear" item 5 narrowed.
- `AUDIT_STAGE1.md` — this section.

**Caveats carried forward, not resolved by this pass** (unchanged scope, per
`PRE_REGISTRATION.md` Section 9): no seed replication was run for MID/UPPERMID tiers, no
ell_max=2 comparison, no new mechanism search. The magnitude non-replication itself is a
substantive open finding, not a caveat to be explained away — it should be reported in the paper
exactly as measured.
