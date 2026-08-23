<title>Claim 1 Paper Data Layer</title>

# Claim 1 — paper-facing data layer

This directory is a **thin consolidation layer**, not a new analysis. Every numeric value in
the five data files below was copied exactly (full precision, no re-derivation) from an
existing, already-validated output file somewhere under `analysis_outputs/` in one of the two
project checkouts. Nothing here was recomputed, refit, or adjusted. Where a number referenced
in `analysis_outputs/claim1_cleanup/CURRENT_PIPELINE.md`'s prose could not be located in a
source file exactly as stated, that is noted explicitly (see "Discrepancies" below) rather than
silently reproduced from the prose.

**Authoritative map of sources:** `analysis_outputs/claim1_cleanup/CURRENT_PIPELINE.md`. Read
that file first — it is the map this directory was built from, not the other way around.

**Two source repos** (read-only; nothing in this directory was written into either of them):
- `dev-equivariant-scaling-laws/` — empirical four-architecture force-scaling fits (Component 1).
- `dev-equivariant-scaling-laws-kernel-pilot-clean/` — checkpoint-owner selection, force-tangent-kernel
  construction, spectral summaries, alignment statistics, and empirical KRR (Components 2-12).

Every row of every CSV below carries a `source_file` column giving the path (relative to one of
the two repo roots above) of the file the value was copied from.

## Files

- **`force_scaling.csv`** — one row per architecture, the fitted force-scaling exponent gamma
  (and its two alternate estimators, R^2, fit domain, common-compute interval) on the identical
  2.16-decade common-compute window [8.49e15, 1.224e18] FLOPs, `force_mse_norm` metric,
  `truncation_frac=1.0`. This is the empirical target Claim 1 is trying to explain. Source:
  `dev-equivariant-scaling-laws/analysis_outputs/four_arch_force_scaling_repaired_2026_08_14/four_arch_force_scaling_fits.csv`
  (the REPAIRED/post-muP-bugfix version — the original `four_arch_force_scaling_2026_08_14/` is
  historical-only per `CURRENT_PIPELINE.md` and was not used here). Verified: the four gamma
  values (MPNN 0.2274, MC-EGNN 0.2964, GemNet-OC 0.3308, eSEN 0.6632) match
  `CURRENT_PIPELINE.md`'s quoted 0.227/0.296/0.331/0.663 to within rounding.

- **`owner_manifest.csv`** — the 12 frontier owners (4 architectures x LOW/MID/HIGH budget
  tiers): architecture, checkpoint path/step, N (trainable params), D (atom tokens), C (FLOPs,
  both the budget target and the actual owner value), utilization, and the
  `owner_reconstruction_audit_status` (a single top-level "PASS" in the source JSON that applies
  to all 12 owners — the audit file does not carry a distinct pass/fail field per owner, only a
  full reconstruction record per owner plus one overall status). Source:
  `dev-equivariant-scaling-laws-kernel-pilot-clean/analysis_outputs/four_arch_matched_compute_frontier_kernel_2026_08_15/selected_frontier_owners.csv`
  cross-checked against `owner_reconstruction_audit.json`.

- **`existing_kernel_inventory.csv`** — every (architecture, checkpoint-state) cell that has an
  actually-constructed force tangent kernel with spectral/alignment summaries on disk. Rows come
  from four distinct sources, distinguished by the `checkpoint_state`/`m` columns:
  - K0 (m=256, eigenvalue-spectrum k50/k80/k90/k95): `four_arch_initial_force_kernel_pilot_2026_08_14/spectral_descriptors.csv`
  - K0/K50k/K500k (m=256, "large" size tier only, target-power k50_y..k95_y and residual k50_residual..k95_residual): `four_arch_trained_force_kernel_pilot_2026_08_15/spectral_evolution_summary.csv`
  - matched-compute LOW/MID/HIGH (m=256, target-power k50_y..k95_y): `four_arch_matched_compute_frontier_kernel_2026_08_15/spectral_summary.csv`
  - task-directed alignment A(K_t,y) at each fixed step (m=256): `bordelon_feature_learning_kernel_audit_2026_08_17/task_directed_alignment.csv` (target="y" rows only; the "r_t" rows are an auxiliary residual-alignment statistic, not A(K,y), and are excluded here)
  - large-m (m=1024) K_x numerics + alignment A(K_x): `large_m_convergence_2026_08_19/{esen,gemnet_oc}_..._large_m_result.json`, GemNet-OC (width256) and eSEN (sphere64) only, both step=500000.
  No row was fabricated for a cell without an on-disk kernel — e.g. there is no MPNN/MC-EGNN
  m=1024 row, because no such kernel was ever constructed (see `MISSING_CELLS.md`).

- **`existing_krr_inventory.csv`** — the full stacked empirical KRR table (1,414 rows), tidy
  long format: architecture, checkpoint-state (K0/K50k/K500k/matched-compute-LOW/MID/HIGH/large-m-K500k),
  budget tier where applicable, m, n_train, ridge setting/value, target (raw_force at K0 in the
  static pilot / "common" vs "residual" target definitions in the trained and matched-compute
  pilots), median and mean test NMSE, split count. Stacked from:
  - `four_arch_initial_force_kernel_pilot_2026_08_14/krr_learning_curves.csv` (K0, m=256, static pilot)
  - `four_arch_trained_force_kernel_pilot_2026_08_15/krr_learning_curves_common_target.csv` + `_residual.csv` (K0/K50k/K500k, m=256)
  - `four_arch_matched_compute_frontier_kernel_2026_08_15/krr_learning_curves_common_target.csv` + `_residual.csv` (LOW/MID/HIGH, m=256)
  - `final_validation_sprint_2026_08_20/control_D_large_m_krr/krr_learning_curves_m1024.csv` (m=1024, GemNet-OC + eSEN only, "large" checkpoint, step=500000)
  This is the spine of Claim 1 (Component 6): the decisive empirical KRR comparison across
  architectures and matched-compute budgets.

- **`MISSING_CELLS.md`** — lays out the full 4-architecture x 3-budget-tier x {m=256, m=1024}
  target grid against what is actually in `existing_krr_inventory.csv`, and states explicitly
  which cells are covered, which are covered at the wrong checkpoint (m=1024 exists for
  GemNet-OC/eSEN only, and at a checkpoint that is NOT any of the 12 LOW/MID/HIGH frontier
  owners), and which are entirely missing (MPNN and MC-EGNN have no m=1024 KRR data at all).
  Ends with an explicit list of the 12 (architecture x budget) cells that would need a new
  matched-compute-owner m=1024 kernel+KRR run — not launched here, information-gathering only.

## Discrepancies / judgment calls made while building this layer

- `owner_reconstruction_audit.json` has one top-level `status: "PASS"` field, not a
  per-owner pass/fail flag as the task description assumed ("owner_reconstruction_audit status
  PASS/FAIL" per row). All 12 owners are covered by that single overall PASS; `owner_manifest.csv`
  applies it uniformly to every row and this is noted rather than fabricating a per-row field
  that doesn't exist in the source.
- `cumulative_target_alignment.png` in `four_arch_initial_force_kernel_pilot_2026_08_14/` indeed
  has **no CSV/JSON twin** on disk, as the task description anticipated. The k50/k90/k95(y)
  "target power" numbers `CURRENT_PIPELINE.md` quotes in Component 4's Notes ("GemNet-OC's
  target power ... k50/k90/k95(y) = 1-3 modes ... MPNN's diffuse spectrum k90(y)=99-208") are
  **not** in that pilot's `spectral_descriptors.csv` (which reports eigenvalue-spectrum k50/k90/k95,
  a different quantity, matching the numbers in that pilot's own `REPORT.md` table instead).
  The actual k50_y/k90_y/k95_y target-power numbers matching the pipeline doc's prose live in
  `four_arch_matched_compute_frontier_kernel_2026_08_15/spectral_summary.csv` (LOW/MID/HIGH
  values) and `four_arch_trained_force_kernel_pilot_2026_08_15/spectral_evolution_summary.csv`
  (K0/K50k/K500k values) — both are included in `existing_kernel_inventory.csv` with clear
  `checkpoint_state` labeling; the K0-pilot's own eigenvalue-spectrum k50/k90/k95 is also kept,
  separately labeled, so no row conflates the two different statistics.
- The `target` column in `existing_krr_inventory.csv` for the K0 static pilot rows is populated
  as `"raw_force"` since that pilot has no `target` column of its own (it predates the
  common/residual target distinction introduced in the trained and matched-compute pilots).
