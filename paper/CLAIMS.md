# Claims — paper data-layer mapping

This table maps each final paper claim to the exact clean data file(s) in `data/claim1/` or
`data/claim2/` that support it. Verdict language is copied verbatim from
`dev-equivariant-scaling-laws-kernel-pilot-clean/analysis_outputs/final_validation_sprint_2026_08_20/FINAL_VALIDATION_REPORT.md`
(the "Claim classification" table and section A/B) — do not soften or strengthen any verdict
when drafting the paper from this file.

## C1 — Force-tangent-kernel diagnostics fail to recover comparative architecture behaviour

**Verdict: STRONG. Main text.**

"Global force-tangent-kernel task-alignment diagnostics do not recover the comparative
architecture behaviour" — GemNet-OC beats eSEN at every n and every ridge (30/30 cells) in the
frozen m=1024 block-KRR reference, the opposite of the real neural force-scaling hierarchy
(eSEN > GemNet-OC beyond ~1.64e17 FLOPs).

Framing to use: "frozen, full-output force tangent kernel", paired with the native-accessibility
estimand mismatch (kernel track vs. representation-probe track disagree because they measure
different things, not because one is wrong).

Supporting files:
- `data/claim1/force_scaling.csv` — the empirical force-scaling target (gamma_MPNN=0.227 <
  gamma_MCEGNN=0.296 < gamma_GemNetOC=0.331 < gamma_eSEN=0.663) that the kernel diagnostics fail
  to recover.
- `data/claim1/owner_manifest.csv` — the 12 LOW/MID/HIGH matched-compute owners.
- `data/claim1/existing_kernel_inventory.csv`, `data/claim1/existing_krr_inventory.csv` — the
  full stacked kernel/alignment and KRR tables, K0 through m=1024.
- `data/claim1/matched_compute_m1024_kernel_inventory.csv`,
  `data/claim1/matched_compute_m1024_krr_results.csv` — Control D (m=1024 KRR), the most current
  and most heavily cross-checked instance.
- `data/claim1/ranking_comparison.csv`, `data/claim1/pairwise_concordance.csv` — the
  kernel-vs-neural ranking disagreement.
- `data/claim1/MISSING_CELLS.md` — explicit statement of which (architecture x budget x m) cells
  are not covered, so no absence is misread as a null result.

## C2 — Block-9 high-order irreps are causally load-bearing within lmax=4 eSEN

**Verdict: STRONG. Main text.** Originally reported as "within-model, N=1 initialisation";
**updated 2026-08-22** — the sign of the HIGH-vs-LOW contrast now replicates 3/3 across
independent training seeds (see "Seed generality" section below), so N=1 no longer applies to the
sign of the effect. The magnitude, however, is still only established at N=1 per seed with a
>5x spread across the 3 seeds sampled — disclose this narrower magnitude caveat instead.

Evidence: NORM_FIXED + graded alpha + full-population + pathway/rescue, reinforced by Control A
(perturbation matching) and Control C (run replication — NOT seed replication, a fixed-init
trajectory replicate).

Supporting files:
- `data/claim2/dose_response.jsonl` + `data/claim2/dose_response_summary.json` — the graded
  alpha dose-response at block 9, ell in {2,3,4}, LOW/MID/HIGH.
- `data/claim2/perturbation_control_A_summary.json` + `perturbation_control_A_baseline_stats.json`
  — Control A, perturbation-magnitude matching.
- `data/claim2/pathway_rescue.jsonl` + `pathway_rescue_trace.jsonl` + `pathway_rescue_summary.json`
  — pathway/rescue evidence (qualified, LOW tier only — see Claims-that-must-not-appear #2 below
  re: localisation breaking down at HIGH).
- `data/claim2/run_replication_control_C.jsonl` + `run_replication_control_C_summary.json` —
  Control C; every row is explicitly tagged
  `run_replication_type: fixed_init_trajectory_replicate_NOT_independent_seed_draw`.

### C2 refinement — exact degree-balanced perturbation closure (2026-08-22)

**Verdict: ROBUST** (verbatim from `frontier_ell4_degree_balanced_perturbation_2026_08_21/REPORT.md`).
The block-9 ell=4 frontier-sensitivity result (w10/50k markedly less sensitive than w16/150k,
w20/200k, w24/300k; the three higher-compute owners not ordered monotonically in compute) is
unchanged in sign, significance, and near-quantitative magnitude when perturbation magnitude is
measured with the exact degree-balanced quadratic form eSEN's own `rms_norm_sh` uses, including
its exact `ell=0` per-atom cross-channel centering step — this closes the one previously-open
"uncentered ell=0" caveat. Centering shifts the `P_bal` denominator by only 0.24-0.65% across the
four owners and changes no sign or significance decision.

Supporting files:
- `data/claim2/frontier_ell4_degree_balanced.csv` — 20 rows (4 owners x 5 alphas), the
  degree-balanced `P_bal`/`delta`/CI points at the exact-centered metric.
- `data/claim2/frontier_ell4_degree_balanced_summary.json` — common-support table, pairwise
  contrasts, and the uncentered-pass comparison, for the exact-centered closure.

This is a companion/refinement of the C2/C3 dose-response evidence above, not a duplicate: it
uses a different (superset) owner set (w10/w16/w20/w24, vs. dose_response.jsonl's w10/w16/w40)
and a different perturbation metric (exact degree-balanced RMS with ell=0 centering, vs. raw
whole-tensor RMS in `dose_response.jsonl`). See `AUDIT_STAGE1.md` for the explicit non-duplication
note.

## C3 — Functional reliance on high-order computation grows along the compute progression

**Verdict: SUPPORTED WITH QUALIFIER. Main text ONLY as "higher-compute models are more reliant
than the lowest-compute model" — the monotone-growth phrasing must NOT be used** (MID->HIGH
reverses under perturbation matching: G goes +0.254 -> -0.045, CI excludes 0).

Supporting files: same as C2 (`data/claim2/dose_response.jsonl`,
`data/claim2/perturbation_control_A_summary.json`). The limitation section must also state that
part of the top-end growth is the ell=4 power fraction doubling (5.4% -> 10.7%), visible in
`perturbation_control_A_baseline_stats.json`'s `power_fraction_by_ell` field per tier.

## C4 — Compute-matched lmax2/lmax4 use systematically different causal-reliance profiles

**Verdict: SUPPORTED WITH QUALIFIER, matched-compute only. Main text only as a matched-compute
claim; the same-width reversal must be reported alongside it** (Control B: at same width the
contrast reverses in 26/32 cells, 30/32 CI-significant).

Supporting files:
- `data/claim2/matched_compute_lmax_owners.csv` — the 4 lmax4 + 4 lmax2 matched-compute owners
  (LOW/MID/UPPERMID/HIGH).
- `data/claim2/compensation.jsonl` + `data/claim2/compensation_summary.json` +
  `compensation_pathway_raw.jsonl` — the matched-compute shared-irrep reliance profiles and
  contrasts (32/32 cells CI-excludes-zero and positive).
- `data/claim2/same_width_control_B.jsonl` + `same_width_control_B_summary.json` — Control B, the
  same-width reversal that qualifies this claim.

## C5 / claim-8 — Within-model causal importance is not sufficient evidence of comparative necessity

**Verdict: STRONG. "This is the safest and most defensible claim in the paper."**

Evidence: gap-erasure (no matched-compute lmax4 advantage to explain) + Control B + Control D.

Supporting files:
- `data/claim2/gap_erasure.jsonl` + `data/claim2/gap_erasure_summary.json` — no matched-compute
  lmax4 advantage exists at any pair for the gap-erasure to explain.
- `data/claim2/same_width_control_B.jsonl` + summary — the reliance-profile difference is not
  attributable to ell_max alone.
- `data/claim1/matched_compute_m1024_krr_results.csv` (Control D) — internally consistent but
  comparatively wrong diagnostic, reinforcing the same point from the Claim-1 side.

## C6 — Neutral-domain vs. support-matched OOD (Val-Comp) generalization of the block-9 ell=4 causal effect

**Verdict** (no single boxed tag in the source; using its own "Bottom line" language verbatim, per
`esen_irrep_ood_2026_08_22/REPORT.md` §8 and `OOD_ROBUSTNESS.md` §4): "G_m (block-9 ell=4
attenuation, OOD-vs-ID, at matched degree-balanced representation perturbation) is significantly
negative for w16/150k, w20/200k, w24/300k and null for w10/50k, converged by M=16384," and "the
low-vs-high interaction is significantly negative at every M and tightens with M — the
compute-trend in G_m is itself statistically supported, not an artifact of one owner." The negative
`G_m` finding "survives the switch from relative to absolute units... and in absolute terms is
significant even for w10/50k." However: "no ℓmax=4 OOD (or ID) baseline force-performance advantage
exists at any of the four compute-matched pairs — ℓmax=2 is uniformly better," so this result
qualifies, and must not be read as explaining, any cross-architecture ℓmax=4 advantage (consistent
with claim #2 in "Claims that must NOT appear" below).

**Precise framing** (terminology correction from `OOD_ROBUSTNESS.md` §0, supersedes `REPORT.md`'s
own "ID vs OOD" shorthand): this compares Neutral-domain validation (eSEN's own training/eval
validation split) against the broader support-matched Val-Comp (OMol25-val) population — not a
literal in-distribution/out-of-distribution split, since Neutral-val's 4 constituent data_id
families are exactly, 100% identical in eligible population to Val-Comp's same 4 families. The
genuine OOD signal comes entirely from 4 composition families (elytes, biomolecules, reactivity,
metal_complexes) absent from Neutral-val by construction; within those families the compute-scaling
contrast is real but **not uniform in magnitude** (biomolecules > elytes > reactivity) — the
headline compute-trend should not be read as one family-independent effect size. A shared-family
positive control (pooled Neutral-like sources inside Val-Comp vs. Neutral-val itself) passes:
statistically indistinguishable contrasts, as expected when the underlying molecules are the same.

**Process-issue disclosure** (carried from `REPORT.md` §0, in substance, not omitted): a
research-only subagent dispatched to map the existing intervention code exceeded its instructions
and, on its own initiative, wrote `id_pool/scan_id_pool.py` and the full `eval/run_gpu_eval.py` /
`eval/analyze_and_report.py` GPU-evaluation implementation. It did not submit any GPU job and did
not modify any tracked repository file. Every load-bearing piece of what it wrote was independently
re-verified against its claimed source before use (degree-balanced weights, orchestration code,
checkpoint/owner identities, loss convention, both frozen pools) — see `REPORT.md` §0 for the full
verification list. The GPU run itself (job 10445341) was submitted and supervised by the primary
agent.

Supporting files (all new, from `analysis_outputs/esen_irrep_ood_2026_08_22/`, a source location
outside both `dev-...` repos — see `AUDIT_STAGE1.md` for this new source-location pattern):
- `data/claim2/ood_pool_provenance.json` — both frozen pools' definitions/sizes/support criteria.
- `data/claim2/ood_full_alpha_curves.csv` — full per-(tag, domain, M, alpha) curves incl. `P_bal`.
- `data/claim2/ood_G_m_convergence.csv` — primary quantity 1, nested-M.
- `data/claim2/ood_interaction_w24_vs_w10.csv` — primary quantity 2, low-vs-high interaction.
- `data/claim2/ood_lmax2_vs_lmax4_baseline.csv` — compute-matched ℓmax2 vs ℓmax4 baseline (no
  ℓmax=4 OOD/ID advantage at any pair).
- `data/claim2/ood_absolute_effect_robustness.csv` + `ood_absolute_effect_A_OOD_minus_A_Neutral.csv`
  — relative-vs-absolute-unit robustness check.
- `data/claim2/ood_domain_decomposition.csv` — per-chemistry-family stratification at M=16384.
- `data/claim2/ood_shared_family_positive_control.csv` — the w24-vs-w10 contrast by stratum,
  including the shared-family positive control.
- `data/claim2/ood_raw_per_config/` — per-configuration (sum, natoms) pairs for both raw JSONLs,
  for bootstrap reproduction (n_boot=2000).

## Seed generality — ESTABLISHED FOR SIGN (magnitude not tightly replicated)

**Status as of 2026-08-22: ANALYSIS COMPLETE.** The 4 seed-replication training runs (LOW
seed2/seed3 at w10/s50000, HIGH seed2/seed3 at w24/s300000; jobs 10429387-10429390, all
`COMPLETED`) plus the downstream block-9 ell=4 NORM_FIXED intervention analysis (5-alpha sweep,
M=1024, same frozen convention as the seed-1 result) have both now been run — see
`dev-equivariant-scaling-laws-kernel-pilot-clean/analysis_outputs/seed_replication_2026_08_20/REPORT.md`
for the full derivation, and `data/claim2/seed_replication_result.csv` +
`data/claim2/seed_replication_result_summary.json` for the canonical tidy result.

`G_s = Delta_HIGH,s(P*=0.15) - Delta_LOW,s(P*=0.15)` (linear interpolation between the two
measured alpha points bracketing P*, no extrapolation; all 6 (seed, endpoint) cells had P* within
measured support):

| seed | G_s | 95% CI |
|---|---|---|
| 1 (original) | +0.477 | [0.429, 0.534] |
| 2 (new) | +0.715 | [0.658, 0.778] |
| 3 (new) | +0.134 | [0.107, 0.166] |

**Sign replicates 3/3** across independent training seeds (both weight init `seed` and data-order
`data.datamodule.data.seed` varied together per replicate) — every seed's individual CI excludes
zero and every seed has the same sign as the original. This is enough to say the *direction* of
the block-9 ell=4 HIGH-relies-more-than-LOW effect is a property of the architecture/training
recipe, not an artifact of one initialization.

**Magnitude does NOT tightly replicate** — `G_s` spans more than 5x across seeds (0.134 to 0.715,
mean 0.442). This must be stated alongside the sign result, not omitted: **"the effect replicates
in sign across 3 independent seeds; it does not replicate in magnitude."** Do not upgrade this to
an unqualified "the effect generalizes across seeds" — the quantitative claim is exactly as strong
as the data supports and no stronger.

Baseline force-loss levels are consistent across all 3 seeds at both endpoints (LOW spread 6.8%,
HIGH spread 1.8%, both within the project's own established 3-13% between-run-noise precedent from
`FINAL_VALIDATION_REPORT.md`) — ruling out a bad/divergent training run as an alternative
explanation.

- `data/claim2/seed_replication_result.csv` — tidy (seed, tier, alpha, P_total, delta,
  L_baseline) table, both the 5 measured alpha points and the P*=0.15 interpolated row per
  (seed, tier), with bootstrap CIs on the interpolated rows.
- `data/claim2/seed_replication_result_summary.json` — G_s per seed + CI, sign-replication
  verdict, across-seed mean/range (descriptive, n=3, no cross-seed CI per the task's own
  "n=3 is too small for a bootstrap" instruction), baseline-loss comparison, full verdict.
- `data/claim2/seed_replication_PENDING.json` — superseded/historical planning file, kept
  unmodified for provenance; do not read a result out of it.
- `manifests/checkpoints.csv` — the 4 seed-replication rows now `status=ANALYSIS_COMPLETE` with
  `N_trainable_params`/`D_atom_tokens`/`C_flops` filled in (verified from the checkpoints' own
  strictly-loaded state dicts, identical architecture to the seed=1 owner at each width).

Any statement implying the effect's **magnitude** (not just its sign) generalizes across training
seeds must NOT appear.

---

## Claims that must NOT appear (verbatim, from FINAL_VALIDATION_REPORT.md section C)

1. "The high-order pathway explains the 0.40 vs 0.35 (or any) scaling-exponent difference." Not
   tested; direction unfavourable.
2. "The high-order pathway explains ell_max=4's advantage over matched ell_max=2." There is no
   such advantage at any matched pair.
3. "Reliance on high-order irreps grows **monotonically** with compute." MID->HIGH reverses under
   perturbation matching.
4. "ell_max=2 compensates by redistributing reliance onto lower orders **because** it lacks
   ell > 2." Not attributable to ell_max on the available evidence (Control B).
5. **UPDATED 2026-08-22 — narrower than before, do not overclaim in the other direction either.**
   The sign of the block-9 ell=4 HIGH-vs-LOW effect (`G_s > 0`) IS established across 3
   independent seeds (see "Seed generality" section above) — a statement that it replicates in
   *sign* is now supported and MAY appear. What must still NOT appear: any statement implying the
   effect's **magnitude** generalizes or is stable across seeds (`G_s` spans 0.134-0.715, >5x,
   across the 3 seeds) — do not say "the effect size is seed-robust" or give a single number
   (e.g. seed-1's 0.477) as if it were architecture-representative without the range.
6. "The tangent kernel is falsified" / "kernel theory is falsified." Only the frozen, full-output
   force tangent kernel as a *comparative* explanation is falsified; the native-head-restricted
   track gives the opposite (correct) ordering.
7. The word "proof" anywhere near compensation. Use "consistent with redistribution".
