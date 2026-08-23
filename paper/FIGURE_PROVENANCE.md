# Figure provenance survey

**No figures are generated in this consolidation pass.** `figures/` is an empty placeholder (see
`figures/README.md`). This document only catalogs the EXISTING precedent figures already sitting
in the two source repos, so a later pass knows what to reproduce/adapt rather than design from
scratch. Paths are repo-relative to `/home/mila/m/michnikr/CODE/`.

## Claim 1

| Figure | Source path | Visualizes |
|---|---|---|
| `four_arch_force_frontiers.png` | `dev-equivariant-scaling-laws/analysis_outputs/four_arch_force_scaling_repaired_2026_08_14/` | The empirical force-scaling frontier -> `data/claim1/force_scaling.csv` |
| `four_arch_matched_compute_ratio.png` | same dir | Matched-compute ratio underlying the LOW/MID/HIGH tiers -> `data/claim1/owner_manifest.csv` |
| `eigenvalue_spectra.png`, `target_power.png`, `local_spectral_slopes.png` | `dev-equivariant-scaling-laws-kernel-pilot-clean/analysis_outputs/four_arch_initial_force_kernel_pilot_2026_08_14/` | K0 static-kernel spectral descriptors -> `data/claim1/existing_kernel_inventory.csv` |
| `krr_learning_curves_large.png`, `krr_learning_curves_medium.png` | same dir | Static K0 empirical KRR -> `data/claim1/existing_krr_inventory.csv` |
| `force_frontiers_with_selected_budgets.png`, `selected_owner_compute_utilization.png` | `four_arch_matched_compute_frontier_kernel_2026_08_15/` | The 12 LOW/MID/HIGH owner selection -> `data/claim1/owner_manifest.csv` |
| `krr_auc_vs_compute.png`, `krr_auc_vs_neural_force_loss.png`, `gemnet_vs_esen_crossover.png` | same dir | Matched-compute KRR vs. neural loss (the PARTIAL SIGNAL / crossover finding) -> `data/claim1/existing_krr_inventory.csv` |
| `eigenvalue_evolution.png`, `krr_auc_evolution.png`, `residual_alignment_evolution.png`, `target_alignment_evolution.png` | `four_arch_trained_force_kernel_pilot_2026_08_15/` | K0/K50k/K500k trained-kernel evolution -> `data/claim1/existing_kernel_inventory.csv`, `existing_krr_inventory.csv` |
| `gemnet_esen_decision_panel.png`, `kernel_norm_growth.png`, `kernel_shape_change.png`, `task_alignment_vs_training.png` | `bordelon_feature_learning_kernel_audit_2026_08_17/` | The A(K_t,y) task-directed alignment anomaly (GemNet-OC > eSEN) -> `data/claim1/existing_kernel_inventory.csv` |
| `Cf_gemnet_vs_esen_m1024.png`, `k90_k95_nested_m.png` | `large_m_convergence_2026_08_19/cumulative_target_alignment/` | m-convergence robustness check (m=64..1024) -> `data/claim1/existing_kernel_inventory.csv` (large-m rows) |
| `krr_learning_curves.png` | `final_validation_sprint_2026_08_20/control_D_large_m_krr/` | Control D, the m=1024 KRR reproduction gate, the decisive C1 evidence -> `data/claim1/matched_compute_m1024_krr_results.csv` |

## Claim 2

| Figure | Source path | Visualizes |
|---|---|---|
| `dose_response_M1024.png`, `ell_depth_heatmap_M1024.png`, `ell_effect_vs_compute_M1024.png`, `mechanism_summary_M1024.png`, `power_vs_dependence_M1024.png` | `stage3a_esen_irrep_intervention_2026_08_16/figures/` | The base irrep-sector intervention primitive (superseded by Stage-3A.1's NORM_FIXED correction — see AUDIT note) |
| `A_natural_vs_normfixed_block9.png`, `B_natural_vs_normfixed_block11.png`, `C_normalization_mediated_difference.png`, `D_fixed_width_temporal.png`, `E_fixed_width_mechanism_alignment.png`, `F_large_population_replication.png` | `stage3a1_norm_control_temporal_2026_08_16/figures/` | The NORM_FIXED correction actually used by the paper -> `data/claim2/dose_response.jsonl` (mechanism), norm-fixed convention |
| `block9_dose_response.png` | `stage3_block9_dose_response_2026_08_19/` | The block-9 alpha dose-response, LOW/MID/HIGH -> `data/claim2/dose_response.jsonl` (backs C2/C3, likely the central paper figure candidate for this claim) |
| `perturbation_damage.png` | `final_validation_sprint_2026_08_20/control_A_perturbation_magnitude/` | Control A perturbation-matched damage -> `data/claim2/perturbation_control_A_summary.json` (backs C2/C3) |
| `same_width_profiles.png` | `final_validation_sprint_2026_08_20/control_B_same_width/` | Control B same-width reversal -> `data/claim2/same_width_control_B.jsonl` (backs C4/C5 qualifier) |
| `seed_replication.png` | `final_validation_sprint_2026_08_20/control_C_seeds/` | Control C run-replication (NOT seed replication) -> `data/claim2/run_replication_control_C.jsonl` |
| `A_all_checkpoints_frontiers.png`, `B_Hforce_common_support.png`, `C_zoom_near_common_max.png`, `G_combined_mechanism_panel.png` | `stage3b1_true_lmax_frontiers_2026_08_16/figures/` | The matched-compute lmax2/lmax4 frontier selection -> `data/claim2/matched_compute_lmax_owners.csv` |
| `A_force_performance.png`, `C_ell2_compensation.png`, `D_lmax4_ell34_pathway.png`, `F_mechanism_summary.png` | `stage3b_lmax2_vs_lmax4_2026_08_16/figures/` | Precursor lmax2-vs-4 mechanism figures (Stage-3B, partly superseded/re-scoped by Stage-3B.1's true-frontier reconstruction — see `frontier_owner_manifest.json`'s dominance notes) |
| `shared_irrep_reliance.png`, `compensation_contrasts.png`, `compensation_pathway.png` | `stage3_lmax2_lmax4_compensation_2026_08_20/` | The matched-compute shared-irrep reliance/contrast figures -> `data/claim2/compensation.jsonl` (backs C4, likely central paper figure candidate) |
| `gap_erasure.png` | `stage3_high_order_gap_erasure_2026_08_20/` | Gap-erasure -> `data/claim2/gap_erasure.jsonl` (backs C5) |
| `ell4_downstream_trace.png`, `ell4_rescue.png` | `stage3_block9_pathway_rescue_2026_08_19/` | Pathway/rescue -> `data/claim2/pathway_rescue.jsonl` (qualified, LOW-tier-only claim) |

No figure exists yet for the seed-replication PENDING data (`data/claim2/seed_replication_PENDING.json`)
— there is nothing to plot until the experiment runs.

This is a survey for later work, not a generation step: none of the above PNGs were copied,
regenerated, or modified by this consolidation pass.
