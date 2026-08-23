#!/usr/bin/env python3
"""
Build the paper-facing Claim-1 consolidation layer under paper_results/claim1/.

This is a READ-ONLY consolidation: every value written to the output CSVs is copied
verbatim from an existing, already-validated source file identified in
analysis_outputs/claim1_cleanup/CURRENT_PIPELINE.md. No recomputation, no refitting.
"""
import csv
import json
import os

ROOT = "/home/mila/m/michnikr/CODE"
REPO1 = f"{ROOT}/dev-equivariant-scaling-laws"
REPO2 = f"{ROOT}/dev-equivariant-scaling-laws-kernel-pilot-clean"
OUT = f"{ROOT}/paper_results/claim1"
os.makedirs(OUT, exist_ok=True)


def rel1(p):
    return "dev-equivariant-scaling-laws/" + os.path.relpath(p, REPO1)


def rel2(p):
    return "dev-equivariant-scaling-laws-kernel-pilot-clean/" + os.path.relpath(p, REPO2)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# 2. force_scaling.csv
# ---------------------------------------------------------------------------
fits_path = f"{REPO1}/analysis_outputs/four_arch_force_scaling_repaired_2026_08_14/four_arch_force_scaling_fits.csv"
fits_rows = read_csv(fits_path)

# The row CURRENT_PIPELINE.md quotes: metric=force_mse_norm, fit_domain=common, truncation_frac=1.0
# (identical 2.16-decade common-compute interval [8.49e15,1.224e18] FLOPs across all 4 archs)
selected = [
    r for r in fits_rows
    if r["metric"] == "force_mse_norm" and r["fit_domain"] == "common" and r["truncation_frac"] == "1.0"
]
assert len(selected) == 4, len(selected)

out_rows = []
for r in selected:
    out_rows.append({
        "architecture": r["architecture_label"],
        "architecture_key": r["architecture"],
        "metric": r["metric"],
        "fit_domain": r["fit_domain"],
        "truncation_frac": r["truncation_frac"],
        "gamma": r["gamma"],
        "gamma_event_point_ols": r["gamma_event_point_ols"],
        "gamma_nls_rawloss": r["gamma_nls_rawloss"],
        "logA": r["logA"],
        "r2": r["r2"],
        "rmse_log": r["rmse_log"],
        "C_min_flops": r["C_min"],
        "C_max_flops": r["C_max"],
        "decades": r["decades"],
        "n_frontier_owners_in_range": r["n_frontier_owners_in_range"],
        "source_file": rel1(fits_path),
    })

write_csv(
    f"{OUT}/force_scaling.csv",
    list(out_rows[0].keys()),
    out_rows,
)

# Sanity check against CURRENT_PIPELINE.md-quoted values
quoted = {"mpnn": 0.227, "egnn": 0.296, "gemnet_oc": 0.331, "esen": 0.663}
for r in out_rows:
    g = float(r["gamma"])
    q = quoted[r["architecture_key"]]
    assert abs(g - q) < 0.001, (r["architecture_key"], g, q)
print("force_scaling.csv OK, gammas match CURRENT_PIPELINE.md prose to 1e-3")

# ---------------------------------------------------------------------------
# 3. owner_manifest.csv  (12 frontier owners)
# ---------------------------------------------------------------------------
owners_csv_path = f"{REPO2}/analysis_outputs/four_arch_matched_compute_frontier_kernel_2026_08_15/selected_frontier_owners.csv"
audit_json_path = f"{REPO2}/analysis_outputs/four_arch_matched_compute_frontier_kernel_2026_08_15/owner_reconstruction_audit.json"

owners_rows = read_csv(owners_csv_path)
audit = json.load(open(audit_json_path))
audit_status_overall = audit["status"]  # single top-level "PASS" applies to all 12 owners
audit_by_tag = {o["tag"]: o for o in audit["owners"]}

out_rows = []
for r in owners_rows:
    tag = r["tag"]
    a = audit_by_tag.get(tag, {})
    prov = a.get("checkpoint_provenance", {})
    out_rows.append({
        "architecture": r["architecture_label"],
        "architecture_key": r["architecture"],
        "budget_tier": r["budget_label"],
        "tag": tag,
        "width_label": r["width_label"],
        "width": r["width"],
        "global_step": r["step"],
        "N_trainable_params": r["N"],
        "D_atom_tokens": r["D_atom_tokens"],
        "C_budget_flops": r["C_budget"],
        "C_owner_flops": r["C_owner"],
        "utilization": r["utilization"],
        "force_mse_norm": r["force_mse_norm"],
        "ckpt_path": r["ckpt_path"],
        "checkpoint_sha256": prov.get("checkpoint_sha256", ""),
        "checkpoint_global_step_from_audit": prov.get("checkpoint_global_step", ""),
        "owner_reconstruction_audit_status": audit_status_overall,
        "source_file": rel2(owners_csv_path) + " ; " + rel2(audit_json_path),
    })

write_csv(f"{OUT}/owner_manifest.csv", list(out_rows[0].keys()), out_rows)
assert len(out_rows) == 12
print("owner_manifest.csv OK, 12 owners, audit status:", audit_status_overall)

# ---------------------------------------------------------------------------
# 4. existing_kernel_inventory.csv
# ---------------------------------------------------------------------------
kernel_rows_out = []
fieldnames_kernel = [
    "architecture", "checkpoint_state", "budget_tier", "m", "width_or_size",
    "trace_raw_max_eig", "max_eigenvalue_operator", "min_eigenvalue_raw",
    "numerical_rank", "effective_rank", "symmetry_relative_max_error",
    "psd_pass", "k50", "k80", "k90", "k95", "k50_y", "k80_y", "k90_y", "k95_y",
    "k50_residual", "k80_residual", "k90_residual", "k95_residual",
    "alignment_A_Kt_y", "alignment_A_Kx", "source_file",
]

# --- K0 static pilot: spectral_descriptors.csv (eigenvalue spectrum, m=256) ---
sd_path = f"{REPO2}/analysis_outputs/four_arch_initial_force_kernel_pilot_2026_08_14/spectral_descriptors.csv"
for r in read_csv(sd_path):
    kernel_rows_out.append({
        "architecture": r["architecture"],
        "checkpoint_state": "K0",
        "budget_tier": "",
        "m": r["m"],
        "width_or_size": f"{r['size']}/{r['width_label']}={r['width']}",
        "trace_raw_max_eig": "",
        "max_eigenvalue_operator": r["max_eigenvalue_operator"],
        "min_eigenvalue_raw": r["min_eigenvalue_raw"],
        "numerical_rank": r["numerical_rank"],
        "effective_rank": r["effective_rank"],
        "symmetry_relative_max_error": r["symmetry_relative_max_error"],
        "psd_pass": r["psd_pass"],
        "k50": r["k50"], "k80": r["k80"], "k90": r["k90"], "k95": r["k95"],
        "k50_y": "", "k80_y": "", "k90_y": "", "k95_y": "",
        "k50_residual": "", "k80_residual": "", "k90_residual": "", "k95_residual": "",
        "alignment_A_Kt_y": "", "alignment_A_Kx": "",
        "source_file": rel2(sd_path),
    })

# --- trained pilot: spectral_evolution_summary.csv (K0/K50k/K500k, m=256, "large" size only) ---
sev_path = f"{REPO2}/analysis_outputs/four_arch_trained_force_kernel_pilot_2026_08_15/spectral_evolution_summary.csv"
for r in read_csv(sev_path):
    kernel_rows_out.append({
        "architecture": r["architecture"],
        "checkpoint_state": r["time"],
        "budget_tier": "",
        "m": "256",
        "width_or_size": f"width={r['width']}",
        "trace_raw_max_eig": "",
        "max_eigenvalue_operator": r["max_eigenvalue_operator"],
        "min_eigenvalue_raw": r["min_raw_eigenvalue"],
        "numerical_rank": r["numerical_rank"],
        "effective_rank": r["effective_rank"],
        "symmetry_relative_max_error": r["symmetry_relative_max_error"],
        "psd_pass": r["psd_pass"],
        "k50": "", "k80": "", "k90": "", "k95": "",
        "k50_y": r["k50_y"], "k80_y": r["k80_y"], "k90_y": r["k90_y"], "k95_y": r["k95_y"],
        "k50_residual": r["k50_residual"], "k80_residual": r["k80_residual"],
        "k90_residual": r["k90_residual"], "k95_residual": r["k95_residual"],
        "alignment_A_Kt_y": "", "alignment_A_Kx": "",
        "source_file": rel2(sev_path),
    })

# --- matched-compute frontier: spectral_summary.csv (LOW/MID/HIGH owners, m=256) ---
msum_path = f"{REPO2}/analysis_outputs/four_arch_matched_compute_frontier_kernel_2026_08_15/spectral_summary.csv"
for r in read_csv(msum_path):
    kernel_rows_out.append({
        "architecture": r["architecture"],
        "checkpoint_state": f"matched-compute-{r['budget']}",
        "budget_tier": r["budget"],
        "m": "256",
        "width_or_size": f"width={r['width']}/step={r['step']}",
        "trace_raw_max_eig": "",
        "max_eigenvalue_operator": r["max_eigenvalue_operator"],
        "min_eigenvalue_raw": r["min_raw_eigenvalue"],
        "numerical_rank": r["numerical_rank"],
        "effective_rank": r["effective_rank"],
        "symmetry_relative_max_error": r["symmetry_relative_max_error"],
        "psd_pass": r["psd_pass"],
        "k50": "", "k80": "", "k90": "", "k95": "",
        "k50_y": r["k50_y"], "k80_y": r["k80_y"], "k90_y": r["k90_y"], "k95_y": r["k95_y"],
        "k50_residual": r["k50_residual"], "k80_residual": r["k80_residual"],
        "k90_residual": r["k90_residual"], "k95_residual": r["k95_residual"],
        "alignment_A_Kt_y": "", "alignment_A_Kx": "",
        "source_file": rel2(msum_path),
    })

# --- bordelon task-directed alignment A(K_t,y): task_directed_alignment.csv (K0..K500k, m=256, target=y only) ---
tda_path = f"{REPO2}/analysis_outputs/bordelon_feature_learning_kernel_audit_2026_08_17/task_directed_alignment.csv"
for r in read_csv(tda_path):
    if r["target"] != "y":
        continue  # r_t rows are the auxiliary residual-alignment statistic, not A(K,y)
    kernel_rows_out.append({
        "architecture": r["architecture"],
        "checkpoint_state": f"step={r['step']}",
        "budget_tier": "",
        "m": r["m"],
        "width_or_size": "",
        "trace_raw_max_eig": "",
        "max_eigenvalue_operator": "",
        "min_eigenvalue_raw": "",
        "numerical_rank": "",
        "effective_rank": "",
        "symmetry_relative_max_error": "",
        "psd_pass": "",
        "k50": "", "k80": "", "k90": "", "k95": "",
        "k50_y": "", "k80_y": "", "k90_y": "", "k95_y": "",
        "k50_residual": "", "k80_residual": "", "k90_residual": "", "k95_residual": "",
        "alignment_A_Kt_y": r["value"], "alignment_A_Kx": "",
        "source_file": rel2(tda_path),
    })

# --- large-m convergence (m=1024): esen/gemnet_oc large_m_result.json, K_x numerics + A_Kx at m=1024 ---
for tag, fn in [
    ("esen", "esen_large_sphere64_large_m_result.json"),
    ("gemnet_oc", "gemnet_oc_large_width256_large_m_result.json"),
]:
    p = f"{REPO2}/analysis_outputs/large_m_convergence_2026_08_19/{fn}"
    d = json.load(open(p))
    nk = d["numerics_K_x"]
    kernel_rows_out.append({
        "architecture": d["architecture"],
        "checkpoint_state": f"large-m-K500k-step{d['gate1']['global_step']}",
        "budget_tier": "",
        "m": d["m"],
        "width_or_size": d["tag"],
        "trace_raw_max_eig": nk["max_eigenvalue"],
        "max_eigenvalue_operator": "",
        "min_eigenvalue_raw": nk["min_eigenvalue"],
        "numerical_rank": "",
        "effective_rank": "",
        "symmetry_relative_max_error": nk["symmetry_relative_max_error"],
        "psd_pass": nk["psd_pass"],
        "k50": "", "k80": "", "k90": "", "k95": "",
        "k50_y": "", "k80_y": "", "k90_y": "", "k95_y": "",
        "k50_residual": "", "k80_residual": "", "k90_residual": "", "k95_residual": "",
        "alignment_A_Kt_y": "", "alignment_A_Kx": d["nested_m"]["1024"]["A_Kx"],
        "source_file": rel2(p),
    })

write_csv(f"{OUT}/existing_kernel_inventory.csv", fieldnames_kernel, kernel_rows_out)
print(f"existing_kernel_inventory.csv OK, {len(kernel_rows_out)} rows")

# ---------------------------------------------------------------------------
# 5. existing_krr_inventory.csv
# ---------------------------------------------------------------------------
krr_fieldnames = [
    "architecture", "checkpoint_state", "budget_tier", "m", "n_train",
    "ridge_setting", "ridge_value", "target", "test_nmse_median",
    "test_nmse_mean", "split_count", "source_file",
]
krr_rows_out = []

# K0 static pilot (m=256)
p = f"{REPO2}/analysis_outputs/four_arch_initial_force_kernel_pilot_2026_08_14/krr_learning_curves.csv"
for r in read_csv(p):
    krr_rows_out.append({
        "architecture": r["architecture"],
        "checkpoint_state": "K0",
        "budget_tier": "",
        "m": "256",
        "n_train": r["n"],
        "ridge_setting": r["ridge_setting"],
        "ridge_value": r["rho"],
        "target": "raw_force",
        "test_nmse_median": r["nmse_median"],
        "test_nmse_mean": r["nmse_mean"],
        "split_count": r["split_count"],
        "source_file": rel2(p),
    })

# Trained pilot: common_target and residual (K0/K50k/K500k, m=256)
for target_label, fn in [("common", "krr_learning_curves_common_target.csv"),
                          ("residual", "krr_learning_curves_residual.csv")]:
    p = f"{REPO2}/analysis_outputs/four_arch_trained_force_kernel_pilot_2026_08_15/{fn}"
    for r in read_csv(p):
        krr_rows_out.append({
            "architecture": r["architecture"],
            "checkpoint_state": r["time"],
            "budget_tier": "",
            "m": "256",
            "n_train": r["n"],
            "ridge_setting": r["ridge_setting"],
            "ridge_value": r["rho"],
            "target": r.get("target", target_label),
            "test_nmse_median": r["nmse_median"],
            "test_nmse_mean": r["nmse_mean"],
            "split_count": r["split_count"],
            "source_file": rel2(p),
        })

# Matched-compute frontier: common_target and residual (LOW/MID/HIGH, m=256)
for target_label, fn in [("common", "krr_learning_curves_common_target.csv"),
                          ("residual", "krr_learning_curves_residual.csv")]:
    p = f"{REPO2}/analysis_outputs/four_arch_matched_compute_frontier_kernel_2026_08_15/{fn}"
    for r in read_csv(p):
        krr_rows_out.append({
            "architecture": r["architecture"],
            "checkpoint_state": f"matched-compute-{r['budget']}",
            "budget_tier": r["budget"],
            "m": "256",
            "n_train": r["n"],
            "ridge_setting": r["ridge_setting"],
            "ridge_value": r["rho"],
            "target": r.get("target", target_label),
            "test_nmse_median": r["nmse_median"],
            "test_nmse_mean": r["nmse_mean"],
            "split_count": r["split_count"],
            "source_file": rel2(p),
        })

# Control D: m=1024 KRR extension (GemNet-OC width256, eSEN sphere64, step=500000 only)
p = f"{REPO2}/analysis_outputs/final_validation_sprint_2026_08_20/control_D_large_m_krr/krr_learning_curves_m1024.csv"
for r in read_csv(p):
    krr_rows_out.append({
        "architecture": r["architecture"],
        "checkpoint_state": "large-m-K500k",
        "budget_tier": "",
        "m": "1024",
        "n_train": r["n"],
        "ridge_setting": r["ridge_setting"],
        "ridge_value": r["rho"],
        "target": "common",
        "test_nmse_median": r["nmse_median"],
        "test_nmse_mean": r["nmse_mean"],
        "split_count": r["split_count"],
        "source_file": rel2(p),
    })

write_csv(f"{OUT}/existing_krr_inventory.csv", krr_fieldnames, krr_rows_out)
print(f"existing_krr_inventory.csv OK, {len(krr_rows_out)} rows")

print("DONE")
