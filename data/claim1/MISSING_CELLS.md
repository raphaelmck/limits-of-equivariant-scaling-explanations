<title>Claim 1 KRR Coverage Grid</title>

# Claim 1 — KRR coverage grid and missing cells

Read-only analysis of `existing_krr_inventory.csv` in this directory. No runs were launched
to produce this document; it only inventories what already exists on disk.

## Target grid

The paper's spine (Component 6 of `CURRENT_PIPELINE.md`) wants empirical force-NTK KRR
coverage across:

- **4 architectures**: MPNN, MC-EGNN, GemNet-OC, eSEN
- **3 matched-compute budget tiers**: LOW, MID, HIGH (from `owner_manifest.csv` / the 12
  frontier owners)
- **probe pool size** m=256 (canonical/frozen) and m=1024 (the "strongest treatment" large-m
  extension)

That is a 4 x 3 x 2 = 24-cell grid (architecture x budget x m).

## What actually exists, by checkpoint-state axis

`existing_krr_inventory.csv` has KRR data along **two different checkpoint-state axes**, and
they are not the same axis as the LOW/MID/HIGH budget tiers:

1. **Fixed-step / time axis** (`checkpoint_state` = K0 / K50k / K500k): 4 architectures at a
   single fixed width per architecture ("large" size tier: MPNN width 2048, MC-EGNN width 512,
   GemNet-OC width 256, eSEN width 64), all at m=256. Source:
   `four_arch_initial_force_kernel_pilot_2026_08_14/krr_learning_curves.csv` (K0 only) and
   `four_arch_trained_force_kernel_pilot_2026_08_15/krr_learning_curves_common_target.csv` /
   `_residual.csv` (K0/K50k/K500k).
2. **Matched-compute / budget axis** (`checkpoint_state` = matched-compute-LOW/MID/HIGH): 4
   architectures at the 12 frontier owners (see `owner_manifest.csv`), all at m=256. Source:
   `four_arch_matched_compute_frontier_kernel_2026_08_15/krr_learning_curves_common_target.csv`
   / `_residual.csv`.
3. **Large-m extension** (`checkpoint_state` = large-m-K500k, m=1024): **only 2 of 4
   architectures** — GemNet-OC and eSEN — at their "large" fixed-step checkpoint (width 256 /
   width 64 respectively, step=500000). Source:
   `final_validation_sprint_2026_08_20/control_D_large_m_krr/krr_learning_curves_m1024.csv`.

## Coverage matrix (architecture x budget, m=256 vs m=1024)

| Architecture | LOW (m=256) | MID (m=256) | HIGH (m=256) | m=1024 (any tier) |
|---|---|---|---|---|
| MPNN | YES (matched-compute) | YES (matched-compute) | YES (matched-compute) | **NO — no m=1024 KRR treatment at all** |
| MC-EGNN | YES (matched-compute) | YES (matched-compute) | YES (matched-compute) | **NO — no m=1024 KRR treatment at all** |
| GemNet-OC | YES (matched-compute) | YES (matched-compute) | YES (matched-compute) | YES, but at a **separate, non-frontier-owner checkpoint** (width 256, step=500000) — see below |
| eSEN | YES (matched-compute) | YES (matched-compute) | YES (matched-compute) | YES, but at a **separate, non-frontier-owner checkpoint** (sphere 64, step=500000) — see below |

All four architectures also have K0/K50k/K500k fixed-step m=256 coverage at their "large" size
tier (not a LOW/MID/HIGH matched-compute owner either — see below).

## Which budget tier does the m=1024 treatment actually correspond to?

The m=1024 large-m and Control-D KRR extensions were run on exactly two checkpoints:

- `esen_large_sphere64` at step=500000 (width label `sphere_channels=64`)
- `gemnet_oc_large_width256` at step=500000 (width label `width=256`)

Cross-referencing `owner_manifest.csv` (the 12 LOW/MID/HIGH frontier owners):

- eSEN owners are: LOW = sphere_channels 10 (step 50000), MID = sphere_channels 16 (step
  150000), HIGH = sphere_channels 40 (step 275000). **None of these is sphere_channels=64.**
- GemNet-OC owners are: LOW = width 48 (step 200000), MID = width 64 (step 500000), HIGH =
  width 144 (step 500000). **None of these is width=256.**

Directly checked in `selected_frontier_owners.csv` for `esen-sphere-64` and
`gemnet-oc-atom256` checkpoint-path substrings — no match in either case.

**Conclusion: the m=1024 KRR treatment is NOT at any of the 12 LOW/MID/HIGH matched-compute
frontier owners.** It is at the same fixed "large" size-tier checkpoints used throughout the
K0/K50k/K500k fixed-step kernel-evolution line (Component 9's kernel-trajectory manifest and
Component 10's large-m convergence check), which is a different compute point per architecture
than any of LOW/MID/HIGH. It happens to be the single largest width tested for each of these
two architectures in the four-arch pilot suite, but it was never matched to a common FLOPs
budget the way the LOW/MID/HIGH owners were.

## Explicit list of missing cells (m=1024 KRR, matched-compute tiers)

None of the following cells exist on disk; all would require a new kernel construction + KRR
run on a matched-compute-owner checkpoint at m=1024 (not attempted here — this is
information-gathering only):

- MPNN x LOW x m=1024
- MPNN x MID x m=1024
- MPNN x HIGH x m=1024
- MC-EGNN x LOW x m=1024
- MC-EGNN x MID x m=1024
- MC-EGNN x HIGH x m=1024
- GemNet-OC x LOW x m=1024
- GemNet-OC x MID x m=1024
- GemNet-OC x HIGH x m=1024 (closest existing data: GemNet-OC width=256/step=500000 m=1024,
  which is NOT the HIGH owner width=144/step=500000 — a different checkpoint entirely)
- eSEN x LOW x m=1024
- eSEN x MID x m=1024
- eSEN x HIGH x m=1024 (closest existing data: eSEN sphere_channels=64/step=500000 m=1024,
  which is NOT the HIGH owner sphere_channels=40/step=275000 — a different checkpoint entirely)

12 of 12 (architecture x budget) cells lack a true matched-compute m=1024 KRR result. MPNN and
MC-EGNN additionally have **zero** m=1024 KRR data of any kind (not even at an unmatched
checkpoint) — no large-m Jacobian/Gram was ever constructed for either architecture in
`large_m_convergence_2026_08_19/`, which only covers GemNet-OC and eSEN.
