"""Figure 2 -- causal dependence on high-order features does not imply better performance at
matched compute.

Panel A: block-9 ell=4 dose response in the four ell_max=4 eSEN frontier models. The horizontal
         axis is the degree-balanced relative perturbation magnitude P; the vertical axis is
         log[L(alpha)/L(1)], the force-loss damage caused by the intervention. The legend is keyed
         to each model's own training compute (the scientifically relevant quantity), not its
         internal checkpoint tag.
Panel B: log(L_2 / L_4) on Neutral validation, across the FULL dense ell_max=4-vs-ell_max=2
         frontier (every matched pair with a valid Neutral-validation point estimate), not just
         the four causal-intervention checkpoints from Panel A -- those four are highlighted,
         colored to match their own curve in Panel A. One training run (ell_max=4, width=16,
         checkpointed every 500 steps from step 500-50000) is excluded from the scatter: it is a
         single run's own checkpoint trajectory, not independent matched-compute observations,
         and would otherwise dominate the low-compute end of the panel by sheer checkpoint count.

Plotting only: every point and interval was computed and validated by `make analysis`. The
matched-compute values on the horizontal axis of Panel B are each pair's ell_max=4 checkpoint's
own training compute, joined from the checkpoint table. `lmax24_all_frontier_gaps.csv` and
`lmax24_frontier_dependence.csv` are frozen inputs (data/claim2/), not analysis_out/ outputs --
they are the raw matched-pair table itself, not a derived summary, so this reads them from
data/, the same way build_lmax24_frontier.py does.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from _style import ANALYSIS_OUT, REPO_ROOT, apply_style, read_csv, read_json, save_fig

OWNERS = ["w10/50k", "w16/150k", "w20/200k", "w24/300k"]
OWNER_COLORS = {
    "w10/50k": "#4c78a8",
    "w16/150k": "#72b7b2",
    "w20/200k": "#e45756",
    "w24/300k": "#b279a2",
}
TIERS = ["LOW", "MID", "UPPERMID", "HIGH"]
OWNER_BY_TIER = dict(zip(TIERS, OWNERS))

DENSE_RUN_NAME = "esen-sphere-16-neutral-epoch-lmax4-dense0to50k"

DATA_CLAIM2 = os.path.join(REPO_ROOT, "data", "claim2")


def lmax4_compute():
    """Training compute of the ell_max=4 model in each matched-compute pair."""
    rows = read_csv(os.path.join(DATA_CLAIM2, "matched_compute_lmax_checkpoints.csv"))
    return {r["role_tier"].replace("matched_compute_", ""): float(r["C_flops"])
            for r in rows if r["lmax"] == "4"}


def _format_flops(x):
    exponent = int(np.floor(np.log10(x)))
    mantissa = x / 10 ** exponent
    return rf"${mantissa:.2g}\times10^{{{exponent}}}$ FLOPs"


def panel_a(ax, dose_response):
    flops = lmax4_compute()
    flops_by_owner = {OWNER_BY_TIER[tier]: c for tier, c in flops.items()}

    for owner in OWNERS:
        rows = sorted((r for r in dose_response["per_alpha"] if r["owner"] == owner),
                      key=lambda r: r["P_bal"])
        xs = [r["P_bal"] for r in rows]
        ys = [r["delta"] for r in rows]
        yerr = [[y - r["delta_ci_lo"] for y, r in zip(ys, rows)],
                [r["delta_ci_hi"] - y for y, r in zip(ys, rows)]]
        ax.errorbar(xs, ys, yerr=yerr, fmt="-o", color=OWNER_COLORS[owner],
                    label=_format_flops(flops_by_owner[owner]), markersize=5, linewidth=1.6,
                    capsize=3)

    ax.set_xlabel(r"$\ell{=}4$ perturbation magnitude $P$")
    ax.set_ylabel(r"Force-loss damage $\log[L(\alpha)/L(1)]$")
    ax.set_title("A. High-order features are causally\nrequired in trained eSEN")
    ax.legend(loc="upper left", frameon=False, title="Training compute",
              fontsize=6.8, title_fontsize=7)


def _dense_run_keys():
    """(width, step) keys belonging to the single dense ell_max=4 training run to exclude from
    the Panel B scatter -- identified by run_name, not guessed from a step-count heuristic."""
    dep = read_csv(os.path.join(DATA_CLAIM2, "lmax24_frontier_dependence.csv"))
    return {(r["width"], r["step"]) for r in dep if r["run_name"] == DENSE_RUN_NAME}


def panel_b(ax, causal_rows):
    gaps = read_csv(os.path.join(DATA_CLAIM2, "lmax24_all_frontier_gaps.csv"))
    dense_keys = _dense_run_keys()

    neutral = [r for r in gaps if r["domain"] == "Neutral_val" and r["valid_match"] == "True"
               and r["H_force"] != "" and (r["l4_width"], r["l4_step"]) not in dense_keys]
    neutral.sort(key=lambda r: float(r["l4_flops"]))

    ax.axhline(0.0, color="black", linewidth=1.0, zorder=1)

    xs = np.array([float(r["l4_flops"]) for r in neutral])
    ys = np.array([float(r["H_force"]) for r in neutral])
    ax.scatter(xs, ys, s=14, color="#4c78a8", alpha=0.45, linewidths=0, zorder=2)

    # The four Panel A causal-intervention checkpoints, colored to match their own curve there,
    # so a reader can connect a Panel B point back to its Panel A dose-response curve directly.
    by_owner = {r["l4_owner_tag"]: r for r in causal_rows}
    cx = np.array([float(by_owner[owner]["C_flops"]) for owner in OWNERS])
    cy = np.array([float(by_owner[owner]["H_force_Neutral"]) for owner in OWNERS])
    colors = [OWNER_COLORS[owner] for owner in OWNERS]
    ax.scatter(cx, cy, s=90, facecolor=colors, linewidths=0, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Training compute (FLOPs)")
    ax.set_ylabel(r"$\log\left(L_{\ell_{\max}=2}/L_{\ell_{\max}=4}\right)$")
    ax.set_title("B. Causal use does not imply\nmatched-compute advantage")

    ylo, yhi = ax.get_ylim()
    offset = 0.04 * (yhi - ylo)
    xlo = ax.get_xlim()[0]
    ax.text(xlo * 1.3, offset, r"$\ell_{\max}{=}4$ better", fontsize=7.2, color="#555555",
            ha="left", va="bottom", style="italic")
    ax.text(xlo * 1.3, -offset, r"$\ell_{\max}{=}2$ better", fontsize=7.2, color="#555555",
            ha="left", va="top", style="italic")


def main():
    apply_style()
    dose_response = read_json(os.path.join(
        ANALYSIS_OUT, "claim2", "frontier_ell4_degree_balanced_recomputed.json"))
    causal_rows = read_csv(os.path.join(DATA_CLAIM2, "lmax4_causal_vs_comparative.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    panel_a(axes[0], dose_response)
    panel_b(axes[1], causal_rows)

    fig.tight_layout()
    for path in save_fig(fig, "figure2"):
        print("wrote", path)


if __name__ == "__main__":
    main()
