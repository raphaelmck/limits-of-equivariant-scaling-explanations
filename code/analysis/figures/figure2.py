"""Figure 2 -- causal dependence on high-order features does not imply better performance at
matched compute.

Panel A: block-9 ell=4 dose response in the four ell_max=4 eSEN frontier models. The horizontal
         axis is the degree-balanced relative perturbation magnitude P; the vertical axis is
         log[L(alpha)/L(1)], the force-loss damage caused by the intervention.
Panel B: at the four approximately compute-matched pairs corresponding to those checkpoints,
         log(L_2 / L_4) on Neutral validation and on support-matched validation. Negative values
         favour ell_max=2. Error bars are 95% paired-bootstrap confidence intervals.

Plotting only: every point and interval was computed and validated by `make analysis`. The
matched-compute values on the horizontal axis of Panel B are the ell_max=4 checkpoint's own
training compute, joined from the checkpoint table.
"""
from __future__ import annotations

import os

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

# The two evaluation populations, and a small multiplicative x-offset so their points do not
# overlap at the same matched compute.
DOMAINS = {
    "ID": {"label": "Neutral validation", "color": "#4c78a8", "marker": "o", "dx": 0.94},
    "OOD": {"label": "support-matched validation", "color": "#e45756", "marker": "s", "dx": 1.06},
}

EVAL_SUBSET = "16384"


def panel_a(ax, dose_response):
    for owner in OWNERS:
        rows = sorted((r for r in dose_response["per_alpha"] if r["owner"] == owner),
                      key=lambda r: r["P_bal"])
        xs = [r["P_bal"] for r in rows]
        ys = [r["delta"] for r in rows]
        yerr = [[y - r["delta_ci_lo"] for y, r in zip(ys, rows)],
                [r["delta_ci_hi"] - y for y, r in zip(ys, rows)]]
        ax.errorbar(xs, ys, yerr=yerr, fmt="-o", color=OWNER_COLORS[owner], label=owner,
                    markersize=5, linewidth=1.6, capsize=3)

    ax.set_xlabel(r"$\ell{=}4$ perturbation magnitude $P$")
    ax.set_ylabel(r"Force-loss damage $\log[L(\alpha)/L(1)]$")
    ax.set_title("A. High-order features are causally\nrequired in trained eSEN")
    ax.legend(loc="upper left", frameon=False, title=r"$\ell_{\max}{=}4$ model",
              fontsize=6.8, title_fontsize=7)


def lmax4_compute():
    """Training compute of the ell_max=4 model in each matched-compute pair."""
    rows = read_csv(os.path.join(REPO_ROOT, "data", "claim2", "matched_compute_lmax_checkpoints.csv"))
    return {r["role_tier"].replace("matched_compute_", ""): float(r["C_flops"])
            for r in rows if r["lmax"] == "4"}


def panel_b(ax, matched_rows):
    rows = [r for r in matched_rows if r["M"] == EVAL_SUBSET]
    flops = lmax4_compute()

    ax.axhline(0.0, color="black", linewidth=1.0, zorder=1)
    for domain, style in DOMAINS.items():
        xs, ys, lo_err, hi_err = [], [], [], []
        for tier in TIERS:
            row = next(r for r in rows
                       if r["lmax4_tag"] == OWNER_BY_TIER[tier] and r["domain"] == domain)
            h = float(row["H_force"])
            xs.append(flops[tier] * style["dx"])
            ys.append(h)
            lo_err.append(h - float(row["H_ci_lo"]))
            hi_err.append(float(row["H_ci_hi"]) - h)
        ax.errorbar(xs, ys, yerr=[lo_err, hi_err], fmt=style["marker"], color=style["color"],
                    markersize=7, linewidth=1.6, capsize=3, linestyle="none",
                    label=style["label"], zorder=3)

    ax.set_xscale("log")
    ax.set_xlim(1e16, 1e18)
    ax.set_xlabel("Matched training compute (FLOPs)")
    ax.set_ylabel(r"$\log(L_{\ell_{\max}=2}\,/\,L_{\ell_{\max}=4})$")
    ax.set_title("B. No matched-compute advantage\nfor $\\ell_{\\max}{=}4$")
    ax.legend(loc="lower right", frameon=False, fontsize=7)

    ylo, yhi = ax.get_ylim()
    offset = 0.04 * (yhi - ylo)
    xlo = ax.get_xlim()[0]
    ax.text(xlo * 1.3, offset, r"$\ell_{\max}{=}4$ better", fontsize=6.8, color="#555555",
            ha="left", va="bottom", style="italic")
    ax.text(xlo * 1.3, -offset, r"$\ell_{\max}{=}2$ better", fontsize=6.8, color="#555555",
            ha="left", va="top", style="italic")


def main():
    apply_style()
    dose_response = read_json(os.path.join(
        ANALYSIS_OUT, "claim2", "frontier_ell4_degree_balanced_recomputed.json"))
    matched_rows = read_csv(os.path.join(
        ANALYSIS_OUT, "claim2", "matched_compute_lmax2_vs_lmax4_baseline.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    panel_a(axes[0], dose_response)
    panel_b(axes[1], matched_rows)

    fig.tight_layout()
    for path in save_fig(fig, "figure2"):
        print("wrote", path)


if __name__ == "__main__":
    main()
