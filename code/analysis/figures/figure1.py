"""Figure 1 -- frozen force-NTK regression does not track the neural scaling transition.

Panel A: the empirical compute-optimal checkpoints of all four architectures (scatter) with the
         fitted power law per architecture (dashed guide line), plus the eight evaluation budgets
         as larger markers. Establishes that the architectures scale differently and that
         GemNet-OC and eSEN cross.
Panel B: the force-NTK KRR learning-curve area at the same eight budgets. Shows that KRR
         recovers the broad MPNN < MC-EGNN < {GemNet-OC, eSEN} ordering.
Panel C: the signed GemNet-OC/eSEN gap, neural (top) versus force-NTK KRR at n=512 (bottom), on a
         shared compute axis and a shared zero line. The neural gap changes sign at the observed
         crossover; the KRR gap and its 95% confidence interval stay GemNet-OC-favored at every
         budget.

Sign convention in Panel C, for both series: log(L_GemNet-OC / L_eSEN), so positive favors eSEN.

Plotting only: every quantity read here was computed and validated by `make analysis`
(analysis_out/claim1/*.csv). Log axes and the exponentiation of the frozen power-law fit are
display transforms, not new statistics.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from _style import ANALYSIS_OUT, ARCH_COLORS, ARCH_MARKERS, apply_style, read_csv, save_fig
from common.compute_axis import BUDGETS, budget_flops

ARCHS = ["MPNN", "MC-EGNN", "GemNet-OC", "eSEN"]

# Observed crossover: the compute of eSEN's first evaluation-budget checkpoint with lower
# normalized force MSE than GemNet-OC's. C_FIT is where the two fitted power laws cross.
C_OBS = 1.2435507843387587e17
C_FIT = 1.657746173616457e17

OBS_STYLE = {"linestyle": (0, (5, 5)), "linewidth": 0.9}
FIT_STYLE = {"linestyle": ":", "linewidth": 0.7}

# MPNN's highest budget reuses a checkpoint at only 50% of that budget's compute, so no MPNN
# point exists at the budget it nominally names; it is dropped from the display rather than
# plotted at a compute it did not reach.
UNREALIZED_BUDGETS = {("MPNN", "TOP")}

# Four eSEN widths were checkpointed twice as often as every other width. Thinning them for
# display keeps the plotted checkpoint cadence uniform across widths; the underlying table is
# untouched.
DENSE_CHECKPOINT_WIDTHS = {("eSEN", w): 2 for w in ("8", "40", "48", "64")}


def thin_dense_runs(points, arch):
    out = []
    by_width = {}
    for row in points:
        by_width.setdefault(row["width"], []).append(row)
    for width, rows in by_width.items():
        stride = DENSE_CHECKPOINT_WIDTHS.get((arch, width))
        if stride is None:
            out.extend(rows)
        else:
            out.extend(sorted(rows, key=lambda r: float(r["global_step"]))[::stride])
    return out


def panel_a(ax, frontier_points, ranking, flops, fits):
    # Restrict the scatter to the common fit domain, the interval on which all four exponents
    # were fitted. Below it only MPNN has checkpoints, so no cross-architecture comparison is
    # possible there.
    common_c_min = min(float(f["C_min_flops"]) for f in fits)
    for arch in ARCHS:
        pts = thin_dense_runs([r for r in frontier_points if r["architecture"] == arch], arch)
        xs = np.array([float(r["C_flops"]) for r in pts])
        ys = np.array([float(r["force_mse_norm"]) for r in pts])
        keep = xs >= common_c_min
        ax.scatter(xs[keep], ys[keep], s=8, color=ARCH_COLORS[arch], alpha=0.35,
                   linewidths=0, zorder=2)

    fit_by_arch = {r["architecture"].replace(" (lmax=4)", ""): r for r in fits}
    for arch in ARCHS:
        fit = fit_by_arch.get(arch)
        if fit is None:
            continue
        xs = np.geomspace(float(fit["C_min_flops"]), float(fit["C_max_flops"]), 50)
        ys = np.exp(float(fit["logA"])) * xs ** (-float(fit["gamma"]))
        ax.plot(xs, ys, "--", color=ARCH_COLORS[arch], linewidth=1.8, alpha=0.85, zorder=3)

    for arch in ARCHS:
        xs, ys = [], []
        for budget in BUDGETS:
            row = next((r for r in ranking
                        if r["architecture"] == arch and r["budget_tier"] == budget), None)
            if row is None:
                continue
            xs.append(flops[(arch, budget)])
            ys.append(float(row["force_mse_norm"]))
        ax.plot(xs, ys, ARCH_MARKERS[arch], color=ARCH_COLORS[arch], markersize=5,
                markeredgecolor="black", markeredgewidth=0.4, zorder=4)

    ax.axvline(C_OBS, color="#333333", alpha=0.6, zorder=0, **OBS_STYLE)
    ax.axvline(C_FIT, color="#333333", alpha=0.35, zorder=0, **FIT_STYLE)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training compute (FLOPs)")
    ax.set_ylabel("Normalized force MSE\n(lower is better)")
    ax.set_title("A. Architecture-dependent force scaling")
    handles = [
        Line2D([0], [0], color=ARCH_COLORS[arch], linestyle="--", linewidth=1.8,
               marker=ARCH_MARKERS[arch], markersize=6, markeredgecolor="black",
               markeredgewidth=0.4)
        for arch in ARCHS
    ]
    ax.legend(handles, ARCHS, loc="upper right", frameon=False, fontsize=6.5)
    ax.text(C_OBS, ax.get_ylim()[1], " $C_{\\mathrm{obs}}$", fontsize=6.5, color="#555555",
            ha="left", va="top")


def panel_b(ax, ranking, flops):
    for arch in ARCHS:
        xs, ys = [], []
        for budget in BUDGETS:
            row = next((r for r in ranking
                        if r["architecture"] == arch and r["budget_tier"] == budget), None)
            if row is None:
                continue
            xs.append(flops[(arch, budget)])
            ys.append(float(row["krr_auc_log_nmse_n32_512"]))
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "-", color=ARCH_COLORS[arch],
                marker=ARCH_MARKERS[arch], markersize=5, markeredgecolor="black",
                markeredgewidth=0.4, linewidth=1.3, zorder=3)

    ax.axvline(C_OBS, color="#333333", alpha=0.6, zorder=0, **OBS_STYLE)
    ax.axvline(C_FIT, color="#333333", alpha=0.35, zorder=0, **FIT_STYLE)

    ax.set_xscale("log")
    ax.set_xlabel("Training compute (FLOPs)")
    ax.set_ylabel("KRR learning-curve AUC\n(lower is better)")
    ax.set_title("B. Frozen force-NTK learnability\ndoes not track the crossover")


def panel_c(ax_top, ax_bot, gap_rows, flops):
    by_budget = {r["budget_tier"]: r for r in gap_rows}
    # GemNet-OC and eSEN own near-equal compute at each budget by construction; the GemNet-OC
    # value is used as the shared x-position of the pair.
    xs = np.array([flops[("GemNet-OC", b)] for b in BUDGETS])
    neural = np.array([float(by_budget[b]["neural_signed_gap_log_gemnet_over_esen"])
                       for b in BUDGETS])
    krr = np.array([float(by_budget[b]["krr_signed_gap_log_gemnet_over_esen_at_n512"])
                    for b in BUDGETS])
    lo = np.array([float(by_budget[b]["krr_bootstrap_ci_lo_2.5pct"]) for b in BUDGETS])
    hi = np.array([float(by_budget[b]["krr_bootstrap_ci_hi_97.5pct"]) for b in BUDGETS])

    order = np.argsort(xs)
    xs, neural, krr, lo, hi = (a[order] for a in (xs, neural, krr, lo, hi))

    for ax in (ax_top, ax_bot):
        ax.axhline(0.0, color="black", linewidth=0.9, zorder=1)
        ax.axvline(C_OBS, color="#333333", alpha=0.7, zorder=0, **OBS_STYLE)
        ax.set_xscale("log")

    ax_top.plot(xs, neural, "-o", color="#333333", markersize=5, linewidth=1.6,
                label="neural (force loss)")
    ax_top.set_ylabel("Neural log-loss gap", fontsize=7.5)
    ax_top.tick_params(labelbottom=False)
    ax_top.legend(loc="upper left", frameon=False, fontsize=6.5)
    ax_top.text(C_OBS, ax_top.get_ylim()[1] * 0.85, " $C_{\\mathrm{obs}}$", fontsize=6.5,
                color="#555555", ha="left", va="top")
    ax_top.text(0.02, 0.78, "positive = eSEN better", transform=ax_top.transAxes, fontsize=6.5,
                color="#555555", ha="left", va="top", style="italic")
    ax_top.set_title("C. Kernel learnability does not\ntrack the neural transition")

    ax_bot.errorbar(xs, krr, yerr=[krr - lo, hi - krr], fmt="-s", color="#e45756", markersize=5,
                    linewidth=1.6, capsize=2.5, label="KRR ($m$=1024, $n$=512, 95% CI)")
    ax_bot.set_ylabel("KRR log-NMSE gap", fontsize=7.5)
    ax_bot.set_xlabel("Training compute (FLOPs)")
    ax_bot.legend(loc="lower left", frameon=False, fontsize=6.5)
    # errorbar() over-pads the autoscale on a log axis; pin the limits to the real data range.
    ax_bot.set_xlim(xs.min() / 1.5, xs.max() * 1.5)


def main():
    apply_style()
    claim1 = lambda name: read_csv(os.path.join(ANALYSIS_OUT, "claim1", name))
    ranking = [r for r in claim1("dense_grid_ranking_comparison_recomputed.csv")
               if (r["architecture"], r["budget_tier"]) not in UNREALIZED_BUDGETS]
    gap_rows = claim1("dense_grid_pairwise_gap_recomputed.csv")
    frontier_points = claim1("frontier_points.csv")
    fits = claim1("force_scaling_recomputed.csv")
    flops = budget_flops()

    fig = plt.figure(figsize=(14, 4.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.08, wspace=0.32)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c_top = fig.add_subplot(gs[0, 2])
    ax_c_bot = fig.add_subplot(gs[1, 2], sharex=ax_c_top)

    panel_a(ax_a, frontier_points, ranking, flops, fits)
    panel_b(ax_b, ranking, flops)
    panel_c(ax_c_top, ax_c_bot, gap_rows, flops)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.12)
    for path in save_fig(fig, "figure1"):
        print("wrote", path)


if __name__ == "__main__":
    main()
