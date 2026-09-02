"""Shared plotting style/helpers for paper_repro/scripts/figures/*.

Deterministic by construction: fixed rcParams, fixed DPI, fixed save routine, and a fixed RNG
seed for any deliberate visual jitter (none of the current panels need jitter, but the helper is
here so a future panel does not introduce unseeded randomness).

This module reads ONLY from paper_repro/analysis_out/ (via each figure script) and does no
scientific computation of its own: log10 and simple ratios used here are pure display transforms
of already-validated numbers, not new statistics.
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed seed for any incidental visual jitter (none currently used, kept for determinism-by-policy).
JITTER_SEED = 20260823
random.seed(JITTER_SEED)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
ANALYSIS_OUT = os.path.join(REPO_ROOT, "analysis_out")
FIGURES_OUT = os.path.join(REPO_ROOT, "figures")

# Architecture color map, held constant across every figure in the paper.
ARCH_COLORS = {
    "MPNN": "#8a8f98",
    "MC-EGNN": "#4c78a8",
    "GemNet-OC": "#e45756",
    "eSEN": "#54a24b",
    "eSEN (lmax=4)": "#54a24b",
    "eSEN (lmax=2)": "#b5cf6b",
}
ARCH_MARKERS = {
    "MPNN": "s",
    "MC-EGNN": "D",
    "GemNet-OC": "o",
    "eSEN": "^",
    "eSEN (lmax=4)": "^",
    "eSEN (lmax=2)": "v",
}

TIER_ORDER = ["LOW", "MID", "UPPERMID", "HIGH"]

RC = {
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style():
    matplotlib.rcParams.update(RC)


def save_fig(fig, name: str):
    os.makedirs(FIGURES_OUT, exist_ok=True)
    pdf_path = os.path.join(FIGURES_OUT, f"{name}.pdf")
    png_path = os.path.join(FIGURES_OUT, f"{name}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path) as f:
        return json.load(f)
