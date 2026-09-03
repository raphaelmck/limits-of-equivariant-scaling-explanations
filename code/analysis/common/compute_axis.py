"""Training compute of each evaluation-budget checkpoint -- the shared x-axis join.

At a nominal budget, the frontier checkpoint is the available checkpoint with the lowest
normalized force MSE whose realized compute does not exceed that budget, so adjacent budgets can
own the same checkpoint. A budget that reuses its neighbour's checkpoint has no checkpoint-table
row of its own and resolves through the budget it reuses from.
"""
from __future__ import annotations

from .io_utils import data_path, read_csv

BUDGETS = ["LOW", "MID", "PRECROSS", "CROSS", "FITX", "POSTCROSS", "HIGH", "TOP"]


def budget_flops():
    """Map (architecture, budget) to the realized training compute of its checkpoint."""
    out = {}
    for row in read_csv(data_path("checkpoints.csv")):
        if row["claim"] != "1":
            continue
        budget = row["role_tier"].split(" ")[0]
        if budget in BUDGETS:
            out.setdefault((row["architecture"], budget), float(row["C_flops"]))

    for row in read_csv(data_path("claim1", "dense_grid_frontier_force_mse.csv")):
        key = (row["architecture"], row["budget_tier"])
        if key in out or row["reused_checkpoint"] != "True" or not row["reused_from_budget_tier"]:
            continue
        source = row["reused_from_budget_tier"].split(" ")[0].split("(")[0].strip()
        if (row["architecture"], source) in out:
            out[key] = out[(row["architecture"], source)]
    return out
