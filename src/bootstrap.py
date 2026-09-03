"""Configuration-level paired bootstrap.

Resample configuration indices with replacement, apply the same resampled index set to every
cell, tier, or architecture in a comparison so cross-cell contrasts stay paired, recompute the
per-atom-weighted loss L = sum(per_config_sum) / sum(per_config_natoms) on the resample, then
recompute whatever scalar statistic the caller needs (Delta = log(L_int / L_base), a difference
of two such Deltas, and so on).

The bootstrap uses numpy.random.RandomState, the same generator class and the same `.choice`
API as the analyses that produced the frozen tables. Where a seed is documented for a specific
bootstrap -- for example the KRR pairwise bootstrap at seed 20260822 -- matching both the seed
and the exact call sequence (one `.choice` call per replicate on a single shared RandomState,
never one vectorized call across replicates) reproduces the frozen interval bit-exactly. Where
no seed was recorded, a fixed seed is chosen here and stated at the call site: that makes this
repository's own bootstrap deterministic and re-runnable, but the resulting interval matches the
frozen one statistically rather than bit-for-bit.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np

DEFAULT_N_BOOT = 2000


def weighted_L(per_config_sum: Sequence[float], per_config_natoms: Sequence[int]) -> float:
    """Per-atom-weighted mean loss: sum(sum_i) / sum(natoms_i). Matches the L_baseline/
    L_intervened values already stored in every claim2 raw JSONL row (verified against
    dose_response.jsonl's own L_original field to full float precision)."""
    return float(np.sum(per_config_sum)) / float(np.sum(per_config_natoms))


def log_delta(l_base: float, l_int: float) -> float:
    """Delta = log(L_intervened / L_baseline). Matches every 'delta'/'Delta' field observed
    across data/claim2/*.jsonl and *_summary.json (verified numerically against
    dose_response.jsonl's own alpha=0.75 row: log(0.024478.../0.023626...) == 0.03545...)."""
    return math.log(l_int / l_base)


def ci95(values: Sequence[float]) -> tuple[float, float]:
    """2.5/97.5 percentile with numpy's default ('linear') interpolation -- directly comparable
    to any numpy-computed frozen CI (e.g. np.quantile(boot_means, [0.025, 0.975]))."""
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


def paired_config_bootstrap(
    arrays: Mapping[str, tuple[Sequence[float], Sequence[int]]],
    statistic_fn: Callable[[Mapping[str, float]], float],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    rng: "np.random.RandomState | None" = None,
) -> list[float]:
    """Generic paired configuration-level bootstrap.

    arrays: {name -> (per_config_sum, per_config_natoms)}. All arrays MUST be aligned by
      configuration index (same molecule at position i across every name) -- this holds for
      every claim2 file that carries per_config_sum/per_config_natoms, since they all draw
      from the same frozen M=1024 (or M=16384, for the OOD raw files) population.
    statistic_fn: given {name -> resampled L value}, returns the scalar statistic of interest
      for this bootstrap replicate (e.g. a single Delta, or Delta_l4 - Delta_l2).
    rng: an existing numpy.random.RandomState to draw from (advances its state) -- pass this
      when a shared RNG stream must be threaded across multiple bootstrap calls in a fixed
      order to reproduce an upstream computation bit-exactly (see
      scripts/claim1/build_dense_grid_pairwise_gap.py, which threads one RandomState across
      all eight budgets in a fixed order). If omitted, a fresh numpy.random.RandomState(seed)
      is created for this call alone.
    Returns the list of n_boot replicate statistic values (caller computes the CI and, if
    wanted, the point estimate separately from the *unresampled* data).

    Resampling convention: for each replicate, draw n_configs indices into [0, n_configs) with
    replacement via `rng.randint(0, n_configs, size=n_configs)` (equivalent to
    `rng.choice(n_configs, size=n_configs, replace=True)` for an integer population -- both
    reduce to the same RandomState.randint call internally), apply the SAME index draw to every
    named array so cross-array contrasts stay paired, then sum per_config_sum/per_config_natoms
    over the drawn indices (sums, not means, since sampling with replacement duplicates
    denominators as well as numerators -- this matches how weighted_L is defined on the
    unresampled data).
    """
    names = list(arrays.keys())
    n_configs = len(arrays[names[0]][0])
    for name in names:
        if len(arrays[name][0]) != n_configs or len(arrays[name][1]) != n_configs:
            raise ValueError(f"array length mismatch for '{name}': expected {n_configs} configs")

    if rng is None:
        rng = np.random.RandomState(seed)

    np_arrays = {
        name: (np.asarray(s, dtype=np.float64), np.asarray(na, dtype=np.float64))
        for name, (s, na) in arrays.items()
    }

    replicates = []
    for _ in range(n_boot):
        idx = rng.randint(0, n_configs, size=n_configs)
        Ls = {}
        for name in names:
            s, na = np_arrays[name]
            Ls[name] = float(s[idx].sum() / na[idx].sum())
        replicates.append(statistic_fn(Ls))
    return replicates


def bootstrap_delta_ci(
    base_sum: Sequence[float],
    base_natoms: Sequence[int],
    int_sum: Sequence[float],
    int_natoms: Sequence[int],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    rng: "np.random.RandomState | None" = None,
) -> dict:
    """Convenience wrapper: point estimate + CI for Delta = log(L_int/L_base) on a single
    baseline/intervened pair of per-config arrays."""
    l_base_point = weighted_L(base_sum, base_natoms)
    l_int_point = weighted_L(int_sum, int_natoms)
    delta_point = log_delta(l_base_point, l_int_point)

    def stat(Ls):
        return log_delta(Ls["base"], Ls["int"])

    reps = paired_config_bootstrap(
        {"base": (base_sum, base_natoms), "int": (int_sum, int_natoms)},
        stat,
        n_boot=n_boot,
        seed=seed,
        rng=rng,
    )
    lo, hi = ci95(reps)
    return {
        "L_baseline": l_base_point,
        "L_intervened": l_int_point,
        "delta": delta_point,
        "delta_ci95": [lo, hi],
        "n_boot": n_boot,
        "bootstrap_seed": seed,
    }


def bootstrap_paired_log_ratio_ci(
    log_ratios: Sequence[float],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    rng: "np.random.RandomState | None" = None,
) -> dict:
    """Paired bootstrap CI over an already-paired list of per-split log-ratios
    (log(test_nmse_A[s]) - log(test_nmse_B[s]) for split index s), matching -- BIT-EXACTLY, when
    `rng` is threaded across pairs in the same order the upstream script processed them -- the
    procedure that produced the frozen dense-grid pairwise-gap table:

        rng = np.random.RandomState(RNG_SEED)                      # created ONCE, outside any
                                                                     # per-budget loop
        for budget in BUDGETS:                                     # all 8 budgets, fixed order
                ...
                boot_means = np.array([
                    rng.choice(log_ratios, size=len(log_ratios), replace=True).mean()
                    for _ in range(N_BOOT)
                ])                                                  # ONE rng.choice() CALL PER
                                                                     # REPLICATE, in a loop --
                                                                     # NOT a single vectorized
                                                                     # call across all N_BOOT
                                                                     # reps at once. These two
                                                                     # produce DIFFERENT draws
                                                                     # from the same seed, so the
                                                                     # loop form must be matched
                                                                     # exactly for bit-exactness.

    This function performs exactly the inner list-comprehension for ONE budget; the caller
    (scripts/claim1/build_dense_grid_pairwise_gap.py) is responsible for creating a single shared
    `np.random.RandomState(RNG_SEED)` and threading it through all 8 budgets in the SAME order
    the upstream script used, via the `rng=` argument, for bit-exact reproduction. If `rng` is
    omitted, a fresh RandomState(seed) is created for this call alone (used for tables where no
    shared cross-budget RNG stream is required/documented).
    """
    arr = np.asarray(log_ratios, dtype=np.float64)
    n = len(arr)
    point_estimate = float(arr.mean())
    if rng is None:
        rng = np.random.RandomState(seed)
    reps = [float(rng.choice(arr, size=n, replace=True).mean()) for _ in range(n_boot)]
    lo, hi = ci95(reps)
    return {
        "point_estimate": point_estimate,
        "ci95": [lo, hi],
        "n_boot": n_boot,
        "bootstrap_seed": seed,
        "n_paired": n,
    }


def ci_excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


def ci_overlap(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
    return lo1 <= hi2 and lo2 <= hi1
