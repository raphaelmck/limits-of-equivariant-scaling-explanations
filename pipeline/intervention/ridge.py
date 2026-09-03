# Adapted for anonymous release from analysis_scripts/repr_probe/ridge.py
# source revision withheld for anonymous review, original SHA256 4bd6ef448ba0b7c1f78cbe29d488c185d335cf16e34bc2b20948457e36bfffda
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Ridge-regression probe fitting and its regularization protocol.

Two closed-form fitting primitives cover every probe class in this pilot (native for all 4
architectures, common for all 4):

- `fit_multi_output`: features X:[n,D] (one row per atom), targets Y:[n,3] -- a SEPARATE
  weight row per Cartesian output component (W in R^{3xD}), optional shared intercept.
  This is MPNN's native function class (g(h)=Wh+b) and is the only probe class in this
  pilot with this shape -- all three other architectures' native heads, and the common
  tier's aggregated-direction construction, share ONE weight vector across the 3 Cartesian
  (or m) output rows instead (see fit_pooled_shared).

- `fit_pooled_shared`: features X:[n,3,D] (one [3,D] feature block per atom -- 3 rows already
  aligned to the 3 output components), targets Y:[n,3] -- ONE shared weight vector w in
  R^{1xD} applied identically to each of the 3 rows, pooled over all (atom,component) pairs
  before fitting (matches the micro-averaged weighting convention, the architecture audit). This is
  MC-EGNN's native class (w contracts the channel axis identically for x/y/z), eSEN's native
  class (W_1 channel-mixes identically across the 3 m-rows of the l=1 sector -- Schur's lemma
  forbids mixing m), and GemNet-OC's native class AND every architecture's common-tier class
  once their per-edge/per-neighbor features have been aggregated into a per-(atom,component)
  vector via the shared external direction (see decoder.py) -- these two are the same
  underlying linear-algebra shape, just built by different upstream aggregations.

Bias/intercept handled by the standard augmented-column trick (bias term is NOT regularized).
All fitting happens in float64 (fixed by protocol: precision_protocol).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _to_np64(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def lambda_grid(X_fit: np.ndarray) -> np.ndarray:
    """9-point log grid, lambda_k = 10^k * s, k in {-6,...,2}, s = trace(X^T X)/(n*d).
    the regularization_protocol (grid_definition), applied identically to
    every (architecture, budget, probe_class) cell -- only the resulting scale s differs."""
    n, d = X_fit.shape
    s = float(np.trace(X_fit.T @ X_fit) / (n * d))
    s = max(s, 1e-300)  # guard against a degenerate all-zero feature block
    return np.array([10.0**k * s for k in range(-6, 3)], dtype=np.float64)


def _ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float, bias: bool) -> np.ndarray:
    """Single-output ridge: w = argmin ||y - Xw||^2 + lam*||w||^2 (bias column, if present,
    excluded from the penalty). Returns w (with an appended intercept as the last entry if
    bias=True). X:[n,d], y:[n]."""
    n, d = X.shape
    if bias:
        Xb = np.concatenate([X, np.ones((n, 1), dtype=np.float64)], axis=1)
        reg = np.eye(d + 1, dtype=np.float64) * lam
        reg[-1, -1] = 0.0  # never regularize the intercept
    else:
        Xb = X
        reg = np.eye(d, dtype=np.float64) * lam
    A = Xb.T @ Xb + reg
    b = Xb.T @ y
    w = np.linalg.solve(A, b)
    return w


def _apply(X: np.ndarray, w: np.ndarray, bias: bool) -> np.ndarray:
    if bias:
        return X @ w[:-1] + w[-1]
    return X @ w


@dataclass
class RidgeFitResult:
    kind: str            # "multi_output" | "pooled_shared"
    bias: bool
    W: np.ndarray         # multi_output: [3,D] (+intercept row appended if bias); pooled_shared: [D] (+intercept)
    selected_lambda: float
    selected_lambda_grid_index: int
    grid: np.ndarray
    e_probe_val: float
    e_probe_test: float | None
    grid_e_val: np.ndarray | None = None  # validation E_probe at EVERY grid point (diagnostic
        # only; purely additive capture of a value already computed inside the selection loop
        # below -- does not change the selection rule, argmin, or any returned W/lambda).


def _e_probe(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """the architecture audit: uncentered, micro-averaged (flat mean over every (atom,component)
    pair) normalized error. y_pred/y_true: [n,3] or [n]."""
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    num = float(np.mean((y_true - y_pred) ** 2))
    den = float(np.mean(y_true ** 2))
    return num / den


def predict(X, fit_result: "RidgeFitResult") -> np.ndarray:
    """Apply an already-fitted g (fit_result.W, no refit) to new rows. Returns [n,3].
    Used by the bootstrap-CI protocol (the uncertainty_protocol:
    "recompute E_probe on each replicate WITHOUT refitting g") -- resampling only needs to
    re-average already-computed per-molecule residuals, never re-solve the ridge system."""
    X = _to_np64(X)
    if fit_result.kind == "multi_output":
        return np.stack([_apply(X, fit_result.W[c], fit_result.bias) for c in range(3)], axis=1)
    n, three, d = X.shape
    assert three == 3
    Xp = X.reshape(n * three, d)
    yp = _apply(Xp, fit_result.W, fit_result.bias)
    return yp.reshape(n, three)


def per_molecule_residuals(X, Y, fit_result: "RidgeFitResult") -> tuple[np.ndarray, np.ndarray]:
    """Per-molecule squared-error and squared-true-norm, summed over the 3 Cartesian
    components (the architecture audit micro-averaged E_probe = SE.sum()/SS.sum() over any subset of
    molecules, since the (atom,component)-count cancels in the denominator). Feeding a
    bootstrap-resampled subset of rows into these two sums and taking their ratio reproduces
    exactly what re-scoring E_probe on that resample would give, without refitting g."""
    Y = _to_np64(Y)
    Y_pred = predict(X, fit_result)
    se = ((Y - Y_pred) ** 2).sum(axis=1)
    ss = (Y ** 2).sum(axis=1)
    return se, ss


def fit_multi_output(
    X_fit, Y_fit, X_val, Y_val, X_test=None, Y_test=None, bias: bool = True
) -> RidgeFitResult:
    X_fit, Y_fit = _to_np64(X_fit), _to_np64(Y_fit)
    X_val, Y_val = _to_np64(X_val), _to_np64(Y_val)
    grid = lambda_grid(X_fit)
    best = None
    grid_e_val = np.empty(len(grid), dtype=np.float64)
    for k_idx, lam in enumerate(grid):
        W_cols = [_ridge_closed_form(X_fit, Y_fit[:, c], lam, bias) for c in range(3)]
        Y_val_pred = np.stack([_apply(X_val, W_cols[c], bias) for c in range(3)], axis=1)
        e_val = _e_probe(Y_val_pred, Y_val)
        grid_e_val[k_idx] = e_val
        if best is None or e_val < best[0]:
            best = (e_val, k_idx, lam, W_cols)
    e_val, k_idx, lam, W_cols = best
    W = np.stack(W_cols, axis=0)  # [3, D(+1)]
    e_test = None
    if X_test is not None:
        X_test, Y_test = _to_np64(X_test), _to_np64(Y_test)
        Y_test_pred = np.stack([_apply(X_test, W_cols[c], bias) for c in range(3)], axis=1)
        e_test = _e_probe(Y_test_pred, Y_test)
    return RidgeFitResult("multi_output", bias, W, float(lam), k_idx, grid, float(e_val), e_test, grid_e_val)


def fit_pooled_shared(
    X_fit, Y_fit, X_val, Y_val, X_test=None, Y_test=None, bias: bool = False
) -> RidgeFitResult:
    """X_*: [n,3,D] (already component-aligned), Y_*: [n,3]. Pools to [3n,D]/[3n] before
    fitting, matching the architecture audit's micro-averaged (atom,component) weighting exactly."""
    X_fit, Y_fit = _to_np64(X_fit), _to_np64(Y_fit)
    X_val, Y_val = _to_np64(X_val), _to_np64(Y_val)
    n_fit, three, d = X_fit.shape
    assert three == 3
    Xp_fit = X_fit.reshape(n_fit * 3, d)
    Yp_fit = Y_fit.reshape(n_fit * 3)
    n_val = X_val.shape[0]
    Xp_val = X_val.reshape(n_val * 3, d)
    Yp_val = Y_val.reshape(n_val * 3)
    grid = lambda_grid(Xp_fit)
    best = None
    grid_e_val = np.empty(len(grid), dtype=np.float64)
    for k_idx, lam in enumerate(grid):
        w = _ridge_closed_form(Xp_fit, Yp_fit, lam, bias)
        y_val_pred = _apply(Xp_val, w, bias)
        e_val = _e_probe(y_val_pred, Yp_val)
        grid_e_val[k_idx] = e_val
        if best is None or e_val < best[0]:
            best = (e_val, k_idx, lam, w)
    e_val, k_idx, lam, w = best
    e_test = None
    if X_test is not None:
        X_test, Y_test = _to_np64(X_test), _to_np64(Y_test)
        n_test = X_test.shape[0]
        Xp_test = X_test.reshape(n_test * 3, d)
        Yp_test = Y_test.reshape(n_test * 3)
        y_test_pred = _apply(Xp_test, w, bias)
        e_test = _e_probe(y_test_pred, Yp_test)
    return RidgeFitResult("pooled_shared", bias, w, float(lam), k_idx, grid, float(e_val), e_test, grid_e_val)
