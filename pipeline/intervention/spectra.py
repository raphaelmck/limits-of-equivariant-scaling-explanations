# Adapted for anonymous release from analysis_scripts/repr_probe/spectra.py
# source revision withheld for anonymous review, original SHA256 497363963674998586fd24b5105b7a46727e9a86c09a3358a6f5a0ce45dd15d1
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Unsupervised representation spectra.

Part 1 (invariant/scalar representations, e.g. MPNN's frame-dependent-but-non-vector scalar
channel): CENTERED covariance spectrum, Sigma = H_c^T H_c / N, H_c = H - mean(H).

Part 2 (vector/irrep representations, e.g. MC-EGNN's vector_features, eSEN's irreps, and
GemNet-OC's aggregated edge->atom [3,D] block): UNCENTERED second-moment / contraction
spectrum, Sigma_cc' = E[sum_a h_{c,a} h_{c',a}]. Centering an l>=1 equivariant quantity over
observations is not rotation-covariant in general (the empirical mean of a vector/irrep
channel is not itself a fixed geometric object the same way a centered scalar residual is),
and -- separately but consistently -- the initial pilot's own ridge fit for these three architectures
uses bias=False (raw, uncentered X), so matching that fitting convention exactly is also
required for target_spectrum.py's exact-reconstruction guarantee (see that module's
docstring). This is not a coincidence: bias_on=True in the initial pilot probe design is exactly
the "MPNN only" case, and bias_on=False is exactly the "vector/irrep, fixed-direction-typed"
case -- Part 1 vs Part 2 here tracks that existing split exactly.

Both spectra are obtained from a single, numerically efficient dual/primal SVD (np.linalg.svd
on the (possibly centered) representation matrix directly -- LAPACK gesdd chooses the cheaper
of the two internally for tall/wide inputs, so no separate primal/dual code path is needed).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass
class SpectrumSummary:
    r_PR: float
    r_eff: float
    d50: int
    d80: int
    d90: int
    d95: int
    trace: float  # sum of eigenvalues = total (centered or uncentered, per caller) power/variance
    n_obs: int
    n_features: int
    rank: int
    eigenvalues: list[float]  # lambda_1 >= ... >= lambda_r >= 0, truncated to `keep` for storage
    p: list[float]            # normalized spectrum p_i = lambda_i / sum_j lambda_j, same truncation
    cumvar: list[float]       # cumulative sum of p, same truncation

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _dims_at(cum: np.ndarray, thresh: float) -> int:
    """Smallest k such that cum[k-1] >= thresh (1-indexed dimension count)."""
    idx = int(np.searchsorted(cum, thresh))
    return int(min(idx + 1, len(cum)))


def spectrum_from_eigenvalues(
    lam: np.ndarray, n_obs: int, n_features: int, keep_for_storage: int = 512
) -> SpectrumSummary:
    lam = np.clip(np.asarray(lam, dtype=np.float64), 0.0, None)
    total = float(lam.sum())
    p = lam / total if total > 0 else np.zeros_like(lam)
    r_PR = float((lam.sum() ** 2) / (np.sum(lam**2) + 1e-300))
    nz = p[p > 1e-300]
    r_eff = float(np.exp(-np.sum(nz * np.log(nz)))) if nz.size else 0.0
    cum = np.cumsum(p)
    k = min(keep_for_storage, len(lam))
    return SpectrumSummary(
        r_PR=r_PR,
        r_eff=r_eff,
        d50=_dims_at(cum, 0.50),
        d80=_dims_at(cum, 0.80),
        d90=_dims_at(cum, 0.90),
        d95=_dims_at(cum, 0.95),
        trace=total,
        n_obs=int(n_obs),
        n_features=int(n_features),
        rank=int(len(lam)),
        eigenvalues=lam[:k].tolist(),
        p=p[:k].tolist(),
        cumvar=cum[:k].tolist(),
    )


def dual_svd(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD X = U @ diag(s) @ V.T, s descending. np.linalg.svd(full_matrices=False) internally
    uses LAPACK gesdd, which is already the numerically efficient choice for either the primal
    (n>>d) or dual/Gram (d>>n) regime -- no separate code path needed. Returns U:[n,r],
    s:[r], V:[d,r] with r = min(n,d)."""
    U, s, Vt = np.linalg.svd(np.asarray(X, dtype=np.float64), full_matrices=False)
    return U, s, Vt.T


def pca_spectrum(
    H: np.ndarray, mean_ref: np.ndarray | None = None
) -> tuple[SpectrumSummary, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Part 1. H: [N,D]. Centers by `mean_ref` if given (the initial pilot-fit-pool convention, see
    target_spectrum.py), else by H's own column mean. Returns (summary, U, s, V, mean_used)
    with Sigma = Hc.T @ Hc / N eigenvalues lambda_i = s_i^2 / N."""
    H64 = np.asarray(H, dtype=np.float64)
    n = H64.shape[0]
    mean_used = mean_ref if mean_ref is not None else H64.mean(axis=0)
    Hc = H64 - mean_used[None, :]
    U, s, V = dual_svd(Hc)
    lam = (s**2) / n
    summary = spectrum_from_eigenvalues(lam, n_obs=n, n_features=H64.shape[1])
    return summary, U, s, V, mean_used


def power_spectrum(
    H_pooled: np.ndarray, n_normalize: int
) -> tuple[SpectrumSummary, np.ndarray, np.ndarray, np.ndarray]:
    """Part 2/8. H_pooled: [N*(2l+1) or 3N, C] -- already pooled over the Cartesian/m axis
    (e.g. MC-EGNN's vector_features reshaped [3N,C], eSEN's irreps[l] reshaped [N*(2l+1),C],
    GemNet-OC's aggregated [3N,D] block). NO centering (see module docstring). n_normalize is
    the number of independent ENTITIES (atoms), not pooled rows, matching the prompt's
    E[sum_a h_{c,a} h_{c',a}] convention (average over atoms, inner sum over the a/m index
    already folded into the Gram matrix) -- P_ell = trace(Sigma) reproduces total activation
    power per atom exactly. Returns (summary, U, s, V) with lambda_i = s_i^2 / n_normalize."""
    Hp = np.asarray(H_pooled, dtype=np.float64)
    U, s, V = dual_svd(Hp)
    lam = (s**2) / n_normalize
    summary = spectrum_from_eigenvalues(lam, n_obs=Hp.shape[0], n_features=Hp.shape[1])
    return summary, U, s, V


def vector_channel_covariance(vector_features: np.ndarray) -> np.ndarray:
    """MC-EGNN / GemNet-OC-aggregated Part 2 formula, C_cc' propto E[sum_a h_{c,a} h_{c',a}].
    vector_features: [N,3,C]. Returns C: [C,C], normalized by N (atoms)."""
    V = np.asarray(vector_features, dtype=np.float64)
    n = V.shape[0]
    Vp = V.reshape(-1, V.shape[-1])  # [3N, C] -- pools (atom,Cartesian) exactly like ridge.py's fit_pooled_shared
    return (Vp.T @ Vp) / n


def irrep_channel_covariance(irrep_l: np.ndarray) -> np.ndarray:
    """eSEN Part 2/8 formula, C_cc'^(l) propto E[sum_m h_{c,m}^(l) h_{c',m}^(l)*]. irrep_l:
    [N,2l+1,C], REAL-valued (e3nn real spherical-harmonic basis -- no complex coefficients
    anywhere in this pipeline, so h* = h; conjugation is a no-op, not silently dropped).
    Returns C: [C,C], normalized by N (atoms)."""
    H = np.asarray(irrep_l, dtype=np.float64)
    n = H.shape[0]
    Hp = H.reshape(-1, H.shape[-1])  # [N*(2l+1), C]
    return (Hp.T @ Hp) / n


def pooled_rows(entity_by_component: np.ndarray) -> np.ndarray:
    """[N,3,C] or [N,2l+1,C] -> [N*3,C] / [N*(2l+1),C], row-major (atom-major, then
    component/m), matching ridge.py's fit_pooled_shared reshape convention exactly."""
    X = np.asarray(entity_by_component)
    return X.reshape(-1, X.shape[-1])
