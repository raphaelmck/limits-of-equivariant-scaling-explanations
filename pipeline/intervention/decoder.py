# Adapted for anonymous release from analysis_scripts/repr_probe/common_decoder.py
# source revision withheld for anonymous review, original SHA256 fd96c29442e2f66d215938444833ae2cc0985e333af7996a25c061ea656285f3
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Shared machinery for probes whose function class is a learned scalar gate times a fixed
external direction, summed over neighbours or edges.

IMPLEMENTATION DECISION (flagged, not silently resolved -- the theory contract left this
open): the architecture audit describes the common tier's per-edge nonlinearity phi as "a small fixed
nonlinearity (e.g. an MLP with a fixed, shared architecture across all four probes)" -- an
example, not a mandate -- while this implementation task's Part 7 explicitly forbids "a
high-capacity MLP probe." This module resolves that tension with a FIXED (non-learned,
zero trainable parameters), architecture-common feature map: phi(h_i,h_j,||r_ij||) =
concat(h_i, h_j, RBF(||r_ij||), 1). Only h's CONTENT differs by architecture (each
architecture's own scalar channel, per checkpoint_manifest.yaml's representation_extraction_
contract); the RBF basis, its width/cutoff, and the functional FORM of phi are identical
across all four. The trailing constant-1 feature is how this module implements the design
freeze's "bias ON, uniformly" rule (the architecture audit) for this probe class WITHOUT breaking
equivariance: a raw additive Cartesian-vector bias would not transform correctly under
rotation, but a learned bias on the SCALAR gate w.phi+b, still multiplied by the fixed
direction and summed, preserves the same CG 0(x)1->1 equivariance mechanism GemNet-OC's own
native head already uses (only the gate itself gains an intercept). This is an explicit,
documented implementation choice, not a silent deviation -- flag it in any writeup that uses
the common tier's results.

Used for:
- GemNet-OC's NATIVE probe: aggregate x_F (learned) against the model's own fixed
  edge_vector/idx_t (no fresh graph needed -- reuses exactly the graph the checkpoint itself
  built).
- Every architecture's COMMON probe: aggregate each architecture's own atom-level scalar
  channel against a FRESH, common cutoff=6.0 graph built here from raw coordinates (never the
  architecture's own native graph, which may use a different cutoff, e.g. GemNet-OC's 12.0).
"""
from __future__ import annotations

import torch

COMMON_CUTOFF = 6.0
RBF_DIM = 16


def build_common_graph_dense(
    pos: torch.Tensor, cutoff: float = COMMON_CUTOFF
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense (all-pairs) neighbor search -- fine at probe-molecule scale (tens of atoms).
    Returns (center_idx, neighbor_idx, dist, r_hat) for every ordered pair within cutoff
    (self-pairs excluded). r_hat[e] points FROM center_idx[e] TO neighbor_idx[e]."""
    diff = pos[None, :, :] - pos[:, None, :]  # diff[i,j] = pos_j - pos_i
    dist_mat = diff.norm(dim=-1)
    mask = dist_mat <= cutoff
    mask.fill_diagonal_(False)
    center_idx, neighbor_idx = mask.nonzero(as_tuple=True)
    r = diff[center_idx, neighbor_idx]
    dist = dist_mat[center_idx, neighbor_idx].clamp_min(1e-8)
    r_hat = r / dist[:, None]
    return center_idx, neighbor_idx, dist, r_hat


def rbf_expand(dist: torch.Tensor, cutoff: float = COMMON_CUTOFF, num_basis: int = RBF_DIM) -> torch.Tensor:
    centers = torch.linspace(0, cutoff, num_basis, device=dist.device, dtype=dist.dtype)
    width = cutoff / num_basis
    return torch.exp(-((dist[:, None] - centers[None, :]) ** 2) / (2 * width**2))


def aggregate_to_atoms(
    edge_features: torch.Tensor, direction: torch.Tensor, target_idx: torch.Tensor, num_atoms: int
) -> torch.Tensor:
    """edge_features: [E,D], direction: [E,3] (unit vectors), target_idx: [E].
    Returns A: [num_atoms,3,D] with A[i,c,:] = sum_{e: target_idx[e]==i} edge_features[e]*direction[e,c].
    This is the aggregation both GemNet-OC's native head and this module's common-decoder
    construction reduce to (scalar-times-fixed-direction, summed over neighbors)."""
    D = edge_features.shape[-1]
    A = torch.zeros(num_atoms, 3, D, dtype=edge_features.dtype, device=edge_features.device)
    for c in range(3):
        A[:, c, :].index_add_(0, target_idx, edge_features * direction[:, c : c + 1])
    return A


def common_decoder_atom_features(
    h: torch.Tensor, pos: torch.Tensor, cutoff: float = COMMON_CUTOFF
) -> torch.Tensor:
    """h: [N,C] atom-level scalar channel (this architecture's own common_probe_input).
    Returns A: [N,3,2C+RBF_DIM+1] -- per-atom, per-Cartesian-component aggregated feature
    block, ready for ridge.fit_pooled_shared (bias=False; the bias is already the trailing
    constant-1 column inside phi, see module docstring)."""
    center_idx, neighbor_idx, dist, r_hat = build_common_graph_dense(pos, cutoff)
    rbf = rbf_expand(dist, cutoff)
    ones = torch.ones(center_idx.shape[0], 1, dtype=h.dtype, device=h.device)
    phi = torch.cat([h[center_idx], h[neighbor_idx], rbf, ones], dim=-1)
    return aggregate_to_atoms(phi, r_hat, center_idx, num_atoms=h.shape[0])
