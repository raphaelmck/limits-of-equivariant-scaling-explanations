# Adapted for anonymous release from analysis_scripts/repr_probe/rotation.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 c871c5c52f70b930ff0ef7ee575acd3b20f3e5ccc191699716de3f06e5f95386
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Rotation-equivariance checks for an intervened forward pass.

Rotation generator reproduces src/model/omol_module.py's GraphModel._random_rotations exactly
(QR of a Gaussian matrix, sign-fixed, det-flipped into SO(3)) -- per the architecture audit
Sec.8, this is both the canonical choice (it's literally the only source of rotation
consistency MPNN's training ever saw) and guarantees a proper rotation. Seeded here for a
deterministic, reproducible R, using a use a deterministic nontrivial 3-D
rotation R.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch


def random_rotation(seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    sign = torch.sign(torch.diagonal(R))
    Q = Q * sign.unsqueeze(0)
    if torch.linalg.det(Q) < 0:
        Q = torch.cat([Q[:, :1] * -1.0, Q[:, 1:]], dim=1)
    det = float(torch.linalg.det(Q))
    assert abs(det - 1.0) < 1e-4, f"rotation generator did not produce det=+1 (got {det})"
    return Q


def rotate_batch(batch, R: torch.Tensor):
    """Returns a deep-copied batch with pos (and forces, if present) rotated by R. Leaves the
    original batch untouched."""
    rotated = copy.deepcopy(batch)
    rotated.pos = torch.einsum("ij,nj->ni", R, rotated.pos)
    if hasattr(rotated, "forces") and rotated.forces is not None:
        rotated.forces = torch.einsum("ij,nj->ni", R, rotated.forces)
    return rotated


@dataclass
class RotationCheck:
    name: str
    max_abs_error: float
    max_rel_error: float
    tolerance: float
    passed: bool
    note: str = ""


def _err(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff = (a - b).abs()
    max_abs = float(diff.max().item())
    denom = b.abs().max().clamp_min(1e-8)
    max_rel = float((diff.max() / denom).item())
    return max_abs, max_rel


def check_force_equivariance(net_forward, batch, R: torch.Tensor, atol: float = 1e-4) -> RotationCheck:
    """Architecture-agnostic, unambiguous, decisive check (audit Sec.8's recommended
    "simplest, most decisive test" -- avoids any irrep basis-convention ambiguity since force
    is a genuine Cartesian vector by physical definition): F(RX) ~= R F(X)."""
    with torch.no_grad():
        f_x = net_forward(batch)["forces"]
        f_rx = net_forward(rotate_batch(batch, R))["forces"]
    f_x_rotated = torch.einsum("ij,nj->ni", R, f_x)
    max_abs, max_rel = _err(f_rx, f_x_rotated)
    return RotationCheck("force_equivariance", max_abs, max_rel, atol, max_rel <= atol)


def check_invariant(net_forward, batch, R: torch.Tensor, key: str, atol: float = 1e-4) -> RotationCheck:
    with torch.no_grad():
        a = net_forward(batch)[key]
        b = net_forward(rotate_batch(batch, R))[key]
    max_abs, max_rel = _err(b, a)
    return RotationCheck(f"invariant[{key}]", max_abs, max_rel, atol, max_rel <= atol)


def check_vector_equivariant(
    net_forward, batch, R: torch.Tensor, key: str, atol: float = 1e-4
) -> RotationCheck:
    """key's tensor must be [N,3,C] (Cartesian axis before channel axis, per the architecture
    audit's verified MC-EGNN convention) -- rotates the Cartesian axis (axis 1) only."""
    with torch.no_grad():
        x = net_forward(batch)[key]
        x_r = net_forward(rotate_batch(batch, R))[key]
    x_rotated = torch.einsum("ij,njc->nic", R, x)
    max_abs, max_rel = _err(x_r, x_rotated)
    return RotationCheck(f"vector_equivariant[{key}]", max_abs, max_rel, atol, max_rel <= atol)


def check_esen_l1_wigner(
    net_forward, batch, R: torch.Tensor, atol: float = 1e-3
) -> RotationCheck:
    """eSEN l=1 sector: h_1(RX) ~= D^1(R) h_1(X), using e3nn.o3.Irrep(1,1).D_from_matrix(R) in
    the network's own basis. installed fairchem-core (fairchem_core==2.14.0) verified
    BYTE-IDENTICAL to the vendored nets/uma/{nn,common}/*.py this pilot's adapters import
    convention from (diff exit 0) -- resolves the architecture audit Sec.10 point 3 open risk;
    the Wigner-D convention below is checked against the ACTUALLY-imported package, not an
    inference. A failure here with check_force_equivariance PASSING should be read as a
    real-SH basis/ordering mismatch in this check's own D-matrix construction, not a model bug
    -- the force-level check is the decisive one (audit Sec.8)."""
    import e3nn.o3

    with torch.no_grad():
        h_x = net_forward(batch)["node_embedding"]
        h_rx = net_forward(rotate_batch(batch, R))["node_embedding"]
    l1_x = h_x[:, 1:4, :]
    l1_rx = h_rx[:, 1:4, :]
    # e3nn's wigner_D builds its generator matrices on CPU internally regardless of the input
    # rotation's device (a library quirk, not something the original CPU-only smoke test could
    # have caught) -- compute on CPU explicitly, then move the small (3,3) result to h_x's device.
    D1 = e3nn.o3.Irrep(1, 1).D_from_matrix(R.detach().cpu().to(torch.float64)).to(h_x.device, h_x.dtype)
    l1_x_rotated = torch.einsum("ij,njc->nic", D1, l1_x)
    max_abs, max_rel = _err(l1_rx, l1_x_rotated)
    return RotationCheck("esen_l1_wigner", max_abs, max_rel, atol, max_rel <= atol)


def check_mpnn_negative_control(net_forward, batch, R: torch.Tensor) -> RotationCheck:
    """MPNN's native head has NO structural equivariance guarantee (architecture audit Sec.1/
    Sec.8) -- there is no law to assert here. This check is a NEGATIVE CONTROL only: verifies
    the harness is actually applying a nontrivial rotation (forces(RX) must differ
    meaningfully from forces(X) -- if it doesn't, the rotation likely isn't being applied)
    and reports the deviation ratio ||F(RX)-R F(X)|| / ||F(X)|| as a diagnostic, matching the
    audit's prescribed test. Does not gate PASS/FAIL on a tight tolerance since none is
    structurally expected; "passed" here means only "the harness demonstrably rotated the
    input," not "the model is equivariant."""
    with torch.no_grad():
        f_x = net_forward(batch)["forces"]
        f_rx = net_forward(rotate_batch(batch, R))["forces"]
    f_x_rotated = torch.einsum("ij,nj->ni", R, f_x)
    diff = (f_rx - f_x_rotated).norm()
    ratio = float((diff / f_x.norm().clamp_min(1e-8)).item())
    harness_applies_rotation = float((rotate_batch(batch, R).pos - batch.pos).abs().max().item()) > 1e-3
    return RotationCheck(
        "mpnn_negative_control",
        max_abs_error=ratio,
        max_rel_error=ratio,
        tolerance=float("nan"),
        passed=harness_applies_rotation,
        note=(
            f"no structural equivariance law expected; deviation ratio "
            f"||F(RX)-R.F(X)||/||F(X)|| = {ratio:.4f} reported as a diagnostic only "
            f"(architecture audit). PASS here means only that the harness "
            f"demonstrably applied a nontrivial rotation."
        ),
    )
