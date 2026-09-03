# Adapted for anonymous release from analysis_scripts/repr_probe/bundle.py
# source revision withheld for anonymous review, original SHA256 89f9b12b973540cbaee25da7800b858f4d4ceae53081baeae30b7284a61a42a6
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""RepresentationBundle: a common, auditable container for one architecture's frozen
per-checkpoint representation, produced by a single no-grad forward pass.

Design follows the architecture audit_architecture_interface_audit.md Sec. 9 exactly. It deliberately does
NOT flatten heterogeneous representations into one shape: MC-EGNN's vector channels,
GemNet-OC's scalar-plus-fixed-direction split, and eSEN's irrep-indexed tensor are different
mathematical objects, and collapsing them would lose exactly the information that audit
establishes as load-bearing for the probe-class design in
the architecture audit_representation_accessibility_measurement_spec.md Sec. 3.

Only the field(s) that actually apply to a given architecture are populated; the others stay
None. `geometry` is populated only when the architecture's NATIVE head structurally requires
external (non-learned) geometry to reconstruct its output (GemNet-OC only, per audit Sec. 6) --
a populated `geometry` field on any other architecture's native-tier bundle is itself a signal
that the tier boundary has drifted from native toward common/enriched.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RepresentationBundle:
    architecture: str            # "mpnn" | "mcegnn" | "gemnet_oc" | "esen"
    checkpoint_tag: str          # owner tag (e.g. "mpnn_LOW_w607_s400000"), for provenance
    layer: str                   # e.g. "final" -- matches the interface audit's inventory table

    entity_type: str             # "atom" | "edge" -- GemNet-OC's native force-relevant tensor
                                  # is EDGE-indexed; code that assumes "one row per atom" breaks
                                  # for it, so this is not architecture-constant.

    # Symmetry-typed payloads -- populate ONLY the field(s) that apply.
    scalar_features: torch.Tensor | None = None
    # invariant-or-frame-dependent (see metadata["scalar_is_invariant"]), shape (n_entity, C)
    vector_features: torch.Tensor | None = None
    # equivariant Cartesian, shape (n_entity, 3, C) -- axis order Cartesian-then-channel,
    # verified by the architecture audit's einsum trace (Sec. 2.2) -- MC-EGNN only.
    irreps: dict[int, torch.Tensor] | None = None
    # {l: tensor of shape (n_entity, 2l+1, C)}, already split by degree (not the network's
    # internal flattened-(lmax+1)^2 layout) -- eSEN only.

    geometry: dict[str, torch.Tensor] | None = None
    # fixed (non-learned) geometry the entity's NATIVE head structurally requires, e.g.
    # {"edge_vector": (E,3), "idx_t": (E,)} for GemNet-OC. None for MPNN/MC-EGNN/eSEN's native
    # heads, which need none.

    index_maps: dict[str, torch.Tensor] = field(default_factory=dict)
    # batch (atom/edge -> graph), edge_index, and whatever else the intended SUFFIX (not
    # already applied upstream of this bundle) needs to run.

    metadata: dict = field(default_factory=dict)
    # REQUIRED keys: "width_mult" (float, mandatory -- MuReadout/SO3_MuReadout divides its
    # input by this before the linear map; omitting it silently produces a wrong suffix
    # reconstruction rather than an error, per audit Sec. 1/Sec. 10).
    # Conventional additional keys used by this pilot: "dtype", "bias_on" (bool, native-head
    # bias convention per audit Sec. 5), "scalar_is_invariant" (bool), "lmax", "mmax",
    # "sphere_channels", "git_sha", "checkpoint_sha256".

    def __post_init__(self) -> None:
        if "width_mult" not in self.metadata:
            raise ValueError(
                f"{self.architecture}/{self.checkpoint_tag}: RepresentationBundle.metadata "
                "must include width_mult -- MuReadout/SO3_MuReadout divides by it before the "
                "linear map (nets/mup.py); omitting it silently produces a wrong suffix "
                "reconstruction instead of raising."
            )
        if self.entity_type not in ("atom", "edge"):
            raise ValueError(f"entity_type must be 'atom' or 'edge', got {self.entity_type!r}")
