# Adapted for anonymous release from analysis_scripts/repr_probe/esen_intervention.py
# source revision withheld for anonymous review, original SHA256 08234b6d15003ee229889d4374a4c29943c0d7561a0e042d67bc3008e7f8d271
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Symmetry-preserving intervention on eSEN irreducible-representation sectors.

Reproduces `nets.uma.backbone.eSCNMDBackbone.forward` ORCHESTRATION exactly (same calls, same
order, same arguments) but inserts one masking operation on the block-`layer` output before it
is fed to block `layer+1` (or, for the last block, before the shared final `self.norm`). Every
sub-module invoked (`_generate_graph`, `_get_rotmat_and_wigner`, `sphere_embedding`, `envelope`,
`distance_expansion`, `source_embedding`, `target_embedding`, `edge_degree_embedding`,
`blocks[i]`, `norm`, `energy_head`, `force_head`) is called UNMODIFIED and UNRETRAINED, directly
off the loaded checkpoint's own `model.net.backbone` / `model.net` objects -- this module adds
no new parameters and duplicates no sub-module logic, only the glue that decides which tensor
flows into which call. Verbatim source for the orchestration this replicates:
nets/uma/backbone.py:389-505 (eSCNMDBackbone.forward), nets/uma/model.py:74-79
(EquivariantNet.forward). Deliberately NOT implemented as an edit to backbone.py/model.py
(both already carry the layerwise study's own uncommitted diff) -- see the intervention manifest for the repo-state
rationale.

Coefficient-axis layout (dim=1 of x_message, size (lmax+1)^2=25 for lmax=4): verified via
fairchem.core.models.uma.common.so3.CoefficientMapping's construction loop (each degree l
occupies a contiguous run of (2l+1) m-slots, l=0 first) and cross-checked against the force
head's own `node_embedding.narrow(1, 0, 4)` = exactly the l<=1 slots (nets/uma/backbone.py:757).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from fairchem.core.common import gp_utils

# (start, end) into the size-25 coefficient axis for each degree l, lmax=4.
# sizes 1,3,5,7,9 for l=0..4; verified against nets/uma/backbone.py's sph_feature_size=(lmax+1)**2
# and the force head's narrow(1,0,4) = l0+l1 slots.
IRREP_SLICES: dict[int, tuple[int, int]] = {0: (0, 1), 1: (1, 4), 2: (4, 9), 3: (9, 16), 4: (16, 25)}
NUM_COEFFS = 25


def _ell_list(ell) -> list[int]:
    return [ell] if isinstance(ell, int) else list(ell)


def deterministic_channel_permutation(num_channels: int, tag: str, ell) -> np.ndarray:
    """Fixed, seeded, outcome-independent channel ordering for dose-response nesting.
    Seed derives only from (checkpoint tag, ell) -- never from any measured effect."""
    ell_key = "-".join(str(e) for e in _ell_list(ell))
    seed = abs(hash(("stage3a-channel-perm", tag, ell_key))) % (2**32)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_channels)
    return perm


def channels_for_fraction(num_channels: int, tag: str, ell, frac: float) -> np.ndarray:
    """Nested channel subset: prefix of a fixed permutation, so the 25%/50%/75%/100% masks are
    literally nested subsets (25% subset ⊂ 50% subset ⊂ ...)."""
    perm = deterministic_channel_permutation(num_channels, tag, ell)
    k = int(round(frac * num_channels))
    return np.sort(perm[:k])


def sector_power(x_message: torch.Tensor, ell) -> float:
    total = 0.0
    for e in _ell_list(ell):
        s, en = IRREP_SLICES[e]
        total += float((x_message[:, s:en, :] ** 2).sum().item())
    return total


def total_power(x_message: torch.Tensor) -> float:
    return float((x_message ** 2).sum().item())


@dataclass
class InterventionSpec:
    layer: int
    ell: object  # int or list[int]
    alpha: float = 0.0
    channel_idx: np.ndarray | None = None  # None = all channels

    def describe(self) -> dict:
        return {
            "layer": self.layer,
            "ell": self.ell if isinstance(self.ell, int) else list(self.ell),
            "alpha": self.alpha,
            "channel_idx": None if self.channel_idx is None else [int(c) for c in self.channel_idx],
            "n_channels_masked": None if self.channel_idx is None else int(len(self.channel_idx)),
        }


def apply_intervention(x_message: torch.Tensor, spec: InterventionSpec) -> tuple[torch.Tensor, dict]:
    """Zero (or alpha-scale) the complete degree-ell irrep sector(s), for every atom and every
    selected channel, leaving all other degrees and all other channels bit-identical. Returns
    the modified tensor (new allocation; input is not mutated) plus power-accounting diagnostics."""
    power_total_pre = total_power(x_message)
    x_new = x_message.clone()
    power_removed = 0.0
    power_sector_full = 0.0
    for e in _ell_list(spec.ell):
        s, en = IRREP_SLICES[e]
        sector = x_new[:, s:en, :]
        power_sector_full += float((sector ** 2).sum().item())
        if spec.channel_idx is None:
            removed = sector - spec.alpha * sector
            power_removed += float((removed ** 2).sum().item())
            sector.mul_(spec.alpha)
        else:
            idx = torch.as_tensor(spec.channel_idx, device=x_message.device, dtype=torch.long)
            sub = sector[:, :, idx]
            removed = sub - spec.alpha * sub
            power_removed += float((removed ** 2).sum().item())
            sub_scaled = spec.alpha * sub
            sector[:, :, idx] = sub_scaled
        x_new[:, s:en, :] = sector
    diagnostics = {
        "power_total_pre": power_total_pre,
        "power_sector_full": power_sector_full,
        "power_removed": power_removed,
        "q_power": (power_removed / power_total_pre) if power_total_pre > 0 else float("nan"),
    }
    return x_new, diagnostics


class ESenInterventionForward:
    """Callable: (data_dict, intervention: InterventionSpec | None) -> (outputs, diagnostics).

    `outputs` has the same keys as `nets.uma.model.EquivariantNet.forward`'s return dict
    (energy, forces, node_embedding, node_embedding_per_block). With `intervention=None` this
    is bit-identical to `model.net(data_dict)` (Validation 9.1)."""

    def __init__(self, model):
        self.model = model
        self.net = model.net
        self.backbone = model.net.backbone

    @torch.no_grad()
    def __call__(self, data_dict, intervention: InterventionSpec | None = None):
        backbone = self.backbone
        data_dict["atomic_numbers"] = data_dict["atomic_numbers"].long()
        data_dict["atomic_numbers_full"] = data_dict["atomic_numbers"]
        data_dict["batch_full"] = data_dict["batch"]

        graph_dict = backbone._generate_graph(data_dict)
        if graph_dict["edge_index"].numel() == 0:
            raise ValueError("No edges found in input system.")

        (edge_rot_mat, wigner_and_M_mapping_full, wigner_and_M_mapping_inv_full) = (
            backbone._get_rotmat_and_wigner(
                graph_dict["edge_distance_vec_full"],
                use_cuda_graph=False,
            )
        )
        if gp_utils.initialized():
            wigner_and_M_mapping = wigner_and_M_mapping_full[graph_dict["edge_partition"]]
            wigner_and_M_mapping_inv = wigner_and_M_mapping_inv_full[graph_dict["edge_partition"]]
        else:
            wigner_and_M_mapping = wigner_and_M_mapping_full
            wigner_and_M_mapping_inv = wigner_and_M_mapping_inv_full

        x_message = torch.zeros(
            data_dict["atomic_numbers"].shape[0],
            backbone.sph_feature_size,
            backbone.sphere_channels,
            device=data_dict["pos"].device,
            dtype=data_dict["pos"].dtype,
        )
        x_message[:, 0, :] = backbone.sphere_embedding(data_dict["atomic_numbers"])

        dist_scaled = graph_dict["edge_distance"] / backbone.cutoff
        edge_envelope = backbone.envelope(dist_scaled).reshape(-1, 1, 1)
        edge_distance_embedding = backbone.distance_expansion(graph_dict["edge_distance"])
        source_embedding = backbone.source_embedding(
            data_dict["atomic_numbers_full"][graph_dict["edge_index"][0]]
        )
        target_embedding = backbone.target_embedding(
            data_dict["atomic_numbers_full"][graph_dict["edge_index"][1]]
        )
        x_edge = torch.cat((edge_distance_embedding, source_embedding, target_embedding), dim=1)

        node_offset = (
            data_dict["gp_node_offset"] if "gp_node_offset" in data_dict else graph_dict.get("node_offset", 0)
        )

        x_message = backbone.edge_degree_embedding(
            x_message, x_edge, graph_dict["edge_index"], wigner_and_M_mapping_inv, edge_envelope, node_offset,
        )

        diagnostics: dict = {}
        node_embedding_per_block = []
        for i in range(backbone.num_layers):
            x_message = backbone.blocks[i](
                x_message,
                x_edge,
                graph_dict["edge_distance"],
                graph_dict["edge_index"],
                wigner_and_M_mapping,
                wigner_and_M_mapping_inv,
                edge_envelope,
                data_dict["atomic_numbers_full"].shape[0],
                sys_node_embedding=None,
                node_offset=node_offset,
            )
            if intervention is not None and intervention.layer == i:
                x_message, diagnostics = apply_intervention(x_message, intervention)
            node_embedding_per_block.append(x_message)

        x_message = backbone.norm(x_message)
        outputs = {
            "node_embedding": x_message,
            "batch": data_dict["batch"],
            "node_embedding_per_block": node_embedding_per_block,
        }
        energy = self.net.energy_head(data_dict, outputs)["energy"]
        forces = self.net.force_head(data_dict, outputs)["forces"]
        result = {
            "energy": energy,
            "forces": forces,
            "node_embedding": outputs["node_embedding"],
            "node_embedding_per_block": outputs["node_embedding_per_block"],
        }
        return result, diagnostics
