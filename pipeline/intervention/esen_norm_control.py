# Adapted for anonymous release from analysis_scripts/repr_probe/esen_norm_control.py
# source revision 3cd17ea1aa59336bef0252504ebbfb1bc8c6a412, original SHA256 09748e10310338e8f4fe03e2d12a53103474c7e0e105e0044ce6a8c58aa363f9
# Logic unchanged; import paths, filesystem paths, and identifying names rewritten.
# Verify with: python3 provenance/check_pipeline.py
"""Exact reconstruction of eSEN's shared final normalization
(`EquivariantRMSNormArraySphericalHarmonicsV2`, `fairchem_core==2.14.0`,
`fairchem/core/models/uma/nn/layer_norm.py:305-402`), plus the fixed-normalization control
used by every intervention: the normalization statistic is held at its unperturbed value so
that attenuating one degree cannot rescale the untouched degrees.

All functions here operate on the SAME `backbone.norm` nn.Module instance the checkpoint was
trained with (its `affine_weight`, `affine_bias`, `balance_degree_weight`, `expand_index`,
`eps` are read directly off that object, never re-derived) -- no new parameters, no retraining.
"""
from __future__ import annotations

import torch

from pipeline.intervention.esen_intervention import IRREP_SLICES, _ell_list


def compute_centered_and_rms(norm_module, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Steps 1-4 of EquivariantRMSNormArraySphericalHarmonicsV2.forward: l=0 centering, then
    the shared (all-degree, all-channel) inverse-RMS statistic. Returns (feature_centered
    [N,25,C], rms [N,1,1])."""
    feature = h
    if norm_module.centering:
        feature_l0 = feature.narrow(1, 0, 1)
        feature_l0_mean = feature_l0.mean(dim=2, keepdim=True)
        feature_l0 = feature_l0 - feature_l0_mean
        feature = torch.cat((feature_l0, feature.narrow(1, 1, feature.shape[1] - 1)), dim=1)
    if norm_module.normalization == "component" and norm_module.std_balance_degrees:
        feature_norm = feature.pow(2)
        feature_norm = torch.einsum("nic,ia->nac", feature_norm, norm_module.balance_degree_weight)
    else:
        raise NotImplementedError(
            "checkpoint's norm config (normalization=%r, std_balance_degrees=%r) differs from "
            "the audited default (component, True) -- exact_norm_audit.md does not cover this "
            "case; do not silently approximate." % (norm_module.normalization, norm_module.std_balance_degrees)
        )
    feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)
    rms = (feature_norm + norm_module.eps).pow(-0.5)
    return feature, rms


def apply_norm_with_stat(norm_module, feature_centered: torch.Tensor, rms: torch.Tensor) -> torch.Tensor:
    """Steps 5-7: learned affine gain/bias applied to `feature_centered`, scaled by an
    EXTERNALLY SUPPLIED `rms` (may or may not be the one computed from feature_centered itself)."""
    feature_norm = rms
    if norm_module.affine:
        weight = norm_module.affine_weight.view(1, norm_module.lmax + 1, norm_module.num_channels)
        weight = torch.index_select(weight, dim=1, index=norm_module.expand_index)
        feature_norm = feature_norm * weight
    out = feature_centered * feature_norm
    if norm_module.affine and norm_module.centering:
        out = out.clone()
        out[:, 0:1, :] = out.narrow(1, 0, 1) + norm_module.affine_bias.view(1, 1, norm_module.num_channels)
    return out


def norm_natural(norm_module, h: torch.Tensor) -> torch.Tensor:
    """NATURAL: norm_module's own, unmodified forward. Provided for identity-check parity;
    the fixed-normalization control's NATURAL cells should just call norm_module(h) directly."""
    return norm_module(h)


def norm_fixed_output(norm_module, h_ablated_prenorm: torch.Tensor, h_baseline_prenorm: torch.Tensor) -> torch.Tensor:
    """NORM_FIXED counterfactual (task Section 3): apply the trained norm's transformation to
    the ABLATED pre-norm hidden state, but substitute the BASELINE (fully unablated, same
    molecule/checkpoint/block) RMS statistic instead of recomputing it from the ablated state.
    h_tilde_normfixed = Norm(h_tilde; stats=s_base, trained_parameters unchanged)."""
    _, rms_base = compute_centered_and_rms(norm_module, h_baseline_prenorm)
    feature_ablated_centered, _ = compute_centered_and_rms(norm_module, h_ablated_prenorm)
    return apply_norm_with_stat(norm_module, feature_ablated_centered, rms_base)


def postnorm_mask(h_postnorm: torch.Tensor, ell) -> torch.Tensor:
    """POSTNORM negative control (task Section 4): zero degree(s) `ell` in an ALREADY-normalized
    node_embedding, strictly after self.norm and before the energy/force heads. Since the force
    head only ever reads node_embedding.narrow(1,0,4) (l<=1), this must leave forces unchanged
    for ell in {2,3,4} -- a pure code-path sanity check, not a scientific result."""
    out = h_postnorm.clone()
    for e in _ell_list(ell):
        s, en = IRREP_SLICES[e]
        out[:, s:en, :] = 0.0
    return out


def run_heads(net, data_dict, node_embedding: torch.Tensor) -> dict:
    """Re-run only the (unmodified, unretrained) energy_head/force_head given an already-computed
    post-norm node_embedding -- avoids re-running the 12-block backbone when only the final norm
    output changes (NORM_FIXED, POSTNORM)."""
    outputs = {"node_embedding": node_embedding, "batch": data_dict["batch"]}
    energy = net.energy_head(data_dict, outputs)["energy"]
    forces = net.force_head(data_dict, outputs)["forces"]
    return {"energy": energy, "forces": forces}
