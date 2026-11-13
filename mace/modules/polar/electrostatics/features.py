###########################################################################################
# Functions for computing global electrostatic features from atomic multipoles
# and gaussian type orbitals
# Authors: Will Baldwin
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Callable, List, Literal, Optional

import torch
from scipy.constants import pi

from .gto_utils import (
    GTOBasis,
    GTOSelfInteractionBlock,
)
from .realspace_electrostatics import (
    RealSpaceAnalyticalElectrostaticFeatures,
    RealSpaceFiniteDifferenceElectrostaticFeatures,
    RealSpaceGeometry,
    RealSpaceMethod,
    _match_tensor_placement,
    _upgrade_legacy_realspace,
)
from .utils import FIELD_CONSTANT

# realspace: open structures; pbc: fully periodic cells (reciprocal space); auto: one of
# the two per batch (a batch must be all open or all fully periodic).
FeaturePBCHandling = Literal["realspace", "pbc", "auto"]


def assemble_fourier_series_batch(
    source_feats: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    density_basis_fs: torch.Tensor,
    volume_per_k: torch.Tensor,
) -> torch.Tensor:
    """Assemble rho(k) for flattened batch.

    Args:
        source_feats: [n_nodes, m_dim]
        cosines: [n_k_total, n_nodes]
        sines: [n_k_total, n_nodes]
        density_basis_fs: [n_k_total, n_sigma(=1), m_dim, 2]
        volume_per_k: [n_k_total]
    Returns:
        density: [n_k_total, 2]
    """
    n_nodes = source_feats.size(0)
    n_sigma = density_basis_fs.size(1)
    m_dim = density_basis_fs.size(2)
    sm_dim = n_sigma * m_dim

    coeff_cos = torch.matmul(cosines, source_feats)
    coeff_sin = torch.matmul(sines, source_feats)

    density_basis_r = density_basis_fs[..., 0].reshape(density_basis_fs.size(0), sm_dim)
    density_basis_i = density_basis_fs[..., 1].reshape(density_basis_fs.size(0), sm_dim)

    rho_real = (density_basis_r * coeff_cos).sum(dim=-1) + (
        density_basis_i * coeff_sin
    ).sum(dim=-1)
    rho_imag = (density_basis_i * coeff_cos).sum(dim=-1) - (
        density_basis_r * coeff_sin
    ).sum(dim=-1)

    density = torch.stack([rho_real, rho_imag], dim=-1)
    density = (2 * pi) ** 3 * density / volume_per_k.unsqueeze(-1)
    return density


def compute_coulomb_factor(
    k_norm2: torch.Tensor,
    k0_mask: torch.Tensor,
) -> torch.Tensor:
    """Build the masked Coulomb factor 1/|k|^2 with the k=0 mode set to zero."""
    k0_mask_bool = k0_mask > 0.0
    safe_k_norm2 = torch.where(k0_mask_bool, torch.ones_like(k_norm2), k_norm2)
    return (~k0_mask_bool).to(k_norm2.dtype) / safe_k_norm2


def apply_coulomb_kernel_batch(
    density: torch.Tensor,
    k_factor_coulomb: torch.Tensor,
) -> torch.Tensor:
    """Apply Coulomb kernel in k-space for flattened batch."""
    factor = k_factor_coulomb.reshape(-1, *([1] * (density.dim() - 1)))
    potential = density * factor
    return potential * FIELD_CONSTANT


def project_to_features_batch(
    potential: torch.Tensor,
    feature_basis_fs: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
) -> torch.Tensor:
    """Project potential to local GTO features for flattened batch."""
    n_k = feature_basis_fs.size(0)
    n_sigma = feature_basis_fs.size(1)
    m_dim = feature_basis_fs.size(2)
    sm_dim = n_sigma * m_dim

    proj_basis_r = feature_basis_fs[..., 0].reshape(n_k, sm_dim)
    proj_basis_i = feature_basis_fs[..., 1].reshape(n_k, sm_dim)

    A = potential[:, 0].unsqueeze(-1) * proj_basis_r + (
        potential[:, 1].unsqueeze(-1) * proj_basis_i
    )
    B = potential[:, 0].unsqueeze(-1) * proj_basis_i - (
        potential[:, 1].unsqueeze(-1) * proj_basis_r
    )

    A = A * k_factor_proj.unsqueeze(-1)
    B = B * k_factor_proj.unsqueeze(-1)

    proj_cos = torch.matmul(A.t(), cosines)
    proj_sin = torch.matmul(B.t(), sines)
    proj_total = 2.0 * (proj_cos + proj_sin)
    projections = proj_total.t().reshape(cosines.size(1), n_sigma, m_dim)
    return projections / (2 * pi) ** 3


def reconstruct_esps_batch(
    potential: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k0_mask: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct electrostatic potential at node positions."""
    summand = (
        2.0 * potential[:, 0].unsqueeze(-1) * cosines
        - 2.0 * potential[:, 1].unsqueeze(-1) * sines
    )
    k0_factor = torch.ones_like(k0_mask)
    k0_factor[k0_mask > 0.0] = 0.5
    summand = summand * k0_factor.unsqueeze(-1)
    return torch.sum(summand, dim=0) / (2 * pi) ** 3


class GTOElectrostaticFeatures(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        feature_max_l: int,
        feature_smearing_widths: List[float],
        include_self_interaction: bool,
        kspace_cutoff: float,
        integral_normalization: str = "receiver",
        pbc_handling: FeaturePBCHandling = "auto",
        realspace_method: RealSpaceMethod = "analytical",
    ):
        super().__init__()
        self.density_basis = GTOBasis(
            max_l=density_max_l,
            sigmas=[density_smearing_width],
            kspace_cutoff=kspace_cutoff,
            normalize="multipoles",
        )
        self.feature_basis = GTOBasis(
            max_l=feature_max_l,
            sigmas=feature_smearing_widths,
            kspace_cutoff=kspace_cutoff,
            normalize=integral_normalization,
        )
        self.kspace_cutoff = kspace_cutoff
        self.include_self_interaction = include_self_interaction
        self.pbc_handling = pbc_handling
        self.realspace_method = realspace_method

        self.self_interaction_terms = GTOSelfInteractionBlock(
            l_source=density_max_l,
            sigma_source=density_smearing_width,
            l_receive=feature_max_l,
            sigmas_receive=feature_smearing_widths,
            normalize_source="multipoles",
            normalize_receive=integral_normalization,
        )
        self.realspace_features = self._build_realspace_features()
        self.register_buffer(
            "output_permutation",
            self._build_output_permutation(
                max_l=feature_max_l,
                n_radial=len(feature_smearing_widths),
            ),
        )

        self.static_quantities = None
        self._precompute_geometry_impl = self._select_precompute_geometry_impl()
        self._forward_dynamic_impl = self._select_forward_dynamic_impl()

    def _build_realspace_features(self, method=None, *, dtype=None):
        # Read configuration from the bases, including on old whole-model pickles.
        if method is None:
            method = getattr(self, "realspace_method", "finite_difference")
        kwargs = dict(
            density_max_l=self.density_basis.max_l,
            density_smearing_width=self.density_basis.sigmas[0],
            projection_max_l=self.feature_basis.max_l,
            projection_smearing_widths=self.feature_basis.sigmas,
            include_self_interaction=self.include_self_interaction,
            integral_normalization=self.feature_basis.normalize,
            dtype=dtype,
        )
        if method == "finite_difference":
            return RealSpaceFiniteDifferenceElectrostaticFeatures(**kwargs)
        if method == "analytical":
            return RealSpaceAnalyticalElectrostaticFeatures(**kwargs)
        raise ValueError(f"Unsupported realspace_method: {method}")

    @staticmethod
    def _build_output_permutation(max_l: int, n_radial: int) -> torch.Tensor:
        indices = []
        block = (max_l + 1) ** 2
        for l in range(max_l + 1):
            for c in range(n_radial):
                offset = c * block
                indices += range(l**2 + offset, (l + 1) ** 2 + offset)
        return torch.tensor(indices, dtype=torch.long)

    def _permute_output_channels(self, features_flat: torch.Tensor) -> torch.Tensor:
        return torch.index_select(features_flat, dim=-1, index=self.output_permutation)

    def _select_precompute_geometry_impl(self) -> Callable:
        if self.pbc_handling == "realspace":
            return self._precompute_geometry_realspace
        if self.pbc_handling == "pbc":
            return self._precompute_geometry_pbc
        if self.pbc_handling == "auto":
            return self._precompute_geometry_auto
        raise ValueError(f"Unsupported pbc_handling: {self.pbc_handling}")

    def _select_forward_dynamic_impl(self) -> Callable:
        if self.pbc_handling == "realspace":
            return self._forward_dynamic_realspace
        if self.pbc_handling == "pbc":
            return self._forward_dynamic_pbc
        if self.pbc_handling == "auto":
            return self._forward_dynamic_auto
        raise ValueError(f"Unsupported pbc_handling: {self.pbc_handling}")

    def set_pbc_handling(self, pbc_handling: FeaturePBCHandling) -> None:
        self.pbc_handling = pbc_handling
        self._precompute_geometry_impl = self._select_precompute_geometry_impl()
        self._forward_dynamic_impl = self._select_forward_dynamic_impl()

    def __setstate__(self, state):
        super().__setstate__(state)
        _upgrade_legacy_realspace(self, state)

    def set_realspace_method(self, realspace_method: RealSpaceMethod) -> None:
        """Swap the non-periodic evaluator on an already-built block.

        The analytical evaluator is exactly rotationally equivariant; the
        finite-difference one (graph_longrange <= v0.4.4) is orientation dependent.
        Analytical is the default; legacy pickles are upgraded on load.
        """
        old = self.realspace_features
        reference = next(
            buffer for buffer in old.buffers() if buffer.is_floating_point()
        )
        replacement = self._build_realspace_features(
            realspace_method, dtype=reference.dtype
        )
        replacement = _match_tensor_placement(replacement, old)
        replacement.train(old.training)
        # Leave the current evaluator and flag unchanged if construction fails.
        self.realspace_features = replacement
        self.realspace_method = realspace_method

    def forward(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> torch.Tensor:
        cache = self._precompute_geometry_impl(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )
        return self._forward_dynamic_impl(
            source_feats=source_feats,
            cache=cache,
        )

    def precompute_geometry(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
        realspace_geometry: Optional[RealSpaceGeometry] = None,
    ) -> dict:
        """Everything the features need from the geometry, returned as a value.

        In real space this holds the pair kernels (RealSpaceGeometry), formed once and
        reused by every forward_dynamic call on the returned value. A caller that
        shares one real-space geometry between several modules (the features and the
        energy) prepares it itself, with the merged kernel widths of both, and passes
        it as realspace_geometry.
        """
        cache = self._precompute_geometry_impl(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )
        if cache["mode"] == "realspace" and self._has_prepared_realspace():
            cache["realspace_geometry"] = (
                realspace_geometry
                if realspace_geometry is not None
                else self.realspace_features.prepare(node_positions, batch)
            )
        return cache

    def _has_prepared_realspace(self) -> bool:
        """Whether the real-space evaluator takes a prepared geometry (analytical)."""
        return isinstance(
            self.realspace_features, RealSpaceAnalyticalElectrostaticFeatures
        )

    def forward_dynamic(
        self,
        cache: dict,
        source_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Features of one source set [n_nodes, C], or of several independent sets
        [n_nodes, n_sets, C] (the two spin channels) -> [n_nodes, n_sets, F]."""
        if source_feats.dim() == 3 and cache.get("realspace_geometry") is None:
            return torch.stack(
                [
                    self._forward_dynamic_impl(
                        source_feats=source_feats[:, index], cache=cache
                    )
                    for index in range(source_feats.shape[1])
                ],
                dim=1,
            )
        return self._forward_dynamic_impl(
            source_feats=source_feats,
            cache=cache,
        )

    def _precompute_geometry_realspace(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> dict:
        return {
            "mode": "realspace",
            "node_positions": node_positions,
            "batch": batch,
        }

    def _precompute_geometry_periodic_common(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> dict:
        inner_products = torch.matmul(
            k_vectors, node_positions.t()
        )  # [n_k_total, n_nodes]
        mask = k_vector_batch[:, None] == batch[None, :]
        mask_f = mask.to(dtype=inner_products.dtype)
        cosines = torch.cos(inner_products) * mask_f
        sines = torch.sin(inner_products) * mask_f

        density_basis_fs = self.density_basis(k_vectors, k_norm2, k0_mask)
        feature_basis_fs = self.feature_basis(k_vectors, k_norm2, k0_mask)

        volume_per_k = volume.reshape(-1)[k_vector_batch]
        k0_mask_bool = k0_mask > 0.0
        k_factor_coulomb = compute_coulomb_factor(k_norm2, k0_mask)
        k_factor_proj = torch.ones_like(k_norm2)
        k_factor_proj[k0_mask_bool] = 0.5

        return {
            "mode": "pbc",
            "k_norm2": k_norm2,
            "k_vector_batch": k_vector_batch,
            "k0_mask": k0_mask,
            "volume_per_k": volume_per_k,
            "k_factor_coulomb": k_factor_coulomb,
            "k_factor_proj": k_factor_proj,
            "volumes": volume.reshape(-1),
            "batch": batch,
            "node_positions": node_positions,
            "pbc": pbc,
            "cosines": cosines,
            "sines": sines,
            "density_basis_fs": density_basis_fs,
            "feature_basis_fs": feature_basis_fs,
        }

    def _precompute_geometry_pbc(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> dict:
        return self._precompute_geometry_periodic_common(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )

    def _precompute_geometry_auto(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> dict:
        if torch.any(pbc):
            if not bool(torch.all(pbc)):
                raise ValueError(
                    "pbc_handling 'auto' needs a batch of open or of fully periodic "
                    "structures (slabs and mixed batches are not supported)"
                )
            return self._precompute_geometry_pbc(
                k_vectors=k_vectors,
                k_norm2=k_norm2,
                k_vector_batch=k_vector_batch,
                k0_mask=k0_mask,
                node_positions=node_positions,
                batch=batch,
                volume=volume,
                pbc=pbc,
            )
        return self._precompute_geometry_realspace(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )

    @staticmethod
    def _require_cache_mode(cache: dict, expected_mode: str) -> None:
        cache_mode = cache.get("mode")
        if cache_mode != expected_mode:
            raise ValueError(f"Expected a {expected_mode!r} cache, got {cache_mode!r}.")

    def compute_esps(
        self, cache: dict, source_feats: torch.Tensor, pbc: torch.Tensor
    ) -> torch.Tensor:
        # if cache.get("mode") == "realspace":
        #    raise NotImplementedError("Real-space ESP evaluation is not implemented yet.")
        density = assemble_fourier_series_batch(
            source_feats=source_feats,
            cosines=cache["cosines"],
            sines=cache["sines"],
            density_basis_fs=cache["density_basis_fs"],
            volume_per_k=cache["volume_per_k"],
        )
        potential = apply_coulomb_kernel_batch(
            density=density,
            k_factor_coulomb=cache["k_factor_coulomb"],
        )
        esps = reconstruct_esps_batch(
            potential=potential,
            cosines=cache["cosines"],
            sines=cache["sines"],
            k0_mask=cache["k0_mask"],
        )
        return esps

    def _forward_dynamic_periodic_common(
        self, source_feats: torch.Tensor, cache: dict
    ) -> torch.Tensor:
        self._require_cache_mode(cache, "pbc")
        density = assemble_fourier_series_batch(
            source_feats=source_feats,
            cosines=cache["cosines"],
            sines=cache["sines"],
            density_basis_fs=cache["density_basis_fs"],
            volume_per_k=cache["volume_per_k"],
        )
        potential = apply_coulomb_kernel_batch(
            density=density,
            k_factor_coulomb=cache["k_factor_coulomb"],
        )
        features_si = project_to_features_batch(
            potential=potential,
            feature_basis_fs=cache["feature_basis_fs"],
            cosines=cache["cosines"],
            sines=cache["sines"],
            k_factor_proj=cache["k_factor_proj"],
        )
        features_flat = features_si.reshape(features_si.size(0), -1)
        features_flat = self._permute_output_channels(features_flat)
        if not self.include_self_interaction:
            si_terms = self.self_interaction_terms(source_feats)
            features_flat = features_flat - si_terms
        return features_flat

    def _forward_dynamic_realspace(
        self, source_feats: torch.Tensor, cache: dict
    ) -> torch.Tensor:
        self._require_cache_mode(cache, "realspace")
        geometry = cache.get("realspace_geometry")
        if geometry is not None:
            features, _ = self.realspace_features.evaluate(source_feats, geometry)
            return features
        features, _, _ = self.realspace_features(
            source_feats=source_feats,
            node_positions=cache["node_positions"],
            batch=cache["batch"],
        )
        return features

    def _forward_dynamic_pbc(
        self, source_feats: torch.Tensor, cache: dict
    ) -> torch.Tensor:
        return self._forward_dynamic_periodic_common(
            source_feats=source_feats, cache=cache
        )

    def _forward_dynamic_auto(
        self, source_feats: torch.Tensor, cache: dict
    ) -> torch.Tensor:
        cache_mode = cache.get("mode")
        if cache_mode == "realspace":
            return self._forward_dynamic_realspace(
                source_feats=source_feats,
                cache=cache,
            )
        if cache_mode == "pbc":
            return self._forward_dynamic_pbc(
                source_feats=source_feats,
                cache=cache,
            )
        raise ValueError(f"Unsupported cache mode for auto features: {cache_mode!r}.")
