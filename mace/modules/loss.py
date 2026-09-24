###########################################################################################
# Implementation of different loss functions
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist

from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch


# ------------------------------------------------------------------------------
# Helper function for loss reduction that handles DDP correction
# ------------------------------------------------------------------------------
def is_ddp_enabled():
    return dist.is_initialized() and dist.get_world_size() > 1


def reduce_loss(raw_loss: torch.Tensor, ddp: Optional[bool] = None) -> torch.Tensor:
    """
    Reduces an element-wise loss tensor.

    If ddp is True and distributed is initialized, the function computes:

        loss = (local_sum * world_size) / global_num_elements

    Otherwise, it returns the regular mean.
    """
    ddp = is_ddp_enabled() if ddp is None else ddp
    if ddp and dist.is_initialized():
        world_size = dist.get_world_size()
        n_local = raw_loss.numel()
        loss_sum = raw_loss.sum()
        total_samples = torch.tensor(
            n_local, device=raw_loss.device, dtype=raw_loss.dtype
        )
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        return loss_sum * world_size / total_samples
    return raw_loss.mean()


# ------------------------------------------------------------------------------
# Energy Loss Functions
# ------------------------------------------------------------------------------


def mean_squared_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    raw_loss = torch.square(ref["energy"] - pred["energy"])
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Calculate per-graph number of atoms.
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # shape: [n_graphs]
    raw_loss = (
        ref.weight
        * ref.energy_weight
        * torch.square((ref["energy"] - pred["energy"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


def weighted_mean_absolute_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    raw_loss = (
        ref.weight
        * ref.energy_weight
        * torch.abs((ref["energy"] - pred["energy"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Stress and Virials Loss Functions
# ------------------------------------------------------------------------------


def weighted_mean_squared_stress(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
    raw_loss = (
        configs_weight
        * configs_stress_weight
        * torch.square(ref["stress"] - pred["stress"])
    )
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_virials(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_virials_weight = ref.virials_weight.view(-1, 1, 1)
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)
    raw_loss = (
        configs_weight
        * configs_virials_weight
        * torch.square((ref["virials"] - pred["virials"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Forces Loss Functions
# ------------------------------------------------------------------------------


def mean_squared_error_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Repeat per-graph weights to per-atom level.
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    raw_loss = (
        configs_weight
        * configs_forces_weight
        * torch.square(ref["forces"] - pred["forces"])
    )
    return reduce_loss(raw_loss, ddp)


def mean_normed_error_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    raw_loss = torch.linalg.vector_norm(ref["forces"] - pred["forces"], ord=2, dim=-1)
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Dipole Loss Function
# ------------------------------------------------------------------------------


def weighted_mean_squared_error_dipole(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
    raw_loss = torch.square((ref["dipole"] - pred["dipole"]) / num_atoms)
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Polarizability Loss Function
# ------------------------------------------------------------------------------


def weighted_mean_squared_error_polarizability(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[
        bool
    ] = None,  # ,mean: Optional[torch.Tensor] = None , std: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # polarizability: [n_graphs, ]
    # ref_polar = ref["polarizability"].view(-1, 3, 3) * std.view(1, 3, 3) + mean.view(1, 3, 3) if mean is not None and std is not None else ref["polarizability"]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)  # [n_graphs,1]
    raw_loss = torch.square(
        (ref["polarizability"].view(-1, 3, 3) - pred["polarizability"]) / num_atoms
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Conditional Losses for Forces
# ------------------------------------------------------------------------------


def conditional_mse_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    # Define multiplication factors for different regimes.
    factors = torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=ref["forces"].device, dtype=ref["forces"].dtype
    )
    err = ref["forces"] - pred["forces"]
    se = torch.zeros_like(err)
    norm_forces = torch.norm(ref["forces"], dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    se[c1] = torch.square(err[c1]) * factors[0]
    se[c2] = torch.square(err[c2]) * factors[1]
    se[c3] = torch.square(err[c3]) * factors[2]
    se[~(c1 | c2 | c3)] = torch.square(err[~(c1 | c2 | c3)]) * factors[3]
    raw_loss = configs_weight * configs_forces_weight * se
    return reduce_loss(raw_loss, ddp)


FORCE_MAGNITUDE_BIN_EDGES = (100.0, 200.0, 300.0)
FORCE_MAGNITUDE_HUBER_FACTORS = (1.0, 0.7, 0.4, 0.1)


def conditional_huber_force_thresholds(
    ref_forces: torch.Tensor, huber_delta: float
) -> torch.Tensor:
    """Per-atom Huber threshold ``[n_atoms]`` of :func:`conditional_huber_forces`: the delta
    shrinks by the factors 1, 0.7, 0.4, 0.1 for reference force norms below 100, 200, 300
    and above."""
    norm_forces = torch.norm(ref_forces, dim=-1)
    factors = huber_delta * torch.tensor(
        FORCE_MAGNITUDE_HUBER_FACTORS, device=ref_forces.device, dtype=ref_forces.dtype
    )
    edges = torch.tensor(
        FORCE_MAGNITUDE_BIN_EDGES, device=ref_forces.device, dtype=ref_forces.dtype
    )
    return factors[torch.bucketize(norm_forces, edges, right=True)]


def huber_elementwise(
    ref_values: torch.Tensor, pred_values: torch.Tensor, thresholds: torch.Tensor
) -> torch.Tensor:
    """``torch.nn.functional.huber_loss(reduction="none")`` with a threshold tensor that
    broadcasts against the values: ``0.5 x^2`` for ``|x| < delta``, else
    ``delta (|x| - 0.5 delta)``."""
    absolute_errors = (pred_values - ref_values).abs()
    return torch.where(
        absolute_errors < thresholds,
        0.5 * absolute_errors.square(),
        thresholds * (absolute_errors - 0.5 * thresholds),
    )


def conditional_huber_forces_unreduced(
    ref_forces: torch.Tensor,
    pred_forces: torch.Tensor,
    huber_delta: float,
) -> torch.Tensor:
    """Element-wise Huber loss ``[n_atoms, 3]`` with the magnitude-dependent thresholds of
    :func:`conditional_huber_force_thresholds`."""
    thresholds = conditional_huber_force_thresholds(ref_forces, huber_delta)
    return huber_elementwise(ref_forces, pred_forces, thresholds.unsqueeze(-1))


def conditional_huber_forces(
    ref_forces: torch.Tensor,
    pred_forces: torch.Tensor,
    huber_delta: float,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    se = conditional_huber_forces_unreduced(ref_forces, pred_forces, huber_delta)
    return reduce_loss(se, ddp)


# ------------------------------------------------------------------------------
# Loss Modules Combining Multiple Quantities
# ------------------------------------------------------------------------------


class WeightedEnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        return self.energy_weight * loss_energy + self.forces_weight * loss_forces

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )


class WeightedForcesLoss(torch.nn.Module):
    def __init__(self, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        return self.forces_weight * loss_forces

    def __repr__(self):
        return f"{self.__class__.__name__}(forces_weight={self.forces_weight:.3f})"


class WeightedEnergyForcesStressLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_stress = weighted_mean_squared_stress(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedHuberEnergyForcesStressLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    ) -> None:
        super().__init__()
        # We store the huber_delta rather than a loss with fixed reduction.
        self.huber_delta = huber_delta
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        if ddp:
            loss_energy = torch.nn.functional.huber_loss(
                ref["energy"] / num_atoms,
                pred["energy"] / num_atoms,
                reduction="none",
                delta=self.huber_delta,
            )
            loss_energy = reduce_loss(loss_energy, ddp)
            loss_forces = torch.nn.functional.huber_loss(
                ref["forces"], pred["forces"], reduction="none", delta=self.huber_delta
            )
            loss_forces = reduce_loss(loss_forces, ddp)
            loss_stress = torch.nn.functional.huber_loss(
                ref["stress"], pred["stress"], reduction="none", delta=self.huber_delta
            )
            loss_stress = reduce_loss(loss_stress, ddp)
        else:
            loss_energy = torch.nn.functional.huber_loss(
                ref["energy"] / num_atoms,
                pred["energy"] / num_atoms,
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_forces = torch.nn.functional.huber_loss(
                ref["forces"], pred["forces"], reduction="mean", delta=self.huber_delta
            )
            loss_stress = torch.nn.functional.huber_loss(
                ref["stress"], pred["stress"], reduction="mean", delta=self.huber_delta
            )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class UniversalLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        stress_weight=1.0,
        magforces_weight=1.0,
        huber_delta=0.01,
    ) -> None:
        super().__init__()
        self.huber_delta = huber_delta
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "magforces_weight",
            torch.tensor(magforces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
        configs_energy_weight = ref.energy_weight
        configs_forces_weight = torch.repeat_interleave(
            ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
        ).unsqueeze(-1)
        configs_magforces_weight = torch.repeat_interleave(
            ref.magforces_weight, ref.ptr[1:] - ref.ptr[:-1]
        ).unsqueeze(-1)
        if ddp:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="none",
                delta=self.huber_delta,
            )
            loss_energy = reduce_loss(loss_energy, ddp)
            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )
            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="none",
                delta=self.huber_delta,
            )
            loss_stress = reduce_loss(loss_stress, ddp)
            loss_magforces = 0
            if "magforces" in pred.keys() and (
                pred["magforces"] is not None and ref["magforces"] is not None
            ):
                loss_magforces = torch.nn.functional.huber_loss(
                    configs_magforces_weight * ref["magforces"],
                    configs_magforces_weight * pred["magforces"],
                    reduction="none",
                    delta=self.huber_delta,
                )
                loss_magforces = reduce_loss(loss_magforces, ddp)
        else:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )
            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_magforces = 0
            if "magforces" in pred.keys() and (
                pred["magforces"] is not None and ref["magforces"] is not None
            ):
                loss_magforces = torch.nn.functional.huber_loss(
                    configs_magforces_weight * ref["magforces"],
                    configs_magforces_weight * pred["magforces"],
                    reduction="mean",
                    delta=self.huber_delta,
                )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
            + self.magforces_weight * loss_magforces
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f}, magforces_weight={self.magforces_weight:.3f})"
        )


class WeightedEnergyForcesVirialsLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, virials_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "virials_weight",
            torch.tensor(virials_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_virials = weighted_mean_squared_virials(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.virials_weight * loss_virials
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, virials_weight={self.virials_weight:.3f})"
        )


class DipoleSingleLoss(torch.nn.Module):
    def __init__(self, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss = (
            weighted_mean_squared_error_dipole(ref, pred, ddp) * 100.0
        )  # scale adjustment
        return self.dipole_weight * loss

    def __repr__(self):
        return f"{self.__class__.__name__}(dipole_weight={self.dipole_weight:.3f})"


class DipolePolarLoss(torch.nn.Module):
    def __init__(
        self, dipole_weight=1.0, polarizability_weight=1.0
    ) -> (
        None
    ):  # dipole_mean=None,dipole_std=None,polarizability_mean=None,polarizability_std=None
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarizability_weight",
            torch.tensor(polarizability_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_dipole = weighted_mean_squared_error_dipole(
            ref, pred, ddp
        )  # ,self.dipole_mean,self.dipole_std) #* 100.0  # scale adjustment

        loss_polarizability = weighted_mean_squared_error_polarizability(
            ref, pred, ddp
        )  # ,self.polarizability_mean,self.polarizability_std) #* 100.0  # scale adjustment
        return (
            self.dipole_weight * loss_dipole
            + self.polarizability_weight * loss_polarizability
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dipole_weight={self.dipole_weight:.3f}, polarizability_weight={self.polarizability_weight:.3f})"
        )


class WeightedEnergyForcesDipoleLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_dipole = weighted_mean_squared_error_dipole(ref, pred, ddp) * 100.0
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.dipole_weight * loss_dipole
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.dipole_weight:.3f})"
        )


class WeightedEnergyForcesL1L2Loss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_absolute_error_energy(ref, pred, ddp)
        loss_forces = mean_normed_error_forces(ref, pred, ddp)
        return self.energy_weight * loss_energy + self.forces_weight * loss_forces

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )


class InteractionUniversalLoss(UniversalLoss):
    """``UniversalLoss`` plus Huber terms on the interaction energy and forces of cluster
    records (``mace.data.cluster_records``): ``E_int = sum(sign * E)`` per record, per
    cluster atom, and ``F_int = sum(sign_node * F)`` per cluster atom (slot). Both are
    formed by the same signed scatter on reference and prediction, so the absolute-label
    terms of the parent class and the interaction terms see the same batch. Requires
    batches from ``ClusterCollater`` (``--cluster_records``)."""

    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        stress_weight=1.0,
        magforces_weight=1.0,
        huber_delta=0.01,
        interaction_energy_weight=1.0,
        interaction_forces_weight=1.0,
    ) -> None:
        super().__init__(
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            stress_weight=stress_weight,
            magforces_weight=magforces_weight,
            huber_delta=huber_delta,
        )
        self.register_buffer(
            "interaction_energy_weight",
            torch.tensor(interaction_energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "interaction_forces_weight",
            torch.tensor(interaction_forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        from mace.tools.torch_geometric.cluster_collate import (  # pylint: disable=import-outside-toplevel
            cluster_atom_counts,
            combine_energy,
            combine_forces,
            is_cluster_batch,
            slot_to_cluster,
        )

        if not is_cluster_batch(ref):
            raise ValueError(
                "InteractionUniversalLoss needs cluster-record batches "
                "(train with --cluster_records on a cluster HDF5 dataset)"
            )
        if ddp:
            raise NotImplementedError("InteractionUniversalLoss does not support ddp")
        loss = super().forward(ref, pred, ddp)

        record_weight = ref["cluster_interaction_weight"]  # [n_clusters]
        atom_counts = cluster_atom_counts(ref)  # [n_clusters]
        energy_scale = record_weight / atom_counts
        loss_interaction_energy = torch.nn.functional.huber_loss(
            energy_scale * combine_energy(ref, ref["energy"]),
            energy_scale * combine_energy(ref, pred["energy"]),
            reduction="mean",
            delta=self.huber_delta,
        )
        slot_weight = record_weight[slot_to_cluster(ref)].unsqueeze(-1)  # [n_slots, 1]
        loss_interaction_forces = torch.nn.functional.huber_loss(
            slot_weight * combine_forces(ref, ref["forces"]),
            slot_weight * combine_forces(ref, pred["forces"]),
            reduction="mean",
            delta=self.huber_delta,
        )
        return (
            loss
            + self.interaction_energy_weight * loss_interaction_energy
            + self.interaction_forces_weight * loss_interaction_forces
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f}, "
            f"magforces_weight={self.magforces_weight:.3f}, "
            f"interaction_energy_weight={self.interaction_energy_weight:.3f}, "
            f"interaction_forces_weight={self.interaction_forces_weight:.3f})"
        )


@dataclass
class LossTermStatistics:
    """Components of one :class:`InteractionHuberLoss` term from its last forward pass.

    All tensors are detached scalars on the batch device. ``entry_count`` and
    ``linear_count`` are floats so they can be accumulated alongside the sums."""

    weight: torch.Tensor  # the global term weight used in this forward pass
    unweighted_mean: torch.Tensor  # sum(w_i * loss_i) / sum(w_i), 0 without entries
    weighted_sum: torch.Tensor  # sum(w_i * loss_i)
    weight_sum: torch.Tensor  # sum(w_i) over entries with w_i > 0
    entry_count: torch.Tensor  # number of entries with w_i > 0
    linear_count: torch.Tensor  # those entries whose |error| exceeds their Huber threshold


def weighted_huber_term(
    ref_values: torch.Tensor,
    pred_values: torch.Tensor,
    entry_weights: torch.Tensor,
    thresholds: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted mean of element-wise Huber losses over the entries with a positive weight.

    ``entry_weights`` and ``thresholds`` broadcast against the values; the weights sit
    outside the Huber function. Entries with weight <= 0 neither contribute nor dilute.
    Returns ``(mean, weighted_sum, weight_sum, entry_count, linear_count)``; the mean keeps
    the autograd graph (it is exactly zero, and still connected, without entries), the
    others are detached."""
    absolute_errors = (pred_values - ref_values).abs()
    per_entry_loss = huber_elementwise(ref_values, pred_values, thresholds)
    entry_weights = entry_weights.expand_as(per_entry_loss)
    included = entry_weights > 0
    weighted_sum = torch.where(
        included, entry_weights * per_entry_loss, torch.zeros_like(per_entry_loss)
    ).sum()
    weight_sum = torch.where(
        included, entry_weights, torch.zeros_like(entry_weights)
    ).sum()
    safe_weight_sum = torch.where(
        weight_sum > 0, weight_sum, torch.ones_like(weight_sum)
    )
    mean = weighted_sum / safe_weight_sum
    entry_count = included.sum().to(per_entry_loss.dtype)
    linear_count = (
        (included & (absolute_errors > thresholds)).sum().to(per_entry_loss.dtype)
    )
    return (
        mean,
        weighted_sum.detach(),
        weight_sum.detach(),
        entry_count.detach(),
        linear_count.detach(),
    )


class InteractionHuberLoss(torch.nn.Module):
    """Six Huber terms, each a weighted mean over the entries that carry it.

    ================== =============================================== =====================
    term               entries                                         entry weight
    ================== =============================================== =====================
    frame_energy       periodic configurations with sign +1 (per atom) weight * energy_weight
    frame_forces       force components of their atoms                 weight * forces_weight
    monomer_energy     aperiodic configurations with sign +1           weight * energy_weight
    monomer_forces     force components of their atoms                 weight * forces_weight
    interaction_energy records, ``E_int / cluster atoms``              interaction weight
    interaction_forces slot components of ``F_int``                    interaction weight
    ================== =============================================== =====================

    A term is ``sum(w_i * huber_i) / sum(w_i)`` over its entries with ``w_i > 0``, so
    zero-weight entries neither contribute nor dilute; the global term weight multiplies it.
    In-place monomers (sign -1) get no absolute terms: they enter only through ``E_int`` and
    ``F_int`` (``mace.tools.torch_geometric.cluster_collate``). Batches without ``sign``
    (extxyz) count every configuration as +1 and have empty interaction terms. The force
    terms of frames and monomers use the magnitude-dependent thresholds of
    :func:`conditional_huber_forces`; ``F_int`` uses a plain threshold. There is no stress
    term: neither ``ref["stress"]`` nor ``pred["stress"]`` is read.

    Every forward pass stores the detached components of each term in ``last_statistics``.
    """

    term_names: Tuple[str, ...] = (
        "frame_energy",
        "frame_forces",
        "monomer_energy",
        "monomer_forces",
        "interaction_energy",
        "interaction_forces",
    )
    # buffer holding the global weight of each term
    _weight_buffer_names: Dict[str, str] = {
        "frame_energy": "energy_weight",
        "frame_forces": "forces_weight",
        "monomer_energy": "monomer_energy_weight",
        "monomer_forces": "monomer_forces_weight",
        "interaction_energy": "interaction_energy_weight",
        "interaction_forces": "interaction_forces_weight",
    }

    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        monomer_energy_weight=1.0,
        monomer_forces_weight=1.0,
        interaction_energy_weight=1.0,
        interaction_forces_weight=1.0,
        huber_delta_frame_energy=0.01,
        huber_delta_frame_forces=0.01,
        huber_delta_monomer_energy=0.01,
        huber_delta_monomer_forces=0.01,
        huber_delta_interaction_energy=0.01,
        huber_delta_interaction_forces=0.01,
    ) -> None:
        super().__init__()
        weights = {
            "energy_weight": energy_weight,
            "forces_weight": forces_weight,
            "monomer_energy_weight": monomer_energy_weight,
            "monomer_forces_weight": monomer_forces_weight,
            "interaction_energy_weight": interaction_energy_weight,
            "interaction_forces_weight": interaction_forces_weight,
        }
        for buffer_name, value in weights.items():
            self.register_buffer(
                buffer_name, torch.tensor(value, dtype=torch.get_default_dtype())
            )
        self.huber_deltas: Dict[str, float] = {
            "frame_energy": float(huber_delta_frame_energy),
            "frame_forces": float(huber_delta_frame_forces),
            "monomer_energy": float(huber_delta_monomer_energy),
            "monomer_forces": float(huber_delta_monomer_forces),
            "interaction_energy": float(huber_delta_interaction_energy),
            "interaction_forces": float(huber_delta_interaction_forces),
        }
        self.last_statistics: Dict[str, LossTermStatistics] = {}

    def term_weights(self) -> Dict[str, float]:
        return {
            name: float(getattr(self, buffer_name))
            for name, buffer_name in self._weight_buffer_names.items()
        }

    def _absolute_terms(
        self, ref: Batch, pred: TensorDict, configuration_mask: torch.Tensor, kind: str
    ):
        """Energy and force terms of the configurations in ``configuration_mask``."""
        num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).to(pred["energy"].dtype)
        selected = configuration_mask.to(pred["energy"].dtype)
        energy_entry_weights = ref.weight * ref.energy_weight * selected
        energy_term = weighted_huber_term(
            ref["energy"] / num_atoms,
            pred["energy"] / num_atoms,
            energy_entry_weights,
            torch.full_like(pred["energy"], self.huber_deltas[f"{kind}_energy"]),
        )
        atom_entry_weights = (ref.weight * ref.forces_weight * selected)[ref.batch]
        forces_term = weighted_huber_term(
            ref["forces"],
            pred["forces"],
            atom_entry_weights.unsqueeze(-1),
            conditional_huber_force_thresholds(
                ref["forces"], self.huber_deltas[f"{kind}_forces"]
            ).unsqueeze(-1),
        )
        return energy_term, forces_term

    def _interaction_terms(self, ref: Batch, pred: TensorDict):
        from mace.tools.torch_geometric.cluster_collate import (  # pylint: disable=import-outside-toplevel
            cluster_atom_counts,
            combine_energy,
            combine_forces,
            is_cluster_batch,
            slot_to_cluster,
        )

        if not is_cluster_batch(ref):
            # empty slices keep the terms connected to the prediction's graph
            empty_energy = pred["energy"][:0]
            empty_forces = pred["forces"][:0]
            energy_term = weighted_huber_term(
                empty_energy.detach(), empty_energy, empty_energy.detach(), empty_energy.detach()
            )
            forces_term = weighted_huber_term(
                empty_forces.detach(), empty_forces, empty_forces.detach(), empty_forces.detach()
            )
            return energy_term, forces_term

        record_weights = ref["cluster_interaction_weight"]  # [n_clusters]
        atom_counts = cluster_atom_counts(ref).to(pred["energy"].dtype)  # [n_clusters]
        energy_term = weighted_huber_term(
            combine_energy(ref, ref["energy"]) / atom_counts,
            combine_energy(ref, pred["energy"]) / atom_counts,
            record_weights,
            torch.full_like(record_weights, self.huber_deltas["interaction_energy"]),
        )
        slot_weights = record_weights[slot_to_cluster(ref)].unsqueeze(-1)  # [n_slots, 1]
        forces_term = weighted_huber_term(
            combine_forces(ref, ref["forces"]),
            combine_forces(ref, pred["forces"]),
            slot_weights,
            torch.full_like(slot_weights, self.huber_deltas["interaction_forces"]),
        )
        return energy_term, forces_term

    def compute_terms(self, ref: Batch, pred: TensorDict) -> Dict[str, torch.Tensor]:
        """The six weighted terms (with autograd graph); fills ``last_statistics``."""
        periodic = ref.pbc.any(dim=-1)  # [n_graphs]
        sign = getattr(ref, "sign", None)
        positive = (
            sign > 0 if sign is not None else torch.ones_like(periodic, dtype=torch.bool)
        )
        frame_energy, frame_forces = self._absolute_terms(
            ref, pred, periodic & positive, "frame"
        )
        monomer_energy, monomer_forces = self._absolute_terms(
            ref, pred, ~periodic & positive, "monomer"
        )
        interaction_energy, interaction_forces = self._interaction_terms(ref, pred)
        components = dict(
            zip(
                self.term_names,
                (
                    frame_energy,
                    frame_forces,
                    monomer_energy,
                    monomer_forces,
                    interaction_energy,
                    interaction_forces,
                ),
            )
        )

        terms: Dict[str, torch.Tensor] = {}
        statistics: Dict[str, LossTermStatistics] = {}
        for name, (mean, weighted_sum, weight_sum, entry_count, linear_count) in components.items():
            weight = getattr(self, self._weight_buffer_names[name]).to(
                device=mean.device, dtype=mean.dtype
            )
            terms[name] = weight * mean
            statistics[name] = LossTermStatistics(
                weight=weight.detach(),
                unweighted_mean=mean.detach(),
                weighted_sum=weighted_sum,
                weight_sum=weight_sum,
                entry_count=entry_count,
                linear_count=linear_count,
            )
        self.last_statistics = statistics
        return terms

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        ddp = is_ddp_enabled() if ddp is None else ddp
        if ddp:
            raise NotImplementedError(
                "InteractionHuberLoss does not support distributed training: its "
                "per-term weighted means would need a global reduction of the weight sums"
            )
        return torch.stack(list(self.compute_terms(ref, pred).values())).sum()

    def __repr__(self):
        weights = ", ".join(
            f"{buffer_name}={float(getattr(self, buffer_name)):.3f}"
            for buffer_name in self._weight_buffer_names.values()
        )
        thresholds = ", ".join(
            f"huber_delta_{name}={delta:g}" for name, delta in self.huber_deltas.items()
        )
        return f"{self.__class__.__name__}({weights}, {thresholds})"
