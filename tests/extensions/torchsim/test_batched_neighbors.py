"""The fully periodic path of ``MaceTorchSimModel``: ``BatchedNeighborList`` and the
wrapper around it, against torch-sim's neighbour list and the wrapper's torch-sim path."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from tests.helpers import GRAPH_LONGRANGE_AVAILABLE, TORCHSIM_AVAILABLE

pytestmark = [
    pytest.mark.torchsim,
    pytest.mark.skipif(not TORCHSIM_AVAILABLE, reason="torch-sim not installed"),
]

CELLS = [
    np.diag([7.0, 7.5, 8.0]),
    np.array([[7.0, 0.0, 0.0], [1.3, 6.8, 0.0], [0.9, -1.1, 6.9]]),
    # Smaller than twice the cutoff: atoms see several images of each other.
    np.array([[5.1, 0.0, 0.0], [0.4, 4.9, 0.0], [-0.3, 0.6, 5.3]]),
]
ATOMS_PER_SYSTEM = [9, 12, 7]


def _periodic_batch(device: torch.device, dtype=torch.float64):
    """Positions (unwrapped: some atoms outside their cell), row-vector cells, system
    index and species of three periodic systems."""
    generator = torch.Generator().manual_seed(11)
    cells = torch.tensor(np.stack(CELLS), dtype=dtype)
    system_idx = torch.repeat_interleave(
        torch.arange(len(CELLS)), torch.tensor(ATOMS_PER_SYSTEM)
    )
    fractional = torch.rand(system_idx.shape[0], 3, generator=generator, dtype=dtype)
    fractional = fractional * 1.6 - 0.3
    positions = torch.einsum("ni,nij->nj", fractional, cells[system_idx])
    species = torch.tensor([8, 1, 1, 6] * 10, dtype=torch.long)[: system_idx.shape[0]]
    return (
        positions.to(device),
        cells.to(device),
        system_idx.to(device),
        species.to(device),
    )


def _pairs(edge_index, unit_shifts) -> set:
    return {
        (int(i), int(j), *(int(round(float(value))) for value in shift))
        for i, j, shift in zip(edge_index[0], edge_index[1], unit_shifts)
    }


def _system_pointer(system_idx: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(system_idx)
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)


DEVICES = [torch.device("cpu")] + (
    [torch.device("cuda")] if torch.cuda.is_available() else []
)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("initial_capacity", [0, 16], ids=["estimated", "too_small"])
def test_batched_neighbor_list_finds_torch_sims_pairs(device, initial_capacity):
    from torch_sim.neighbors import torchsim_nl

    from mace.calculators.batched_neighbors import BatchedNeighborList, wrap_positions

    positions, cells, system_idx, _ = _periodic_batch(device)
    wrapped = wrap_positions(positions, cells, system_idx)
    # Wrapped into [0, 1) fractional coordinates, by whole lattice vectors.
    fractional = torch.einsum(
        "ni,nij->nj", wrapped, torch.linalg.inv(cells)[system_idx]
    )
    assert bool(((fractional >= -1e-12) & (fractional < 1 + 1e-12)).all())
    moved = torch.einsum(
        "ni,nij->nj", wrapped - positions, torch.linalg.inv(cells)[system_idx]
    )
    torch.testing.assert_close(moved, moved.round(), rtol=0, atol=1e-10)

    cutoff = 4.0
    neighbor_list = BatchedNeighborList(cutoff)
    neighbor_list.maximum_neighbors = initial_capacity
    edge_index, unit_shifts = neighbor_list(
        wrapped, cells, system_idx, _system_pointer(system_idx), max(ATOMS_PER_SYSTEM)
    )
    periodic = torch.ones(len(CELLS), 3, dtype=torch.bool, device=device)
    expected_index, _, expected_shifts = torchsim_nl(
        wrapped, cells, periodic, cutoff, system_idx
    )
    assert _pairs(edge_index.cpu(), unit_shifts.cpu()) == _pairs(
        expected_index.cpu(), expected_shifts.cpu()
    )
    assert neighbor_list.maximum_neighbors >= int(
        torch.bincount(edge_index[0].cpu()).max()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_batched_neighbor_list_reads_the_host_once():
    from mace.calculators.batched_neighbors import BatchedNeighborList, wrap_positions

    positions, cells, system_idx, _ = _periodic_batch(torch.device("cuda"))
    pointer = _system_pointer(system_idx)
    neighbor_list = BatchedNeighborList(4.0)
    neighbor_list(
        wrap_positions(positions, cells, system_idx),
        cells,
        system_idx,
        pointer,
        max(ATOMS_PER_SYSTEM),
    )
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            neighbor_list(
                wrap_positions(positions, cells, system_idx),
                cells,
                system_idx,
                pointer,
                max(ATOMS_PER_SYSTEM),
            )
        finally:
            torch.cuda.set_sync_debug_mode("default")
    synchronising = [
        w for w in caught if "called a synchronizing CUDA operation" in str(w.message)
    ]
    assert len(synchronising) == 1


@pytest.mark.skipif(
    not GRAPH_LONGRANGE_AVAILABLE, reason="graph_longrange is not installed"
)
def test_wrapper_periodic_path_equals_the_torch_sim_path():
    """Energies, forces and stress of the fully periodic path (BatchedNeighborList, own
    wrapping, no state copy) equal the torch-sim path's, and the state is untouched."""
    import torch_sim as ts
    from torch_sim.neighbors import torchsim_nl

    from mace.calculators.mace_torchsim import MaceTorchSimModel
    from tests.extensions.polar.test_polar_shared_geometry import _polar_model

    device = torch.device("cpu")
    positions, cells, system_idx, species = _periodic_batch(device)
    state = ts.SimState(
        positions=positions,
        masses=torch.ones_like(positions[:, 0]),
        cell=cells.transpose(-1, -2),
        pbc=True,
        atomic_numbers=species,
        system_idx=system_idx,
    )
    before = state.positions.clone()
    model = _polar_model()
    model.set_electrostatic_pbcs("pbc")
    periodic_path = MaceTorchSimModel(model, device=device, dtype=torch.float64)
    torch_sim_path = MaceTorchSimModel(
        model, device=device, dtype=torch.float64, neighbor_list_fn=torchsim_nl
    )
    actual = periodic_path(state)
    expected = torch_sim_path(state)
    assert periodic_path._fully_periodic  # pylint: disable=protected-access
    for key in ("energy", "forces", "stress"):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(state.positions, before, rtol=0, atol=0)
    assert not state.positions.requires_grad
    # A second call reuses the checks and the capacities.
    again = periodic_path(state)
    torch.testing.assert_close(again["energy"], actual["energy"], rtol=0, atol=0)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_wrapper_device_carries_its_index():
    """``device="cuda"`` is the state's ``cuda:0``: compared unindexed, every forward
    copied (and revalidated) the whole MD state to "its" device."""
    from mace.calculators.mace_torchsim import (  # pylint: disable=protected-access
        _indexed_device,
    )

    index = torch.cuda.current_device()
    assert _indexed_device("cuda") == torch.device("cuda", index)
    assert _indexed_device(torch.device("cuda", index)) == torch.device("cuda", index)
    assert _indexed_device("cpu") == torch.device("cpu")
