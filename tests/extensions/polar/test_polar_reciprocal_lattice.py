"""k-vectors from integer lattice indices (``reciprocal_lattice_indices`` outside the
forward, ``lattice_k_vectors`` inside it) instead of ``compute_k_vectors_flat`` inside it.

The lattice indices are what a compiled MD step needs: fixed shapes, no host reads in the
forward, and a set that stays valid while an NPT cell breathes (``margin``), with the
cutoff applied at the cell of each evaluation. The k-vectors that enter the sums, and so
every output, must be those of ``compute_k_vectors_flat``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import mace.modules.polar.model as polar_model_module
from mace.modules.polar.electrostatics.kspace import (
    compute_k_vectors_flat,
    lattice_indices_cover,
    lattice_k_vectors,
    reciprocal_lattice_indices,
)
from tests.extensions.polar.test_polar_shared_geometry import (
    _batch,
    _clone,
    _molecules,
    _polar_model,
)

CELLS = [
    np.diag([7.0, 7.5, 8.0]),
    np.array([[7.0, 0.0, 0.0], [1.3, 6.8, 0.0], [0.9, -1.1, 6.9]]),
    np.array([[14.4, 0.0, 0.0], [0.0, 14.5, 0.0], [0.3, 0.2, 14.3]]),
]


def _reciprocal(cell: torch.Tensor) -> torch.Tensor:
    return 2 * torch.pi * torch.linalg.inv(cell.transpose(-1, -2))


def _included_k_vectors(k_vectors, excluded) -> np.ndarray:
    kept = k_vectors[excluded == 0].numpy()
    return kept[np.lexsort(kept.T[::-1])]


@pytest.mark.parametrize("cutoff", [1.2, 1.85, 2.77])
@pytest.mark.parametrize("cell_index", range(len(CELLS)))
def test_lattice_indices_give_the_k_vectors_of_compute_k_vectors_flat(
    cell_index, cutoff
):
    cell = torch.tensor(CELLS[cell_index], dtype=torch.float64).unsqueeze(0)
    rcell = _reciprocal(cell)
    expected_k, _, _, expected_excluded = compute_k_vectors_flat(cutoff, cell, rcell)
    indices = reciprocal_lattice_indices(cell, cutoff, margin=0.1)
    k_vectors, k_norm2, structure, excluded = lattice_k_vectors(indices, rcell, cutoff)
    # The origin is never listed; every other row of compute_k_vectors_flat is.
    expected = _included_k_vectors(expected_k, expected_excluded)
    actual = _included_k_vectors(k_vectors, excluded)
    assert len(actual) == len(expected) > 0
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    # Rows the margin lists beyond the cutoff are excluded here.
    assert bool((k_norm2[excluded == 0] <= cutoff**2).all())
    assert bool((structure == 0).all())


def test_padding_rows_are_excluded_and_rows_must_fit():
    cells = torch.tensor(np.stack(CELLS[:2]), dtype=torch.float64)
    unpadded = reciprocal_lattice_indices(cells, 1.85)
    padded = reciprocal_lattice_indices(cells, 1.85, rows=len(unpadded.miller) + 5)
    _, _, structure, excluded = lattice_k_vectors(padded, _reciprocal(cells), 1.85)
    assert bool(excluded[-5:].all()) and bool(padded.padding[-5:].all())
    assert bool((structure[-5:] == 1).all())
    # Rows grouped by structure.
    assert bool((torch.diff(unpadded.structure) >= 0).all())
    with pytest.raises(ValueError, match="do not fit"):
        reciprocal_lattice_indices(cells, 1.85, rows=len(unpadded.miller) - 1)


def _periodic_batch() -> dict:
    molecules = _molecules()
    for atoms, cell in zip(molecules, CELLS[:2] + [np.diag([6.5, 6.5, 6.5])]):
        atoms.cell = cell
        atoms.pbc = True
        atoms.positions = atoms.positions + 2.0
    return _batch(molecules)


def _outputs(model, batch: dict) -> dict:
    output = model(
        _clone(batch), training=False, compute_force=True, compute_stress=True
    )
    return {key: output[key].detach() for key in ("energy", "forces", "stress")}


@pytest.fixture(name="periodic_model")
def _periodic_model():
    model = _polar_model()
    model.set_electrostatic_pbcs("pbc")
    return model


def test_model_matches_compute_k_vectors_flat(periodic_model, monkeypatch):
    batch = _periodic_batch()
    actual = _outputs(periodic_model, batch)

    def flat_k_vectors(_indices, rcell, cutoff):
        cell = 2 * torch.pi * torch.linalg.inv(rcell).transpose(-1, -2)
        return compute_k_vectors_flat(cutoff, cell, rcell)

    monkeypatch.setattr(polar_model_module, "lattice_k_vectors", flat_k_vectors)
    expected = _outputs(periodic_model, batch)
    assert float(actual["stress"].abs().max()) > 1e-6
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=1e-11, atol=1e-12)


def test_lattice_indices_from_the_batch_match_the_forward_listing(periodic_model):
    batch = _periodic_batch()
    expected = _outputs(periodic_model, batch)
    listed = reciprocal_lattice_indices(
        batch["cell"].view(-1, 3, 3), periodic_model.kspace_cutoff_value()
    )
    indices = reciprocal_lattice_indices(
        batch["cell"].view(-1, 3, 3),
        periodic_model.kspace_cutoff_value(),
        margin=0.15,
        rows=len(listed.miller) * 2,
    )
    batch.update(
        kspace_miller=indices.miller,
        kspace_structure=indices.structure,
        kspace_padding=indices.padding,
    )
    actual = _outputs(periodic_model, batch)
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=1e-11, atol=1e-12)


def test_stress_matches_finite_differences_of_the_strain(periodic_model):
    """NPT needs the strain derivative through the reciprocal cell: the stress is
    (1/V) dE/d(strain) with positions and cell strained together."""
    batch = _periodic_batch()
    stress = _outputs(periodic_model, batch)["stress"]
    cells = batch["cell"].view(-1, 3, 3)
    volumes = torch.linalg.det(cells).abs()
    step = 1e-5

    def energy_at(strain: torch.Tensor) -> torch.Tensor:
        strained = _clone(batch)
        deformation = torch.eye(3, dtype=torch.float64) + strain
        strained["positions"] = batch["positions"] @ deformation
        strained["cell"] = (cells @ deformation).reshape(batch["cell"].shape)
        strained["shifts"] = batch["shifts"] @ deformation
        strained_cells = strained["cell"].view(-1, 3, 3)
        strained["rcell"] = _reciprocal(strained_cells).reshape(batch["rcell"].shape)
        strained["volume"] = torch.linalg.det(strained_cells).abs()
        return _outputs(periodic_model, strained)["energy"]

    numerical = torch.zeros_like(stress)
    for row in range(3):
        for column in range(3):
            strain = torch.zeros(3, 3, dtype=torch.float64)
            strain[row, column] += 0.5 * step
            strain[column, row] += 0.5 * step
            difference = energy_at(strain) - energy_at(-strain)
            numerical[:, row, column] = difference / (2 * step) / volumes
    torch.testing.assert_close(stress, numerical, rtol=0, atol=1e-8)


def test_selection_uses_the_kspace_cutoff_buffer(periodic_model, monkeypatch):
    """Fine-tuned models carry the foundation's kspace_cutoff buffer, which can differ
    from the cutoff their GTO bases were built for; the k-vectors are selected with the
    buffer, as compute_k_vectors_flat(self.kspace_cutoff, ...) always did."""
    built_for = periodic_model.electric_potential_descriptor.kspace_cutoff
    periodic_model.kspace_cutoff.fill_(0.8 * built_for)
    assert periodic_model.kspace_cutoff_value() == pytest.approx(0.8 * built_for)
    batch = _periodic_batch()
    actual = _outputs(periodic_model, batch)

    def flat_k_vectors_at_the_buffer(_indices, rcell, cutoff):
        assert float(cutoff) == pytest.approx(0.8 * built_for)
        cell = 2 * torch.pi * torch.linalg.inv(rcell).transpose(-1, -2)
        return compute_k_vectors_flat(0.8 * built_for, cell, rcell)

    monkeypatch.setattr(
        polar_model_module, "lattice_k_vectors", flat_k_vectors_at_the_buffer
    )
    expected = _outputs(periodic_model, batch)
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=1e-11, atol=1e-12)


def test_lattice_indices_cover_a_strained_cell_within_the_margin():
    """An NPT run lists the indices once with a margin and re-lists only when the cell
    has left it: inside the margin the k-set at the new cell is the listed set's."""
    cutoff, margin = 1.85, 0.05
    listed_cell = torch.tensor(CELLS[2], dtype=torch.float64).unsqueeze(0)
    indices = reciprocal_lattice_indices(listed_cell, cutoff, margin=margin)
    generator = torch.Generator().manual_seed(5)
    for scale in (0.02, 0.04, 0.08):
        strain = scale * (
            torch.rand(3, 3, generator=generator, dtype=torch.float64) - 0.5
        )
        cell = listed_cell @ (torch.eye(3, dtype=torch.float64) + strain)
        rcell = _reciprocal(cell)
        expected_k, _, _, expected_excluded = compute_k_vectors_flat(
            cutoff, cell, rcell
        )
        expected = _included_k_vectors(expected_k, expected_excluded)
        k_vectors, _, _, excluded = lattice_k_vectors(indices, rcell, cutoff)
        covered = bool(lattice_indices_cover(listed_cell, cell, margin))
        if covered:
            np.testing.assert_allclose(
                _included_k_vectors(k_vectors, excluded), expected, rtol=0, atol=1e-12
            )
    # A growing cell brings k-vectors into the cutoff, a shrinking one only removes some.
    assert bool(lattice_indices_cover(listed_cell, listed_cell * 1.04, margin))
    assert bool(lattice_indices_cover(listed_cell, listed_cell * 0.9, margin))
    assert not bool(lattice_indices_cover(listed_cell, listed_cell * 1.1, margin))
