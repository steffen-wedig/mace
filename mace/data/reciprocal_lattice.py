"""Reciprocal-lattice indices of fully periodic structures, listed when a batch is collated.

PolarMACE's periodic electrostatics sum over the k-vectors within the model's reciprocal
cutoff. Selected inside the forward, that set has a data-dependent size and is read from the
cell on the host, which torch.compile cannot trace. Listed here, in the data-loader workers,
the integer lattice indices arrive with the batch (``kspace_miller``, ``kspace_structure``,
``kspace_padding``; ``mace.modules.polar.electrostatics.kspace``) and the forward forms
``k = m B`` from the reciprocal cell and applies the cutoff in the graph: the k-vectors that
enter the sums are exactly those the model selects from the cell itself.

The cutoff is the model's ``kspace_cutoff`` buffer (``PolarMACE.kspace_cutoff_value``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from mace.modules.polar.electrostatics.kspace import (
    ReciprocalLatticeIndices,
    pad_reciprocal_lattice_indices,
    reciprocal_lattice_indices,
)
from mace.tools.torch_geometric.data import Data

# The batch keys of the indices, in the order of ReciprocalLatticeIndices.
RECIPROCAL_LATTICE_KEYS = ("kspace_miller", "kspace_structure", "kspace_padding")


class OpenStructureInReciprocalSpaceBatch(ValueError):
    """A structure that is not fully periodic in a batch evaluated in reciprocal space."""


def periodic_lattice_indices(
    cell: torch.Tensor, pbc: torch.Tensor, kspace_cutoff: float
) -> ReciprocalLatticeIndices:
    """Lattice indices of every k-vector within ``kspace_cutoff`` of each cell.

    cell: [n_structures * 3, 3] or [n_structures, 3, 3]; pbc: [n_structures, 3]. Training
    with pbc_handling "pbc" treats every structure as fully periodic, so a structure with
    an open direction (a cluster, a slab) is refused rather than given images it lacks.
    """
    periodic = pbc.reshape(-1, 3).all(dim=-1)
    if not bool(periodic.all()):
        raise OpenStructureInReciprocalSpaceBatch(
            f"{int((~periodic).sum())} structure(s) without periodic boundaries in all "
            "three directions in a batch for reciprocal-space electrostatics "
            "(pbc_handling 'pbc' trains fully periodic structures only)"
        )
    return reciprocal_lattice_indices(cell.reshape(-1, 3, 3), kspace_cutoff)


def structure_kspace_rows(structure: Data, kspace_cutoff: float) -> int:
    """Number of k-vector rows one structure adds to a batch."""
    indices = periodic_lattice_indices(structure.cell, structure.pbc, kspace_cutoff)
    return int(indices.miller.shape[0])


def attach_reciprocal_lattice_indices(
    batch,
    kspace_cutoff: float,
    rows: Optional[int] = None,
    real_graphs: Optional[int] = None,
):
    """Add the lattice-index keys to a collated batch (in place) and return it.

    ``real_graphs``: list the first ``real_graphs`` structures only (the others are
    padding, ``mace.data.padding``); ``rows``: pad to a fixed row count with excluded rows
    owned by the first structure after them (contributing exactly zero).
    """
    graphs = batch.num_graphs if real_graphs is None else real_graphs
    if rows is not None and graphs >= batch.num_graphs:
        raise ValueError("padding rows need a padding structure after the listed ones")
    indices = periodic_lattice_indices(
        batch.cell.reshape(-1, 3, 3)[:graphs],
        batch.pbc.reshape(-1, 3)[:graphs],
        kspace_cutoff,
    )
    if rows is not None:
        indices = pad_reciprocal_lattice_indices(indices, rows, owner=graphs)
    for key, value in zip(RECIPROCAL_LATTICE_KEYS, indices, strict=True):
        batch[key] = value
    return batch


def batch_kspace_rows(data_list: Sequence[Data], kspace_cutoff: float) -> int:
    """Number of k-vector rows of the batch ``data_list`` collates to."""
    return sum(
        structure_kspace_rows(structure, kspace_cutoff) for structure in data_list
    )


__all__ = [
    "OpenStructureInReciprocalSpaceBatch",
    "RECIPROCAL_LATTICE_KEYS",
    "attach_reciprocal_lattice_indices",
    "batch_kspace_rows",
    "periodic_lattice_indices",
    "structure_kspace_rows",
]
