"""Periodic neighbour pairs of a batch of systems with one host read per call.

Built on nvalchemiops' batched naive neighbour matrix. The wrapped positions, the
periodic-image ranges and the matrix are computed on the device; the only value the host
needs is the number of pairs (the size of the edge list), read once per call together with
the two capacity checks (the widest matrix row and the most periodic images any system
needs). A capacity that turns out too small grows and the call is repeated, so no pair is
ever dropped.

The default path of ``MaceTorchSimModel`` for fully periodic batches (MD); torch-sim's
``torchsim_nl`` reads ~20 values back per call (system counts, image ranges, the neighbour
estimate, validation) and the wrapper around it another ~15 (experiment 10 of
cluster_bulk_generalization).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

# Headroom of a capacity that has to grow: a breathing NPT cell or a denser region
# would otherwise trigger a regrowth every few steps.
CAPACITY_HEADROOM = 1.25


def wrap_positions(
    positions: torch.Tensor, row_vector_cell: torch.Tensor, system_idx: torch.Tensor
) -> torch.Tensor:
    """Positions wrapped into the primary cell of their system (all axes periodic).

    ``r = f C`` with the lattice vectors as the rows of ``C``; ``floor(f) C`` is
    subtracted. ``inv_ex`` because ``inv`` checks for singular matrices on the host.
    """
    inverse = torch.linalg.inv_ex(row_vector_cell).inverse
    fractional = torch.einsum("ni,nij->nj", positions, inverse[system_idx])
    return positions - torch.einsum(
        "ni,nij->nj", torch.floor(fractional), row_vector_cell[system_idx]
    )


def periodic_image_ranges(
    row_vector_cell: torch.Tensor, cutoff: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per system, the image range along each lattice vector and the number of images
    of nvalchemiops' half-space enumeration (``compute_naive_num_shifts``, on the
    device): ``ceil(cutoff |b_i|)`` with ``b_i`` the rows of ``C^-T``."""
    inverse_transposed = torch.linalg.inv_ex(row_vector_cell).inverse.transpose(-1, -2)
    ranges = torch.ceil(inverse_transposed.norm(dim=-1) * cutoff).to(torch.int32)
    first, second, third = ranges.unbind(-1)
    second_count = 2 * second + 1
    third_count = 2 * third + 1
    images = first * second_count * third_count + second * third_count + third + 1
    return ranges, images.to(torch.int32)


class BatchedNeighborList:
    """Neighbour pairs ``(sender, receiver)`` within ``cutoff`` of a fully periodic batch.

    Returns the edge index ``(2, edges)`` (row atom first, as torch-sim's
    ``alchemiops_nl_n2``) and the integer cell shifts ``(edges, 3)`` of the second atom.
    The positions must already be wrapped (:func:`wrap_positions`).
    """

    def __init__(self, cutoff: float) -> None:
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        self.cutoff = float(cutoff)
        self.maximum_neighbors = 0
        self.maximum_images = 0

    def __call__(
        self,
        positions: torch.Tensor,
        row_vector_cell: torch.Tensor,
        system_idx: torch.Tensor,
        system_pointer: torch.Tensor,
        maximum_atoms_per_system: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
        from nvalchemiops.torch.neighbors import batch_naive_neighbor_list

        systems = row_vector_cell.shape[0]
        periodic = torch.ones(systems, 3, dtype=torch.bool, device=positions.device)
        ranges, images = periodic_image_ranges(row_vector_cell, self.cutoff)
        if self.maximum_neighbors == 0:
            self.maximum_neighbors = estimate_max_neighbors(self.cutoff)
        if self.maximum_images == 0:
            # First call: one extra read for the image capacity.
            self.maximum_images = int(images.max())
        system_index = system_idx.to(torch.int32)
        while True:
            neighbor_matrix, neighbor_counts, neighbor_shifts = (
                batch_naive_neighbor_list(
                    positions=positions,
                    cutoff=self.cutoff,
                    batch_idx=system_index,
                    batch_ptr=system_pointer,
                    cell=row_vector_cell,
                    pbc=periodic,
                    max_neighbors=self.maximum_neighbors,
                    shift_range_per_dimension=ranges,
                    num_shifts_per_system=images,
                    max_shifts_per_system=self.maximum_images,
                    max_atoms_per_system=maximum_atoms_per_system,
                    wrap_positions=False,
                )
            )
            # The one host read: pairs, the widest row, the most images.
            edges, widest, most_images = (
                torch.stack(
                    [
                        neighbor_counts.sum(),
                        neighbor_counts.max(),
                        images.max().to(neighbor_counts.dtype),
                    ]
                )
                .cpu()
                .tolist()
            )
            if widest <= self.maximum_neighbors and most_images <= self.maximum_images:
                break
            if widest > self.maximum_neighbors:
                self.maximum_neighbors = 16 * math.ceil(widest * CAPACITY_HEADROOM / 16)
            if most_images > self.maximum_images:
                self.maximum_images = math.ceil(most_images * CAPACITY_HEADROOM)
        fill = positions.shape[0]
        pairs = torch.nonzero_static(neighbor_matrix != fill, size=edges)
        rows, slots = pairs[:, 0], pairs[:, 1]
        edge_index = torch.stack([rows, neighbor_matrix[rows, slots].to(torch.long)])
        return edge_index, neighbor_shifts[rows, slots]
