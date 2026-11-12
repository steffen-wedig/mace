"""Real-space pair lists of open structures, built on the host when a batch is collated.

PolarMACE's open-boundary electrostatics evaluate every directed atom pair of each open
structure. Built on the device inside the forward, that list costs host syncs and has a
data-dependent size, which torch.compile cannot trace. Built here, in the data-loader
workers, it arrives with the batch as ``real_space_pair_index`` ([2, n_pairs], batch node
numbering) and the forward only indexes with it.
"""

from __future__ import annotations

import torch


class PeriodicStructureInRealSpaceBatch(ValueError):
    """A periodic structure in a batch evaluated with open-boundary electrostatics."""


def real_space_pair_index(ptr: torch.Tensor, pbc: torch.Tensor) -> torch.Tensor:
    """Every directed pair (sender, receiver), sender != receiver, within each structure.

    ptr: [n_structures + 1] node offsets of the structures in the batch; pbc:
    [n_structures, 3]. Training with pbc_handling "realspace" treats every structure as
    open, so a structure with any periodic direction is refused rather than evaluated
    without its images.
    """
    periodic = pbc.reshape(-1, 3).any(dim=-1)
    if bool(periodic.any()):
        raise PeriodicStructureInRealSpaceBatch(
            f"{int(periodic.sum())} periodic structure(s) in a batch for open-boundary "
            "electrostatics (pbc_handling 'realspace' trains open structures only)"
        )
    sizes = ptr[1:] - ptr[:-1]
    # Each node sends to every other node of its structure: (size - 1) pairs per node.
    sender_structure_sizes = torch.repeat_interleave(sizes, sizes)
    node_starts = torch.repeat_interleave(ptr[:-1], sizes)
    sender = torch.repeat_interleave(
        torch.arange(int(ptr[-1]), dtype=torch.long), sender_structure_sizes - 1
    )
    # Receivers of one sender: its structure's nodes in order, skipping the sender.
    pair_start = torch.repeat_interleave(
        torch.cumsum(sender_structure_sizes - 1, dim=0) - (sender_structure_sizes - 1),
        sender_structure_sizes - 1,
    )
    position = torch.arange(sender.numel(), dtype=torch.long) - pair_start
    first_node = torch.repeat_interleave(node_starts, sender_structure_sizes - 1)
    receiver = first_node + position
    receiver = receiver + (receiver >= sender).long()
    return torch.stack([sender, receiver])


def attach_real_space_pair_index(batch):
    """Add ``real_space_pair_index`` to a collated batch (in place) and return it."""
    batch.real_space_pair_index = real_space_pair_index(batch.ptr, batch.pbc)
    return batch


__all__ = [
    "PeriodicStructureInRealSpaceBatch",
    "attach_real_space_pair_index",
    "real_space_pair_index",
]
