"""Collate cluster records into one ``Batch`` and combine per-subsystem outputs with signs.

A record (``mace.data.cluster_records``) is a ``List[AtomicData]``: the cluster graph
(``sign = +1``) followed by its monomers (``sign = -1``), every subsystem carrying
``cluster_id == 0`` and a record-local ``slot``. A minibatch is a ``List[List[AtomicData]]``.
:class:`ClusterCollater` flattens it into a single ``Batch`` -- a cluster's subsystems are
ordinary graphs of the disconnected union, so the model forward is untouched -- and makes
the bookkeeping disjoint across records: ``cluster_id`` becomes the record's position in
the minibatch and ``slot`` is shifted by a running offset, both on shallow copies so the
input records are never modified. Neither name matches
``(index|face)``, so ``Batch.from_data_list`` leaves both alone; the collater owns them.

mace's own ``torch_geometric.dataloader.DataLoader`` deletes any ``collate_fn``, so the
collater is attached to a stock ``torch.utils.data.DataLoader`` (:func:`get_cluster_data_loader`).
"""

from __future__ import annotations

import copy
from typing import List, Sequence

import torch
from torch.utils.data import DataLoader

from mace.tools.scatter import scatter_sum

from .batch import Batch
from .data import Data


class ClusterCollater:
    """``List[List[AtomicData]]`` -> one ``Batch`` with ``n_clusters``, ``n_slots`` and the
    per-record ``cluster_interaction_weight`` (``[n_clusters]``) attached."""

    def __init__(self, follow_batch: Sequence = (), exclude_keys: Sequence = ()) -> None:
        self.follow_batch = list(follow_batch)
        self.exclude_keys = list(exclude_keys)

    def __call__(self, records: List[List[Data]]) -> Batch:
        flat: List[Data] = []
        interaction_weights: List[torch.Tensor] = []
        slot_offset = 0
        for record_index, record in enumerate(records):
            if not record:
                raise ValueError(f"record {record_index} has no subsystems")
            cluster_subsystems = [subsystem for subsystem in record if float(subsystem.sign) > 0]
            if len(cluster_subsystems) != 1:
                raise ValueError(
                    f"record {record_index} has {len(cluster_subsystems)} cluster graphs "
                    "(sign = +1); expected exactly one"
                )
            record_slot_count = int(cluster_subsystems[0].num_nodes)
            for subsystem in record:
                if int(subsystem.slot.max()) >= record_slot_count:
                    raise ValueError(
                        f"record {record_index}: a slot exceeds the cluster's atom count"
                    )
                # a shallow copy: the dataset's records (possibly cached) stay untouched
                shifted = copy.copy(subsystem)
                shifted.cluster_id = torch.tensor(record_index, dtype=torch.long)
                shifted.slot = subsystem.slot + slot_offset
                flat.append(shifted)
            interaction_weights.append(
                torch.as_tensor(cluster_subsystems[0].interaction_weight).reshape(())
            )
            slot_offset += record_slot_count

        batch = Batch.from_data_list(
            flat, follow_batch=self.follow_batch, exclude_keys=self.exclude_keys
        )
        batch.n_clusters = len(records)
        batch.n_slots = slot_offset
        batch.cluster_interaction_weight = torch.stack(interaction_weights)
        return batch


def is_cluster_batch(batch) -> bool:
    """Whether ``batch`` came through :class:`ClusterCollater`."""
    return hasattr(batch, "cluster_id") and hasattr(batch, "n_clusters")


def combine_energy(batch, energy: torch.Tensor) -> torch.Tensor:
    """Signed per-record sum of per-graph energies -> ``E_int [n_clusters]``."""
    return scatter_sum(
        batch.sign * energy, batch.cluster_id, dim=0, dim_size=int(batch.n_clusters)
    )


def combine_forces(batch, forces: torch.Tensor) -> torch.Tensor:
    """Signed per-slot sum of per-atom forces -> ``F_int [n_slots, 3]``."""
    sign_node = batch.sign[batch.batch]
    return scatter_sum(
        sign_node.unsqueeze(-1) * forces, batch.slot, dim=0, dim_size=int(batch.n_slots)
    )


def cluster_atom_counts(batch) -> torch.Tensor:
    """Atoms of each record's cluster graph ``[n_clusters]`` (for intensive E_int terms)."""
    node_cluster = batch.cluster_id[batch.batch]
    is_cluster_node = (batch.sign[batch.batch] > 0).to(batch.positions.dtype)
    counts = scatter_sum(is_cluster_node, node_cluster, dim=0, dim_size=int(batch.n_clusters))
    return counts.clamp_min(1.0)


def slot_to_cluster(batch) -> torch.Tensor:
    """Owning record of every slot ``[n_slots]`` (long)."""
    node_cluster = batch.cluster_id[batch.batch]
    cluster_node_mask = batch.sign[batch.batch] > 0
    owner = torch.zeros(int(batch.n_slots), dtype=torch.long, device=batch.slot.device)
    owner[batch.slot[cluster_node_mask]] = node_cluster[cluster_node_mask]
    return owner


def get_cluster_data_loader(
    dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    drop_last: bool = False,
    follow_batch: Sequence = (),
    exclude_keys: Sequence = (),
    **kwargs,
) -> DataLoader:
    """A stock torch ``DataLoader`` over a record dataset with the :class:`ClusterCollater`."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=ClusterCollater(follow_batch, exclude_keys),
        **kwargs,
    )


__all__ = [
    "ClusterCollater",
    "is_cluster_batch",
    "combine_energy",
    "combine_forces",
    "cluster_atom_counts",
    "slot_to_cluster",
    "get_cluster_data_loader",
]
