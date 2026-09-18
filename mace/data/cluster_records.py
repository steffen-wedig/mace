"""Cluster records: one training example is a cluster graph plus its monomers.

A *cluster record* bundles a full system (a periodic bulk frame or a gas-phase cluster,
``sign = +1``) with the fragments it decomposes into (``sign = -1``). Every subsystem stays an
ordinary MACE graph carrying its own absolute labels, and the interaction quantities are a
single signed scatter applied identically to predictions and references::

    E_int = scatter_sum(sign * energy,      cluster_id)   # [n_clusters]
    F_int = scatter_sum(sign_node * forces, slot)         # [n_slots, 3]

Fields carried on every subsystem, as ``Configuration.properties`` (so they serialise with
the stock HDF5 writer and pass through ``AtomicData.from_config``):

==================  ==============  =====================================================
field               shape           meaning
==================  ==============  =====================================================
``sign``            scalar          ``+1`` cluster graph, ``-1`` monomer
``cluster_id``      scalar (long)   record-local ``0``; the collater offsets it per batch
``slot``            ``[n_atoms]``   the atom's index in the cluster graph (monomer atom k
                                    <-> cluster atom ``slot[k]``)
``interaction_weight`` scalar       per-record weight of the interaction terms (``0`` for
                                    a record without monomers, i.e. a lone molecule)
==================  ==============  =====================================================

The names deliberately avoid ``index``/``face``: ``Batch.from_data_list`` cumulatively
increments any attribute matching ``(index|face)`` by the node count, which would corrupt
``slot`` and ``cluster_id``. The collater (``mace.tools.torch_geometric.cluster_collate``)
owns their per-batch offsetting instead.

On disk a record is one HDF5 group holding one subgroup per subsystem::

    config_batch_0/
        cluster_0/
            subsystem_0/   # sign = +1 (the cluster graph), always first
            subsystem_1/   # monomer
            ...
        cluster_1/ ...

Each ``subsystem_j`` is byte-for-byte the per-configuration record of
``save_configurations_as_HDF5``.
"""

from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from typing import List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from mace.data.atomic_data import AtomicData
from mace.data.hdf5_dataset import unpack_value
from mace.data.utils import DEFAULT_CONFIG_TYPE, Configuration, write_value
from mace.tools.utils import AtomicNumberTable

CLUSTER_SIGN = 1.0
MONOMER_SIGN = -1.0
CLUSTER_RECORD_FIELDS = ("sign", "cluster_id", "slot", "interaction_weight")


@dataclass(frozen=True)
class MonomerSubsystem:
    """One fragment of a cluster: its own geometry and labels, plus where its atoms sit
    in the cluster graph (``cluster_atom_indices[k]`` is the cluster index of monomer atom
    ``k``). Positions may differ from the cluster's by lattice vectors (unwrapped
    molecules of a periodic frame)."""

    atomic_numbers: np.ndarray
    positions: np.ndarray
    energy: float
    forces: np.ndarray
    cluster_atom_indices: np.ndarray
    total_charge: float = 0.0
    config_type: str = DEFAULT_CONFIG_TYPE


def subsystem_configuration(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    energy: float,
    forces: np.ndarray,
    sign: float,
    slot: np.ndarray,
    *,
    stress: Optional[np.ndarray] = None,
    cell: Optional[np.ndarray] = None,
    pbc: Optional[Sequence[bool]] = None,
    total_charge: float = 0.0,
    interaction_weight: float = 1.0,
    config_type: str = DEFAULT_CONFIG_TYPE,
    head: str = "Default",
    weight: float = 1.0,
    energy_weight: float = 1.0,
    forces_weight: float = 1.0,
    stress_weight: float = 1.0,
) -> Configuration:
    """One subsystem ``Configuration`` with absolute labels and the record fields."""
    properties = {
        "energy": float(energy),
        "forces": np.asarray(forces, dtype=float),
        "total_charge": float(total_charge),
        "sign": float(sign),
        "cluster_id": 0,
        "slot": np.asarray(slot, dtype=np.int64),
        "interaction_weight": float(interaction_weight),
    }
    if stress is not None:
        properties["stress"] = np.asarray(stress, dtype=float)
    return Configuration(
        atomic_numbers=np.asarray(atomic_numbers, dtype=int),
        positions=np.asarray(positions, dtype=float),
        properties=properties,
        property_weights={
            "energy": float(energy_weight),
            "forces": float(forces_weight),
            "stress": float(stress_weight),
        },
        weight=float(weight),
        config_type=config_type,
        head=head,
        cell=None if cell is None else np.asarray(cell, dtype=float),
        pbc=None if pbc is None else tuple(bool(flag) for flag in pbc),
    )


def _lattice_shift_residual(
    difference: np.ndarray, cell: Optional[np.ndarray], pbc: Optional[Sequence[bool]]
) -> np.ndarray:
    """``difference`` minus its nearest lattice-vector part (Angstrom); the difference
    itself for a non-periodic cluster."""
    if cell is None or pbc is None or not any(pbc):
        return difference
    cell = np.asarray(cell, dtype=float)
    fractional = np.linalg.solve(cell.T, difference.T).T
    return difference - np.round(fractional) @ cell


def build_cluster_record(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    energy: float,
    forces: np.ndarray,
    monomers: Sequence[MonomerSubsystem],
    *,
    stress: Optional[np.ndarray] = None,
    cell: Optional[np.ndarray] = None,
    pbc: Optional[Sequence[bool]] = None,
    total_charge: float = 0.0,
    interaction_weight: Optional[float] = None,
    config_type: str = DEFAULT_CONFIG_TYPE,
    head: str = "Default",
    weight: float = 1.0,
    energy_weight: float = 1.0,
    forces_weight: float = 1.0,
    stress_weight: float = 1.0,
    position_tolerance: float = 1e-3,
) -> List[Configuration]:
    """Assemble ``[cluster_subsystem, *monomer_subsystems]`` for one record.

    Every cluster atom must be claimed by exactly one monomer atom of the same species at
    the same position (up to a lattice vector for periodic clusters); anything else raises.
    A record without monomers is a lone molecule: its interaction weight defaults to 0 so
    the interaction terms ignore it while the absolute-label terms still see it.
    """
    atomic_numbers = np.asarray(atomic_numbers, dtype=int)
    positions = np.asarray(positions, dtype=float)
    atom_count = len(atomic_numbers)
    if interaction_weight is None:
        interaction_weight = 1.0 if monomers else 0.0

    claimed = np.zeros(atom_count, dtype=int)
    for monomer_index, monomer in enumerate(monomers):
        indices = np.asarray(monomer.cluster_atom_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) != len(monomer.atomic_numbers):
            raise ValueError(
                f"monomer {monomer_index}: cluster_atom_indices must map every monomer atom"
            )
        if indices.min() < 0 or indices.max() >= atom_count:
            raise ValueError(f"monomer {monomer_index}: cluster atom index out of range")
        if not np.array_equal(atomic_numbers[indices], np.asarray(monomer.atomic_numbers)):
            raise ValueError(f"monomer {monomer_index}: species differ from the cluster atoms")
        residual = _lattice_shift_residual(
            positions[indices] - np.asarray(monomer.positions, dtype=float), cell, pbc
        )
        if np.abs(residual).max() > position_tolerance:
            raise ValueError(
                f"monomer {monomer_index}: positions differ from the cluster atoms by up to "
                f"{np.abs(residual).max():.3e} Angstrom (tolerance {position_tolerance:.1e})"
            )
        claimed[indices] += 1
    if monomers and not np.all(claimed == 1):
        raise ValueError(
            f"cluster atoms claimed {sorted(set(claimed.tolist()))} times by the monomers; "
            "every atom must belong to exactly one monomer"
        )

    record = [
        subsystem_configuration(
            atomic_numbers,
            positions,
            energy,
            forces,
            CLUSTER_SIGN,
            np.arange(atom_count, dtype=np.int64),
            stress=stress,
            cell=cell,
            pbc=pbc,
            total_charge=total_charge,
            interaction_weight=interaction_weight,
            config_type=config_type,
            head=head,
            weight=weight,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            stress_weight=stress_weight,
        )
    ]
    for monomer in monomers:
        record.append(
            subsystem_configuration(
                monomer.atomic_numbers,
                monomer.positions,
                monomer.energy,
                monomer.forces,
                MONOMER_SIGN,
                np.asarray(monomer.cluster_atom_indices, dtype=np.int64),
                total_charge=monomer.total_charge,
                interaction_weight=interaction_weight,
                config_type=monomer.config_type,
                head=head,
                weight=weight,
                energy_weight=energy_weight,
                forces_weight=forces_weight,
                stress_weight=0.0,
            )
        )
    return record


def write_configuration_to_group(group: h5py.Group, config: Configuration) -> None:
    """The per-configuration body of ``save_configurations_as_HDF5``."""
    group["atomic_numbers"] = write_value(config.atomic_numbers)
    group["positions"] = write_value(config.positions)
    properties_group = group.create_group("properties")
    for key, value in config.properties.items():
        properties_group[key] = write_value(value)
    group["cell"] = write_value(config.cell)
    group["pbc"] = write_value(config.pbc)
    group["weight"] = write_value(config.weight)
    weights_group = group.create_group("property_weights")
    for key, value in config.property_weights.items():
        weights_group[key] = write_value(value)
    group["config_type"] = write_value(config.config_type)


def save_cluster_records_as_HDF5(
    records: Sequence[Sequence[Configuration]], h5_file: h5py.File
) -> None:
    """Write cluster records (each a list of subsystem configurations, cluster first)."""
    batch_group = h5_file.create_group("config_batch_0")
    for record_index, record in enumerate(records):
        if not record or record[0].properties.get("sign") != CLUSTER_SIGN:
            raise ValueError(f"record {record_index}: the first subsystem must be the cluster")
        record_group = batch_group.create_group(f"cluster_{record_index}")
        for subsystem_index, config in enumerate(record):
            write_configuration_to_group(
                record_group.create_group(f"subsystem_{subsystem_index}"), config
            )


def _suffix_int(name: str) -> int:
    return int(name.rsplit("_", 1)[-1])


class ClusterHDF5Dataset(Dataset):
    """``HDF5Dataset`` whose item is a whole record: ``List[AtomicData]``, cluster first."""

    def __init__(
        self,
        file_path,
        r_max: float,
        z_table: AtomicNumberTable,
        atomic_dataclass=AtomicData,
        **kwargs,
    ):
        super().__init__()
        self.file_path = file_path
        self._file = None
        self._index: List[tuple] = []
        for batch_name in sorted(self.file.keys(), key=_suffix_int):
            for record_name in sorted(self.file[batch_name].keys(), key=_suffix_int):
                self._index.append((batch_name, record_name))
        self.length = len(self._index)
        self.r_max = r_max
        self.z_table = z_table
        self.atomic_dataclass = atomic_dataclass
        self.kwargs = kwargs

    @property
    def file(self):
        if self._file is None:  # opened lazily so the dataset survives being pickled to workers
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        return state

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> List[AtomicData]:
        batch_name, record_name = self._index[index]
        record_group = self.file[batch_name][record_name]
        return [
            self._build_subsystem(record_group[name])
            for name in sorted(record_group.keys(), key=_suffix_int)
        ]

    def _build_subsystem(self, group) -> AtomicData:
        properties = {key: unpack_value(group["properties"][key][()]) for key in group["properties"]}
        property_weights = {
            key: unpack_value(group["property_weights"][key][()])
            for key in group["property_weights"]
        }
        record_fields = {key: properties.pop(key) for key in CLUSTER_RECORD_FIELDS}
        config = Configuration(
            atomic_numbers=group["atomic_numbers"][()],
            positions=group["positions"][()],
            properties=properties,
            weight=unpack_value(group["weight"][()]),
            property_weights=property_weights,
            config_type=unpack_value(group["config_type"][()]),
            pbc=unpack_value(group["pbc"][()]),
            cell=unpack_value(group["cell"][()]),
        )
        if config.head is None:
            config.head = self.kwargs.get("head")
        atomic_data = self.atomic_dataclass.from_config(
            config,
            z_table=self.z_table,
            cutoff=self.r_max,
            heads=self.kwargs.get("heads", ["Default"]),
            **{key: value for key, value in self.kwargs.items() if key != "heads"},
        )
        # Attached explicitly with fixed shapes/dtypes (from_config would promote the
        # per-atom slot to [n, 1] and guess dtypes).
        dtype = torch.get_default_dtype()
        atomic_data.sign = torch.tensor(float(record_fields["sign"]), dtype=dtype)
        atomic_data.cluster_id = torch.tensor(int(record_fields["cluster_id"]), dtype=torch.long)
        atomic_data.slot = torch.as_tensor(
            np.asarray(record_fields["slot"], dtype=np.int64).reshape(-1), dtype=torch.long
        )
        atomic_data.interaction_weight = torch.tensor(
            float(record_fields["interaction_weight"]), dtype=dtype
        )
        return atomic_data


def cluster_dataset_from_sharded_hdf5(
    directory: str, z_table: AtomicNumberTable, r_max: float, **kwargs
) -> ConcatDataset:
    """``ConcatDataset`` of :class:`ClusterHDF5Dataset` over every ``*.h5`` shard."""
    files = sorted(glob(directory + "/*.h5") + glob(directory + "/*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no HDF5 shards under {directory}")
    return ConcatDataset(
        [ClusterHDF5Dataset(file, z_table=z_table, r_max=r_max, **kwargs) for file in files]
    )


__all__ = [
    "CLUSTER_SIGN",
    "MONOMER_SIGN",
    "CLUSTER_RECORD_FIELDS",
    "MonomerSubsystem",
    "subsystem_configuration",
    "build_cluster_record",
    "write_configuration_to_group",
    "save_cluster_records_as_HDF5",
    "ClusterHDF5Dataset",
    "cluster_dataset_from_sharded_hdf5",
]
