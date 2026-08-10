"""Fused multi-level training data: one configuration, every level's labels.

The stock multihead pipeline duplicates a structure once per head, so a
structure labelled at K levels is K configurations: its graph is built K
times, an epoch is K times longer, and the levels' losses reach the trunk
through separate forward passes in different batches. This loader builds
ONE AtomicData per structure carrying all level labels plus presence masks,
for training with ``MultiLevelScaleShiftMACE`` (which produces every level
in one forward pass) and the ``multilevel_weighted`` loss.

File convention: an extended-xyz file whose per-level labels use suffixed
keys -- ``<energy_key>_<head>`` in ``info`` and ``<forces_key>_<head>`` in
``arrays`` -- with a missing key meaning the structure carries no label at
that level (mask zero). The first head is the base level and must be
labelled on every structure.

Extra tensors attached to each AtomicData (names deliberately free of
"index"/"face", which the collation offsets by node count):

- ``energy_levels``          [1, num_heads]  absolute energy per level
- ``energy_levels_weight``   [1, num_heads]  presence mask times config weight
- ``forces_levels``          [n_atoms, num_force_levels, 3]
- ``forces_levels_weight``   [1, num_force_levels]

where the force columns follow ``force_carrying_heads`` order.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import ase.io
import torch

from mace.data.atomic_data import AtomicData
from mace.data.utils import Configuration
from mace.tools.utils import AtomicNumberTable


def load_multilevel_dataset(
    file_path: str,
    r_max: float,
    z_table: AtomicNumberTable,
    heads: Sequence[str],
    force_carrying_heads: Sequence[str],
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    config_weight_key: str = "config_weight",
) -> Tuple[List[AtomicData], Dict[str, int]]:
    """Read a suffixed-key extended-xyz file into fused multi-level AtomicData.

    Returns the dataset and, for logging, the number of labelled structures
    per head.
    """
    if len(heads) < 2:
        raise ValueError(f"A multi-level dataset needs at least two heads, got {heads}")
    base_head = heads[0]
    unknown = [head for head in force_carrying_heads if head not in heads]
    if unknown:
        raise ValueError(
            f"force_carrying_heads {unknown} are not in heads {list(heads)}"
        )
    if base_head not in force_carrying_heads:
        raise ValueError(
            f"the base head {base_head!r} must be force-carrying; got "
            f"{list(force_carrying_heads)}"
        )

    dtype = torch.get_default_dtype()
    atoms_list = ase.io.read(file_path, index=":")
    dataset: List[AtomicData] = []
    labelled_counts = {head: 0 for head in heads}

    for atoms_index, atoms in enumerate(atoms_list):
        base_energy: Optional[float] = atoms.info.get(f"{energy_key}_{base_head}")
        base_forces = atoms.arrays.get(f"{forces_key}_{base_head}")
        if base_energy is None or base_forces is None:
            raise ValueError(
                f"structure {atoms_index} in {file_path} is missing the base "
                f"level's energy or forces ({energy_key}_{base_head} / "
                f"{forces_key}_{base_head}); the base level must label every "
                "structure"
            )
        config_weight = float(atoms.info.get(config_weight_key, 1.0))

        configuration = Configuration(
            atomic_numbers=atoms.get_atomic_numbers(),
            positions=atoms.get_positions(),
            properties={"energy": float(base_energy), "forces": base_forces},
            property_weights={"energy": 1.0, "forces": 1.0},
            cell=atoms.get_cell().array,
            pbc=tuple(atoms.get_pbc()),
            weight=config_weight,
            head=base_head,
        )
        atomic_data = AtomicData.from_config(
            configuration, z_table=z_table, cutoff=r_max, heads=list(heads)
        )
        num_atoms = len(atoms)

        energy_levels = torch.zeros(1, len(heads), dtype=dtype)
        energy_levels_weight = torch.zeros(1, len(heads), dtype=dtype)
        for level, head in enumerate(heads):
            level_energy = atoms.info.get(f"{energy_key}_{head}")
            if level_energy is not None:
                energy_levels[0, level] = float(level_energy)
                energy_levels_weight[0, level] = config_weight
                labelled_counts[head] += 1

        forces_levels = torch.zeros(
            num_atoms, len(force_carrying_heads), 3, dtype=dtype
        )
        forces_levels_weight = torch.zeros(1, len(force_carrying_heads), dtype=dtype)
        for column, head in enumerate(force_carrying_heads):
            level_forces = atoms.arrays.get(f"{forces_key}_{head}")
            if level_forces is not None:
                forces_levels[:, column, :] = torch.tensor(level_forces, dtype=dtype)
                forces_levels_weight[0, column] = config_weight

        atomic_data.energy_levels = energy_levels
        atomic_data.energy_levels_weight = energy_levels_weight
        atomic_data.forces_levels = forces_levels
        atomic_data.forces_levels_weight = forces_levels_weight
        dataset.append(atomic_data)

    if not dataset:
        raise ValueError(f"{file_path} contains no structures")
    return dataset, labelled_counts
