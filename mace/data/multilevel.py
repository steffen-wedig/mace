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

``LevelQuotaBatchSampler`` (below) composes batches from this dataset so a
sparsely labelled level is guaranteed a fixed number of structures in every
batch instead of the handful a random draw would give it.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import ase.io
import torch
from torch.utils.data import Sampler

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
) -> Tuple[List[AtomicData], Dict[str, int], Dict[str, List[int]]]:
    """Read a suffixed-key extended-xyz file into fused multi-level AtomicData.

    Returns the dataset, the number of labelled structures per head (for
    logging), and the indices into the dataset of the structures each head
    labels (what ``LevelQuotaBatchSampler`` draws its quota from).
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
    labelled_indices: Dict[str, List[int]] = {head: [] for head in heads}

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
                # len(dataset) is the index this structure gets below.
                labelled_indices[head].append(len(dataset))

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
    return dataset, labelled_counts, labelled_indices


class LevelQuotaBatchSampler(Sampler[List[int]]):
    """Batch sampler guaranteeing a fixed quota of one level's labels.

    At 1 % coupled-cluster coverage a random batch of 32 carries 0.3 CC
    labels on average: most batches teach the CC head nothing and the few
    that do carry a single structure, so the CC gradient is almost pure
    noise. This sampler partitions the training set into the structures a
    given level labels (the *pool*) and the rest, and composes every batch
    from ``batch_size - effective_quota`` rest structures and
    ``effective_quota`` pool structures.

    The quota is a floor, never a cap: ``effective_quota`` is raised to the
    pool's natural share of a batch (``batch_size * len(pool) /
    dataset_size``) whenever that is larger, because a hard partition at a
    small quota would UNDERsample a well-covered level relative to plain
    shuffling -- at 50 % coverage a shuffled batch of 32 already holds ~16
    pool structures.

    One epoch is one full pass over the rest partition in a fresh shuffled
    order, dropping the final incomplete batch (as ``drop_last=True``
    does). The pool is drawn from a cycling iterator -- shuffled, handed
    out in order, reshuffled when exhausted -- whose state carries ACROSS
    epochs, so every pool structure is seen equally often in the long run
    even though the pool is revisited many times per epoch. A structure
    cannot appear twice in a batch: the two partitions are disjoint, and a
    reshuffle in the middle of a batch keeps the indices already handed out
    to that batch out of the front of the new cycle.
    """

    def __init__(
        self,
        dataset_size: int,
        quota_pool_indices: Sequence[int],
        batch_size: int,
        quota: int,
        seed: int,
    ) -> None:
        super().__init__()
        if dataset_size < 1:
            raise ValueError(f"dataset_size must be positive, got {dataset_size}")
        if quota < 1:
            raise ValueError(f"the quota must be at least one structure, got {quota}")
        if batch_size <= quota:
            raise ValueError(
                f"batch_size {batch_size} must leave room for structures outside "
                f"the quota pool, but the quota is {quota}"
            )
        pool = sorted(set(int(index) for index in quota_pool_indices))
        if not pool:
            raise ValueError(
                "the quota pool is empty: no structure carries that level's label"
            )
        if pool[0] < 0 or pool[-1] >= dataset_size:
            raise ValueError(
                f"quota pool indices {pool[0]}..{pool[-1]} fall outside the "
                f"dataset of {dataset_size} structures"
            )

        natural_share = round(batch_size * len(pool) / dataset_size)
        self.effective_quota = min(
            max(quota, natural_share), batch_size - 1, len(pool)
        )
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.requested_quota = quota
        self.seed = seed
        self.pool_indices = pool
        pool_membership = set(pool)
        self.rest_indices = [
            index for index in range(dataset_size) if index not in pool_membership
        ]
        self.rest_per_batch = batch_size - self.effective_quota
        if len(self.rest_indices) < self.rest_per_batch:
            raise ValueError(
                f"{len(self.rest_indices)} structures outside the quota pool "
                f"cannot fill a single batch, which needs {self.rest_per_batch}"
            )

        self.epoch_counter = 0
        self._pool_order: List[int] = []
        self._pool_cursor = 0
        self._pool_cycle_counter = 0

    def __len__(self) -> int:
        return len(self.rest_indices) // self.rest_per_batch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self._derive_seed("epoch", self.epoch_counter))
        self.epoch_counter += 1
        permutation = torch.randperm(len(self.rest_indices), generator=generator)
        shuffled_rest = [self.rest_indices[position] for position in permutation.tolist()]
        for batch_number in range(len(self)):
            start = batch_number * self.rest_per_batch
            batch = shuffled_rest[start : start + self.rest_per_batch]
            batch.extend(self._draw_from_pool())
            yield batch

    def _derive_seed(self, stream: str, counter: int) -> int:
        """A distinct but reproducible seed per (run seed, stream, counter)."""
        offset = 0 if stream == "epoch" else 1
        return (self.seed * 2 + offset) * 1_000_003 + counter

    def _draw_from_pool(self) -> List[int]:
        drawn: List[int] = []
        while len(drawn) < self.effective_quota:
            if self._pool_cursor >= len(self._pool_order):
                self._reshuffle_pool(excluded=set(drawn))
            available = len(self._pool_order) - self._pool_cursor
            take = min(self.effective_quota - len(drawn), available)
            drawn.extend(self._pool_order[self._pool_cursor : self._pool_cursor + take])
            self._pool_cursor += take
        return drawn

    def _reshuffle_pool(self, excluded: Sequence[int]) -> None:
        generator = torch.Generator()
        generator.manual_seed(self._derive_seed("pool", self._pool_cycle_counter))
        self._pool_cycle_counter += 1
        permutation = torch.randperm(len(self.pool_indices), generator=generator)
        order = [self.pool_indices[position] for position in permutation.tolist()]
        if excluded:
            # The cycle wrapped in the middle of a batch: the indices already
            # in that batch go to the back of the new cycle so no batch can
            # carry the same structure twice. Every pool index is still handed
            # out exactly once per cycle.
            excluded_set = set(excluded)
            order = [index for index in order if index not in excluded_set] + [
                index for index in order if index in excluded_set
            ]
        self._pool_order = order
        self._pool_cursor = 0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(batch_size={self.batch_size}, "
            f"effective_quota={self.effective_quota} "
            f"(requested {self.requested_quota}), pool={len(self.pool_indices)} of "
            f"{self.dataset_size} structures, batches_per_epoch={len(self)})"
        )

    @property
    def pool_passes_per_epoch(self) -> float:
        """How often the average pool structure is trained on per epoch."""
        return self.effective_quota * len(self) / len(self.pool_indices)
