"""Batches padded to fixed sizes, for compiled training with CUDA graphs.

A CUDA graph replays fixed shapes, so every batch is padded up to one of a few budgets
(numbers of nodes, edges, real-space pairs, reciprocal-space k-vector rows and structures),
chosen once from the training set (``plan_padding_budgets``). The padding is appended as
extra structures behind the real ones:

- the first padding structure holds the padding atoms (at least two), the padding edges
  and the padding pairs; any further padding structure (a batch with fewer structures
  than the budget) is empty;
- padding atoms sit on a cubic grid, edges and pairs run between consecutive padding
  atoms, so every separation is nonzero and every term of the model stays finite;
- periodic batches (``kspace_cutoff``): the padding structures take the cell of a real
  structure, and the k-vector rows are padded with excluded rows owned by the first padding
  structure (``mace.data.reciprocal_lattice``), so they contribute exactly zero;
- no edge or pair connects a padding atom to a real one, and every per-structure sum of
  the model (E0, charge constraint, Fukui normalisation, energies) keeps the padding
  structures apart: the real structures' outputs are those of the unpadded batch;
- the padding structures are neutral singlets with zero reference energy and forces and
  zero weights, and the batch carries ``real_node_mask`` / ``real_graph_mask`` (1 for
  real entries, 0 for padding) for the loss reductions.

Padding structures are built from a real structure of the batch (``Data`` objects with
the keys of ``mace.data.AtomicData``), so the collation treats every key as usual.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from mace.data.real_space_pairs import (
    attach_real_space_pair_index,
    real_space_pair_index,
)
from mace.data.reciprocal_lattice import (
    attach_reciprocal_lattice_indices,
    batch_kspace_rows,
)
from mace.tools.torch_geometric.batch import Batch
from mace.tools.torch_geometric.data import Data

NODE_KEYS = frozenset(
    {
        "positions",
        "node_attrs",
        "atomic_numbers",
        "forces",
        "charges",
        "magmom",
        "magforces",
        "density_coefficients",
    }
)
EDGE_KEYS = frozenset({"edge_index", "shifts", "unit_shifts"})
GRAPH_KEYS = frozenset(
    {
        "cell",
        "rcell",
        "volume",
        "pbc",
        "head",
        "energy",
        "stress",
        "virials",
        "dipole",
        "polarizability",
        "total_charge",
        "total_spin",
        "elec_temp",
        "fermi_level",
        "external_field",
        "weight",
        "energy_weight",
        "forces_weight",
        "stress_weight",
        "virials_weight",
        "dipole_weight",
        "charges_weight",
        "polarizability_weight",
        "magforces_weight",
    }
)
# Graph-level values of a padding structure: a neutral singlet without reference data.
ZEROED_GRAPH_KEYS = frozenset(
    {"energy", "stress", "virials", "dipole", "polarizability", "total_charge"}
)
# Multiplicity 2S+1 = 1 (PolarMACE uses total_spin - 1 as the spin excess).
SINGLET_MULTIPLICITY = 1.0

# Every padding batch needs at least one padding structure with two atoms: pairs and
# edges need two distinct ends.
MINIMUM_PADDING_ATOMS = 2


@dataclass(frozen=True)
class BatchShape:
    """Sizes of a collated batch, or the budget it is padded to.

    kspace_rows: k-vector rows of the reciprocal-space electrostatics (periodic batches;
    0 otherwise).
    """

    nodes: int
    edges: int
    pairs: int
    graphs: int
    kspace_rows: int = 0

    def fits_within(self, budget: "BatchShape") -> bool:
        """Whether a batch of these sizes can be padded to ``budget``.

        Padding needs one extra structure with at least two atoms.
        """
        return (
            budget.graphs > self.graphs
            and budget.nodes - self.nodes >= MINIMUM_PADDING_ATOMS
            and budget.edges >= self.edges
            and budget.pairs >= self.pairs
            and budget.kspace_rows >= self.kspace_rows
        )


class BatchExceedsPaddingBudgets(ValueError):
    """A batch larger than the padding budget it was to be padded to."""


@dataclass(frozen=True)
class PaddingPlan:
    """Budgets a batch is padded to: the first (cheapest) one it fits within.

    budgets: ascending in cost. A batch that fits none (rare: the largest budget covers
    every batch of many sampled epochs) is collated unpadded and trained eagerly.
    padding_atom_spacing: grid spacing of the padding atoms in Angstrom.
    """

    budgets: Tuple[BatchShape, ...]
    padding_atom_spacing: float = 3.0

    def budget_for(self, shape: BatchShape) -> Optional[BatchShape]:
        for budget in self.budgets:
            if shape.fits_within(budget):
                return budget
        return None


def structure_pair_count(num_nodes: int) -> int:
    """Directed real-space pairs of one open structure."""
    return num_nodes * (num_nodes - 1)


def batch_shape(
    data_list: Sequence[Data],
    real_space_pairs: bool,
    kspace_cutoff: Optional[float] = None,
) -> BatchShape:
    """Sizes of the batch ``data_list`` collates to (k-vector rows with a
    ``kspace_cutoff``)."""
    nodes = [int(data.num_nodes) for data in data_list]
    return BatchShape(
        nodes=sum(nodes),
        edges=sum(int(data["edge_index"].shape[1]) for data in data_list),
        pairs=(
            sum(structure_pair_count(count) for count in nodes)
            if real_space_pairs
            else 0
        ),
        graphs=len(data_list),
        kspace_rows=(
            batch_kspace_rows(data_list, kspace_cutoff)
            if kspace_cutoff is not None
            else 0
        ),
    )


def _grid_positions(count: int, spacing: float, dtype: torch.dtype) -> torch.Tensor:
    side = max(1, math.ceil(count ** (1.0 / 3.0)))
    index = torch.arange(count)
    grid = torch.stack(
        [index % side, (index // side) % side, index // (side * side)], dim=1
    )
    return grid.to(dtype) * spacing


def _ring(count: int, atoms: int) -> torch.Tensor:
    """``count`` directed links between consecutive atoms of ``atoms`` (>= 2), cycling."""
    sender = torch.arange(count, dtype=torch.long) % atoms
    return torch.stack([sender, (sender + 1) % atoms])


def padding_structure(
    template: Data, num_nodes: int, num_edges: int, spacing: float
) -> Data:
    """One padding structure with the keys of ``template`` (a real structure)."""
    unknown = set(template.keys) - NODE_KEYS - EDGE_KEYS - GRAPH_KEYS
    if unknown:
        raise ValueError(
            f"cannot pad batches with keys {sorted(unknown)} (known node, edge and "
            "graph keys: mace.data.padding)"
        )
    if num_edges > 0 and num_nodes < MINIMUM_PADDING_ATOMS:
        raise ValueError("padding edges need a padding structure of two atoms or more")
    padding = copy.copy(template)
    padding.num_nodes = num_nodes
    for key in template.keys:
        value = template[key]
        if key == "positions":
            padding[key] = _grid_positions(num_nodes, spacing, value.dtype)
        elif key in ("node_attrs", "atomic_numbers"):
            # One element for all padding atoms: the template's first atom.
            padding[key] = value[:1].expand(num_nodes, *value.shape[1:]).clone()
        elif key in NODE_KEYS:
            padding[key] = value.new_zeros((num_nodes, *value.shape[1:]))
        elif key == "edge_index":
            padding[key] = _ring(num_edges, max(num_nodes, 1))
        elif key in EDGE_KEYS:
            padding[key] = value.new_zeros((num_edges, *value.shape[1:]))
        elif key in ZEROED_GRAPH_KEYS or key.endswith("weight"):
            padding[key] = torch.zeros_like(value)
        elif key == "total_spin":
            padding[key] = torch.full_like(value, SINGLET_MULTIPLICITY)
        else:
            padding[key] = value.clone()
    return padding


def padding_structures(
    template: Data, shape: BatchShape, budget: BatchShape, spacing: float
) -> List[Data]:
    """Structures that pad a batch of ``shape`` to ``budget`` nodes, edges and graphs."""
    if not shape.fits_within(budget):
        raise BatchExceedsPaddingBudgets(f"batch {shape} does not fit {budget}")
    first = padding_structure(
        template,
        num_nodes=budget.nodes - shape.nodes,
        num_edges=budget.edges - shape.edges,
        spacing=spacing,
    )
    empty = [
        padding_structure(template, num_nodes=0, num_edges=0, spacing=spacing)
        for _ in range(budget.graphs - shape.graphs - 1)
    ]
    return [first, *empty]


def collate_padded(
    data_list: Sequence[Data],
    plan: PaddingPlan,
    real_space_pairs: bool,
    kspace_cutoff: Optional[float] = None,
    follow_batch=None,
    exclude_keys=None,
) -> Batch:
    """Collate ``data_list`` padded to the plan's budget for it (unpadded if none fits).

    The batch carries ``real_node_mask`` / ``real_graph_mask`` (1 for real entries) and,
    with ``real_space_pairs``, ``real_space_pair_index``: the pairs of the real
    structures followed by padding pairs within the first padding structure; with a
    ``kspace_cutoff`` (fully periodic structures), the reciprocal-lattice indices of the
    real structures followed by excluded rows of the first padding structure.
    """
    if real_space_pairs and kspace_cutoff is not None:
        raise ValueError("a batch has real-space pairs or k-vectors, not both")
    shape = batch_shape(data_list, real_space_pairs, kspace_cutoff)
    budget = plan.budget_for(shape)
    if budget is None:
        # Larger than every budget: unpadded, without masks (trained eagerly).
        batch = Batch.from_data_list(
            list(data_list),
            follow_batch=follow_batch or [],
            exclude_keys=exclude_keys or [],
        )
        if real_space_pairs:
            attach_real_space_pair_index(batch)
        if kspace_cutoff is not None:
            attach_reciprocal_lattice_indices(batch, kspace_cutoff)
        return batch
    padding = padding_structures(
        data_list[0], shape, budget, plan.padding_atom_spacing
    )
    batch = Batch.from_data_list(
        [*data_list, *padding],
        follow_batch=follow_batch or [],
        exclude_keys=exclude_keys or [],
    )
    dtype = batch.positions.dtype
    batch.real_node_mask = (torch.arange(budget.nodes) < shape.nodes).to(dtype)
    batch.real_graph_mask = (torch.arange(budget.graphs) < shape.graphs).to(dtype)
    if real_space_pairs:
        real_pairs = real_space_pair_index(
            batch.ptr[: shape.graphs + 1], batch.pbc.reshape(-1, 3)[: shape.graphs]
        )
        padding_pairs = shape.nodes + _ring(
            budget.pairs - shape.pairs, budget.nodes - shape.nodes
        )
        batch.real_space_pair_index = torch.cat([real_pairs, padding_pairs], dim=1)
    if kspace_cutoff is not None:
        attach_reciprocal_lattice_indices(
            batch, kspace_cutoff, rows=budget.kspace_rows, real_graphs=shape.graphs
        )
    return batch


# ---------------------------------------------------------------------------------------
# Budgets from the training set
# ---------------------------------------------------------------------------------------


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _with_padding_room(shape: BatchShape, multiple: int) -> BatchShape:
    """The smallest rounded budget a batch of ``shape`` fits within (one padding graph)."""
    return BatchShape(
        nodes=_round_up(shape.nodes + MINIMUM_PADDING_ATOMS, multiple),
        edges=_round_up(max(shape.edges, 1), multiple),
        pairs=_round_up(max(shape.pairs, 1), multiple),
        graphs=shape.graphs + 1,
        kspace_rows=(
            _round_up(shape.kspace_rows, multiple) if shape.kspace_rows > 0 else 0
        ),
    )


@dataclass(frozen=True)
class StepCostModel:
    """Relative cost of a padded step per node, edge, pair and k-vector row (fitted or
    assumed)."""

    per_node: float
    per_edge: float
    per_pair: float
    per_kspace_row: float = 0.0

    @property
    def weights(self) -> np.ndarray:
        """Weights of the size columns (nodes, edges, pairs, k-vector rows)."""
        return np.array(
            [self.per_node, self.per_edge, self.per_pair, self.per_kspace_row]
        )

    def cost(self, shape: BatchShape) -> float:
        return (
            self.per_node * shape.nodes
            + self.per_edge * shape.edges
            + self.per_pair * shape.pairs
            + self.per_kspace_row * shape.kspace_rows
        )

    def costs(self, sizes: np.ndarray) -> np.ndarray:
        """Costs of rows (nodes, edges, pairs[, k-vector rows])."""
        return sizes @ self.weights[: sizes.shape[-1]]


def sample_batch_shapes(
    structure_nodes: np.ndarray,
    structure_edges: np.ndarray,
    batch_size: int,
    epochs: int,
    real_space_pairs: bool,
    seed: int = 0,
    structure_kspace_rows: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sizes [n_batches, 4] (nodes, edges, pairs, k-vector rows) of shuffled, drop-last
    batches."""
    generator = np.random.default_rng(seed)
    pairs = (
        structure_nodes * (structure_nodes - 1)
        if real_space_pairs
        else np.zeros_like(structure_nodes)
    )
    kspace_rows = (
        np.asarray(structure_kspace_rows, dtype=np.int64)
        if structure_kspace_rows is not None
        else np.zeros_like(structure_nodes)
    )
    per_structure = np.stack(
        [structure_nodes, structure_edges, pairs, kspace_rows], axis=1
    )
    shapes = []
    batches_per_epoch = len(structure_nodes) // batch_size
    for _ in range(epochs):
        order = generator.permutation(len(structure_nodes))
        members = order[: batches_per_epoch * batch_size].reshape(-1, batch_size)
        shapes.append(per_structure[members].sum(axis=1))
    return np.concatenate(shapes, axis=0)


def plan_padding_budgets(
    structure_nodes: Sequence[int],
    structure_edges: Sequence[int],
    batch_size: int,
    bucket_count: int,
    cost_model: StepCostModel,
    real_space_pairs: bool,
    structure_kspace_rows: Optional[Sequence[int]] = None,
    sampled_epochs: int = 20,
    coverage_epochs: int = 500,
    multiple: int = 32,
    seed: int = 0,
    padding_atom_spacing: float = 3.0,
) -> PaddingPlan:
    """``bucket_count`` budgets for random batches of the training set.

    ``structure_kspace_rows``: k-vector rows of each structure (periodic training sets,
    ``mace.data.reciprocal_lattice.structure_kspace_rows``); None for none.

    Samples the batch sizes of ``sampled_epochs`` shuffled epochs, sorts them by step
    cost and splits them into ``bucket_count`` groups whose boundaries minimise the
    expected padded cost (dynamic programming over cost-sorted batches; each group's
    budget is the elementwise maximum of its members). The largest budget is then
    widened to the elementwise maximum of ``coverage_epochs`` sampled epochs, so that
    a batch fitting no budget (trained unpadded) is rare.
    """
    if bucket_count < 1:
        raise ValueError("bucket_count must be at least 1")
    nodes = np.asarray(structure_nodes, dtype=np.int64)
    edges = np.asarray(structure_edges, dtype=np.int64)
    if len(nodes) < batch_size:
        raise ValueError("fewer structures than one batch")
    kspace_rows = (
        np.asarray(structure_kspace_rows, dtype=np.int64)
        if structure_kspace_rows is not None
        else None
    )
    if kspace_rows is not None and len(kspace_rows) != len(nodes):
        raise ValueError("one k-vector row count per structure")
    sampled = sample_batch_shapes(
        nodes, edges, batch_size, sampled_epochs, real_space_pairs, seed, kspace_rows
    )

    def shape_of(row: np.ndarray) -> BatchShape:
        return BatchShape(
            int(row[0]), int(row[1]), int(row[2]), batch_size, kspace_rows=int(row[3])
        )

    envelopes = [
        shape_of(row)
        for row in _cost_optimal_envelopes(sampled, bucket_count, cost_model)
    ]
    coverage = sample_batch_shapes(
        nodes,
        edges,
        batch_size,
        coverage_epochs,
        real_space_pairs,
        seed + 1,
        kspace_rows,
    ).max(axis=0)
    largest = envelopes[-1]
    envelopes[-1] = shape_of(
        np.maximum(
            [largest.nodes, largest.edges, largest.pairs, largest.kspace_rows], coverage
        )
    )
    budgets = sorted(
        {_with_padding_room(envelope, multiple) for envelope in envelopes},
        key=cost_model.cost,
    )
    return PaddingPlan(tuple(budgets), padding_atom_spacing)


def _cost_optimal_envelopes(
    sizes: np.ndarray, group_count: int, cost_model: StepCostModel
) -> np.ndarray:
    """Envelopes [groups, 4] of contiguous groups of cost-sorted ``sizes`` [n, 4]
    (nodes, edges, pairs, k-vector rows; three columns: no k-vector rows).

    Every row is padded to its group's envelope (elementwise maximum); the groups
    minimise the summed padded cost. Groups are contiguous in cost order, which is exact
    for one size dimension and a heuristic for several. Dynamic programming over distinct
    rows (with multiplicities): O(n^2) envelope costs, O(groups n^2) for the splits.
    """
    unique, multiplicity = np.unique(sizes, axis=0, return_counts=True)
    order = np.argsort(cost_model.costs(unique.astype(float)), kind="stable")
    unique, multiplicity = unique[order], multiplicity[order]
    count = len(unique)
    group_count = min(group_count, count)
    # envelope_cost[start, end - 1]: padded cost of rows start..end-1 together.
    envelope_cost = np.full((count, count), np.inf)
    cumulative_multiplicity = np.concatenate([[0], np.cumsum(multiplicity)])
    for start in range(count):
        running_maximum = np.maximum.accumulate(unique[start:], axis=0)
        members = cumulative_multiplicity[start + 1 :] - cumulative_multiplicity[start]
        envelope_cost[start, start:] = cost_model.costs(running_maximum) * members
    best = np.full((group_count + 1, count + 1), np.inf)
    split = np.zeros((group_count + 1, count + 1), dtype=np.int64)
    best[0, 0] = 0.0
    for groups in range(1, group_count + 1):
        for end in range(1, count + 1):
            candidates = best[groups - 1, :end] + envelope_cost[:end, end - 1]
            start = int(np.argmin(candidates))
            best[groups, end] = candidates[start]
            split[groups, end] = start
    # Fewer groups can be as good (e.g. identical rows): take the best group count.
    groups = int(np.argmin(best[1:, count])) + 1
    boundaries = []
    end = count
    for level in range(groups, 0, -1):
        start = int(split[level, end])
        boundaries.append((start, end))
        end = start
    return np.array(
        [unique[start:end].max(axis=0) for start, end in reversed(boundaries)]
    )


__all__ = [
    "BatchExceedsPaddingBudgets",
    "BatchShape",
    "PaddingPlan",
    "StepCostModel",
    "batch_shape",
    "collate_padded",
    "padding_structure",
    "padding_structures",
    "plan_padding_budgets",
    "sample_batch_shapes",
]
