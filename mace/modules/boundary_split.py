"""Evaluate a batch that mixes periodic and aperiodic graphs as two homogeneous batches.

PolarMACE picks its electrostatics evaluator from the boundary conditions of the *batch*
(``pbc_handling="auto"``: reciprocal space as soon as any graph is periodic). A gas-phase
molecule that shares a batch with a periodic frame is then evaluated inside the bounding
cell mace fabricates for it, with a Makov-Payne correction whose residual is a q^2
finite-size error of tens of meV for a charged fragment -- while the same molecule alone
in a batch (as at inference) takes the real-space path. Training data that mixes both
kinds of structures in one batch (bulk frames plus monomers, or cluster records) would
otherwise teach the model a batch-dependent energy.

:func:`forward_by_boundary` splits such a batch by graph into its periodic and its
aperiodic part, runs the model on each, and merges the outputs back into the original
graph/node/edge order, so every structure is evaluated exactly as it would be on its own.
Gradients flow through the indexing, so forces, stress and the training loss are unchanged
in form.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

# Batch entries with one row per graph (mace.data.AtomicData plus the cluster-record
# fields); everything else per-node or per-edge is recognised by its first dimension.
PER_GRAPH_KEYS = frozenset(
    {
        "weight",
        "head",
        "energy_weight",
        "forces_weight",
        "stress_weight",
        "virials_weight",
        "dipole_weight",
        "charges_weight",
        "polarizability_weight",
        "magforces_weight",
        "energy",
        "stress",
        "virials",
        "dipole",
        "polarizability",
        "elec_temp",
        "total_charge",
        "total_spin",
        "pbc",
        "volume",
        "fermi_level",
        "external_field",
        "sign",
        "cluster_id",
        "interaction_weight",
    }
)
PER_NODE_KEYS = frozenset(
    {
        "positions",
        "node_attrs",
        "forces",
        "charges",
        "magmom",
        "magforces",
        "density_coefficients",
        "slot",
    }
)
PER_EDGE_KEYS = frozenset({"shifts", "unit_shifts"})
# 3x3 blocks stacked along the first dimension, one per graph.
STACKED_CELL_KEYS = frozenset({"cell", "rcell"})
# Model outputs with one row per graph (the rest is per node or per edge).
PER_GRAPH_OUTPUT_KEYS = frozenset(
    {
        "energy",
        "interaction_energy",
        "virials",
        "stress",
        "displacement",
        "dipole",
        "total_charge",
        "fermi_level",
        "external_field",
        "electrostatic_energy",
        "electron_energy",
    }
)


def graph_periodicity(data: Dict[str, torch.Tensor]) -> torch.Tensor:
    """``[n_graphs]`` bool: whether each graph is periodic along any direction."""
    return data["pbc"].view(-1, 3).any(dim=-1)


def select_graphs(
    data: Dict[str, torch.Tensor], graph_mask: torch.Tensor
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """The sub-batch of the graphs where ``graph_mask`` is True.

    Returns the sub-batch dict and the original graph, node and edge indices it consists
    of (for merging outputs back). Keys the model does not need and whose layout cannot be
    inferred are dropped from the sub-batch.
    """
    batch = data["batch"]
    ptr = data["ptr"]
    num_graphs = ptr.numel() - 1
    num_nodes = batch.numel()
    edge_index = data["edge_index"]
    num_edges = edge_index.shape[1]

    graph_indices = torch.nonzero(graph_mask, as_tuple=False).view(-1)
    node_mask = graph_mask[batch]
    node_indices = torch.nonzero(node_mask, as_tuple=False).view(-1)
    edge_mask = node_mask[edge_index[0]]
    edge_indices = torch.nonzero(edge_mask, as_tuple=False).view(-1)

    new_node_of_old = torch.full((num_nodes,), -1, dtype=torch.long, device=batch.device)
    new_node_of_old[node_indices] = torch.arange(node_indices.numel(), device=batch.device)
    new_graph_of_old = torch.full((num_graphs,), -1, dtype=torch.long, device=batch.device)
    new_graph_of_old[graph_indices] = torch.arange(graph_indices.numel(), device=batch.device)

    sub: Dict[str, torch.Tensor] = {}
    sub["batch"] = new_graph_of_old[batch[node_indices]]
    counts = torch.bincount(sub["batch"], minlength=graph_indices.numel())
    sub["ptr"] = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
    sub["edge_index"] = new_node_of_old[edge_index[:, edge_indices]]

    for key, value in data.items():
        if key in ("batch", "ptr", "edge_index"):
            continue
        if not isinstance(value, torch.Tensor):
            sub[key] = value
            continue
        if key in STACKED_CELL_KEYS:
            sub[key] = value.view(-1, 3, 3)[graph_indices].reshape(-1, 3)
        elif key in PER_GRAPH_KEYS or (
            key not in PER_NODE_KEYS
            and key not in PER_EDGE_KEYS
            and value.dim() > 0
            and value.shape[0] == num_graphs
            and value.shape[0] not in (num_nodes, num_edges)
        ):
            sub[key] = value[graph_indices]
        elif key in PER_NODE_KEYS or (value.dim() > 0 and value.shape[0] == num_nodes):
            sub[key] = value[node_indices]
        elif key in PER_EDGE_KEYS or (value.dim() > 0 and value.shape[0] == num_edges):
            sub[key] = value[edge_indices]
        # anything else (e.g. per-cluster bookkeeping) is not needed by the model forward
    return sub, graph_indices, node_indices, edge_indices


def merge_outputs(
    parts: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Optional[torch.Tensor]]]],
    num_graphs: int,
    num_nodes: int,
    num_edges: int,
) -> Dict[str, Optional[torch.Tensor]]:
    """Scatter the sub-batch outputs back into the original graph/node/edge order."""
    merged: Dict[str, Optional[torch.Tensor]] = {}
    keys: List[str] = []
    for _, _, _, output in parts:
        for key in output:
            if key not in keys:
                keys.append(key)
    for key in keys:
        values = [(part, part[3].get(key)) for part in parts]
        if all(value is None for _, value in values):
            merged[key] = None
            continue
        if any(value is None for _, value in values):
            raise ValueError(f"output {key!r} is present for one boundary type only")
        template = values[0][1]
        first = None
        for (graph_indices, node_indices, edge_indices, _), value in values:
            sub_graphs = graph_indices.numel()
            sub_nodes = node_indices.numel()
            sub_edges = edge_indices.numel()
            if key in PER_GRAPH_OUTPUT_KEYS or (
                value.shape[0] == sub_graphs and value.shape[0] not in (sub_nodes, sub_edges)
            ):
                target_size, target_indices = num_graphs, graph_indices
            elif value.shape[0] == sub_nodes:
                target_size, target_indices = num_nodes, node_indices
            elif value.shape[0] == sub_edges:
                target_size, target_indices = num_edges, edge_indices
            else:
                raise NotImplementedError(
                    f"cannot merge output {key!r} of shape {tuple(value.shape)} across "
                    "boundary types"
                )
            if first is None:
                first = template.new_zeros((target_size, *template.shape[1:]))
            first = first.index_copy(0, target_indices, value.to(first.dtype))
        merged[key] = first
    return merged


def forward_by_boundary(
    forward: Callable[..., Dict[str, Optional[torch.Tensor]]],
    data: Dict[str, torch.Tensor],
    graph_is_periodic: torch.Tensor,
    **kwargs,
) -> Dict[str, Optional[torch.Tensor]]:
    """Run ``forward`` separately on the periodic and the aperiodic graphs of ``data``."""
    num_graphs = data["ptr"].numel() - 1
    num_nodes = data["batch"].numel()
    num_edges = data["edge_index"].shape[1]
    parts = []
    for mask in (graph_is_periodic, ~graph_is_periodic):
        if not bool(mask.any()):
            continue
        sub, graph_indices, node_indices, edge_indices = select_graphs(data, mask)
        parts.append((graph_indices, node_indices, edge_indices, forward(sub, **kwargs)))
    return merge_outputs(parts, num_graphs, num_nodes, num_edges)


__all__ = [
    "PER_GRAPH_KEYS",
    "PER_NODE_KEYS",
    "PER_EDGE_KEYS",
    "PER_GRAPH_OUTPUT_KEYS",
    "graph_periodicity",
    "select_graphs",
    "merge_outputs",
    "forward_by_boundary",
]
