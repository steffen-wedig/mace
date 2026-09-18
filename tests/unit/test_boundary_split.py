"""Splitting a mixed periodic/aperiodic batch by boundary type must not change a model
whose outputs do not depend on batch mates (plain MACE), for every output layout."""

import numpy as np
import torch
from e3nn import o3

from mace import data, modules, tools
from mace.modules.boundary_split import (
    forward_by_boundary,
    graph_periodicity,
    select_graphs,
)
from mace.tools import torch_geometric

torch.set_default_dtype(torch.float64)

TABLE = tools.AtomicNumberTable([1, 8])
WATER_NUMBERS = np.array([8, 1, 1])
WATER_POSITIONS = np.array([[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _model():
    model_config = dict(
        r_max=3.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([1.0, 3.0]),
        avg_num_neighbors=3,
        atomic_numbers=TABLE.zs,
        correlation=2,
        radial_type="bessel",
    )
    return modules.MACE(**model_config)


def _configurations():
    rng = np.random.default_rng(0)
    periodic_a = data.Configuration(
        atomic_numbers=np.concatenate([WATER_NUMBERS, WATER_NUMBERS]),
        positions=np.concatenate([WATER_POSITIONS + 1.0, WATER_POSITIONS + 3.5]) + 0.05 * rng.normal(size=(6, 3)),
        properties={"energy": -1.0, "forces": rng.normal(size=(6, 3))},
        property_weights={"energy": 1.0, "forces": 1.0},
        cell=6.0 * np.eye(3),
        pbc=(True, True, True),
    )
    aperiodic = data.Configuration(
        atomic_numbers=WATER_NUMBERS,
        positions=WATER_POSITIONS + 0.05 * rng.normal(size=(3, 3)),
        properties={"energy": -0.5, "forces": rng.normal(size=(3, 3))},
        property_weights={"energy": 1.0, "forces": 1.0},
    )
    single_atom = data.Configuration(
        atomic_numbers=np.array([8]),
        positions=np.zeros((1, 3)),
        properties={"energy": -0.1, "forces": np.zeros((1, 3))},
        property_weights={"energy": 1.0, "forces": 1.0},
    )
    periodic_b = data.Configuration(
        atomic_numbers=WATER_NUMBERS,
        positions=WATER_POSITIONS + 2.0,
        properties={"energy": -0.7, "forces": rng.normal(size=(3, 3))},
        property_weights={"energy": 1.0, "forces": 1.0},
        cell=5.0 * np.eye(3),
        pbc=(True, True, True),
    )
    return [periodic_a, aperiodic, single_atom, periodic_b]


def _batch():
    atomic_data = [data.AtomicData.from_config(c, z_table=TABLE, cutoff=3.0) for c in _configurations()]
    loader = torch_geometric.dataloader.DataLoader(dataset=atomic_data, batch_size=4, shuffle=False)
    return next(iter(loader))


def test_select_graphs_reindexes_consistently():
    batch = _batch()
    batch_dict = batch.to_dict()
    periodic = graph_periodicity(batch_dict)
    assert periodic.tolist() == [True, False, False, True]
    sub, graph_indices, node_indices, edge_indices = select_graphs(batch_dict, ~periodic)
    assert graph_indices.tolist() == [1, 2]
    assert node_indices.tolist() == [6, 7, 8, 9]
    assert sub["batch"].tolist() == [0, 0, 0, 1]
    assert sub["ptr"].tolist() == [0, 3, 4]
    assert sub["cell"].shape == (6, 3) and sub["pbc"].shape == (2, 3)
    assert int(sub["edge_index"].max()) < 4 and sub["edge_index"].shape[1] == edge_indices.numel()
    assert torch.equal(sub["positions"], batch_dict["positions"][node_indices])


def test_forward_by_boundary_matches_joint_forward():
    model = _model()
    batch = _batch()
    kwargs = dict(training=True, compute_force=True, compute_stress=True, compute_virials=True)
    joint = model(batch.to_dict(), **kwargs)
    split = forward_by_boundary(model, batch.to_dict(), graph_periodicity(batch.to_dict()), **kwargs)
    for key in ("energy", "forces", "stress", "virials", "node_energy"):
        assert split[key] is not None, key
        assert torch.allclose(joint[key], split[key], atol=1e-10, rtol=0), key
    assert set(joint) == set(split)
    for key, value in joint.items():
        if value is None:
            assert split[key] is None, key

    # gradients reach the parameters through the split path as well
    loss = split["energy"].sum() + split["forces"].square().sum()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
