"""PolarMACE prepares one real-space geometry per forward and evaluates both spin channels
in one call; neither changes the result.

The pair kernels of the open-boundary electrostatics depend only on the positions. The
forward evaluates them once, for the union of the combined widths of the field features
and the Coulomb energy, and every feature evaluation (each recursion step, both spin
channels) and the energy read that one geometry. The reference evaluates every channel
and the energy on its own, as before.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

from mace import data
from mace.modules import interaction_classes
from mace.modules.boundary_split import select_graphs
from mace.tools import torch_geometric, utils
from tests.helpers import GRAPH_LONGRANGE_AVAILABLE

pytestmark = [
    pytest.mark.polar,
    pytest.mark.skipif(
        not GRAPH_LONGRANGE_AVAILABLE, reason="graph_longrange is not installed"
    ),
]

ATOMIC_NUMBERS = [1, 6, 8]


def _polar_model(dtype=torch.float64):
    from mace.modules.extensions import PolarMACE  # noqa: PLC0415

    torch.manual_seed(3)
    return PolarMACE(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=interaction_classes[
            "RealAgnosticResidualNonLinearInteractionBlock"
        ],
        interaction_cls_first=interaction_classes[
            "RealAgnosticResidualNonLinearInteractionBlock"
        ],
        num_interactions=2,
        num_elements=len(ATOMIC_NUMBERS),
        hidden_irreps=o3.Irreps("4x0e + 4x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=torch.zeros(len(ATOMIC_NUMBERS), dtype=dtype),
        avg_num_neighbors=3.0,
        atomic_numbers=ATOMIC_NUMBERS,
        correlation=1,
        gate=torch.nn.functional.silu,
        radial_MLP=[16, 16],
        radial_type="bessel",
        kspace_cutoff_factor=1.0,
        # POLAR-M widths: the energy's combined width sqrt(2) * 1.5 equals the first
        # feature width sqrt(1.5^2 + 1.5^2), so two kernel columns serve all.
        atomic_multipoles_max_l=1,
        atomic_multipoles_smearing_width=1.5,
        field_feature_max_l=1,
        field_feature_widths=[1.5, 3.0],
        field_feature_norms=[2.0, 2.0, 0.5, 0.5],
        num_recursion_steps=2,
        field_si=False,
        include_electrostatic_self_interaction=True,
        add_local_electron_energy=True,
        return_electrostatic_potentials=False,
        heads=["Default"],
        field_norm_factor=1.0,
        fixedpoint_update_config={
            "type": "AgnosticEmbeddedOneBodyVariableUpdate",
            "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding",
            "nonlinearity_cls": "MLPNonLinearity",
        },
        field_readout_config={"type": "OneBodyMLPFieldReadout"},
    ).to(dtype=dtype)


def _molecules() -> list[Atoms]:
    rng = np.random.default_rng(7)
    water = Atoms(
        numbers=[8, 1, 1],
        positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]],
    )
    methanol = Atoms(
        numbers=[6, 8, 1, 1, 1, 1],
        positions=[
            [0.0, 0.0, 0.0],
            [1.42, 0.0, 0.0],
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89],
            [1.74, 0.0, 0.92],
        ],
    )
    dimer = Atoms(numbers=[8, 1, 1, 8, 1, 1])
    dimer.positions = np.concatenate(
        [water.positions, water.positions + np.array([2.9, 0.3, 0.2])]
    )
    molecules = [water, methanol, dimer]
    for atoms in molecules:
        atoms.positions = atoms.positions + 0.05 * rng.standard_normal(
            atoms.positions.shape
        )
    molecules[1].info["charge"] = 1.0
    return molecules


def _batch(molecules: list[Atoms], dtype=torch.float64) -> dict:
    z_table = utils.AtomicNumberTable(ATOMIC_NUMBERS)
    keyspec = data.KeySpecification(
        info_keys={"total_charge": "charge", "total_spin": "spin"}, arrays_keys={}
    )
    dataset = [
        data.AtomicData.from_config(
            data.config_from_atoms(atoms, key_specification=keyspec, head_name="Default"),
            z_table=z_table,
            cutoff=4.0,
            heads=["Default"],
        )
        for atoms in molecules
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=len(dataset), shuffle=False, drop_last=False
    )
    batch = next(iter(loader)).to_dict()
    return {
        key: value.to(dtype)
        if torch.is_tensor(value) and value.dtype.is_floating_point
        else value
        for key, value in batch.items()
    }


def _host_pair_index(ptr: torch.Tensor, order_seed: int) -> torch.Tensor:
    """Directed pairs within each structure, built from ptr on the host, in an order
    unrelated to the device builder's."""
    pairs = []
    for start, stop in zip(ptr[:-1].tolist(), ptr[1:].tolist()):
        nodes = torch.arange(start, stop)
        sender, receiver = torch.meshgrid(nodes, nodes, indexing="ij")
        keep = sender != receiver
        pairs.append(torch.stack([sender[keep], receiver[keep]]))
    pair_index = torch.cat(pairs, dim=1)
    permutation = torch.randperm(
        pair_index.shape[1], generator=torch.Generator().manual_seed(order_seed)
    )
    return pair_index[:, permutation]


def _clone(batch: dict) -> dict:
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _outputs_and_gradients(model, batch: dict) -> dict:
    """Energy, forces and the parameter gradients of a force-dependent loss (the
    backward through the forces, as in training)."""
    model.zero_grad(set_to_none=True)
    output = model(_clone(batch), training=True, compute_force=True)
    loss = (output["energy"] ** 2).sum() + (output["forces"] ** 2).sum()
    loss.backward()
    return {
        "energy": output["energy"].detach(),
        "forces": output["forces"].detach(),
        "charges": output["charges"].detach(),
        **{
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        },
    }


def _evaluate_every_channel_on_its_own(model, monkeypatch) -> None:
    """The previous evaluation: no shared geometry (features and energy each prepare
    their own) and one feature evaluation per spin channel."""
    descriptor = model.electric_potential_descriptor
    forward_dynamic = descriptor.forward_dynamic

    def per_channel(cache, source_feats):
        assert source_feats.dim() == 3
        return torch.stack(
            [
                forward_dynamic(cache=cache, source_feats=source_feats[:, channel])
                for channel in range(source_feats.shape[1])
            ],
            dim=1,
        )

    monkeypatch.setattr(model, "prepare_real_space_geometry", lambda *_, **__: None)
    monkeypatch.setattr(descriptor, "forward_dynamic", per_channel)


def _assert_all_close(actual: dict, expected: dict, tolerance: float) -> None:
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        scale = max(value.abs().max().item(), 1.0)
        difference = (actual[key] - value).abs().max().item()
        assert difference <= tolerance * scale, (key, difference, scale)


@pytest.mark.parametrize("pbc_handling", ["auto", "realspace"])
def test_shared_geometry_matches_separate_evaluation(pbc_handling, monkeypatch):
    model = _polar_model()
    model.set_electrostatic_pbcs(pbc_handling)
    batch = _batch(_molecules())
    shared = _outputs_and_gradients(model, batch)
    _evaluate_every_channel_on_its_own(model, monkeypatch)
    separate = _outputs_and_gradients(model, batch)
    _assert_all_close(shared, separate, tolerance=1e-12)


def test_one_geometry_per_forward(monkeypatch):
    """One preparation per forward, with both kernel widths; the blocks never prepare
    their own."""
    from graph_longrange.realspace_electrostatics import (  # noqa: PLC0415
        RealSpaceAnalyticalElectrostaticFeatures,
        RealSpaceAnalyticalEnergy,
    )

    from mace.modules import extensions  # noqa: PLC0415

    model = _polar_model()
    prepared = []
    prepare = extensions.prepare_real_space_geometry

    def counting_prepare(*args, **kwargs):
        geometry = prepare(*args, **kwargs)
        prepared.append(geometry)
        return geometry

    def refuse(*_, **__):
        raise AssertionError("a block prepared its own geometry")

    monkeypatch.setattr(extensions, "prepare_real_space_geometry", counting_prepare)
    monkeypatch.setattr(RealSpaceAnalyticalElectrostaticFeatures, "prepare", refuse)
    monkeypatch.setattr(RealSpaceAnalyticalEnergy, "prepare", refuse)
    model(_clone(_batch(_molecules())), training=True, compute_force=True)
    assert len(prepared) == 1
    assert prepared[0].kernel_widths == pytest.approx((2.0**0.5 * 1.5, 45.0**0.5 / 2))


def test_pair_index_from_the_batch_matches_device_pairs():
    model = _polar_model()
    batch = _batch(_molecules())
    device_pairs = _outputs_and_gradients(model, batch)
    batch["real_space_pair_index"] = _host_pair_index(batch["ptr"], order_seed=1)
    host_pairs = _outputs_and_gradients(model, batch)
    _assert_all_close(host_pairs, device_pairs, tolerance=1e-12)


def test_split_by_boundary_keeps_the_pairs_of_the_selected_graphs():
    """Two graphs, so the pair list's leading 2 equals the graph count: it must still be
    split by node, not taken as one row per graph."""
    molecules = _molecules()[:2]
    batch = _batch(molecules)
    batch["real_space_pair_index"] = _host_pair_index(batch["ptr"], order_seed=2)
    second_graph = torch.tensor([False, True])
    sub, _, node_indices, _ = select_graphs(batch, second_graph)
    expected = _host_pair_index(sub["ptr"], order_seed=0)
    selected = sub["real_space_pair_index"]
    assert node_indices.tolist() == list(range(3, 9))
    assert selected.shape == expected.shape
    assert set(map(tuple, selected.t().tolist())) == set(
        map(tuple, expected.t().tolist())
    )


def test_split_by_boundary_with_pair_index_matches_joint_forward():
    """A mixed batch evaluated by boundary type, with a pair list for its open graphs,
    gives the open graphs the same result as a batch of open graphs alone."""
    molecules = _molecules()
    periodic = molecules[0].copy()
    periodic.cell = np.eye(3) * 8.0
    periodic.pbc = True
    model = _polar_model()
    open_batch = _batch(molecules[1:])
    open_batch["real_space_pair_index"] = _host_pair_index(open_batch["ptr"], 3)
    open_only = model(_clone(open_batch), training=False, compute_force=True)

    mixed_batch = _batch([periodic, *molecules[1:]])
    open_ptr = mixed_batch["ptr"][1:]
    mixed_batch["real_space_pair_index"] = _host_pair_index(open_ptr, 4)
    mixed = model(_clone(mixed_batch), training=False, compute_force=True)
    assert torch.allclose(mixed["energy"][1:], open_only["energy"], atol=1e-10)
    assert torch.allclose(mixed["forces"][3:], open_only["forces"], atol=1e-10)
