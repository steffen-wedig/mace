"""Batches padded to fixed sizes (mace.data.padding) for CUDA-graph training.

The padding must leave the real structures' predictions, the loss and its parameter
gradients exactly as for the unpadded batch, keep every padding term finite, and the
budget planner must cover every batch of the training set.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from ase import Atoms

from mace import data, modules
from mace.data.padding import (
    BatchShape,
    PaddingPlan,
    StepCostModel,
    _cost_optimal_envelopes,
    batch_shape,
    collate_padded,
    plan_padding_budgets,
    sample_batch_shapes,
)
from mace.data.real_space_pairs import attach_real_space_pair_index
from mace.tools import compile as mace_compile
from mace.tools import torch_geometric, utils
from mace.tools.compiled.training import (
    CompiledForceTrainingLoss,
    configure_dynamo_for_training,
)
from tests.extensions.polar.test_polar_shared_geometry import (
    ATOMIC_NUMBERS,
    _polar_model,
)
from tests.helpers import GRAPH_LONGRANGE_AVAILABLE

WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
FORWARD_KWARGS = {"training": True, "compute_force": True}

requires_graph_longrange = pytest.mark.skipif(
    not GRAPH_LONGRANGE_AVAILABLE, reason="graph_longrange is not installed"
)


def _water_clusters(sizes, seed):
    rng = np.random.default_rng(seed)
    keyspec = data.KeySpecification(
        info_keys={"energy": "REF_energy"}, arrays_keys={"forces": "REF_forces"}
    )
    structures = []
    for size in sizes:
        positions = np.concatenate(
            [WATER + 3.1 * index * np.array([1.0, 0.3, 0.1]) for index in range(size)]
        )
        atoms = Atoms(
            numbers=[8, 1, 1] * size,
            positions=positions + rng.normal(0, 0.05, positions.shape),
        )
        atoms.info["REF_energy"] = float(rng.normal())
        atoms.new_array("REF_forces", rng.normal(size=atoms.positions.shape))
        structures.append(
            data.AtomicData.from_config(
                data.config_from_atoms(atoms, key_specification=keyspec, head_name="Default"),
                z_table=utils.AtomicNumberTable(ATOMIC_NUMBERS),
                cutoff=4.0,
                heads=["Default"],
            )
        )
    return structures


def _to_float64(batch):
    for key in batch.keys:
        value = batch[key]
        if torch.is_tensor(value) and value.is_floating_point():
            batch[key] = value.to(torch.float64)
    return batch


def _unpadded(structures):
    batch = next(
        iter(
            torch_geometric.dataloader.DataLoader(
                structures, batch_size=len(structures)
            )
        )
    )
    return _to_float64(attach_real_space_pair_index(batch))


def _padded(structures, budget):
    return _to_float64(
        collate_padded(structures, PaddingPlan((budget,)), real_space_pairs=True)
    )


def _roomy_budget(structures, extra_graphs=1):
    shape = batch_shape(structures, real_space_pairs=True)
    return BatchShape(
        nodes=shape.nodes + 7,
        edges=shape.edges + 13,
        pairs=shape.pairs + 29,
        graphs=shape.graphs + extra_graphs,
    )


# ---------------------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("extra_graphs", [1, 3])
def test_padded_batch_has_the_budget_shape_and_keeps_padding_apart(extra_graphs):
    structures = _water_clusters([3, 2, 4], seed=0)
    budget = _roomy_budget(structures, extra_graphs)
    shape = batch_shape(structures, real_space_pairs=True)
    padded = _padded(structures, budget)
    unpadded = _unpadded(structures)

    assert padded.positions.shape[0] == budget.nodes
    assert padded.edge_index.shape[1] == budget.edges
    assert padded.real_space_pair_index.shape[1] == budget.pairs
    assert padded.num_graphs == budget.graphs
    assert padded.real_node_mask.sum().item() == shape.nodes
    assert padded.real_graph_mask.sum().item() == shape.graphs

    # The real part is the unpadded batch.
    real_pairs = padded.real_space_pair_index[:, : shape.pairs]
    assert torch.equal(real_pairs, unpadded.real_space_pair_index)
    assert torch.equal(padded.edge_index[:, : shape.edges], unpadded.edge_index)
    assert torch.equal(padded.positions[: shape.nodes], unpadded.positions)

    # Padding edges and pairs stay within the first padding structure, between
    # distinct atoms at nonzero separation.
    for links in (
        padded.edge_index[:, shape.edges :],
        padded.real_space_pair_index[:, shape.pairs :],
    ):
        assert bool((links >= shape.nodes).all())
        assert bool((padded.batch[links] == shape.graphs).all())
        assert bool((links[0] != links[1]).all())
        separation = padded.positions[links[1]] - padded.positions[links[0]]
        assert bool((separation.norm(dim=-1) > 0).all())
    graph_count = torch.bincount(padded.batch, minlength=budget.graphs)
    assert bool((graph_count[shape.graphs + 1 :] == 0).all())

    # Padding structures: neutral singlets without reference data or weight.
    padding_graphs = slice(shape.graphs, None)
    assert bool((padded.energy[padding_graphs] == 0).all())
    assert bool((padded.total_charge[padding_graphs] == 0).all())
    assert bool((padded.total_spin[padding_graphs] == 1).all())
    assert bool((padded.weight[padding_graphs] == 0).all())
    assert bool((padded.forces[shape.nodes :] == 0).all())


def test_batch_larger_than_every_budget_is_collated_unpadded():
    structures = _water_clusters([3, 2], seed=0)
    shape = batch_shape(structures, real_space_pairs=True)
    no_room = BatchShape(shape.nodes + 1, shape.edges, shape.pairs, shape.graphs + 1)
    batch = collate_padded(structures, PaddingPlan((no_room,)), real_space_pairs=True)
    assert batch["real_graph_mask"] is None
    assert batch.num_graphs == shape.graphs
    assert torch.equal(
        batch.real_space_pair_index, _unpadded(structures).real_space_pair_index
    )


def test_plan_picks_the_first_budget_that_fits():
    small = BatchShape(10, 10, 10, 3)
    large = BatchShape(100, 100, 100, 3)
    plan = PaddingPlan((small, large))
    assert plan.budget_for(BatchShape(8, 10, 10, 2)) == small
    assert plan.budget_for(BatchShape(9, 10, 10, 2)) == large  # needs two padding atoms
    assert plan.budget_for(BatchShape(8, 11, 10, 2)) == large
    assert plan.budget_for(BatchShape(8, 101, 10, 2)) is None


# ---------------------------------------------------------------------------------------
# Budget planning
# ---------------------------------------------------------------------------------------


def test_planned_budgets_cover_every_sampled_batch():
    generator = np.random.default_rng(3)
    nodes = generator.integers(10, 200, size=120)
    edges = nodes * generator.integers(8, 20, size=120)
    cost = StepCostModel(per_node=1.0, per_edge=0.05, per_pair=0.002)
    plan = plan_padding_budgets(
        nodes, edges, batch_size=8, bucket_count=4, cost_model=cost,
        real_space_pairs=True, sampled_epochs=5, coverage_epochs=50,
    )
    assert 1 <= len(plan.budgets) <= 4
    costs = [cost.cost(budget) for budget in plan.budgets]
    assert costs == sorted(costs)
    for seed in (0, 1):  # the planned sample and the coverage sample
        for row in sample_batch_shapes(nodes, edges, 8, 5, True, seed=seed):
            shape = BatchShape(int(row[0]), int(row[1]), int(row[2]), 8)
            assert plan.budget_for(shape) is not None


def test_cost_optimal_envelopes_match_brute_force():
    generator = np.random.default_rng(5)
    cost = StepCostModel(per_node=1.0, per_edge=0.1, per_pair=0.01)
    sizes = generator.integers(1, 100, size=(9, 3))
    ordered = sizes[np.argsort(cost.costs(sizes.astype(float)), kind="stable")]

    def padded_cost(envelopes):
        # Every row padded to the cheapest envelope that holds it.
        total = 0.0
        for row in ordered:
            fitting = [e for e in envelopes if bool((e >= row).all())]
            total += min(cost.costs(np.array(fitting, float)))
        return total

    best = min(
        padded_cost(
            [ordered[start:end].max(axis=0) for start, end in zip((0, *cuts), (*cuts, 9))]
        )
        for cuts in itertools.combinations(range(1, 9), 2)
    )
    envelopes = _cost_optimal_envelopes(sizes, 3, cost)
    assert len(envelopes) <= 3
    assert padded_cost(list(envelopes)) <= best * (1 + 1e-12)


# ---------------------------------------------------------------------------------------
# The model and the loss on a padded batch
# ---------------------------------------------------------------------------------------


def _huber():
    return modules.WeightedHuberEnergyForcesStressLoss(
        energy_weight=1.0, forces_weight=100.0, stress_weight=0.0, huber_delta=0.01
    )


def _loss_and_gradients(model, loss_fn, batch):
    model.zero_grad(set_to_none=True)
    batch_dict = batch.to_dict()
    positions = batch_dict["positions"].detach().requires_grad_(True)
    output = model({**batch_dict, "positions": positions}, **FORWARD_KWARGS)
    loss = loss_fn(pred=output, ref=batch)
    loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output, loss.detach(), gradients


def _assert_same_loss_and_gradients(reference, candidate, tolerance):
    assert abs(candidate[1].item() - reference[1].item()) <= tolerance * max(
        abs(reference[1].item()), 1.0
    )
    assert candidate[2].keys() == reference[2].keys()
    for name, gradient in reference[2].items():
        assert bool(torch.isfinite(candidate[2][name]).all()), name
        scale = max(gradient.abs().max().item(), 1e-30)
        difference = (candidate[2][name] - gradient).abs().max().item()
        assert difference <= tolerance * scale, name


@requires_graph_longrange
@pytest.mark.polar
@pytest.mark.parametrize("extra_graphs", [1, 3])
def test_padding_leaves_predictions_loss_and_gradients_unchanged(extra_graphs):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = _polar_model()
        model.set_electrostatic_pbcs("realspace")
        loss_fn = _huber()
        structures = _water_clusters([3, 2, 4], seed=1)
        shape = batch_shape(structures, real_space_pairs=True)
        unpadded = _loss_and_gradients(model, loss_fn, _unpadded(structures))
        padded = _loss_and_gradients(
            model, loss_fn, _padded(structures, _roomy_budget(structures, extra_graphs))
        )
        assert torch.allclose(
            padded[0]["energy"][: shape.graphs], unpadded[0]["energy"], rtol=0, atol=1e-10
        )
        assert torch.allclose(
            padded[0]["forces"][: shape.nodes], unpadded[0]["forces"], rtol=0, atol=1e-10
        )
        assert bool(torch.isfinite(padded[0]["energy"]).all())
        assert bool(torch.isfinite(padded[0]["forces"]).all())
        _assert_same_loss_and_gradients(unpadded, padded, tolerance=1e-12)
    finally:
        torch.set_default_dtype(previous_dtype)


@requires_graph_longrange
@pytest.mark.polar
@pytest.mark.slow
def test_static_compiled_loss_on_padded_batches_matches_eager_unpadded():
    """Two batches padded to one budget: one graph, eager parity (CPU: no CUDA graph)."""
    from torch._dynamo.utils import counters  # noqa: PLC0415

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        configure_dynamo_for_training()
        torch._dynamo.reset()
        with mace_compile.disable_e3nn_codegen():
            model = _polar_model()
        model.set_electrostatic_pbcs("realspace")
        loss_fn = _huber()
        compiled_loss = CompiledForceTrainingLoss(
            model, FORWARD_KWARGS, static_shapes=True
        )
        batches = [_water_clusters(sizes, seed) for sizes, seed in (([4, 6, 3], 0), ([5, 7], 1))]
        envelope = BatchShape(
            nodes=max(batch_shape(s, True).nodes for s in batches) + 9,
            edges=max(batch_shape(s, True).edges for s in batches) + 9,
            pairs=max(batch_shape(s, True).pairs for s in batches) + 9,
            graphs=4,
        )
        graphs_before = counters["stats"]["unique_graphs"]
        for structures in batches:
            reference = _loss_and_gradients(model, loss_fn, _unpadded(structures))
            padded_batch = _padded(structures, envelope)
            model.zero_grad(set_to_none=True)
            loss = compiled_loss(model, loss_fn, padded_batch)
            loss.backward()
            compiled = (
                None,
                loss.detach(),
                {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                },
            )
            _assert_same_loss_and_gradients(reference, compiled, tolerance=1e-11)
        assert counters["stats"]["unique_graphs"] - graphs_before == 1
        # A batch larger than every budget arrives unpadded and runs eagerly.
        reference = _loss_and_gradients(model, loss_fn, _unpadded(batches[0]))
        model.zero_grad(set_to_none=True)
        loss = compiled_loss(model, loss_fn, _unpadded(batches[0]))
        assert compiled_loss.eager_batches == 1
        assert abs(loss.item() - reference[1].item()) <= 1e-12 * max(
            abs(reference[1].item()), 1.0
        )
        assert counters["stats"]["unique_graphs"] - graphs_before == 1
    finally:
        torch.set_default_dtype(previous_dtype)


@requires_graph_longrange
@pytest.mark.polar
@pytest.mark.slow
def test_run_train_with_padding_buckets(tmp_path):
    from tests.extensions.polar.test_polar_models import (  # noqa: PLC0415
        _base_train_params,
        _run_train,
    )
    from tests.extensions.polar.test_polar_realspace_training import (  # noqa: PLC0415
        _write_open_molecules,
    )

    train_file = _write_open_molecules(tmp_path)
    params = _base_train_params(tmp_path, train_file, "polar_padded")
    params.update(
        {
            "loss": "huber",
            "stress_weight": 0.0,
            "swa_stress_weight": 0.0,
            "pbc_handling": "realspace",
            "compile_training": True,
            "padding_buckets": 2,
            "max_num_epochs": 2,
        }
    )
    del params["stress_key"]
    _run_train(params)
    model = torch.load(tmp_path / "polar_padded.model", weights_only=False)
    assert model.pbc_handling == "auto"


def test_padding_buckets_need_compiled_training(tmp_path):
    from mace.cli.run_train import run  # noqa: PLC0415
    from mace.tools import build_default_arg_parser  # noqa: PLC0415

    args = build_default_arg_parser().parse_args(
        ["--name", "x", "--train_file", "unused.xyz", "--padding_buckets", "2",
         "--model", "PolarMACE", "--pbc_handling", "realspace",
         "--log_dir", str(tmp_path), "--results_dir", str(tmp_path)]
    )
    with pytest.raises(ValueError, match="padding_buckets"):
        run(args)
