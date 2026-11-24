"""Compiled force training of PolarMACE on fully periodic structures (pbc_handling 'pbc').

The reciprocal-lattice indices of the k-vectors within the model's cutoff are listed when
a batch is collated (``mace.data.reciprocal_lattice``), so the k-space electrostatics,
their force gradient and the parameter backward through it are one compiled graph. The
compiled energies, forces, loss and parameter gradients must equal the eager 'pbc' path
(the model selecting its k-vectors from the cell itself) to float64 round-off; padding
(atoms, structures, k-vector rows) must change nothing; cells varying within one padding
budget must not recompile.
"""

from __future__ import annotations

import subprocess

import ase.io
import numpy as np
import pytest
import torch
from ase import Atoms

from mace import data, modules
from mace.data.padding import (
    BatchShape,
    PaddingPlan,
    StepCostModel,
    batch_shape,
    collate_padded,
    plan_padding_budgets,
    sample_batch_shapes,
)
from mace.data.reciprocal_lattice import (
    RECIPROCAL_LATTICE_KEYS,
    OpenStructureInReciprocalSpaceBatch,
    structure_kspace_rows,
)
from mace.modules.polar.electrostatics.kspace import reciprocal_lattice_indices
from mace.tools import compile as mace_compile
from mace.tools import torch_geometric, utils
from mace.tools.compiled.training import (
    CompiledForceTrainingLoss,
    configure_dynamo_for_training,
)
from tests.extensions.polar.test_polar_models import _base_train_params, _run_train
from tests.extensions.polar.test_polar_shared_geometry import (
    ATOMIC_NUMBERS,
    _polar_model,
)
from tests.helpers import GRAPH_LONGRANGE_AVAILABLE

pytestmark = [
    pytest.mark.polar,
    pytest.mark.skipif(
        not GRAPH_LONGRANGE_AVAILABLE, reason="graph_longrange is not installed"
    ),
]

WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
FORWARD_KWARGS = {"training": True, "compute_force": True}
KEYSPEC = data.KeySpecification(
    info_keys={"energy": "REF_energy"}, arrays_keys={"forces": "REF_forces"}
)
# (waters per box side, box length in Angstrom): the boxes of three batches. The cells
# differ (and are sheared at random), so every batch has its own k-vector set.
BATCH_BOXES = (
    ((2, 6.0), (2, 6.5)),
    ((2, 6.2), (3, 8.5), (2, 5.8)),
    ((3, 8.8), (2, 6.4)),
)


def _water_box(waters_per_side: int, length: float, rng) -> Atoms:
    spacing = length / waters_per_side
    grid = np.stack(
        np.meshgrid(*[np.arange(waters_per_side)] * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    positions = np.concatenate([WATER + spacing * site + 0.3 for site in grid])
    atoms = Atoms(
        numbers=[8, 1, 1] * len(grid),
        positions=positions + rng.normal(0, 0.05, positions.shape),
        cell=np.diag([length] * 3) + rng.normal(0, 0.2, (3, 3)),
        pbc=True,
    )
    atoms.info["REF_energy"] = float(rng.normal())
    atoms.new_array("REF_forces", rng.normal(size=atoms.positions.shape))
    return atoms


def _structures(boxes, seed):
    rng = np.random.default_rng(seed)
    return [
        data.AtomicData.from_config(
            data.config_from_atoms(
                _water_box(per_side, length, rng),
                key_specification=KEYSPEC,
                head_name="Default",
            ),
            z_table=utils.AtomicNumberTable(ATOMIC_NUMBERS),
            cutoff=4.0,
            heads=["Default"],
        )
        for per_side, length in boxes
    ]


def _to_float64(batch):
    for key in batch.keys:
        value = batch[key]
        if torch.is_tensor(value) and value.is_floating_point():
            batch[key] = value.to(torch.float64)
    return batch


def _collated(structures, kspace_cutoff=None):
    loader = torch_geometric.dataloader.DataLoader(
        structures, batch_size=len(structures), kspace_cutoff=kspace_cutoff
    )
    return _to_float64(next(iter(loader)))


def _padded(structures, budget, kspace_cutoff):
    return _to_float64(
        collate_padded(
            structures,
            PaddingPlan((budget,)),
            real_space_pairs=False,
            kspace_cutoff=kspace_cutoff,
        )
    )


def _roomy_budget(structures, kspace_cutoff, extra_graphs=1, extra_rows=11):
    shape = batch_shape(structures, real_space_pairs=False, kspace_cutoff=kspace_cutoff)
    return BatchShape(
        nodes=shape.nodes + 7,
        edges=shape.edges + 13,
        pairs=0,
        graphs=shape.graphs + extra_graphs,
        kspace_rows=shape.kspace_rows + extra_rows,
    )


@pytest.fixture(name="float64")
def _float64():
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous_dtype)


def _periodic_model(compiled: bool):
    if compiled:
        configure_dynamo_for_training()
        torch._dynamo.reset()
        with mace_compile.disable_e3nn_codegen():
            model = _polar_model()
    else:
        model = _polar_model()
    model.set_electrostatic_pbcs("pbc")
    return model


def _huber():
    return modules.WeightedHuberEnergyForcesStressLoss(
        energy_weight=1.0, forces_weight=100.0, stress_weight=0.0, huber_delta=0.01
    )


def _eager_loss_and_gradients(model, loss_fn, batch):
    """The eager 'pbc' path: without lattice keys the model lists its k-vectors itself."""
    model.zero_grad(set_to_none=True)
    batch_dict = batch.to_dict()
    positions = batch_dict["positions"].detach().requires_grad_(True)
    output = model({**batch_dict, "positions": positions}, **FORWARD_KWARGS)
    loss = loss_fn(pred=output, ref=batch)
    loss.backward()
    return output, loss.detach(), _gradients(model)


def _gradients(model):
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def _compiled_loss_and_gradients(model, compiled_loss, loss_fn, batch):
    model.zero_grad(set_to_none=True)
    loss = compiled_loss(model, loss_fn, batch)
    loss.backward()
    return None, loss.detach(), _gradients(model)


def _largest_relative_gradient_difference(reference, candidate) -> float:
    """Largest over parameter tensors of max |difference| / max |reference gradient|."""
    assert candidate[2].keys() == reference[2].keys()
    largest = 0.0
    for name, gradient in reference[2].items():
        assert bool(torch.isfinite(candidate[2][name]).all()), name
        scale = max(gradient.abs().max().item(), 1e-30)
        difference = (candidate[2][name] - gradient).abs().max().item()
        largest = max(largest, difference / scale)
    return largest


def _assert_same_loss_and_gradients(reference, candidate, tolerance, label):
    loss_difference = abs(candidate[1].item() - reference[1].item())
    gradient_difference = _largest_relative_gradient_difference(reference, candidate)
    print(
        f"{label}: loss {reference[1].item():.6f}, |loss difference| "
        f"{loss_difference:.2e}, largest relative gradient difference "
        f"{gradient_difference:.2e}"
    )
    assert loss_difference <= tolerance * max(abs(reference[1].item()), 1.0)
    assert gradient_difference <= tolerance


# ---------------------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------------------


def test_collated_batch_carries_the_lattice_indices_of_the_cutoff(float64):
    structures = _structures(BATCH_BOXES[1], seed=1)
    batch = _collated(structures, kspace_cutoff=1.3)
    expected = reciprocal_lattice_indices(batch.cell.reshape(-1, 3, 3), 1.3)
    for key, value in zip(RECIPROCAL_LATTICE_KEYS, expected, strict=True):
        assert torch.equal(batch[key], value), key
    assert sum(structure_kspace_rows(s, 1.3) for s in structures) == len(
        batch.kspace_miller
    )


def test_open_structures_are_refused():
    open_water = data.AtomicData.from_config(
        data.config_from_atoms(
            Atoms(numbers=[8, 1, 1], positions=WATER), key_specification=KEYSPEC
        ),
        z_table=utils.AtomicNumberTable(ATOMIC_NUMBERS),
        cutoff=4.0,
        heads=["Default"],
    )
    periodic = _structures(BATCH_BOXES[0], seed=0)
    with pytest.raises(OpenStructureInReciprocalSpaceBatch, match="1 structure"):
        _collated([*periodic, open_water], kspace_cutoff=1.3)
    with pytest.raises(ValueError, match="not both"):
        torch_geometric.dataloader.DataLoader(
            periodic, real_space_pairs=True, kspace_cutoff=1.3
        )


@pytest.mark.parametrize("extra_graphs", [1, 3])
def test_padded_batch_pads_k_vector_rows_in_the_first_padding_structure(
    extra_graphs, float64
):
    structures = _structures(BATCH_BOXES[1], seed=1)
    budget = _roomy_budget(structures, 1.3, extra_graphs)
    shape = batch_shape(structures, real_space_pairs=False, kspace_cutoff=1.3)
    padded = _padded(structures, budget, 1.3)
    unpadded = _collated(structures, kspace_cutoff=1.3)
    assert padded.positions.shape[0] == budget.nodes
    assert padded.num_graphs == budget.graphs
    assert padded.kspace_miller.shape == (budget.kspace_rows, 3)
    rows = shape.kspace_rows
    for key in RECIPROCAL_LATTICE_KEYS:
        assert torch.equal(padded[key][:rows], unpadded[key]), key
    assert bool(padded.kspace_padding[rows:].all())
    assert bool((padded.kspace_structure[rows:] == shape.graphs).all())
    # The padding structures take a real structure's cell: finite volume and k-vectors.
    assert bool((padded.volume[shape.graphs :].abs() > 1.0).all())


def test_planned_budgets_cover_every_sampled_batch_with_k_vector_rows():
    generator = np.random.default_rng(4)
    nodes = generator.integers(30, 300, size=80)
    edges = nodes * generator.integers(8, 20, size=80)
    rows = (nodes * generator.uniform(2.0, 4.0, size=80)).astype(np.int64)
    cost = StepCostModel(per_node=1.0, per_edge=0.03, per_pair=0.0, per_kspace_row=0.05)
    plan = plan_padding_budgets(
        nodes,
        edges,
        batch_size=4,
        bucket_count=3,
        cost_model=cost,
        real_space_pairs=False,
        structure_kspace_rows=rows,
        sampled_epochs=5,
        coverage_epochs=50,
    )
    assert all(budget.kspace_rows > 0 for budget in plan.budgets)
    for seed in (0, 1):
        for row in sample_batch_shapes(nodes, edges, 4, 5, False, seed, rows):
            shape = BatchShape(int(row[0]), int(row[1]), 0, 4, int(row[3]))
            assert plan.budget_for(shape) is not None


# ---------------------------------------------------------------------------------------
# Padding and the eager model
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("extra_graphs", [1, 3])
def test_padding_leaves_periodic_predictions_loss_and_gradients_unchanged(
    extra_graphs, float64
):
    model = _periodic_model(compiled=False)
    cutoff = model.kspace_cutoff_value()
    loss_fn = _huber()
    structures = _structures(BATCH_BOXES[1], seed=2)
    shape = batch_shape(structures, real_space_pairs=False, kspace_cutoff=cutoff)
    unpadded = _eager_loss_and_gradients(model, loss_fn, _collated(structures))
    padded = _eager_loss_and_gradients(
        model,
        loss_fn,
        _padded(structures, _roomy_budget(structures, cutoff, extra_graphs), cutoff),
    )
    energy_difference = (
        (padded[0]["energy"][: shape.graphs] - unpadded[0]["energy"]).abs().max().item()
    )
    force_difference = (
        (padded[0]["forces"][: shape.nodes] - unpadded[0]["forces"]).abs().max().item()
    )
    print(
        f"padded vs unpadded: energy {energy_difference:.2e}, forces "
        f"{force_difference:.2e}"
    )
    assert energy_difference <= 1e-11 and force_difference <= 1e-11
    assert bool(torch.isfinite(padded[0]["energy"]).all())
    assert bool(torch.isfinite(padded[0]["forces"]).all())
    _assert_same_loss_and_gradients(unpadded, padded, 1e-12, "padded vs unpadded")


# ---------------------------------------------------------------------------------------
# The compiled step
# ---------------------------------------------------------------------------------------


def test_compiled_loss_refuses_a_periodic_batch_without_lattice_indices(float64):
    model = _periodic_model(compiled=False)
    compiled_loss = CompiledForceTrainingLoss(model, FORWARD_KWARGS)
    with pytest.raises(ValueError, match="reciprocal-lattice indices"):
        compiled_loss(model, _huber(), _collated(_structures(BATCH_BOXES[0], 0)))


@pytest.mark.slow
def test_compiled_energy_and_forces_match_the_eager_pbc_path(float64):
    """Forward and force gradient (training mode) as one fullgraph graph, vs eager."""
    model = _periodic_model(compiled=True)
    cutoff = model.kspace_cutoff_value()
    view = mace_compile.simplified_view(model)

    def energy_and_forces(batch_dict, positions):
        output = view({**batch_dict, "positions": positions}, **FORWARD_KWARGS)
        return output["energy"], output["forces"]

    compiled = torch.compile(energy_and_forces, fullgraph=True, dynamic=True)
    for seed, boxes in enumerate(BATCH_BOXES[:2]):
        structures = _structures(boxes, seed)
        reference = model(_collated(structures).to_dict(), **FORWARD_KWARGS)
        batch_dict = _collated(structures, kspace_cutoff=cutoff).to_dict()
        positions = batch_dict["positions"].detach().requires_grad_(True)
        energy, forces = compiled(batch_dict, positions)
        energy_difference = (energy - reference["energy"]).abs().max().item()
        force_difference = (forces - reference["forces"]).abs().max().item()
        print(
            f"compiled vs eager, batch {seed}: |energy| max "
            f"{reference['energy'].abs().max().item():.3e}, difference "
            f"{energy_difference:.2e}; |forces| max "
            f"{reference['forces'].abs().max().item():.3e}, difference "
            f"{force_difference:.2e}"
        )
        # float64 round-off relative to the largest value (random weights give large
        # energies and forces on these boxes).
        assert energy_difference <= 1e-12 * max(
            reference["energy"].abs().max().item(), 1.0
        )
        assert force_difference <= 1e-11 * max(
            reference["forces"].abs().max().item(), 1.0
        )


@pytest.mark.slow
def test_compiled_padded_loss_matches_eager_and_cells_do_not_recompile(float64):
    """Three batches of different cells (and k-vector counts) padded to one budget: one
    graph (padded_per_budget, static shapes), loss and gradients of the eager pbc path."""
    from torch._dynamo.utils import counters  # noqa: PLC0415

    model = _periodic_model(compiled=True)
    cutoff = model.kspace_cutoff_value()
    loss_fn = _huber()
    compiled_loss = CompiledForceTrainingLoss(
        model, FORWARD_KWARGS, shape_mode="padded_per_budget"
    )
    batches = [_structures(boxes, seed) for seed, boxes in enumerate(BATCH_BOXES)]
    shapes = [batch_shape(s, False, cutoff) for s in batches]
    assert len({shape.kspace_rows for shape in shapes}) == len(shapes)
    envelope = BatchShape(
        nodes=max(shape.nodes for shape in shapes) + 9,
        edges=max(shape.edges for shape in shapes) + 9,
        pairs=0,
        graphs=max(shape.graphs for shape in shapes) + 1,
        kspace_rows=max(shape.kspace_rows for shape in shapes) + 9,
    )
    graphs_before = counters["stats"]["unique_graphs"]
    for index, structures in enumerate(batches):
        reference = _eager_loss_and_gradients(model, loss_fn, _collated(structures))
        compiled = _compiled_loss_and_gradients(
            model, compiled_loss, loss_fn, _padded(structures, envelope, cutoff)
        )
        _assert_same_loss_and_gradients(
            reference, compiled, 1e-11, f"compiled padded vs eager, batch {index}"
        )
    assert counters["stats"]["unique_graphs"] - graphs_before == 1
    assert compiled_loss.eager_batches == 0


@pytest.mark.slow
def test_shared_and_dynamic_compiled_losses_match_eager(float64):
    """padded_shared: batches padded to two budgets, one graph; dynamic: unpadded batches
    with their own k-vector counts, one graph."""
    from torch._dynamo.utils import counters  # noqa: PLC0415

    model = _periodic_model(compiled=True)
    cutoff = model.kspace_cutoff_value()
    loss_fn = _huber()
    batches = [_structures(boxes, seed) for seed, boxes in enumerate(BATCH_BOXES[:2])]
    for shape_mode in ("padded_shared", "dynamic"):
        torch._dynamo.reset()
        compiled_loss = CompiledForceTrainingLoss(
            model, FORWARD_KWARGS, shape_mode=shape_mode
        )
        graphs_before = counters["stats"]["unique_graphs"]
        for index, structures in enumerate(batches):
            batch = (
                _padded(
                    structures, _roomy_budget(structures, cutoff, 1 + index), cutoff
                )
                if shape_mode == "padded_shared"
                else _collated(structures, kspace_cutoff=cutoff)
            )
            reference = _eager_loss_and_gradients(model, loss_fn, _collated(structures))
            compiled = _compiled_loss_and_gradients(
                model, compiled_loss, loss_fn, batch
            )
            _assert_same_loss_and_gradients(
                reference,
                compiled,
                1e-11,
                f"compiled {shape_mode} vs eager, batch {index}",
            )
        assert counters["stats"]["unique_graphs"] - graphs_before == 1


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (CUDA graphs)")
def test_compiled_padded_loss_with_cuda_graphs_matches_eager(float64):
    """padded_shared on the GPU: the CUDA graph of each budget replayed for new cells."""
    from torch._dynamo.utils import counters  # noqa: PLC0415

    model = _periodic_model(compiled=True).to("cuda")
    cutoff = model.kspace_cutoff_value()
    loss_fn = _huber().to("cuda")
    compiled_loss = CompiledForceTrainingLoss(
        model, FORWARD_KWARGS, shape_mode="padded_shared"
    )
    batches = [_structures(boxes, seed) for seed, boxes in enumerate(BATCH_BOXES)]
    envelope = BatchShape(
        nodes=max(batch_shape(s, False).nodes for s in batches) + 9,
        edges=max(batch_shape(s, False).edges for s in batches) + 9,
        pairs=0,
        graphs=4,
        kspace_rows=max(batch_shape(s, False, cutoff).kspace_rows for s in batches) + 9,
    )
    graphs_before = counters["stats"]["unique_graphs"]
    for repeat in range(2):
        for index, structures in enumerate(batches):
            reference = _eager_loss_and_gradients(
                model, loss_fn, _collated(structures).to("cuda")
            )
            compiled = _compiled_loss_and_gradients(
                model,
                compiled_loss,
                loss_fn,
                _padded(structures, envelope, cutoff).to("cuda"),
            )
            _assert_same_loss_and_gradients(
                reference,
                compiled,
                1e-10,
                f"CUDA graphs vs eager, pass {repeat} batch {index}",
            )
    assert counters["stats"]["unique_graphs"] - graphs_before == 1


# ---------------------------------------------------------------------------------------
# run_train
# ---------------------------------------------------------------------------------------


def _write_water_boxes(tmp_path, count=8):
    rng = np.random.default_rng(6)
    configs = []
    for index in range(count):
        atoms = _water_box(2, 6.0 + 0.1 * index, rng)
        atoms.info["REF_energy"] = rng.normal(0, 1e-2)
        atoms.arrays["REF_forces"] = rng.normal(0, 1e-2, size=atoms.positions.shape)
        configs.append(atoms)
    path = tmp_path / "water_boxes.xyz"
    ase.io.write(path, configs)
    return path


def _periodic_train_params(tmp_path, train_file, name):
    params = _base_train_params(tmp_path, train_file, name)
    params.update(
        {
            "loss": "huber",
            "stress_weight": 0.0,
            "swa_stress_weight": 0.0,
            "pbc_handling": "pbc",
            "field_feature_widths": "[1.5, 3.0]",
            "max_num_epochs": 2,
        }
    )
    del params["stress_key"]
    return params


@pytest.mark.slow
@pytest.mark.parametrize("padding_buckets", [0, 2])
def test_run_train_compiles_periodic_training(tmp_path, padding_buckets):
    train_file = _write_water_boxes(tmp_path)
    params = _periodic_train_params(tmp_path, train_file, "polar_periodic")
    params.update({"compile_training": True, "padding_buckets": padding_buckets})
    _run_train(params)
    model = torch.load(tmp_path / "polar_periodic.model", weights_only=False)
    assert model.pbc_handling == "auto"


def test_run_train_pbc_refuses_open_structures(tmp_path):
    train_file = _write_water_boxes(tmp_path)
    configs = ase.io.read(train_file, index=":")
    configs[0].pbc = False
    configs[0].set_cell(np.zeros((3, 3)))
    ase.io.write(train_file, configs)
    params = _periodic_train_params(tmp_path, train_file, "polar_open")
    with pytest.raises(subprocess.CalledProcessError) as failure:
        _run_train(params)
    assert "fully periodic structures only" in failure.value.stderr
