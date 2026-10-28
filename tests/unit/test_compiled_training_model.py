"""CompiledTrainingModel: compiled training forwards match the eager model.

The training step of ``tools.train`` calls the model in the distributed-model
slot with ``training=True`` and differentiates an energy and forces loss with
respect to the parameters. Over changing batch shapes (one graph, dynamic
shapes) the compiled forward must reproduce the eager loss and gradients, and
``training=False`` must reach the eager model. The aot_eager backend traces what
inductor would compile without needing a C++ toolchain.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from ase import build
from e3nn import o3

from mace import data, modules, tools
from mace.tools import compile as mace_compile
from mace.tools import torch_geometric
from mace.tools.scripts_utils import extract_config_mace_model

TABLE = tools.AtomicNumberTable([1, 8])
CUTOFF = 4.0


def create_mace():
    torch_geometric.seed_everything(11)
    return modules.ScaleShiftMACE(
        r_max=CUTOFF,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=F.silu,
        atomic_energies=np.array([-12.1, -2057.9]),
        avg_num_neighbors=6.0,
        atomic_numbers=TABLE.zs,
        correlation=2,
        radial_type="bessel",
        atomic_inter_scale=1.3,
        atomic_inter_shift=0.1,
    )


def water_batch(sizes, seed):
    generator = np.random.default_rng(seed)
    configs = []
    for size in sizes:
        cluster = build.molecule("H2O")
        for _ in range(size - 1):
            molecule = build.molecule("H2O")
            molecule.rotate(generator.uniform(0.0, 360.0), "z")
            molecule.translate(generator.uniform(-3.0, 3.0, size=3))
            cluster += molecule
        configs.append(data.config_from_atoms(cluster))
    loader = torch_geometric.dataloader.DataLoader(
        [data.AtomicData.from_config(c, z_table=TABLE, cutoff=CUTOFF) for c in configs],
        batch_size=len(configs),
        shuffle=False,
    )
    return next(iter(loader)).to_dict()


def loss_and_gradients(model, call, batch):
    output = call(batch, training=True, compute_force=True)
    loss = output["energy"].pow(2).sum() + output["forces"].pow(2).sum()
    return loss, torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)


@pytest.fixture(name="default_dtype_float64")
def fixture_default_dtype_float64():
    with tools.torch_tools.default_dtype(torch.float64):
        yield


@pytest.mark.usefixtures("default_dtype_float64")
def test_compiled_training_matches_eager():
    with mace_compile.disable_e3nn_codegen():
        model = create_mace()
    compiled = mace_compile.CompiledTrainingModel(model, backend="aot_eager")

    for step, sizes in enumerate(([2, 3], [1, 4, 2], [3, 3])):
        compiled_loss, compiled_gradients = loss_and_gradients(
            model, compiled, water_batch(sizes, step)
        )
        batch = water_batch(sizes, step)
        batch["positions"].requires_grad_(True)
        eager_loss, eager_gradients = loss_and_gradients(model, model, batch)
        torch.testing.assert_close(compiled_loss, eager_loss, rtol=1e-12, atol=1e-12)
        for compiled_gradient, eager_gradient in zip(compiled_gradients, eager_gradients):
            if eager_gradient is None:
                assert compiled_gradient is None
                continue
            torch.testing.assert_close(
                compiled_gradient, eager_gradient, rtol=1e-10, atol=1e-12
            )


@pytest.mark.usefixtures("default_dtype_float64")
def test_evaluation_reaches_the_eager_model():
    with mace_compile.disable_e3nn_codegen():
        model = create_mace()
    compiled = mace_compile.CompiledTrainingModel(model, backend="aot_eager")
    batch = water_batch([2, 2], 0)
    expected = model(water_batch([2, 2], 0), training=False, compute_force=True)
    result = compiled(batch, training=False, compute_force=True)
    torch.testing.assert_close(result["energy"], expected["energy"])
    torch.testing.assert_close(result["forces"], expected["forces"])


@pytest.mark.usefixtures("default_dtype_float64")
def test_the_trained_model_keeps_its_module_types():
    # Saved models and config extraction inspect the model's own modules, so
    # the traced graphs must live only in the compiled view.
    with mace_compile.disable_e3nn_codegen():
        model = create_mace()
    contraction = model.products[0].symmetric_contractions
    mace_compile.CompiledTrainingModel(model, backend="aot_eager")
    assert model.products[0].symmetric_contractions is contraction
    assert not isinstance(contraction, torch.fx.GraphModule)
    view = mace_compile.simplified_view(model)
    assert isinstance(view.products[0].symmetric_contractions, torch.fx.GraphModule)
    assert {id(p) for p in view.parameters()} == {id(p) for p in model.parameters()}
    extract_config_mace_model(model)
