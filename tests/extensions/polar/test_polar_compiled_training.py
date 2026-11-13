"""Compiled force training of PolarMACE: forward, forces and loss as one graph.

The loss backward then runs through the force gradient on the compiled graph (double
backward). It must give the eager loss and parameter gradients, compile once for batches of
different sizes (dynamic shapes), and train end to end through run_train.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms

from mace import data, modules
from mace.data.real_space_pairs import attach_real_space_pair_index
from mace.tools import compile as mace_compile
from mace.tools import torch_geometric, utils
from mace.tools.compiled.training import (
    CompiledForceTrainingLoss,
    configure_dynamo_for_training,
)
from tests.extensions.polar.test_polar_models import _base_train_params, _run_train
from tests.extensions.polar.test_polar_realspace_training import _write_open_molecules
from tests.extensions.polar.test_polar_shared_geometry import (
    ATOMIC_NUMBERS,
    _polar_model,
)
from tests.helpers import GRAPH_LONGRANGE_AVAILABLE

pytestmark = [
    pytest.mark.polar,
    pytest.mark.slow,
    pytest.mark.skipif(
        not GRAPH_LONGRANGE_AVAILABLE, reason="graph_longrange is not installed"
    ),
]

WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def _cluster_batch(sizes, seed):
    rng = np.random.default_rng(seed)
    keyspec = data.KeySpecification(
        info_keys={"energy": "REF_energy"}, arrays_keys={"forces": "REF_forces"}
    )
    dataset = []
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
        dataset.append(
            data.AtomicData.from_config(
                data.config_from_atoms(atoms, key_specification=keyspec, head_name="Default"),
                z_table=utils.AtomicNumberTable(ATOMIC_NUMBERS),
                cutoff=4.0,
                heads=["Default"],
            )
        )
    batch = next(
        iter(torch_geometric.dataloader.DataLoader(dataset, batch_size=len(sizes)))
    )
    attach_real_space_pair_index(batch)
    for key in batch.keys:
        value = batch[key]
        if torch.is_tensor(value) and value.is_floating_point():
            batch[key] = value.to(torch.float64)
    return batch


def _loss_and_gradients(model, loss_of_batch, batch):
    model.zero_grad(set_to_none=True)
    loss = loss_of_batch(batch)
    loss.backward()
    return loss.detach(), {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def test_compiled_loss_matches_eager_and_compiles_once():
    from torch._dynamo.utils import counters  # noqa: PLC0415

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        configure_dynamo_for_training()
        torch._dynamo.reset()
        with mace_compile.disable_e3nn_codegen():
            model = _polar_model()
        model.set_electrostatic_pbcs("realspace")
        loss_fn = modules.WeightedHuberEnergyForcesStressLoss(
            energy_weight=1.0, forces_weight=100.0, stress_weight=0.0, huber_delta=0.01
        )
        forward_kwargs = {"training": True, "compute_force": True}
        compiled_loss = CompiledForceTrainingLoss(model, forward_kwargs)

        def eager_loss(batch):
            batch_dict = batch.to_dict()
            positions = batch_dict["positions"].detach().requires_grad_(True)
            output = model({**batch_dict, "positions": positions}, **forward_kwargs)
            return loss_fn(pred=output, ref=batch)

        graphs_before = counters["stats"]["unique_graphs"]
        # More than 32 atoms per batch: on CPU, Inductor rewrites tiny matrix products
        # (<= 32 elements) and guards on that size, which is not a recompile of interest.
        for sizes, seed in (([4, 6, 3], 0), ([5, 7], 1)):
            batch = _cluster_batch(sizes, seed)
            eager = _loss_and_gradients(model, eager_loss, batch)
            compiled = _loss_and_gradients(
                model, lambda batch: compiled_loss(model, loss_fn, batch), batch
            )
            assert abs(compiled[0].item() - eager[0].item()) <= 1e-12 * max(
                abs(eager[0].item()), 1.0
            )
            assert compiled[1].keys() == eager[1].keys()
            for name, gradient in eager[1].items():
                scale = max(gradient.abs().max().item(), 1e-30)
                difference = (compiled[1][name] - gradient).abs().max().item()
                assert difference <= 1e-11 * scale, name
        assert counters["stats"]["unique_graphs"] - graphs_before == 1
    finally:
        torch.set_default_dtype(previous_dtype)


def test_compiled_training_refuses_stress():
    with pytest.raises(NotImplementedError, match="compute_stress"):
        CompiledForceTrainingLoss(
            torch.nn.Linear(1, 1),
            {"training": True, "compute_force": True, "compute_stress": True},
        )


def test_run_train_with_compiled_training(tmp_path):
    train_file = _write_open_molecules(tmp_path)
    params = _base_train_params(tmp_path, train_file, "polar_compiled")
    params.update(
        {
            "loss": "huber",
            "stress_weight": 0.0,
            "swa_stress_weight": 0.0,
            "pbc_handling": "realspace",
            "compile_training": True,
            "max_num_epochs": 2,
        }
    )
    del params["stress_key"]
    _run_train(params)
    model = torch.load(tmp_path / "polar_compiled.model", weights_only=False)
    assert model.pbc_handling == "auto"
