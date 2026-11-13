"""Training PolarMACE with open-boundary electrostatics only (--pbc_handling realspace).

Batches carry their atom pairs, built on the host at collation; the forward then reads no
boundary flags and builds no pairs. The saved model picks its evaluator from the
structures again ('auto').
"""

from __future__ import annotations

import ase.io
import numpy as np
import pytest
import torch
from ase import Atoms

from mace.data.real_space_pairs import (
    PeriodicStructureInRealSpaceBatch,
    real_space_pair_index,
)
from mace.tools import torch_geometric
from tests.extensions.polar.test_polar_models import _base_train_params, _run_train
from tests.extensions.polar.test_polar_shared_geometry import (
    ATOMIC_NUMBERS,
    _molecules,
)
from tests.helpers import GRAPH_LONGRANGE_AVAILABLE

pytestmark = [
    pytest.mark.polar,
    pytest.mark.skipif(
        not GRAPH_LONGRANGE_AVAILABLE, reason="graph_longrange is not installed"
    ),
]


def _directed_pairs(ptr: torch.Tensor) -> set[tuple[int, int]]:
    return {
        (sender, receiver)
        for start, stop in zip(ptr[:-1].tolist(), ptr[1:].tolist())
        for sender in range(start, stop)
        for receiver in range(start, stop)
        if sender != receiver
    }


@pytest.mark.parametrize("sizes", [[3], [3, 1, 4], [1, 1], [5, 2, 6, 3]])
def test_pair_index_lists_every_directed_pair_once(sizes):
    ptr = torch.tensor([0, *np.cumsum(sizes).tolist()])
    pbc = torch.zeros((len(sizes), 3), dtype=torch.bool)
    pairs = real_space_pair_index(ptr, pbc)
    listed = list(map(tuple, pairs.t().tolist()))
    assert len(listed) == len(set(listed))
    assert set(listed) == _directed_pairs(ptr)
    assert pairs.dtype == torch.long


def test_pair_index_matches_the_device_builder():
    from mace.modules.polar.electrostatics.realspace_electrostatics import (  # noqa: PLC0415
        real_space_pair_index as device_pair_index,
    )

    sizes = [4, 1, 3, 5]
    ptr = torch.tensor([0, *np.cumsum(sizes).tolist()])
    batch = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    host = real_space_pair_index(ptr, torch.zeros((len(sizes), 3), dtype=torch.bool))
    device = device_pair_index(batch)
    assert set(map(tuple, host.t().tolist())) == set(map(tuple, device.t().tolist()))


def test_periodic_structures_are_refused():
    ptr = torch.tensor([0, 3, 6])
    pbc = torch.tensor([[False, False, False], [True, True, True]])
    with pytest.raises(PeriodicStructureInRealSpaceBatch):
        real_space_pair_index(ptr, pbc)


def test_collation_attaches_the_pair_index():
    from mace import data  # noqa: PLC0415
    from mace.tools import utils  # noqa: PLC0415

    keyspec = data.KeySpecification(info_keys={}, arrays_keys={})
    dataset = [
        data.AtomicData.from_config(
            data.config_from_atoms(atoms, key_specification=keyspec),
            z_table=utils.AtomicNumberTable(ATOMIC_NUMBERS),
            cutoff=4.0,
        )
        for atoms in _molecules()
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset, batch_size=3, shuffle=False, real_space_pairs=True
    )
    batch = next(iter(loader))
    assert set(map(tuple, batch.real_space_pair_index.t().tolist())) == (
        _directed_pairs(batch.ptr)
    )
    plain = next(
        iter(torch_geometric.dataloader.DataLoader(dataset, batch_size=3, shuffle=False))
    )
    assert "real_space_pair_index" not in plain.to_dict()


def _write_open_molecules(tmp_path):
    rng = np.random.default_rng(5)
    configs = []
    for index in range(6):
        atoms = Atoms(
            numbers=[8, 1, 1],
            positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        )
        if index % 2:
            atoms = atoms + Atoms(
                numbers=[8, 1, 1],
                positions=[[2.9, 0.2, 0.1], [3.8, 0.2, 0.1], [2.7, 1.1, 0.1]],
            )
        atoms.positions += rng.normal(0, 0.05, size=atoms.positions.shape)
        atoms.info["REF_energy"] = rng.normal(0, 1e-2)
        atoms.new_array("REF_forces", rng.normal(0, 1e-2, size=atoms.positions.shape))
        configs.append(atoms)
    path = tmp_path / "open_molecules.xyz"
    ase.io.write(path, configs)
    return path


def test_run_train_realspace_saves_an_auto_model(tmp_path):
    train_file = _write_open_molecules(tmp_path)
    params = _base_train_params(tmp_path, train_file, "polar_realspace")
    params.update(
        {
            "loss": "weighted",
            "stress_key": None,
            "pbc_handling": "realspace",
            "field_feature_widths": "[1.5, 3.0]",
            "atomic_multipoles_smearing_width": 1.5,
            "num_recursion_steps": 2,
        }
    )
    del params["stress_key"]
    _run_train(params)
    model = torch.load(tmp_path / "polar_realspace.model", weights_only=False)
    assert model.pbc_handling == "auto"


def test_realspace_training_refuses_periodic_structures(tmp_path, capsys):
    import subprocess  # noqa: PLC0415

    from tests.extensions.polar.test_polar_models import (  # noqa: PLC0415
        _write_polar_data,
    )

    train_file, _ = _write_polar_data(tmp_path)  # periodic water
    params = _base_train_params(tmp_path, train_file, "polar_realspace_periodic")
    params["pbc_handling"] = "realspace"
    with pytest.raises(subprocess.CalledProcessError):
        _run_train(params)
    assert "PeriodicStructureInRealSpaceBatch" in capsys.readouterr().out
