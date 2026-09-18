"""End-to-end: train a tiny MACE on cluster records with the interaction loss."""

import json
from pathlib import Path

import numpy as np
import pytest

from mace.data import MonomerSubsystem, build_cluster_record, save_cluster_records_as_HDF5
from tests.helpers import base_mace_params, run_mace_train

h5py = pytest.importorskip("h5py")

WATER_NUMBERS = np.array([8, 1, 1])
WATER_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
CELL = 7.0 * np.eye(3)


def _records(count: int, seed: int):
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(count):
        offsets = [rng.uniform(0.5, 2.0, size=3), rng.uniform(3.5, 5.0, size=3)]
        monomer_positions = [WATER_POSITIONS + offset + 0.05 * rng.normal(size=(3, 3)) for offset in offsets]
        monomer_energies = [-14.0 + 0.02 * rng.normal() for _ in offsets]
        monomer_forces = [0.1 * rng.normal(size=(3, 3)) for _ in offsets]
        cluster_positions = np.concatenate(monomer_positions)
        cluster_forces = np.concatenate(monomer_forces) + 0.02 * rng.normal(size=(6, 3))
        interaction_energy = -0.2 + 0.02 * rng.normal()
        monomers = [
            MonomerSubsystem(WATER_NUMBERS, monomer_positions[0], monomer_energies[0], monomer_forces[0], np.arange(3)),
            MonomerSubsystem(WATER_NUMBERS, monomer_positions[1], monomer_energies[1], monomer_forces[1], np.arange(3, 6)),
        ]
        records.append(
            build_cluster_record(
                np.concatenate([WATER_NUMBERS, WATER_NUMBERS]),
                cluster_positions,
                sum(monomer_energies) + interaction_energy,
                cluster_forces,
                monomers,
                stress=0.001 * rng.normal(size=(3, 3)),
                cell=CELL,
                pbc=(True, True, True),
            )
        )
    # one lone molecule: absolute labels only, interaction weight 0
    records.append(build_cluster_record(WATER_NUMBERS, WATER_POSITIONS + 2.0, -14.0, np.zeros((3, 3)), []))
    return records


def test_run_train_cluster_records(tmp_path: Path):
    train_path = tmp_path / "train.h5"
    valid_path = tmp_path / "valid.h5"
    with h5py.File(train_path, "w") as handle:
        save_cluster_records_as_HDF5(_records(12, seed=1), handle)
    with h5py.File(valid_path, "w") as handle:
        save_cluster_records_as_HDF5(_records(4, seed=2), handle)

    params = base_mace_params()
    params.update(
        {
            "name": "cluster_records",
            "train_file": str(train_path),
            "valid_file": str(valid_path),
            "cluster_records": True,
            "atomic_numbers": "[1, 8]",
            "E0s": "{1: -0.5, 8: -13.0}",
            "loss": "interaction_universal",
            "interaction_energy_weight": 10.0,
            "interaction_forces_weight": 10.0,
            "swa_interaction_energy_weight": 100.0,
            "swa_interaction_forces_weight": 10.0,
            "error_table": "PerAtomRMSEinteraction",
            "max_num_epochs": 4,
            "start_swa": 2,
            "batch_size": 3,
            "valid_batch_size": 3,
            "checkpoints_dir": str(tmp_path),
            "model_dir": str(tmp_path),
            "results_dir": str(tmp_path),
            "log_dir": str(tmp_path),
            "default_dtype": "float64",
            "compute_stress": True,
        }
    )
    params.pop("valid_fraction", None)
    run_mace_train(params, cwd=tmp_path)

    assert (tmp_path / "cluster_records.model").is_file()
    assert (tmp_path / "cluster_records_stagetwo.model").is_file()
    results = sorted(tmp_path.glob("cluster_records_run-*_train.txt"))
    assert results, "no results file written"
    evaluations = [json.loads(line) for line in results[0].read_text().splitlines() if '"mode": "eval"' in line]
    assert evaluations
    assert all(np.isfinite(entry["rmse_e_int_per_atom"]) for entry in evaluations)
    assert all(np.isfinite(entry["rmse_f_int"]) for entry in evaluations)
