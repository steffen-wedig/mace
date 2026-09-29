"""End-to-end: train a tiny MACE on cluster records with each interaction loss."""

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


LOSS_SETTINGS = {
    "interaction_universal": {"loss": "interaction_universal", "compute_stress": True},
    "interaction_huber": {
        "loss": "interaction_huber",
        "monomer_energy_weight": 0.0,
        "swa_monomer_forces_weight": 50.0,
        "huber_delta_interaction_energy": 0.02,
    },
    "likelihood_huber": {
        "loss": "likelihood_huber",
        "energy_weight": 50.0,
        "swa_energy_weight": 50.0,
        "forces_weight": 10.0,
        "swa_forces_weight": 10.0,
        "huber_delta_monomer_energy": 0.02,
        "loss_term_probe_interval": 2,
    },
}


@pytest.mark.parametrize("loss_name", sorted(LOSS_SETTINGS))
def test_run_train_cluster_records(tmp_path: Path, loss_name: str):
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
        }
    )
    params.update(LOSS_SETTINGS[loss_name])
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
    if loss_name == "interaction_universal":
        assert all(np.isfinite(entry["rmse_stress"]) for entry in evaluations)
    else:  # no stress term: stress is never computed
        assert all(entry.get("rmse_stress") is None for entry in evaluations)
    if loss_name == "likelihood_huber":
        loss_term_logs = sorted(tmp_path.glob("cluster_records_run-*_loss_terms.jsonl"))
        assert len(loss_term_logs) == 1
        _check_likelihood_loss_terms(loss_term_logs[0])


def _check_likelihood_loss_terms(path: Path) -> None:
    """In-place monomers carry absolute terms, there are no interaction terms, and both
    stages keep the configured weights."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    kinds = {record["record"] for record in records}
    assert kinds == {"run_start", "term_summary", "gradient_probe", "stage_switch"}
    assert records[0]["term_names"] == [
        "frame_energy",
        "frame_forces",
        "monomer_energy",
        "monomer_forces",
        "in_place_monomer_energy",
        "in_place_monomer_forces",
    ]
    switch = next(record for record in records if record["record"] == "stage_switch")
    assert switch["old_weights"] == switch["new_weights"]
    assert switch["new_weights"]["in_place_monomer_energy"] == 50.0
    train_summary = next(
        record for record in records if record["record"] == "term_summary" and record["split"] == "train"
    )
    terms = train_summary["terms"]
    # every frame holds two waters, each also an in-place monomer at the same atoms
    assert terms["in_place_monomer_energy"]["entry_count"] == 2 * terms["frame_energy"]["entry_count"]
    assert terms["in_place_monomer_forces"]["entry_count"] == terms["frame_forces"]["entry_count"]
    assert sum(term["share_of_total"] for term in terms.values()) == pytest.approx(1.0)
    probe = next(record for record in records if record["record"] == "gradient_probe")
    assert probe["gradient_norms"]["in_place_monomer_forces"] > 0.0
    assert "frame_forces_vs_in_place_monomer_forces" in probe["cosine_similarities"]
