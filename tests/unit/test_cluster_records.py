"""Cluster records: HDF5 round trip, collation, signed combination and the interaction loss."""

import numpy as np
import pytest
import torch

from mace import tools
from mace.data import (
    ClusterHDF5Dataset,
    MonomerSubsystem,
    build_cluster_record,
    save_cluster_records_as_HDF5,
)
from mace.modules import InteractionUniversalLoss
from mace.tools.torch_geometric.cluster_collate import (
    ClusterCollater,
    cluster_atom_counts,
    combine_energy,
    combine_forces,
    get_cluster_data_loader,
    slot_to_cluster,
)

h5py = pytest.importorskip("h5py")

torch.set_default_dtype(torch.float64)

WATER_NUMBERS = np.array([8, 1, 1])
WATER_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
Z_TABLE = tools.AtomicNumberTable([1, 8])
CELL = 8.0 * np.eye(3)


def _periodic_two_water_record(seed: int):
    """A 'bulk' record: two waters in a periodic box, the second monomer given unwrapped
    (shifted by one lattice vector) to exercise the lattice-shift tolerance."""
    rng = np.random.default_rng(seed)
    first = WATER_POSITIONS + np.array([1.0, 1.0, 1.0])
    second = WATER_POSITIONS + np.array([4.0, 4.5, 4.0])
    cluster_positions = np.concatenate([first, second])
    cluster_numbers = np.concatenate([WATER_NUMBERS, WATER_NUMBERS])
    monomer_energies = [-14.0 + 0.1 * seed, -14.2]
    monomer_forces = [rng.normal(size=(3, 3)), rng.normal(size=(3, 3))]
    interaction_energy = -0.25 * (seed + 1)
    cluster_forces = np.concatenate(monomer_forces) + 0.05 * rng.normal(size=(6, 3))
    monomers = [
        MonomerSubsystem(WATER_NUMBERS, first, monomer_energies[0], monomer_forces[0], np.arange(3)),
        MonomerSubsystem(
            WATER_NUMBERS, second + CELL[0], monomer_energies[1], monomer_forces[1], np.arange(3, 6)
        ),
    ]
    record = build_cluster_record(
        cluster_numbers,
        cluster_positions,
        sum(monomer_energies) + interaction_energy,
        cluster_forces,
        monomers,
        stress=0.01 * np.eye(3),
        cell=CELL,
        pbc=(True, True, True),
    )
    expected_interaction_forces = cluster_forces - np.concatenate(monomer_forces)
    return record, interaction_energy, expected_interaction_forces


def _lone_monomer_record():
    return build_cluster_record(WATER_NUMBERS, WATER_POSITIONS, -14.1, np.zeros((3, 3)), [])


def _write_records(tmp_path, records):
    path = tmp_path / "clusters.h5"
    with h5py.File(path, "w") as handle:
        save_cluster_records_as_HDF5(records, handle)
    return path


def test_build_cluster_record_rejects_mismatched_monomers():
    monomers = [MonomerSubsystem(WATER_NUMBERS, WATER_POSITIONS + 0.1, -14.0, np.zeros((3, 3)), np.arange(3))]
    with pytest.raises(ValueError, match="positions differ"):
        build_cluster_record(WATER_NUMBERS, WATER_POSITIONS, -14.0, np.zeros((3, 3)), monomers)
    with pytest.raises(ValueError, match="exactly one monomer"):
        build_cluster_record(
            np.concatenate([WATER_NUMBERS, WATER_NUMBERS]),
            np.concatenate([WATER_POSITIONS, WATER_POSITIONS + 3.0]),
            -28.0,
            np.zeros((6, 3)),
            [MonomerSubsystem(WATER_NUMBERS, WATER_POSITIONS, -14.0, np.zeros((3, 3)), np.arange(3))],
        )


def test_lone_record_has_zero_interaction_weight():
    record = _lone_monomer_record()
    assert len(record) == 1
    assert record[0].properties["interaction_weight"] == 0.0
    assert record[0].properties["sign"] == 1.0


def test_round_trip_and_signed_combination(tmp_path):
    bulk_a, e_int_a, f_int_a = _periodic_two_water_record(0)
    bulk_b, e_int_b, f_int_b = _periodic_two_water_record(1)
    lone = _lone_monomer_record()
    path = _write_records(tmp_path, [bulk_a, lone, bulk_b])

    dataset = ClusterHDF5Dataset(str(path), r_max=3.0, z_table=Z_TABLE)
    assert len(dataset) == 3
    first_record = dataset[0]
    assert [float(item.sign) for item in first_record] == [1.0, -1.0, -1.0]
    assert first_record[0].slot.tolist() == list(range(6))
    assert first_record[2].slot.tolist() == [3, 4, 5]
    assert bool(first_record[0].pbc.all()) and not bool(first_record[1].pbc.any())

    loader = get_cluster_data_loader(dataset, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    assert batch.num_graphs == 7  # 3 + 1 + 3 subsystems
    assert int(batch.n_clusters) == 3 and int(batch.n_slots) == 6 + 3 + 6
    assert batch.cluster_id.tolist() == [0, 0, 0, 1, 2, 2, 2]
    assert batch.cluster_interaction_weight.tolist() == [1.0, 0.0, 1.0]
    # slots are disjoint across records and cover every physical atom exactly once
    cluster_nodes = batch.sign[batch.batch] > 0
    assert sorted(batch.slot[cluster_nodes].tolist()) == list(range(15))
    assert slot_to_cluster(batch).tolist() == [0] * 6 + [1] * 3 + [2] * 6
    assert cluster_atom_counts(batch).tolist() == [6.0, 3.0, 6.0]

    e_int = combine_energy(batch, batch.energy)
    np.testing.assert_allclose(e_int.numpy(), [e_int_a, -14.1, e_int_b])
    f_int = combine_forces(batch, batch.forces)
    np.testing.assert_allclose(f_int[:6].numpy(), f_int_a)
    np.testing.assert_allclose(f_int[6:9].numpy(), 0.0)
    np.testing.assert_allclose(f_int[9:].numpy(), f_int_b)


def test_collater_offsets_do_not_depend_on_record_order(tmp_path):
    bulk, _, _ = _periodic_two_water_record(2)
    path = _write_records(tmp_path, [_lone_monomer_record(), bulk])
    dataset = ClusterHDF5Dataset(str(path), r_max=3.0, z_table=Z_TABLE)
    batch = ClusterCollater()([dataset[1], dataset[0]])
    assert batch.cluster_id.tolist() == [0, 0, 0, 1]
    assert batch.slot[batch.sign[batch.batch] > 0].tolist() == list(range(9))


def test_collater_does_not_modify_its_input_records(tmp_path):
    """Collating the same records twice (a caching dataset, a second epoch) gives identical
    batches and leaves every input's slot and cluster_id as the dataset built them."""
    bulk, _, _ = _periodic_two_water_record(4)
    path = _write_records(tmp_path, [_lone_monomer_record(), bulk])
    dataset = ClusterHDF5Dataset(str(path), r_max=3.0, z_table=Z_TABLE)
    records = [dataset[0], dataset[1]]
    slots_before = [[subsystem.slot.clone() for subsystem in record] for record in records]

    collater = ClusterCollater()
    first = collater(records)
    second = collater(records)

    for key in ("slot", "cluster_id", "sign", "batch", "energy", "forces", "positions"):
        assert torch.equal(first[key], second[key]), key
    assert int(first.n_slots) == int(second.n_slots) == 9
    for record, record_slots in zip(records, slots_before):
        for subsystem, slot in zip(record, record_slots):
            assert torch.equal(subsystem.slot, slot)
            assert int(subsystem.cluster_id) == 0


def test_interaction_loss_sees_only_the_interaction_residual(tmp_path):
    bulk, _, _ = _periodic_two_water_record(3)
    path = _write_records(tmp_path, [bulk, _lone_monomer_record()])
    dataset = ClusterHDF5Dataset(str(path), r_max=3.0, z_table=Z_TABLE)
    batch = next(iter(get_cluster_data_loader(dataset, batch_size=2)))

    loss_fn = InteractionUniversalLoss(
        energy_weight=0.0,
        forces_weight=0.0,
        stress_weight=0.0,
        interaction_energy_weight=1.0,
        interaction_forces_weight=1.0,
    )
    exact = {"energy": batch.energy.clone(), "forces": batch.forces.clone(), "stress": batch.stress.clone()}
    assert float(loss_fn(pred=exact, ref=batch)) == pytest.approx(0.0)

    # A per-atom shift (an E0 change) cancels in E_int for the bulk record and is ignored
    # (weight 0) for the lone record: the interaction loss must stay zero.
    atoms_per_graph = (batch.ptr[1:] - batch.ptr[:-1]).to(batch.energy.dtype)
    shifted = dict(exact, energy=batch.energy + 0.3 * atoms_per_graph)
    assert float(loss_fn(pred=shifted, ref=batch)) == pytest.approx(0.0)

    # Shifting only the cluster graph's energy changes E_int by the shift: the loss is
    # the Huber value of shift / cluster atoms (quadratic regime) averaged over 2 records.
    shift = 0.006
    cluster_only = batch.energy.clone()
    cluster_only[0] += shift
    loss = float(loss_fn(pred=dict(exact, energy=cluster_only), ref=batch))
    assert loss == pytest.approx(0.5 * (shift / 6.0) ** 2 / 2.0)

    plain_batch = tools.torch_geometric.Batch.from_data_list(dataset[1])
    with pytest.raises(ValueError, match="cluster-record batches"):
        loss_fn(pred=exact, ref=plain_batch)
