"""InteractionHuberLoss: per-term weighted means checked against an independent numpy
calculation on extxyz batches and cluster-record batches."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pytest
import torch

from mace import tools
from mace.data import (
    AtomicData,
    ClusterHDF5Dataset,
    Configuration,
    MonomerSubsystem,
    build_cluster_record,
    save_cluster_records_as_HDF5,
)
from mace.modules import InteractionHuberLoss, LossTermStatistics
from mace.tools.torch_geometric.batch import Batch
from mace.tools.torch_geometric.cluster_collate import (
    ClusterCollater,
    cluster_atom_counts,
    combine_energy,
    combine_forces,
)

h5py = pytest.importorskip("h5py")

torch.set_default_dtype(torch.float64)

WATER_NUMBERS = np.array([8, 1, 1])
WATER_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
Z_TABLE = tools.AtomicNumberTable([1, 8])
CELL = 8.0 * np.eye(3)
TERM_NAMES = (
    "frame_energy",
    "frame_forces",
    "monomer_energy",
    "monomer_forces",
    "interaction_energy",
    "interaction_forces",
)
# distinct thresholds per term, so a mixed-up threshold shows in the comparison
DELTAS = {
    "frame_energy": 0.01,
    "frame_forces": 0.02,
    "monomer_energy": 0.015,
    "monomer_forces": 0.012,
    "interaction_energy": 0.02,
    "interaction_forces": 0.03,
}
GLOBAL_WEIGHTS = {
    "frame_energy": 1.0,
    "frame_forces": 100.0,
    "monomer_energy": 2.0,
    "monomer_forces": 50.0,
    "interaction_energy": 3.0,
    "interaction_forces": 70.0,
}


# ---------------------------------------------------------------------------
# synthetic labels, kept in numpy for the independent calculation
# ---------------------------------------------------------------------------


@dataclass
class Subsystem:
    numbers: np.ndarray
    positions: np.ndarray
    energy: float
    forces: np.ndarray
    periodic: bool
    sign: float
    slots: np.ndarray  # atom index in the record's frame


@dataclass
class Record:
    subsystems: List[Subsystem]  # frame (sign +1) first, then in-place monomers
    weight: float = 1.0
    energy_weight: float = 1.0
    forces_weight: float = 1.0
    interaction_weight: Optional[float] = None
    predicted_energies: List[float] = field(default_factory=list)
    predicted_forces: List[np.ndarray] = field(default_factory=list)


def _frame_record(seed: int, large_forces: bool = False, **weights) -> Record:
    """A periodic box with two waters and its two in-place monomers."""
    rng = np.random.default_rng(seed)
    first = WATER_POSITIONS + np.array([1.0, 1.0, 1.0])
    second = WATER_POSITIONS + np.array([4.0, 4.5, 4.0])
    monomer_forces = [rng.normal(size=(3, 3)), rng.normal(size=(3, 3))]
    if large_forces:  # one atom in each of the three upper magnitude bins
        monomer_forces[0][0] = [150.0, 0.0, 0.0]
        monomer_forces[0][1] = [0.0, 250.0, 0.0]
        monomer_forces[1][2] = [0.0, 0.0, 350.0]
    monomer_energies = [-14.0 + 0.1 * seed, -14.2]
    frame_forces = np.concatenate(monomer_forces) + 0.05 * rng.normal(size=(6, 3))
    frame = Subsystem(
        np.concatenate([WATER_NUMBERS, WATER_NUMBERS]),
        np.concatenate([first, second]),
        sum(monomer_energies) - 0.25 * (seed + 1),
        frame_forces,
        True,
        1.0,
        np.arange(6),
    )
    monomers = [
        Subsystem(WATER_NUMBERS, first, monomer_energies[0], monomer_forces[0], False, -1.0, np.arange(3)),
        Subsystem(WATER_NUMBERS, second, monomer_energies[1], monomer_forces[1], False, -1.0, np.arange(3, 6)),
    ]
    return Record([frame, *monomers], **weights)


def _lone_monomer_record(seed: int, **weights) -> Record:
    rng = np.random.default_rng(100 + seed)
    monomer = Subsystem(
        WATER_NUMBERS, WATER_POSITIONS, -14.1 + 0.01 * seed, rng.normal(size=(3, 3)), False, 1.0, np.arange(3)
    )
    return Record([monomer], **weights)


def _add_predictions(records: List[Record], seed: int) -> None:
    """Per-atom energy errors up to 0.03 and force errors of scale 0.02: both Huber regimes
    occur for every threshold in ``DELTAS``."""
    rng = np.random.default_rng(seed)
    for record in records:
        record.predicted_energies = [
            subsystem.energy + len(subsystem.numbers) * rng.uniform(-0.03, 0.03)
            for subsystem in record.subsystems
        ]
        record.predicted_forces = [
            subsystem.forces + 0.02 * rng.normal(size=subsystem.forces.shape)
            for subsystem in record.subsystems
        ]


def _cluster_batch(tmp_path, records: List[Record]) -> Batch:
    """HDF5 round trip + ClusterCollater, as in training."""
    configurations = []
    for record in records:
        frame, *monomers = record.subsystems
        configurations.append(
            build_cluster_record(
                frame.numbers,
                frame.positions,
                frame.energy,
                frame.forces,
                [
                    MonomerSubsystem(m.numbers, m.positions, m.energy, m.forces, m.slots)
                    for m in monomers
                ],
                stress=0.01 * np.eye(3) if frame.periodic else None,
                cell=CELL if frame.periodic else None,
                pbc=(True, True, True) if frame.periodic else None,
                interaction_weight=record.interaction_weight,
                weight=record.weight,
                energy_weight=record.energy_weight,
                forces_weight=record.forces_weight,
            )
        )
    path = tmp_path / f"records_{len(list(tmp_path.iterdir()))}.h5"
    with h5py.File(path, "w") as handle:
        save_cluster_records_as_HDF5(configurations, handle)
    dataset = ClusterHDF5Dataset(str(path), r_max=3.0, z_table=Z_TABLE)
    return ClusterCollater()([dataset[index] for index in range(len(dataset))])


def _extxyz_batch(records: List[Record]) -> Batch:
    """Every subsystem as a plain configuration without record fields."""
    data = []
    for record in records:
        for subsystem in record.subsystems:
            config = Configuration(
                atomic_numbers=subsystem.numbers,
                positions=subsystem.positions,
                properties={"energy": subsystem.energy, "forces": subsystem.forces},
                property_weights={"energy": record.energy_weight, "forces": record.forces_weight},
                cell=CELL if subsystem.periodic else None,
                pbc=(True, True, True) if subsystem.periodic else (False, False, False),
                weight=record.weight,
            )
            data.append(AtomicData.from_config(config, z_table=Z_TABLE, cutoff=3.0))
    return Batch.from_data_list(data)


def _prediction(records: List[Record], requires_grad: bool = False) -> Dict[str, torch.Tensor]:
    energy = torch.tensor([e for record in records for e in record.predicted_energies])
    forces = torch.tensor(np.concatenate([f for record in records for f in record.predicted_forces]))
    return {
        "energy": energy.requires_grad_(requires_grad),
        "forces": forces.requires_grad_(requires_grad),
    }


# ---------------------------------------------------------------------------
# the independent numpy calculation
# ---------------------------------------------------------------------------


def _huber(errors: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    absolute = np.abs(errors)
    return np.where(absolute < thresholds, 0.5 * errors**2, thresholds * (absolute - 0.5 * thresholds))


def _force_thresholds(reference_forces: np.ndarray, delta: float) -> np.ndarray:
    norms = np.linalg.norm(reference_forces, axis=-1)
    factors = np.select([norms < 100, norms < 200, norms < 300], [1.0, 0.7, 0.4], default=0.1)
    return (delta * factors)[:, None] * np.ones((1, 3))


class TermAccumulator:
    def __init__(self):
        self.losses, self.weights, self.linear = [], [], []

    def add(self, errors, thresholds, weight):
        errors = np.asarray(errors, dtype=float)
        thresholds = np.broadcast_to(np.asarray(thresholds, dtype=float), errors.shape).ravel()
        errors = errors.ravel()
        if weight <= 0:
            return
        self.losses.extend(_huber(errors, thresholds))
        self.weights.extend([weight] * errors.size)
        self.linear.extend(np.abs(errors) > thresholds)

    def result(self):
        weights = np.array(self.weights)
        losses = np.array(self.losses)
        weight_sum = weights.sum()
        mean = float((weights * losses).sum() / weight_sum) if weights.size else 0.0
        return {"mean": mean, "entries": len(weights), "linear": int(np.sum(self.linear))}


def _expected_terms(records: List[Record], cluster_batch: bool) -> Dict[str, dict]:
    terms = {name: TermAccumulator() for name in TERM_NAMES}
    for record in records:
        for index, subsystem in enumerate(record.subsystems):
            sign = subsystem.sign if cluster_batch else 1.0
            if sign < 0:
                continue  # in-place monomers carry no absolute terms
            kind = "frame" if subsystem.periodic else "monomer"
            atom_count = len(subsystem.numbers)
            energy_error = (record.predicted_energies[index] - subsystem.energy) / atom_count
            terms[f"{kind}_energy"].add(
                energy_error, DELTAS[f"{kind}_energy"], record.weight * record.energy_weight
            )
            terms[f"{kind}_forces"].add(
                record.predicted_forces[index] - subsystem.forces,
                _force_thresholds(subsystem.forces, DELTAS[f"{kind}_forces"]),
                record.weight * record.forces_weight,
            )
        interaction_weight = record.interaction_weight
        if interaction_weight is None:
            interaction_weight = 1.0 if len(record.subsystems) > 1 else 0.0
        if not cluster_batch or interaction_weight <= 0:
            continue
        frame = record.subsystems[0]
        reference_energy = frame.energy
        predicted_energy = record.predicted_energies[0]
        reference_forces = frame.forces.copy()
        predicted_forces = record.predicted_forces[0].copy()
        for index, monomer in enumerate(record.subsystems[1:], start=1):
            reference_energy -= monomer.energy
            predicted_energy -= record.predicted_energies[index]
            reference_forces[monomer.slots] -= monomer.forces
            predicted_forces[monomer.slots] -= record.predicted_forces[index]
        terms["interaction_energy"].add(
            (predicted_energy - reference_energy) / len(frame.numbers),
            DELTAS["interaction_energy"],
            interaction_weight,
        )
        terms["interaction_forces"].add(
            predicted_forces - reference_forces, DELTAS["interaction_forces"], interaction_weight
        )
    return {name: accumulator.result() for name, accumulator in terms.items()}


def _loss(**overrides) -> InteractionHuberLoss:
    weights = dict(GLOBAL_WEIGHTS, **overrides)
    return InteractionHuberLoss(
        energy_weight=weights["frame_energy"],
        forces_weight=weights["frame_forces"],
        monomer_energy_weight=weights["monomer_energy"],
        monomer_forces_weight=weights["monomer_forces"],
        interaction_energy_weight=weights["interaction_energy"],
        interaction_forces_weight=weights["interaction_forces"],
        **{f"huber_delta_{name}": delta for name, delta in DELTAS.items()},
    )


def _cluster_records() -> List[Record]:
    """Frames with in-place monomers (one with large forces), a frame with interaction
    weight 0 and non-unit configuration weights, and two standalone monomers."""
    records = [
        _frame_record(0, large_forces=True),
        _frame_record(1, weight=2.0, energy_weight=0.5, forces_weight=3.0, interaction_weight=0.0),
        _frame_record(2, interaction_weight=2.5),
        _lone_monomer_record(0),
        _lone_monomer_record(1, weight=0.5, forces_weight=4.0),
    ]
    _add_predictions(records, seed=7)
    return records


def _assert_terms_match(loss_fn, terms, expected):
    for name in TERM_NAMES:
        statistics = loss_fn.last_statistics[name]
        assert float(statistics.unweighted_mean) == pytest.approx(expected[name]["mean"], rel=1e-10, abs=1e-15)
        assert float(terms[name]) == pytest.approx(GLOBAL_WEIGHTS[name] * expected[name]["mean"], rel=1e-10, abs=1e-15)
        assert int(statistics.entry_count) == expected[name]["entries"]
        assert int(statistics.linear_count) == expected[name]["linear"]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_matches_hand_calculation_on_cluster_batch(tmp_path):
    records = _cluster_records()
    batch = _cluster_batch(tmp_path, records)
    loss_fn = _loss()
    terms = loss_fn.compute_terms(batch, _prediction(records))
    expected = _expected_terms(records, cluster_batch=True)
    _assert_terms_match(loss_fn, terms, expected)

    # entries: 3 frames; 18 atoms x 3; 2 standalone monomers; 6 atoms x 3; 2 records with
    # interaction weight > 0; their 12 slots x 3
    counts = {name: int(loss_fn.last_statistics[name].entry_count) for name in TERM_NAMES}
    assert counts == {
        "frame_energy": 3,
        "frame_forces": 54,
        "monomer_energy": 2,
        "monomer_forces": 18,
        "interaction_energy": 2,
        "interaction_forces": 36,
    }
    # both Huber regimes occur in every populated term
    for name in TERM_NAMES:
        linear = expected[name]["linear"]
        assert 0 < linear < expected[name]["entries"], name


def test_matches_hand_calculation_on_extxyz_batch():
    records = [_frame_record(0, large_forces=True), _frame_record(1, weight=2.0), _lone_monomer_record(0)]
    for record in records[:2]:  # periodic frames alone, without their monomers
        record.subsystems = record.subsystems[:1]
    records.append(_lone_monomer_record(1, energy_weight=3.0))
    _add_predictions(records, seed=3)
    batch = _extxyz_batch(records)
    assert getattr(batch, "sign", None) is None
    loss_fn = _loss()
    terms = loss_fn.compute_terms(batch, _prediction(records))
    _assert_terms_match(loss_fn, terms, _expected_terms(records, cluster_batch=False))
    for name in ("interaction_energy", "interaction_forces"):
        assert float(terms[name]) == 0.0
        assert int(loss_fn.last_statistics[name].entry_count) == 0


def test_extxyz_counts_every_configuration_as_absolute():
    """Without ``sign``, the in-place monomers of an exported frame are ordinary monomers."""
    records = [_frame_record(0)]
    _add_predictions(records, seed=4)
    batch = _extxyz_batch(records)
    loss_fn = _loss()
    loss_fn.compute_terms(batch, _prediction(records))
    assert int(loss_fn.last_statistics["monomer_energy"].entry_count) == 2
    assert int(loss_fn.last_statistics["frame_energy"].entry_count) == 1


def test_frame_terms_are_not_diluted_by_monomers(tmp_path):
    """A frame term on a cluster batch equals the same term on the frames alone (the
    dilution defect of InteractionUniversalLoss)."""
    records = _cluster_records()
    loss_fn = _loss()
    cluster_terms = loss_fn.compute_terms(_cluster_batch(tmp_path, records), _prediction(records))

    frames_only = []
    for record in records:
        if record.subsystems[0].periodic:
            frames_only.append(
                Record(
                    record.subsystems[:1],
                    record.weight,
                    record.energy_weight,
                    record.forces_weight,
                    predicted_energies=record.predicted_energies[:1],
                    predicted_forces=record.predicted_forces[:1],
                )
            )
    frame_terms = loss_fn.compute_terms(_extxyz_batch(frames_only), _prediction(frames_only))
    for name in ("frame_energy", "frame_forces"):
        assert float(cluster_terms[name]) == pytest.approx(float(frame_terms[name]), rel=1e-12)


def test_zero_weight_configurations_leave_the_terms_unchanged(tmp_path):
    records = _cluster_records()
    loss_fn = _loss()
    reference_terms = loss_fn.compute_terms(_cluster_batch(tmp_path, records), _prediction(records))

    extra = [
        _frame_record(5, weight=0.0, interaction_weight=0.0),
        _frame_record(6, energy_weight=0.0, forces_weight=0.0, interaction_weight=0.0),
        _lone_monomer_record(5, weight=0.0),
        _lone_monomer_record(6, energy_weight=0.0, forces_weight=0.0),
    ]
    _add_predictions(extra, seed=11)
    padded = records[:2] + extra[:2] + records[2:] + extra[2:]
    padded_terms = loss_fn.compute_terms(_cluster_batch(tmp_path, padded), _prediction(padded))
    for name in TERM_NAMES:
        assert float(padded_terms[name]) == pytest.approx(float(reference_terms[name]), rel=1e-12), name
    assert int(loss_fn.last_statistics["frame_energy"].entry_count) == 3

    # switching the monomer terms off changes only the monomer terms
    no_monomer_fn = _loss(monomer_energy=0.0, monomer_forces=0.0)
    no_monomer_terms = no_monomer_fn.compute_terms(_cluster_batch(tmp_path, records), _prediction(records))
    for name in TERM_NAMES:
        expected = 0.0 if name.startswith("monomer") else float(reference_terms[name])
        assert float(no_monomer_terms[name]) == pytest.approx(expected, rel=1e-12), name


def test_forward_and_statistics_are_consistent(tmp_path):
    records = _cluster_records()
    batch = _cluster_batch(tmp_path, records)
    loss_fn = _loss()
    prediction = _prediction(records)
    total = loss_fn(batch, prediction)
    terms = loss_fn.compute_terms(batch, prediction)
    assert float(total) == pytest.approx(float(sum(terms.values())), rel=1e-12)
    assert tuple(loss_fn.last_statistics) == TERM_NAMES
    assert InteractionHuberLoss.term_names == TERM_NAMES
    assert loss_fn.term_weights() == GLOBAL_WEIGHTS
    for name, statistics in loss_fn.last_statistics.items():
        assert isinstance(statistics, LossTermStatistics)
        for value in vars(statistics).values():
            assert value.shape == () and not value.requires_grad
        assert float(statistics.weight) == GLOBAL_WEIGHTS[name]
        assert float(statistics.weight * statistics.unweighted_mean) == pytest.approx(float(terms[name]), rel=1e-12)
        assert float(statistics.weighted_sum / statistics.weight_sum) == pytest.approx(
            float(statistics.unweighted_mean), rel=1e-12
        )
    assert "huber_delta_interaction_forces=0.03" in repr(loss_fn)
    assert "monomer_forces_weight=50.000" in repr(loss_fn)


def test_interaction_terms_match_combine_functions(tmp_path):
    """A prediction that differs only on in-place monomers leaves every absolute term at zero;
    the interaction terms are the weighted Huber means of combine_energy / combine_forces."""
    records = _cluster_records()
    batch = _cluster_batch(tmp_path, records)
    rng = np.random.default_rng(5)
    in_place = (batch.sign < 0).to(batch.energy.dtype)
    energy = batch.energy + in_place * torch.tensor(rng.uniform(-0.1, 0.1, size=batch.num_graphs))
    forces = batch.forces + in_place[batch.batch].unsqueeze(-1) * torch.tensor(
        rng.normal(scale=0.05, size=tuple(batch.forces.shape))
    )
    loss_fn = _loss()
    terms = loss_fn.compute_terms(batch, {"energy": energy, "forces": forces})
    for name in ("frame_energy", "frame_forces", "monomer_energy", "monomer_forces"):
        assert float(terms[name]) == 0.0

    record_weights = batch.cluster_interaction_weight
    atom_counts = cluster_atom_counts(batch)
    energy_errors = (combine_energy(batch, energy) - combine_energy(batch, batch.energy)) / atom_counts
    energy_losses = torch.nn.functional.huber_loss(
        energy_errors, torch.zeros_like(energy_errors), reduction="none", delta=DELTAS["interaction_energy"]
    )
    expected_energy = (record_weights * energy_losses).sum() / record_weights.sum()
    assert float(loss_fn.last_statistics["interaction_energy"].unweighted_mean) == pytest.approx(
        float(expected_energy), rel=1e-12
    )

    slot_owner_weights = torch.cat(
        [torch.full((int(count),), float(weight)) for count, weight in zip(atom_counts, record_weights)]
    ).unsqueeze(-1)
    force_errors = combine_forces(batch, forces) - combine_forces(batch, batch.forces)
    force_losses = torch.nn.functional.huber_loss(
        force_errors, torch.zeros_like(force_errors), reduction="none", delta=DELTAS["interaction_forces"]
    )
    expected_forces = (slot_owner_weights * force_losses).sum() / (3 * slot_owner_weights.sum())
    assert float(loss_fn.last_statistics["interaction_forces"].unweighted_mean) == pytest.approx(
        float(expected_forces), rel=1e-12
    )


def test_needs_no_stress(tmp_path):
    records = _cluster_records()
    batch = _cluster_batch(tmp_path, records)
    del batch.stress
    del batch.stress_weight
    prediction = _prediction(records)
    assert "stress" not in prediction
    loss = _loss()(batch, prediction)
    assert torch.isfinite(loss)


def test_gradients_flow_and_empty_terms_stay_finite(tmp_path):
    records = _cluster_records()
    batch = _cluster_batch(tmp_path, records)
    prediction = _prediction(records, requires_grad=True)
    loss_fn = _loss()
    loss_fn(batch, prediction).backward()
    assert torch.isfinite(prediction["forces"].grad).all()
    assert prediction["forces"].grad.abs().sum() > 0
    assert prediction["energy"].grad.abs().sum() > 0

    # an extxyz batch of frames only: monomer and interaction terms have no entries, yet are
    # zero, finite, and connected to the graph (a per-term backward must work)
    frames = [_frame_record(0)]
    frames[0].subsystems = frames[0].subsystems[:1]
    _add_predictions(frames, seed=2)
    frame_prediction = _prediction(frames, requires_grad=True)
    terms = loss_fn.compute_terms(_extxyz_batch(frames), frame_prediction)
    for name in ("monomer_energy", "monomer_forces", "interaction_energy", "interaction_forces"):
        assert float(terms[name]) == 0.0
        assert terms[name].requires_grad
        assert float(loss_fn.last_statistics[name].weight_sum) == 0.0
        terms[name].backward(retain_graph=True)
    sum(terms.values()).backward()
    assert torch.isfinite(frame_prediction["forces"].grad).all()
    assert torch.isfinite(frame_prediction["energy"].grad).all()


def test_ddp_raises(tmp_path):
    records = _cluster_records()
    batch = _cluster_batch(tmp_path, records)
    with pytest.raises(NotImplementedError, match="distributed"):
        _loss()(batch, _prediction(records), ddp=True)


def test_both_stage_factories_build_the_loss():
    from mace.tools.arg_parser import build_default_arg_parser  # pylint: disable=import-outside-toplevel
    from mace.tools.scripts_utils import get_loss_fn, get_swa  # pylint: disable=import-outside-toplevel

    args = build_default_arg_parser().parse_args(
        [
            "--name",
            "loss_factories",
            "--loss",
            "interaction_huber",
            "--energy_weight",
            "1.0",
            "--forces_weight",
            "100.0",
            "--swa_energy_weight",
            "1000.0",
            "--swa_forces_weight",
            "100.0",
            "--swa_monomer_energy_weight",
            "7.0",
            "--interaction_energy_weight",
            "2.0",
            "--swa_interaction_energy_weight",
            "20.0",
            "--huber_delta_interaction_energy",
            "0.04",
            "--max_num_epochs",
            "4",
            "--start_swa",
            "2",
        ]
    )
    stage_one = get_loss_fn(args, dipole_only=False, compute_dipole=False)
    assert isinstance(stage_one, InteractionHuberLoss)
    assert stage_one.term_weights()["monomer_energy"] == 1.0  # follows energy_weight
    assert stage_one.term_weights()["monomer_forces"] == 100.0
    assert stage_one.term_weights()["interaction_energy"] == 2.0

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    swa, _ = get_swa(args, model, optimizer, swas=[False])
    stage_two = swa.loss_fn
    assert isinstance(stage_two, InteractionHuberLoss)
    assert stage_two.term_weights() == {
        "frame_energy": 1000.0,
        "frame_forces": 100.0,
        "monomer_energy": 7.0,
        "monomer_forces": 100.0,  # follows swa_forces_weight
        "interaction_energy": 20.0,
        "interaction_forces": args.swa_interaction_forces_weight,
    }
    for loss_fn in (stage_one, stage_two):
        assert loss_fn.huber_deltas["interaction_energy"] == 0.04
        assert loss_fn.huber_deltas["frame_energy"] == 0.01
