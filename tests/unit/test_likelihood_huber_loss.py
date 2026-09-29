"""LikelihoodHuberLoss: one normalizer over every calculation in the batch, checked against an
independent numpy calculation, plus the likelihood properties the loss exists for."""

from typing import Dict, List

import numpy as np
import pytest
import torch

from mace.modules import LikelihoodHuberLoss, LossTermStatistics
from tests.unit.test_interaction_huber_loss import (
    Record,
    _add_predictions,
    _cluster_batch,
    _extxyz_batch,
    _force_thresholds,
    _frame_record,
    _huber,
    _lone_monomer_record,
    _prediction,
)

pytest.importorskip("h5py")

torch.set_default_dtype(torch.float64)

TERM_NAMES = (
    "frame_energy",
    "frame_forces",
    "monomer_energy",
    "monomer_forces",
    "in_place_monomer_energy",
    "in_place_monomer_forces",
)
# distinct thresholds, so a frame/monomer mix-up shows in the comparison
DELTAS = {
    "frame_energy": 0.01,
    "frame_forces": 0.02,
    "monomer_energy": 0.015,
    "monomer_forces": 0.012,
}
ENERGY_WEIGHT = 900.0
FORCES_WEIGHT = 100.0


def _loss(energy_weight: float = ENERGY_WEIGHT, forces_weight: float = FORCES_WEIGHT) -> LikelihoodHuberLoss:
    return LikelihoodHuberLoss(
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        **{f"huber_delta_{name}": delta for name, delta in DELTAS.items()},
    )


def _records() -> List[Record]:
    """Frames with in-place monomers (one with large forces, one with non-unit
    configuration weights and interaction weight 0) and two standalone monomers."""
    records = [
        _frame_record(0, large_forces=True),
        _frame_record(1, weight=2.0, energy_weight=0.5, forces_weight=3.0, interaction_weight=0.0),
        _frame_record(2, interaction_weight=2.5),
        _lone_monomer_record(0),
        _lone_monomer_record(1, weight=0.5, forces_weight=4.0),
    ]
    _add_predictions(records, seed=7)
    return records


# ---------------------------------------------------------------------------
# the independent numpy calculation
# ---------------------------------------------------------------------------


def _expected(records: List[Record], cluster_batch: bool) -> Dict[str, dict]:
    sums = {name: {"sum": 0.0, "entries": 0, "linear": 0} for name in TERM_NAMES}
    observation_count = 0.0
    for record in records:
        for index, subsystem in enumerate(record.subsystems):
            sign = subsystem.sign if cluster_batch else 1.0
            if sign < 0:
                calculation, threshold_group = "in_place_monomer", "monomer"
            elif subsystem.periodic:
                calculation, threshold_group = "frame", "frame"
            else:
                calculation, threshold_group = "monomer", "monomer"
            energy_error = (record.predicted_energies[index] - subsystem.energy) / len(subsystem.numbers)
            force_errors = record.predicted_forces[index] - subsystem.forces
            entries = (
                ("energy", np.array([energy_error]), np.array([DELTAS[f"{threshold_group}_energy"]]),
                 record.weight * record.energy_weight),
                ("forces", force_errors.ravel(),
                 _force_thresholds(subsystem.forces, DELTAS[f"{threshold_group}_forces"]).ravel(),
                 record.weight * record.forces_weight),
            )
            for label_kind, errors, thresholds, entry_weight in entries:
                if entry_weight <= 0:
                    continue
                term = sums[f"{calculation}_{label_kind}"]
                term["sum"] += entry_weight * float(_huber(errors, thresholds).sum())
                term["entries"] += errors.size
                term["linear"] += int(np.sum(np.abs(errors) > thresholds))
                observation_count += entry_weight * errors.size
    for name, term in sums.items():
        weight = ENERGY_WEIGHT if name.endswith("energy") else FORCES_WEIGHT
        term["value"] = weight * term["sum"] / observation_count
    return {"terms": sums, "observation_count": observation_count}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_matches_hand_calculation_on_cluster_batch(tmp_path):
    records = _records()
    loss_fn = _loss()
    terms = loss_fn.compute_terms(_cluster_batch(tmp_path, records), _prediction(records))
    expected = _expected(records, cluster_batch=True)
    assert float(loss_fn.last_observation_count) == pytest.approx(expected["observation_count"], rel=1e-12)
    for name in TERM_NAMES:
        statistics = loss_fn.last_statistics[name]
        assert float(terms[name]) == pytest.approx(expected["terms"][name]["value"], rel=1e-10), name
        assert float(statistics.weighted_sum) == pytest.approx(expected["terms"][name]["sum"], rel=1e-10), name
        assert int(statistics.entry_count) == expected["terms"][name]["entries"], name
        assert int(statistics.linear_count) == expected["terms"][name]["linear"], name
    # 3 frames of 6 atoms, each with 2 in-place monomers of 3 atoms; 2 standalone monomers
    counts = {name: int(loss_fn.last_statistics[name].entry_count) for name in TERM_NAMES}
    assert counts == {
        "frame_energy": 3,
        "frame_forces": 54,
        "monomer_energy": 2,
        "monomer_forces": 18,
        "in_place_monomer_energy": 6,
        "in_place_monomer_forces": 54,
    }
    # both Huber regimes occur in every force term
    for name in ("frame_forces", "monomer_forces", "in_place_monomer_forces"):
        assert 0 < expected["terms"][name]["linear"] < expected["terms"][name]["entries"], name


def test_matches_hand_calculation_on_extxyz_batch():
    """Without ``sign`` every configuration is absolute: in-place terms stay empty."""
    records = [_frame_record(0, large_forces=True), _frame_record(1, weight=2.0), _lone_monomer_record(0)]
    _add_predictions(records, seed=3)
    loss_fn = _loss()
    terms = loss_fn.compute_terms(_extxyz_batch(records), _prediction(records))
    expected = _expected(records, cluster_batch=False)
    for name in TERM_NAMES:
        assert float(terms[name]) == pytest.approx(expected["terms"][name]["value"], rel=1e-10), name
    for name in ("in_place_monomer_energy", "in_place_monomer_forces"):
        assert float(terms[name]) == 0.0
        assert int(loss_fn.last_statistics[name].entry_count) == 0
    assert int(loss_fn.last_statistics["monomer_energy"].entry_count) == 5


def test_an_in_place_monomer_counts_like_a_standalone_monomer(tmp_path):
    """A frame with its in-place monomers gives the same loss as the frame alone plus the
    same monomers as standalone records: the sign only moves the entries between terms."""
    records = [_frame_record(0, large_forces=True), _frame_record(3, weight=2.0, forces_weight=0.5)]
    _add_predictions(records, seed=13)
    split = []
    for record in records:
        frame, *monomers = record.subsystems
        split.append(
            Record(
                [frame],
                record.weight,
                record.energy_weight,
                record.forces_weight,
                predicted_energies=record.predicted_energies[:1],
                predicted_forces=record.predicted_forces[:1],
            )
        )
        for index, monomer in enumerate(monomers, start=1):
            standalone = type(monomer)(**{**vars(monomer), "sign": 1.0, "slots": np.arange(3)})
            split.append(
                Record(
                    [standalone],
                    record.weight,
                    record.energy_weight,
                    record.forces_weight,
                    predicted_energies=record.predicted_energies[index : index + 1],
                    predicted_forces=record.predicted_forces[index : index + 1],
                )
            )

    loss_fn = _loss()
    bundled_terms = loss_fn.compute_terms(_cluster_batch(tmp_path, records), _prediction(records))
    bundled_count = float(loss_fn.last_observation_count)
    split_terms = loss_fn.compute_terms(_cluster_batch(tmp_path, split), _prediction(split))
    assert float(loss_fn.last_observation_count) == pytest.approx(bundled_count, rel=1e-12)
    assert float(sum(bundled_terms.values())) == pytest.approx(float(sum(split_terms.values())), rel=1e-12)
    for label_kind in ("energy", "forces"):
        assert float(bundled_terms[f"in_place_monomer_{label_kind}"]) == pytest.approx(
            float(split_terms[f"monomer_{label_kind}"]), rel=1e-12
        )
        assert float(bundled_terms[f"frame_{label_kind}"]) == pytest.approx(
            float(split_terms[f"frame_{label_kind}"]), rel=1e-12
        )
        assert float(bundled_terms[f"monomer_{label_kind}"]) == 0.0
        assert float(split_terms[f"in_place_monomer_{label_kind}"]) == 0.0


def test_duplicating_an_entry_doubles_its_contribution(tmp_path):
    """Unnormalized, the loss is a sum over observations: a duplicated record adds its own
    sum once more, and the observation count grows by its observation count."""
    base = _records()
    extra = [_frame_record(4)]
    _add_predictions(extra, seed=21)
    loss_fn = _loss()

    def total_and_count(records):
        loss = loss_fn(_cluster_batch(tmp_path, records), _prediction(records))
        return float(loss), float(loss_fn.last_observation_count)

    base_loss, base_count = total_and_count(base)
    extra_loss, extra_count = total_and_count(extra)
    once_loss, once_count = total_and_count(base + extra)
    twice_loss, twice_count = total_and_count(base + extra + extra)
    assert once_count == pytest.approx(base_count + extra_count, rel=1e-12)
    assert twice_count == pytest.approx(base_count + 2 * extra_count, rel=1e-12)
    assert once_loss * once_count == pytest.approx(base_loss * base_count + extra_loss * extra_count, rel=1e-12)
    assert twice_loss * twice_count == pytest.approx(
        base_loss * base_count + 2 * extra_loss * extra_count, rel=1e-12
    )
    # a configuration weight of 2 is the same as the entry present twice
    doubled = [_frame_record(4, weight=2.0)]
    _add_predictions(doubled, seed=21)
    doubled_loss, doubled_count = total_and_count(base + doubled)
    assert doubled_loss == pytest.approx(twice_loss, rel=1e-12)
    assert doubled_count == pytest.approx(twice_count, rel=1e-12)
    # duplicating the whole batch leaves the per-observation loss unchanged
    all_twice_loss, _ = total_and_count(base + base)
    assert all_twice_loss == pytest.approx(base_loss, rel=1e-12)


def test_zero_weight_configurations_neither_contribute_nor_dilute(tmp_path):
    records = _records()
    loss_fn = _loss()
    reference = float(loss_fn(_cluster_batch(tmp_path, records), _prediction(records)))
    extra = [
        _frame_record(5, weight=0.0),
        _frame_record(6, energy_weight=0.0, forces_weight=0.0),
        _lone_monomer_record(5, weight=0.0),
    ]
    _add_predictions(extra, seed=11)
    padded = records[:2] + extra[:2] + records[2:] + extra[2:]
    assert float(loss_fn(_cluster_batch(tmp_path, padded), _prediction(padded))) == pytest.approx(
        reference, rel=1e-12
    )


def test_no_interaction_terms(tmp_path):
    """The record's interaction weight plays no role, and a prediction exact on every
    calculation has zero loss whatever E_int it implies."""
    records = _records()
    loss_fn = _loss()
    reference = float(loss_fn(_cluster_batch(tmp_path, records), _prediction(records)))
    for record in records:
        if len(record.subsystems) > 1:
            record.interaction_weight = 7.0
    reweighted = float(loss_fn(_cluster_batch(tmp_path, records), _prediction(records)))
    assert reweighted == pytest.approx(reference, rel=1e-12)
    batch = _cluster_batch(tmp_path, records)
    exact = loss_fn(batch, {"energy": batch.energy.clone(), "forces": batch.forces.clone()})
    assert float(exact) == 0.0
    assert set(loss_fn.term_names) == set(TERM_NAMES)


def test_forward_statistics_and_weights_are_consistent(tmp_path):
    records = _records()
    batch = _cluster_batch(tmp_path, records)
    loss_fn = _loss()
    prediction = _prediction(records)
    total = loss_fn(batch, prediction)
    terms = loss_fn.compute_terms(batch, prediction)
    assert float(total) == pytest.approx(float(sum(terms.values())), rel=1e-12)
    assert tuple(loss_fn.last_statistics) == TERM_NAMES
    weights = loss_fn.term_weights()
    assert weights == {name: ENERGY_WEIGHT if name.endswith("energy") else FORCES_WEIGHT for name in TERM_NAMES}
    count = loss_fn.last_observation_count
    for name, statistics in loss_fn.last_statistics.items():
        assert isinstance(statistics, LossTermStatistics)
        for value in vars(statistics).values():
            assert value.shape == () and not value.requires_grad
        assert float(statistics.weight * statistics.unweighted_mean) == pytest.approx(float(terms[name]), rel=1e-12)
        assert float(statistics.weighted_sum / count) == pytest.approx(float(statistics.unweighted_mean), rel=1e-12)
    assert "energy_weight=900.000" in repr(loss_fn)
    assert "huber_delta_monomer_forces=0.012" in repr(loss_fn)


def test_needs_no_stress_and_gradients_flow(tmp_path):
    records = _records()
    batch = _cluster_batch(tmp_path, records)
    del batch.stress
    del batch.stress_weight
    prediction = _prediction(records, requires_grad=True)
    loss_fn = _loss()
    loss_fn(batch, prediction).backward()
    assert torch.isfinite(prediction["forces"].grad).all()
    in_place_atoms = (batch.sign < 0)[batch.batch]
    assert prediction["forces"].grad[in_place_atoms].abs().sum() > 0
    assert prediction["energy"].grad[batch.sign < 0].abs().sum() > 0

    # an extxyz batch of frames only: the empty terms are zero, finite and connected
    frames = [_frame_record(0)]
    frames[0].subsystems = frames[0].subsystems[:1]
    _add_predictions(frames, seed=2)
    frame_prediction = _prediction(frames, requires_grad=True)
    terms = loss_fn.compute_terms(_extxyz_batch(frames), frame_prediction)
    for name in ("monomer_energy", "monomer_forces", "in_place_monomer_energy", "in_place_monomer_forces"):
        assert float(terms[name]) == 0.0
        assert terms[name].requires_grad
        terms[name].backward(retain_graph=True)


def test_ddp_raises(tmp_path):
    records = _records()
    with pytest.raises(NotImplementedError, match="distributed"):
        _loss()(_cluster_batch(tmp_path, records), _prediction(records), ddp=True)


def _parse(extra_arguments: List[str]):
    from mace.tools.arg_parser import build_default_arg_parser  # pylint: disable=import-outside-toplevel

    return build_default_arg_parser().parse_args(
        ["--name", "likelihood", "--loss", "likelihood_huber", "--max_num_epochs", "4", "--start_swa", "2"]
        + extra_arguments
    )


def test_both_stage_factories_build_the_loss():
    from mace.tools.scripts_utils import get_loss_fn, get_swa  # pylint: disable=import-outside-toplevel

    args = _parse(
        [
            "--energy_weight", "900.0",
            "--forces_weight", "100.0",
            "--swa_energy_weight", "900.0",
            "--swa_forces_weight", "100.0",
            "--monomer_energy_weight", "900.0",
            "--huber_delta_monomer_forces", "0.03",
        ]
    )
    stage_one = get_loss_fn(args, dipole_only=False, compute_dipole=False)
    assert isinstance(stage_one, LikelihoodHuberLoss)
    swa, _ = get_swa(args, torch.nn.Linear(1, 1), torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1), swas=[False])
    stage_two = swa.loss_fn
    assert isinstance(stage_two, LikelihoodHuberLoss)
    for loss_fn in (stage_one, stage_two):
        assert loss_fn.term_weights()["in_place_monomer_energy"] == 900.0
        assert loss_fn.term_weights()["monomer_forces"] == 100.0
        assert loss_fn.huber_deltas["monomer_forces"] == 0.03
        assert loss_fn.huber_deltas["frame_forces"] == 0.01


@pytest.mark.parametrize(
    "arguments",
    [["--monomer_energy_weight", "0.5"], ["--swa_monomer_forces_weight", "7.0"]],
)
def test_a_separate_monomer_weight_is_refused(arguments):
    from mace.tools.scripts_utils import get_loss_fn, get_swa  # pylint: disable=import-outside-toplevel

    args = _parse(arguments)
    with pytest.raises(ValueError, match="one .* weight for every calculation"):
        get_loss_fn(args, dipole_only=False, compute_dipole=False)
        get_swa(args, torch.nn.Linear(1, 1), torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1), swas=[False])
