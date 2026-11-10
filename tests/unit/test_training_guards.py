"""Guards that stop a run whose loss never sees its data.

A from-scratch run on aperiodic clusters with ``interaction_huber`` (whose frame terms
count periodic configurations only) trained for a whole night on a loss of exactly 0:
zero gradients, with Adam's weight decay alone collapsing the model.
"""

import pytest
import torch

from mace.modules.loss import WeightedHuberEnergyForcesStressLoss
from mace.tools.loss_terms import require_entries_for_weighted_terms
from mace.tools.train import require_nonzero_epoch_loss


class _Batch(dict):
    """The parts of a batch the Huber loss reads: items and ``ptr``."""

    def __init__(self, ptr, **items):
        super().__init__(**items)
        self.ptr = ptr


def _batch_and_prediction():
    torch.manual_seed(0)
    ptr = torch.tensor([0, 3, 5])
    batch = _Batch(
        ptr,
        energy=torch.tensor([-3.0, -2.0]),
        forces=torch.randn(5, 3),
        stress=torch.zeros(2, 3, 3),
    )
    prediction = {
        "energy": torch.tensor([-2.9, -2.2]),
        "forces": batch["forces"] + 0.1,
        "stress": None,
    }
    return batch, prediction


def test_a_zero_loss_epoch_is_refused():
    with pytest.raises(RuntimeError, match="exactly zero on every batch"):
        require_nonzero_epoch_loss(torch.tensor(0.0), epoch=0, loss_fn=torch.nn.Module())


def test_an_epoch_without_batches_is_refused():
    with pytest.raises(RuntimeError, match="no batch"):
        require_nonzero_epoch_loss(None, epoch=3, loss_fn=torch.nn.Module())


def test_a_nonzero_loss_epoch_passes():
    require_nonzero_epoch_loss(torch.tensor(1e-12), epoch=0, loss_fn=torch.nn.Module())


def _term(weight, entry_count):
    return {"weight": weight, "entry_count": entry_count}


def test_a_weighted_term_without_entries_is_refused():
    summary = {
        "frame_energy": _term(1.0, 0.0),
        "frame_forces": _term(100.0, 0.0),
        "interaction_energy": _term(0.0, 0.0),
    }
    with pytest.raises(RuntimeError, match=r"\['frame_energy', 'frame_forces'\]"):
        require_entries_for_weighted_terms(summary, epoch=0)


def test_zero_weight_terms_without_entries_pass():
    summary = {
        "frame_energy": _term(1.0, 400.0),
        "interaction_energy": _term(0.0, 0.0),
        "interaction_forces": _term(None, 0.0),
    }
    require_entries_for_weighted_terms(summary, epoch=0)


def test_huber_without_stress_weight_needs_no_stress_prediction():
    batch, prediction = _batch_and_prediction()
    loss = WeightedHuberEnergyForcesStressLoss(
        energy_weight=1.0, forces_weight=100.0, stress_weight=0.0, huber_delta=0.01
    )
    value = loss(ref=batch, pred=prediction)
    assert float(value) > 0.0
    expected = torch.nn.functional.huber_loss(
        batch["energy"] / torch.tensor([3, 2]),
        prediction["energy"] / torch.tensor([3, 2]),
        delta=0.01,
    ) + 100.0 * torch.nn.functional.huber_loss(
        batch["forces"], prediction["forces"], delta=0.01
    )
    assert float(value) == pytest.approx(float(expected))


def test_huber_with_stress_weight_refuses_a_missing_stress_prediction():
    batch, prediction = _batch_and_prediction()
    loss = WeightedHuberEnergyForcesStressLoss(stress_weight=1.0)
    with pytest.raises(ValueError, match="predicts no stress"):
        loss(ref=batch, pred=prediction)
