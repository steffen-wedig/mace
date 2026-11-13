"""mace.tools.optimizer_probe: Adam's step recomputed from its state equals the step taken."""

import pytest
import torch

from mace.tools.optimizer_probe import OptimizerProbe


def adam_after_steps(amsgrad: bool, steps: int = 7):
    torch.manual_seed(0)
    parameter = torch.nn.Parameter(torch.randn(50, dtype=torch.float64))
    optimizer = torch.optim.Adam(
        [{"params": [parameter], "name": "weights"}], lr=1e-2, amsgrad=amsgrad
    )
    before = None
    for step in range(steps):
        scale = 1.0 if step < steps - 2 else 100.0  # a gradient-scale jump at the end
        parameter.grad = torch.randn(50, dtype=torch.float64) * scale + 0.3
        before = parameter.detach().clone()
        optimizer.step()
    return parameter, optimizer, before


@pytest.mark.parametrize("amsgrad", [True, False])
def test_the_recomputed_update_is_the_step_adam_took(amsgrad):
    parameter, optimizer, before = adam_after_steps(amsgrad)

    record = OptimizerProbe(interval=1).record(optimizer)

    taken = (parameter.detach() - before) / 1e-2
    statistics = record["groups"]["weights"]
    assert statistics["update_rms_over_learning_rate"] == pytest.approx(
        float(taken.pow(2).mean().sqrt()), rel=1e-9
    )
    assert statistics["parameter_count"] == 50
    assert 0.0 <= statistics["signal_fraction"] <= 1.0
    assert record["learning_rate"] == 1e-2


def test_the_ratchet_fraction_counts_raised_maxima():
    parameter, optimizer, _ = adam_after_steps(amsgrad=True)
    probe = OptimizerProbe(interval=1)
    probe.record(optimizer)  # first record: no previous maximum to compare with

    parameter.grad = torch.full((50,), 1e3, dtype=torch.float64)  # raises every maximum
    optimizer.step()
    raised = probe.record(optimizer)["groups"]["weights"]["second_moment_raised_fraction"]

    assert raised == 1.0


def test_dense_after_the_switch_then_every_interval():
    probe = OptimizerProbe(interval=10, dense_steps=3)
    due = []
    for step in range(25):
        if step == 12:
            probe.stage_switch()
        due.append(probe.due())
        probe.advance()

    assert [step for step, is_due in enumerate(due) if is_due] == [0, 10, 12, 13, 14, 20]
