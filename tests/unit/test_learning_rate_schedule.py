"""The per-step two-stage learning rate (mace.tools.learning_rate_schedule), the optimizer
change at the stage switch, and checkpoints that keep the raw and the averaged weights."""

import pytest
import torch
from torch_ema import ExponentialMovingAverage

from mace.tools.checkpoint import CheckpointBuilder, CheckpointState
from mace.tools.learning_rate_schedule import (
    StageLearningRate,
    StageTwoOptimizerChange,
    TwoStageLearningRateSchedule,
    group_learning_rate_factors,
)


def test_a_plain_stage_is_constant():
    stage = StageLearningRate(peak=1e-3)

    assert [stage.factor(step, 100, 10) for step in (0, 1, 50, 99)] == [1.0] * 4


def test_hold_then_linear_warmup():
    stage = StageLearningRate(peak=1e-4, hold_steps=3, warmup_steps=4)

    factors = [stage.factor(step, 1000, 10) for step in range(9)]

    assert factors == pytest.approx([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1])


def test_decay_over_the_last_epochs_reaches_the_final_factor():
    stage = StageLearningRate(peak=1.0, decay_epochs=2, final_factor=0.1)
    steps_per_epoch, stage_steps = 10, 50

    assert stage.factor(29, stage_steps, steps_per_epoch) == pytest.approx(1.0)
    assert stage.factor(30, stage_steps, steps_per_epoch) == pytest.approx(1.0)
    assert stage.factor(40, stage_steps, steps_per_epoch) == pytest.approx(0.55)
    # The last step is one step short of the end of the decay.
    assert stage.factor(49, stage_steps, steps_per_epoch) == pytest.approx(
        1.0 - 0.9 * 19 / 20
    )


def test_invalid_stages_are_refused():
    with pytest.raises(ValueError):
        StageLearningRate(peak=1e-3, final_factor=1.5)
    with pytest.raises(ValueError):
        StageLearningRate(peak=1e-3, warmup_steps=-1)


def two_parameter_groups():
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(3))
    return torch.optim.Adam(
        [{"params": [first], "lr": 1e-3}, {"params": [second], "lr": 0.5e-3}],
        lr=1e-3,
        amsgrad=True,
    )


def test_stage_two_keeps_the_group_factors_and_leaves_stage_one_to_the_epoch_scheduler():
    optimizer = two_parameter_groups()
    schedule = TwoStageLearningRateSchedule(
        group_factors=group_learning_rate_factors(optimizer, 1e-3),
        stage_one=None,
        stage_two=StageLearningRate(peak=1e-4, warmup_steps=2),
        stage_two_start_epoch=3,
        max_num_epochs=5,
        steps_per_epoch=4,
    )

    schedule.apply(optimizer, epoch=2, step_in_epoch=3)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 0.5e-3]

    schedule.apply(optimizer, epoch=3, step_in_epoch=0)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.5e-4, 0.25e-4])

    schedule.apply(optimizer, epoch=4, step_in_epoch=0)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([1e-4, 0.5e-4])


def test_a_piecewise_stage_one_warms_up_and_decays_into_the_switch():
    schedule = TwoStageLearningRateSchedule(
        group_factors=[1.0],
        stage_one=StageLearningRate(peak=1e-3, warmup_steps=10, decay_epochs=1, final_factor=0.1),
        stage_two=StageLearningRate(peak=1e-4),
        stage_two_start_epoch=4,
        max_num_epochs=6,
        steps_per_epoch=10,
    )

    assert schedule.learning_rate(0, 0) == pytest.approx(1e-4)
    assert schedule.learning_rate(1, 0) == pytest.approx(1e-3)
    assert schedule.learning_rate(3, 5) == pytest.approx(1e-3 * (1 - 0.9 * 0.5))
    assert schedule.learning_rate(4, 0) == pytest.approx(1e-4)


def adam_with_state():
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.Adam([parameter], lr=1e-3, amsgrad=True)
    parameter.grad = torch.ones(4)
    optimizer.step()
    return optimizer


def test_the_optimizer_change_resets_and_sets_beta2_and_amsgrad():
    optimizer = adam_with_state()

    StageTwoOptimizerChange(reset_state=True, beta2=0.99, amsgrad=False).apply(optimizer)

    assert len(optimizer.state) == 0
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.99)
    assert optimizer.param_groups[0]["amsgrad"] is False


def test_the_default_optimizer_change_changes_nothing():
    optimizer = adam_with_state()
    before = {key: value for key, value in optimizer.param_groups[0].items() if key != "params"}

    StageTwoOptimizerChange().apply(optimizer)

    assert len(optimizer.state) == 1
    assert {
        key: value for key, value in optimizer.param_groups[0].items() if key != "params"
    } == before


def trained_model_with_ema():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
    for _ in range(5):
        optimizer.zero_grad()
        model(torch.randn(4, 3)).pow(2).sum().backward()
        optimizer.step()
        ema.update()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return model, optimizer, scheduler, ema


def test_a_checkpoint_keeps_the_raw_and_the_averaged_weights():
    model, optimizer, scheduler, ema = trained_model_with_ema()
    raw = [parameter.detach().clone() for parameter in model.parameters()]
    averaged = [shadow.clone() for shadow in ema.shadow_params]
    assert not torch.allclose(raw[0], averaged[0])

    checkpoint = CheckpointBuilder.create_checkpoint(
        CheckpointState(model, optimizer, scheduler, ema)
    )

    # Exports read `model`: the averaged weights.
    torch.testing.assert_close(checkpoint["model"]["weight"], averaged[0])
    torch.testing.assert_close(checkpoint["raw_model"]["weight"], raw[0])
    # Saving leaves the raw weights in the model.
    torch.testing.assert_close(model.weight.detach(), raw[0])

    resumed, resumed_optimizer, resumed_scheduler, resumed_ema = trained_model_with_ema()
    with torch.no_grad():
        for parameter in resumed.parameters():
            parameter.add_(7.0)
    resumed_ema.num_updates = 0
    CheckpointBuilder.load_checkpoint(
        CheckpointState(resumed, resumed_optimizer, resumed_scheduler, resumed_ema),
        checkpoint,
        strict=True,
    )
    for parameter, expected in zip(resumed.parameters(), raw):
        torch.testing.assert_close(parameter.detach(), expected)
    for shadow, expected in zip(resumed_ema.shadow_params, averaged):
        torch.testing.assert_close(shadow, expected)
    assert resumed_ema.num_updates == ema.num_updates


def test_an_export_load_gets_the_averaged_weights():
    model, optimizer, scheduler, ema = trained_model_with_ema()
    averaged = [shadow.clone() for shadow in ema.shadow_params]
    checkpoint = CheckpointBuilder.create_checkpoint(
        CheckpointState(model, optimizer, scheduler, ema)
    )
    export_model, export_optimizer, export_scheduler, _ = trained_model_with_ema()

    CheckpointBuilder.load_checkpoint(
        CheckpointState(export_model, export_optimizer, export_scheduler), checkpoint, strict=True
    )

    for parameter, expected in zip(export_model.parameters(), averaged):
        torch.testing.assert_close(parameter.detach(), expected)


def test_restored_moments_take_their_parameter_shape():
    """An e3nn checkpoint (weights (n,)) restored into the cueq layout ((1, n))."""
    from mace.tools.checkpoint import reshape_state_to_parameters  # noqa: PLC0415

    flat = torch.nn.Parameter(torch.ones(6))
    saved = torch.optim.Adam([flat], lr=1e-3, amsgrad=True)
    flat.grad = torch.arange(6.0)
    saved.step()
    row = torch.nn.Parameter(torch.ones(1, 6))
    restored = torch.optim.Adam([row], lr=1e-3, amsgrad=True)
    restored.load_state_dict(saved.state_dict())

    reshape_state_to_parameters(restored)

    state = restored.state[row]
    assert state["exp_avg"].shape == (1, 6)
    assert state["max_exp_avg_sq"].shape == (1, 6)
    torch.testing.assert_close(state["exp_avg"].flatten(), saved.state[flat]["exp_avg"])
    row.grad = torch.ones(1, 6)
    restored.step()


def test_scheduler_state_of_another_kind_is_dropped(caplog):
    from mace.tools.checkpoint import restore_scheduler_state

    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    placeholder = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    with caplog.at_level("WARNING"):
        restore_scheduler_state(placeholder, plateau.state_dict())
    assert "does not match" in caplog.text
    other_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7)
    restore_scheduler_state(other_plateau, plateau.state_dict())
    assert other_plateau.patience == 3


def test_latest_checkpoint_of_a_missing_kind_is_none(tmp_path):
    from mace.tools.checkpoint import CheckpointIO

    (tmp_path / "run_epoch-4.pt").write_bytes(b"")
    io = CheckpointIO(str(tmp_path), tag="run", swa_start=10)
    assert io._get_latest_checkpoint_path(swa=True) is None  # pylint: disable=W0212
    assert io._get_latest_checkpoint_path(swa=False).endswith("run_epoch-4.pt")  # pylint: disable=W0212
