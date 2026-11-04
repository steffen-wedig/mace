"""Linear learning-rate warm-up of the training loop (--lr_warmup_steps)."""

import pytest
import torch

from mace.tools.arg_parser import build_default_arg_parser
from mace.tools.train import LinearLRWarmup


def make_optimizer() -> torch.optim.Optimizer:
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(3))
    return torch.optim.Adam(
        [{"params": [first], "lr": 1e-3}, {"params": [second], "lr": 1e-2}]
    )


def run_steps(warmup: LinearLRWarmup, optimizer, count: int) -> list:
    learning_rates = []
    for _ in range(count):
        warmup.before_step(optimizer)
        learning_rates.append([group["lr"] for group in optimizer.param_groups])
        warmup.after_step(optimizer)
    return learning_rates


def test_learning_rate_rises_linearly_per_group_then_is_left_alone() -> None:
    optimizer = make_optimizer()
    learning_rates = run_steps(LinearLRWarmup(4), optimizer, 6)
    expected_factors = [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
    for step_rates, factor in zip(learning_rates, expected_factors):
        assert step_rates == pytest.approx([1e-3 * factor, 1e-2 * factor])
    # After the warm-up an epoch-level scheduler owns the learning rate.
    optimizer.param_groups[0]["lr"] = 5e-4
    LinearLRWarmup(4, start_step=4).before_step(optimizer)
    assert optimizer.param_groups[0]["lr"] == 5e-4


def test_zero_warmup_never_touches_the_optimizer() -> None:
    optimizer = make_optimizer()
    assert run_steps(LinearLRWarmup(0), optimizer, 3) == [[1e-3, 1e-2]] * 3


def test_a_restart_resumes_the_warmup() -> None:
    optimizer = make_optimizer()
    learning_rates = run_steps(LinearLRWarmup(4, start_step=2), optimizer, 3)
    assert learning_rates[0] == pytest.approx([0.75e-3, 0.75e-2])
    assert learning_rates[2] == pytest.approx([1e-3, 1e-2])


def test_negative_warmup_is_rejected() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        LinearLRWarmup(-1)


def test_the_cli_flag_defaults_to_off() -> None:
    parser = build_default_arg_parser()
    base = ["--name", "unit", "--train_file", "train.xyz"]
    assert parser.parse_args(base).lr_warmup_steps == 0
    assert parser.parse_args([*base, "--lr_warmup_steps", "500"]).lr_warmup_steps == 500
