"""Per-step learning rate of a two-stage training run, and the optimizer change at the switch.

Each stage has its own peak learning rate (stage one ``--lr``, stage two ``--swa_lr``) and,
counted in optimizer steps from the stage's first step:

* ``hold_steps`` at learning rate 0: the optimizer's moments follow the stage's loss without
  moving the parameters (stage two: Adam's second moments adapt to the new loss scale before
  the first real step);
* ``warmup_steps`` of a linear ramp from 0 to the peak;
* the peak, constant;
* over the stage's last ``decay_epochs`` epochs, a linear decay to ``final_factor`` x peak.

The schedule is a pure function of (epoch, step within the epoch), so a resumed run continues
it without any saved state. Each parameter group keeps its learning-rate factor
(``--lr_params_factors``) in both stages.

With stage one under an epoch scheduler (``--scheduler ReduceLROnPlateau`` or
``ExponentialLR``) the schedule leaves stage one alone and only sets stage two. The stage-two
defaults (no hold, no warm-up, no decay) reproduce the former SWALR(anneal_epochs=1) jump to
``swa_lr`` at the first stage-two step, except that the group factors are kept.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import List, Optional

import torch


@dataclasses.dataclass(frozen=True)
class StageLearningRate:
    peak: float
    hold_steps: int = 0
    warmup_steps: int = 0
    decay_epochs: int = 0
    final_factor: float = 0.0

    def __post_init__(self) -> None:
        if self.peak < 0.0:
            raise ValueError(f"peak learning rate must be >= 0, got {self.peak}")
        if self.hold_steps < 0 or self.warmup_steps < 0 or self.decay_epochs < 0:
            raise ValueError(f"step and epoch counts must be >= 0: {self}")
        if not 0.0 <= self.final_factor <= 1.0:
            raise ValueError(f"final_factor must be in [0, 1], got {self.final_factor}")

    def factor(self, step_in_stage: int, stage_steps: int, steps_per_epoch: int) -> float:
        """Learning rate of step ``step_in_stage`` (0-based) as a fraction of the peak."""
        if step_in_stage < self.hold_steps:
            return 0.0
        ramp = 1.0
        if self.warmup_steps > 0:
            ramp = min(1.0, (step_in_stage - self.hold_steps + 1) / self.warmup_steps)
        decay_steps = min(stage_steps, self.decay_epochs * steps_per_epoch)
        decay_start = stage_steps - decay_steps
        decay = 1.0
        if decay_steps > 0 and step_in_stage >= decay_start:
            progress = (step_in_stage - decay_start) / decay_steps
            decay = 1.0 - (1.0 - self.final_factor) * progress
        return ramp * decay


@dataclasses.dataclass(frozen=True)
class StageTwoOptimizerChange:
    """What happens to the optimizer at the first stage-two step (once per run)."""

    reset_state: bool = False
    beta2: Optional[float] = None  # None: keep stage one's
    amsgrad: Optional[bool] = None  # None: keep stage one's

    def apply(self, optimizer: torch.optim.Optimizer) -> None:
        if self.reset_state:
            optimizer.state.clear()
            logging.info("Stage two: optimizer state reset")
        for group in optimizer.param_groups:
            if self.beta2 is not None and "betas" in group:
                group["betas"] = (group["betas"][0], self.beta2)
            if self.amsgrad is not None and "amsgrad" in group:
                group["amsgrad"] = self.amsgrad
        if self.beta2 is not None or self.amsgrad is not None:
            logging.info(
                f"Stage two: optimizer beta2 {self.beta2}, amsgrad {self.amsgrad} "
                "(None: unchanged)"
            )


class TwoStageLearningRateSchedule:
    """Sets every parameter group's learning rate before each optimizer step."""

    def __init__(
        self,
        group_factors: List[float],
        stage_one: Optional[StageLearningRate],
        stage_two: Optional[StageLearningRate],
        stage_two_start_epoch: Optional[int],
        max_num_epochs: int,
        steps_per_epoch: int,
    ) -> None:
        if steps_per_epoch <= 0:
            raise ValueError(f"steps_per_epoch must be > 0, got {steps_per_epoch}")
        if (stage_two is None) != (stage_two_start_epoch is None):
            raise ValueError("stage_two and stage_two_start_epoch must be given together")
        self.stage_one = stage_one
        self.stage_two = stage_two
        self.stage_two_start_epoch = stage_two_start_epoch
        self.max_num_epochs = max_num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.group_factors = list(group_factors)

    def in_stage_two(self, epoch: int) -> bool:
        return self.stage_two_start_epoch is not None and epoch >= self.stage_two_start_epoch

    def learning_rate(self, epoch: int, step_in_epoch: int) -> Optional[float]:
        """The peak-scaled learning rate at this step, or None where the schedule does not
        own the learning rate (stage one under an epoch scheduler)."""
        if self.in_stage_two(epoch):
            assert self.stage_two is not None and self.stage_two_start_epoch is not None
            stage, first_epoch, last_epoch = (
                self.stage_two,
                self.stage_two_start_epoch,
                self.max_num_epochs,
            )
        else:
            if self.stage_one is None:
                return None
            stage, first_epoch = self.stage_one, 0
            last_epoch = (
                self.stage_two_start_epoch
                if self.stage_two_start_epoch is not None
                else self.max_num_epochs
            )
        step_in_stage = (epoch - first_epoch) * self.steps_per_epoch + step_in_epoch
        stage_steps = max(1, (last_epoch - first_epoch) * self.steps_per_epoch)
        return stage.peak * stage.factor(step_in_stage, stage_steps, self.steps_per_epoch)

    def apply(self, optimizer: torch.optim.Optimizer, epoch: int, step_in_epoch: int) -> None:
        learning_rate = self.learning_rate(epoch, step_in_epoch)
        if learning_rate is None:
            return
        if len(optimizer.param_groups) != len(self.group_factors):
            raise ValueError(
                f"{len(optimizer.param_groups)} parameter groups, "
                f"{len(self.group_factors)} learning-rate factors"
            )
        for group, factor in zip(optimizer.param_groups, self.group_factors):
            group["lr"] = factor * learning_rate


def group_learning_rate_factors(
    optimizer: torch.optim.Optimizer, base_learning_rate: float
) -> List[float]:
    """Each parameter group's learning rate relative to ``base_learning_rate`` (--lr): the
    --lr_params_factors. Read from a new optimizer, before a checkpoint changes its rates."""
    if base_learning_rate <= 0.0:
        raise ValueError(f"base learning rate must be > 0, got {base_learning_rate}")
    return [float(group["lr"]) / base_learning_rate for group in optimizer.param_groups]


__all__ = [
    "StageLearningRate",
    "StageTwoOptimizerChange",
    "TwoStageLearningRateSchedule",
    "group_learning_rate_factors",
]
