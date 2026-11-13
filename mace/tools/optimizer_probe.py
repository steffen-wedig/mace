"""Periodic statistics of Adam's state per parameter group, from the state it already holds.

After an optimizer step the gradient (clipped) is still in ``.grad`` and the moments are in
``optimizer.state``; Adam's normalised update is recomputed from them, so the probe needs no
extra forward or backward pass and no parameter copies. Logged per group:

* ``gradient_norm``: norm of the (clipped) gradient;
* ``update_rms_over_learning_rate``: RMS of m_hat / (sqrt(v_hat) + eps), the step in units of
  the learning rate (1 = sign descent, << 1 = noise-dominated);
* ``update_to_weight``: norm of the step over the norm of the weights;
* ``signal_fraction``: fraction of coordinates with |m_hat| / sqrt(v_hat) > 0.5 (coherent
  gradient: those coordinates move by about the learning rate per step);
* ``second_moment_raised_fraction``: with AMSGrad, fraction of coordinates whose running
  maximum v_max grew since the previous probe (the ratchet);
* ``root_second_moment_mean``: mean of sqrt(v_hat).

Records go to the run's results file as ``mode: "optimizer_probe"`` every ``interval`` steps,
and at every one of the first ``dense_steps`` steps after the stage switch.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch


class OptimizerProbe:
    def __init__(self, interval: int, dense_steps: int = 100) -> None:
        if interval <= 0:
            raise ValueError(f"interval must be > 0, got {interval}")
        self.interval = interval
        self.dense_steps = dense_steps
        self.step = 0
        self.steps_since_switch: Optional[int] = None
        self.previous_maximum: Dict[int, torch.Tensor] = {}

    def stage_switch(self) -> None:
        self.steps_since_switch = 0

    def due(self) -> bool:
        if self.steps_since_switch is not None and self.steps_since_switch < self.dense_steps:
            return True
        return self.step % self.interval == 0

    def advance(self) -> None:
        self.step += 1
        if self.steps_since_switch is not None:
            self.steps_since_switch += 1

    @torch.no_grad()
    def record(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        groups = {}
        for index, group in enumerate(optimizer.param_groups):
            name = group.get("name", f"group_{index}")
            statistics = self._group_statistics(optimizer, group, index)
            if statistics is not None:
                groups[name] = statistics
        return {
            "mode": "optimizer_probe",
            "probe_step": self.step,
            "steps_since_switch": self.steps_since_switch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "groups": groups,
        }

    def _group_statistics(
        self, optimizer: torch.optim.Optimizer, group: Dict[str, Any], index: int
    ) -> Optional[Dict[str, float]]:
        parameters: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []
        first: List[torch.Tensor] = []
        second: List[torch.Tensor] = []
        maximum: List[torch.Tensor] = []
        step_count = None
        amsgrad = bool(group.get("amsgrad", False))
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if parameter.grad is None or not state or "exp_avg" not in state:
                continue
            parameters.append(parameter.detach().flatten())
            gradients.append(parameter.grad.detach().flatten())
            first.append(state["exp_avg"].flatten())
            second.append(state["exp_avg_sq"].flatten())
            if amsgrad and "max_exp_avg_sq" in state:
                maximum.append(state["max_exp_avg_sq"].flatten())
            step_count = float(state["step"])
        if not parameters or step_count is None:
            return None
        beta1, beta2 = group["betas"]
        first_correction = 1.0 - beta1**step_count
        second_correction = 1.0 - beta2**step_count
        weights = torch.cat(parameters)
        gradient = torch.cat(gradients)
        first_moment = torch.cat(first) / first_correction
        use_maximum = amsgrad and len(maximum) == len(first)
        raw_second = torch.cat(maximum) if use_maximum else torch.cat(second)
        root_second = (raw_second / second_correction).sqrt()
        normalised_update = first_moment / (root_second + group["eps"])
        signal = first_moment.abs() / root_second.clamp_min(1e-300)
        learning_rate = float(group["lr"])
        weight_norm = float(weights.norm())
        statistics = {
            "parameter_count": int(weights.numel()),
            "gradient_norm": float(gradient.norm()),
            "update_rms_over_learning_rate": float(normalised_update.pow(2).mean().sqrt()),
            "update_to_weight": (
                learning_rate * float(normalised_update.norm()) / weight_norm
                if weight_norm > 0.0
                else math.nan
            ),
            "signal_fraction": float((signal > 0.5).double().mean()),
            "root_second_moment_mean": float(root_second.mean()),
        }
        if use_maximum:
            previous = self.previous_maximum.get(index)
            if previous is not None and previous.shape == raw_second.shape:
                statistics["second_moment_raised_fraction"] = float(
                    (raw_second > previous).double().mean()
                )
            self.previous_maximum[index] = raw_second.clone()
        return statistics


__all__ = ["OptimizerProbe"]
