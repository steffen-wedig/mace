"""Exponential moving average of the parameters, updated with one foreach operation.

``torch_ema.ExponentialMovingAverage.update`` loops over the parameters in Python with
three kernels each (subtract, scale, subtract): 426 launches and ~1.2 ms of host time
per training step for POLAR-M, more than the compiled step's own launches under CUDA
graphs. The same update as one ``torch._foreach_lerp_`` over all parameters: shadow +
(1 - decay) * (parameter - shadow) = shadow - (1 - decay) * (shadow - parameter).

State dicts, ``average_parameters``, ``copy_to``, ``store``/``restore`` are those of
``torch_ema``; checkpoints stay interchangeable.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch_ema import ExponentialMovingAverage


class ForeachExponentialMovingAverage(ExponentialMovingAverage):
    def update(
        self, parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        parameters = list(self._get_parameters(parameters))
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            torch._foreach_lerp_(
                self.shadow_params, [parameter.detach() for parameter in parameters], 1.0 - decay
            )


__all__ = ["ForeachExponentialMovingAverage"]
