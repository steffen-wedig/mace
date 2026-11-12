"""Compiled force training: forward, force gradient and loss as one torch.compile graph.

Compiling only the model forward does not work for force training: the loss backward
then differentiates an AOT-compiled backward, which AOTAutograd refuses ("does not
currently support double backward"). Compiling the whole loss evaluation does: Dynamo
traces the force ``autograd.grad`` into the graph (``trace_autograd_ops``), and
AOTAutograd compiles the backward of the loss, i.e. the parameter gradients through the
force gradient (MACE reforge experiment 06, design note "compiled training seam").

What stays outside the compiled region: the position leaf (``requires_grad_`` on a tensor
created inside the region breaks the graph), ``loss.backward()`` itself (it runs the
compiled backward), gradient clipping, the optimizer and the EMA.

Shapes are dynamic: cluster batches nearly always have a new number of atoms, edges and
pairs, and one graph with symbolic sizes serves them all without recompiling.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch
import torch._dynamo

from mace.tools import compile as mace_compile

# Recompiles beyond this many are an error, not a silent fall-back to eager (reforge F6).
RECOMPILE_LIMIT = 8


def configure_dynamo_for_training() -> None:
    """Dynamo settings for compiled force training; set once by the trainer.

    - ``autograd.grad`` traced into the graph (``trace_autograd_ops``);
    - Python floats specialised: model hyperparameters (smearing widths, cutoffs) stay
      constants under dynamic shapes instead of becoming symbolic;
    - a recompile beyond the limit raises instead of running eagerly.
    """
    import torch._dynamo  # noqa: PLC0415

    mace_compile.configure_autograd_for_compile(allow_autograd=True)
    torch._dynamo.config.specialize_float = True
    torch._dynamo.config.recompile_limit = RECOMPILE_LIMIT
    torch._dynamo.config.fail_on_recompile_limit_hit = True


class CompiledForceTrainingLoss:
    """The training loss of one batch, computed by one compiled graph.

    ``loss = compiled_loss(model, loss_fn, batch)``, then ``loss.backward()`` as usual.
    The model must have been built without e3nn's code generation
    (``mace.tools.compile.disable_e3nn_codegen``) and read no host values in its forward
    (PolarMACE: pbc_handling 'realspace' with the pair list from the batch). The graph
    runs on a simplified view of the model (``mace.tools.compile.simplified_view``) that
    shares its parameters; the model itself is not modified.
    """

    def __init__(self, model: torch.nn.Module, forward_kwargs: Dict[str, bool]):
        if not forward_kwargs.get("compute_force", False):
            raise ValueError("compiled force training needs compute_force")
        for unsupported in ("compute_stress", "compute_virials"):
            if forward_kwargs.get(unsupported, False):
                raise NotImplementedError(
                    f"compiled training does not support {unsupported} yet (its "
                    "displacement leaf is created inside the forward)"
                )
        self.forward_kwargs = dict(forward_kwargs)
        self.model = model
        self.view = mace_compile.simplified_view(model)
        self._compiled = torch.compile(self._loss, fullgraph=True, dynamic=True)

    def _loss(
        self,
        model: torch.nn.Module,
        loss_fn: Callable[..., torch.Tensor],
        batch: Any,
        batch_dict: Dict[str, torch.Tensor],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        output = model({**batch_dict, "positions": positions}, **self.forward_kwargs)
        return loss_fn(pred=output, ref=batch)

    def __call__(
        self, model: torch.nn.Module, loss_fn: Callable[..., torch.Tensor], batch: Any
    ) -> torch.Tensor:
        if model is not self.model:
            raise ValueError("compiled loss called with a model other than its own")
        batch_dict = batch.to_dict()
        # Every per-graph, per-node, per-edge and per-pair size varies between batches.
        # The tensors also reach the graph through the Batch object (the loss
        # reference), where Dynamo would otherwise specialise them on the first batch.
        for value in batch_dict.values():
            if torch.is_tensor(value) and value.dim() > 0:
                torch._dynamo.maybe_mark_dynamic(value, 0)
        positions = batch_dict["positions"].detach().requires_grad_(True)
        torch._dynamo.maybe_mark_dynamic(positions, 0)
        return self._compiled(self.view, loss_fn, batch, batch_dict, positions)


__all__ = [
    "CompiledForceTrainingLoss",
    "RECOMPILE_LIMIT",
    "configure_dynamo_for_training",
]
