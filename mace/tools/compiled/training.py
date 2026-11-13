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

Two shape modes:

- dynamic (default): cluster batches nearly always have a new number of atoms, edges and
  pairs, and one graph with symbolic sizes serves them all without recompiling;
- static (``static_shapes=True``): batches padded to a few fixed budgets
  (``mace.data.padding``), one graph per budget, replayed as CUDA graphs
  (``mode="reduce-overhead"``). CUDA graphs with dynamic shapes are unsupported (reforge
  experiment 06: segfaults). The rare batch larger than every budget arrives unpadded
  and runs eagerly.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch
import torch._dynamo
import torch._inductor

from mace.tools import compile as mace_compile
from mace.tools.compiled import small_matmuls

# Recompiles beyond this many are an error, not a silent fall-back to eager (reforge F6).
RECOMPILE_LIMIT = 8


def configure_dynamo_for_training(recompile_limit: int = RECOMPILE_LIMIT) -> None:
    """Dynamo settings for compiled force training; set once by the trainer.

    - ``autograd.grad`` traced into the graph (``trace_autograd_ops``);
    - Python floats specialised: model hyperparameters (smearing widths, cutoffs) stay
      constants under dynamic shapes instead of becoming symbolic;
    - a recompile beyond the limit raises instead of running eagerly (static shapes:
      at least one graph per padding budget and loss, the SWA loss being a second one).

    Inductor options of the training graph itself: ``training_inductor_options``.
    """
    import torch._dynamo  # noqa: PLC0415

    mace_compile.configure_autograd_for_compile(allow_autograd=True)
    torch._dynamo.config.specialize_float = True
    torch._dynamo.config.recompile_limit = recompile_limit
    torch._dynamo.config.fail_on_recompile_limit_hit = True


def training_inductor_options(static_shapes: bool) -> Dict[str, Any]:
    """Inductor options of the compiled training loss, passed to its ``torch.compile``:

    - small matrix products as fused reductions (``mace.tools.compiled.small_matmuls``);
    - with static shapes (padded batches), CUDA graphs (``mode="reduce-overhead"``).
    """
    options = small_matmuls.inductor_options()
    if static_shapes:
        options.update(torch._inductor.list_mode_options("reduce-overhead"))
    return options


class CompiledForceTrainingLoss:
    """The training loss of one batch, computed by one compiled graph.

    ``loss = compiled_loss(model, loss_fn, batch)``, then ``loss.backward()`` as usual.
    The model must have been built without e3nn's code generation
    (``mace.tools.compile.disable_e3nn_codegen``) and read no host values in its forward
    (PolarMACE: pbc_handling 'realspace' with the pair list from the batch). The graph
    runs on a simplified view of the model (``mace.tools.compile.simplified_view``) that
    shares its parameters; the model itself is not modified.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        forward_kwargs: Dict[str, bool],
        static_shapes: bool = False,
    ):
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
        self.static_shapes = static_shapes
        # Unpadded batches run eagerly under static shapes (counted for the logs).
        self.eager_batches = 0
        self._compiled = torch.compile(
            self._loss,
            fullgraph=True,
            dynamic=not static_shapes,
            options=training_inductor_options(static_shapes),
        )

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
        if self.static_shapes:
            if batch_dict.get("real_graph_mask") is None:
                # A batch larger than every padding budget (mace.data.padding): eager,
                # rather than a new graph for a shape that may never come back.
                self.eager_batches += 1
                positions = batch_dict["positions"].detach().requires_grad_(True)
                output = model(
                    {**batch_dict, "positions": positions}, **self.forward_kwargs
                )
                return loss_fn(pred=output, ref=batch)
            positions = batch_dict["positions"].detach().requires_grad_(True)
            # A new training step: outputs of the previous replay may be overwritten.
            torch.compiler.cudagraph_mark_step_begin()
            return self._compiled(self.view, loss_fn, batch, batch_dict, positions)
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
