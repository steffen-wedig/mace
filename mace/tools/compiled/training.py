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

Three shape modes (``ShapeMode``):

- ``dynamic``: unpadded cluster batches nearly always have a new number of atoms, edges
  and pairs, and one graph with symbolic sizes serves them all without recompiling; no
  CUDA graphs;
- ``padded_per_budget``: batches padded to a few fixed budgets (``mace.data.padding``),
  one static-shape graph per budget, each replayed as a CUDA graph
  (``mode="reduce-overhead"``). Every budget pays the full compile (Dynamo, AOTAutograd,
  Inductor): ~2 min per budget for POLAR-M in float64;
- ``padded_shared``: the same padded batches, but one graph with symbolic sizes for all
  budgets, compiled once. Inductor's CUDA-graph trees record one CUDA graph per distinct
  budget (``cudagraph_skip_dynamic_graphs`` off), so the replay is the same as
  ``padded_per_budget`` while the compile is paid once. (Reforge experiment 06 saw
  segfaults with CUDA graphs on unpadded dynamic shapes, where every batch is a new
  shape; padded budgets bound the recordings.)

In both padded modes the rare batch larger than every budget arrives unpadded and runs
eagerly.

Electrostatics (PolarMACE): open structures in real space, with the directed atom pairs
listed at collation (``pbc_handling`` 'realspace', ``mace.data.real_space_pairs``), or
fully periodic structures in reciprocal space, with the integer lattice indices of the
k-vectors within the model's cutoff listed at collation (``pbc_handling`` 'pbc',
``mace.data.reciprocal_lattice``): the forward forms the k-vectors from the reciprocal
cell and applies the cutoff in the graph, so the k-vector count is a batch size like the
others (padded to a per-budget row count, excluded padding rows contributing exactly
zero) and the Ewald sums, their force gradient and the backward through it are all in
the graph.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal, get_args

import torch
import torch._dynamo
import torch._inductor

from mace.tools import compile as mace_compile

# Recompiles beyond this many are an error, not a silent fall-back to eager (reforge F6).
RECOMPILE_LIMIT = 8

ShapeMode = Literal["dynamic", "padded_per_budget", "padded_shared"]
SHAPE_MODES: tuple[str, ...] = get_args(ShapeMode)

# Batch tensors whose varying size is the last dimension (COO index pairs, [2, n]);
# every other batch tensor varies along dimension 0.
TRAILING_SIZE_KEYS = frozenset({"edge_index", "real_space_pair_index"})


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
    # padded_shared: one CUDA graph per padding budget is expected, not a sign of an
    # unbounded stream of shapes; the budget count is bounded by --padding_buckets.
    torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None


def training_inductor_options(shape_mode: ShapeMode) -> Dict[str, Any]:
    """Inductor options of the compiled training loss, passed to its ``torch.compile``:
    with padded batches (both padded modes), CUDA graphs (``mode="reduce-overhead"``).
    """
    options: Dict[str, Any] = {}
    if shape_mode != "dynamic":
        options.update(torch._inductor.list_mode_options("reduce-overhead"))
    return options


class CompiledForceTrainingLoss:
    """The training loss of one batch, computed by one compiled graph.

    ``loss = compiled_loss(model, loss_fn, batch)``, then ``loss.backward()`` as usual.
    The model must have been built without e3nn's code generation
    (``mace.tools.compile.disable_e3nn_codegen``) and read no host values in its forward
    (PolarMACE: pbc_handling 'realspace' with the pair list from the batch, or 'pbc' with
    the reciprocal-lattice indices from the batch). The graph
    runs on a simplified view of the model (``mace.tools.compile.simplified_view``) that
    shares its parameters; the model itself is not modified.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        forward_kwargs: Dict[str, bool],
        shape_mode: ShapeMode = "dynamic",
    ):
        if shape_mode not in SHAPE_MODES:
            raise ValueError(
                f"shape_mode must be one of {SHAPE_MODES}, got {shape_mode!r}"
            )
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
        self.shape_mode = shape_mode
        # Unpadded batches run eagerly in the padded modes (counted for the logs).
        self.eager_batches = 0
        self._compiled = torch.compile(
            self._loss,
            fullgraph=True,
            # padded_shared marks the padded sizes dynamic per call (_mark_sizes_dynamic);
            # dynamic=None leaves everything else static instead of guessing.
            dynamic={
                "dynamic": True,
                "padded_per_budget": False,
                "padded_shared": None,
            }[shape_mode],
            options=training_inductor_options(shape_mode),
        )

    @staticmethod
    def _mark_sizes_dynamic(
        batch_dict: Dict[str, torch.Tensor], positions: torch.Tensor
    ):
        """Symbolic leading sizes: every per-graph, per-node, per-edge and per-pair size.

        The tensors also reach the graph through the Batch object (the loss reference),
        where Dynamo would otherwise specialise them on the first batch.
        """
        for key, value in batch_dict.items():
            if torch.is_tensor(value) and value.dim() > 0:
                torch._dynamo.maybe_mark_dynamic(value, 0)
                if key in TRAILING_SIZE_KEYS:
                    torch._dynamo.maybe_mark_dynamic(value, value.dim() - 1)
        torch._dynamo.maybe_mark_dynamic(positions, 0)

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
        if (
            getattr(model, "pbc_handling", None) == "pbc"
            and "kspace_miller" not in batch_dict
        ):
            raise ValueError(
                "compiled training with pbc_handling 'pbc' needs the reciprocal-lattice "
                "indices in the batch (a loader with kspace_cutoff, "
                "mace.data.reciprocal_lattice)"
            )
        if self.shape_mode != "dynamic":
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
            if self.shape_mode == "padded_shared":
                self._mark_sizes_dynamic(batch_dict, positions)
            # A new training step: outputs of the previous replay may be overwritten.
            torch.compiler.cudagraph_mark_step_begin()
            return self._compiled(self.view, loss_fn, batch, batch_dict, positions)
        # Every per-graph, per-node, per-edge and per-pair size varies between batches.
        positions = batch_dict["positions"].detach().requires_grad_(True)
        self._mark_sizes_dynamic(batch_dict, positions)
        return self._compiled(self.view, loss_fn, batch, batch_dict, positions)


__all__ = [
    "CompiledForceTrainingLoss",
    "RECOMPILE_LIMIT",
    "SHAPE_MODES",
    "ShapeMode",
    "configure_dynamo_for_training",
]
