"""Compiled inference: energy, forces and stress of a batch as one torch.compile graph.

The sibling of ``mace.tools.compiled.training`` for MD and single points. The forward, the
force and stress gradient (``autograd.grad`` traced into the graph) and the output
selection are one ``fullgraph=True`` graph; no double backward, so AOTAutograd compiles an
inference graph (nothing it returns requires grad).

What stays outside the compiled region, built by the caller per call:

- the position and displacement leaves (``requires_grad_`` inside the region breaks the
  graph); the displacement is the symmetric strain whose gradient is the virial, so the
  stress comes from the same graph as the forces (NPT and NVT alike);
- everything that reads the host: the neighbour list and, for periodic POLAR-M, the
  reciprocal-lattice indices (``kspace_lattice_inputs``), listed with a margin so one set
  serves an NPT cell over many steps.

Two shape modes, as in training: dynamic (symbolic node, edge and k-vector counts; the
default) and static (fixed sizes per padding budget, CUDA graphs).
"""

from __future__ import annotations

import math
from typing import Any, Dict, NamedTuple, Optional, Sequence

import torch
import torch._dynamo
import torch._inductor

from mace.tools import compile as mace_compile
from mace.tools.compiled import small_matmuls

# Dynamo's and Inductor's settings live in torch's private modules (torch._dynamo.config,
# torch._inductor.list_mode_options); configuring them is this module's purpose.
# pylint: disable=protected-access


RECOMPILE_LIMIT = 8
DEFAULT_OUTPUTS = ("energy", "forces", "stress")


def configure_dynamo_for_inference(recompile_limit: int = RECOMPILE_LIMIT) -> None:
    """Process-wide Dynamo settings for compiled inference (set once): ``autograd.grad``
    traced into the graph, Python floats specialised (model hyperparameters stay
    constants), and a recompile beyond the limit raises instead of running eagerly."""
    mace_compile.configure_autograd_for_compile(allow_autograd=True)
    torch._dynamo.config.specialize_float = True
    torch._dynamo.config.recompile_limit = recompile_limit
    torch._dynamo.config.fail_on_recompile_limit_hit = True


def inference_inductor_options(
    static_shapes: bool, autotune_gemms: bool = False, freeze_weights: bool = False
) -> Dict[str, Any]:
    """Inductor options of the compiled evaluation: small matrix products as fused
    reductions; with static shapes, CUDA graphs (``mode="reduce-overhead"``);
    ``autotune_gemms``: matrix products benchmarked against Triton templates at compile
    time (``max_autotune_gemm``; slower compile)."""
    options = small_matmuls.inductor_options()
    if static_shapes:
        options.update(torch._inductor.list_mode_options("reduce-overhead"))
    if autotune_gemms:
        options["max_autotune_gemm"] = True
    if freeze_weights:
        # Parameters become constants and weight-only subgraphs are folded at compile
        # time; Inductor applies it only to graphs compiled with grad mode off (the
        # evaluation then enables grad itself around the force gradient).
        options["freezing"] = True
    return options


def kspace_lattice_inputs(
    model: torch.nn.Module,
    cell: torch.Tensor,
    margin: float = 0.0,
    rows: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """The batch keys of periodic POLAR-M's reciprocal-lattice indices for ``cell``
    ([n_graphs, 3, 3], rows are lattice vectors). Reads the cell on the host."""
    from mace.modules.polar.electrostatics.kspace import (  # noqa: PLC0415
        reciprocal_lattice_indices,
    )

    indices = reciprocal_lattice_indices(
        cell, model.kspace_cutoff_value(), margin=margin, rows=rows
    )
    return {
        "kspace_miller": indices.miller,
        "kspace_structure": indices.structure,
        "kspace_padding": indices.padding,
    }


class InferenceBudgets(NamedTuple):
    """Fixed sizes of a padded batch: atoms (real + padding), edges and k-vector rows."""

    atoms: int
    edges: int
    kspace_rows: int


# The padding structure: atoms of the first atom's element on a cubic grid (nonzero
# separations) in a cubic cell of their own, enough of them that the padding edges give
# each a few dozen neighbours (thousands of edges on a handful of atoms could overflow
# their features in float32, and the dense k-space products would carry inf * 0 = NaN
# into real rows).
PADDING_ATOMS = 4
PADDING_EDGES_PER_ATOM = 48
PADDING_SPACING = 1.5
PADDING_CELL_LENGTH = 12.0

_PER_GRAPH_ZEROS = ("total_charge", "external_field", "fermi_level")


def inference_budgets(
    data: Dict[str, torch.Tensor],
    edge_headroom: float = 0.03,
    kspace_headroom: float = 0.25,
    edge_multiple: int = 1024,
    kspace_multiple: int = 64,
) -> InferenceBudgets:
    """Budgets for a batch like ``data``: its atoms plus the padding atoms, its edges and
    k-vector rows with headroom (an MD step's counts drift; a batch beyond the budgets
    needs new budgets and a new graph). Padding edges cost as much as real ones: the
    edge count of a 250-atom liquid box drifts by ~2 % over an NPT run (dmso, A1), less
    for the sum over replicas, hence 3 %."""

    def round_up(value: float, multiple: int) -> int:
        return int(math.ceil(value / multiple) * multiple)

    rows = int(data["kspace_miller"].shape[0]) if "kspace_miller" in data else 0
    edges = int(data["edge_index"].shape[1])
    edge_budget = round_up(edges * (1 + edge_headroom), edge_multiple)
    padding_atoms = max(
        PADDING_ATOMS, math.ceil((edge_budget - edges) / PADDING_EDGES_PER_ATOM)
    )
    return InferenceBudgets(
        atoms=int(data["positions"].shape[0]) + padding_atoms,
        edges=edge_budget,
        kspace_rows=(
            round_up(max(rows, 1) * (1 + kspace_headroom), kspace_multiple)
            if rows
            else 0
        ),
    )


def fits_budgets(data: Dict[str, torch.Tensor], budgets: InferenceBudgets) -> bool:
    rows = int(data["kspace_miller"].shape[0]) if "kspace_miller" in data else 0
    return (
        int(data["positions"].shape[0]) + 2 <= budgets.atoms
        and int(data["edge_index"].shape[1]) <= budgets.edges
        and rows <= budgets.kspace_rows
    )


def pad_inference_batch(
    data: Dict[str, torch.Tensor], budgets: InferenceBudgets
) -> Dict[str, torch.Tensor]:
    """``data`` padded to ``budgets`` with one padding structure appended (index
    ``num_graphs``): its atoms, the padding edges (a ring between its atoms, no
    periodic shift) and the padding k-vector rows (always excluded). No edge, pair or
    k-vector couples it to a real structure and every per-structure sum keeps it apart,
    so the real structures' outputs are those of the unpadded batch: take
    ``energy[:num_graphs]``, ``forces[:num_atoms]``, ``stress[:num_graphs]``.
    On the device, no host reads."""
    positions = data["positions"]
    device, dtype = positions.device, positions.dtype
    num_atoms = int(positions.shape[0])
    num_graphs = int(data["ptr"].numel()) - 1
    padding_atoms = budgets.atoms - num_atoms
    padding_edges = budgets.edges - int(data["edge_index"].shape[1])
    if padding_atoms < 2 or padding_edges < 0:
        raise ValueError(f"batch does not fit the budgets {budgets}")
    side = math.ceil(padding_atoms ** (1.0 / 3.0))
    index = torch.arange(padding_atoms, device=device)
    grid = torch.stack([index % side, (index // side) % side, index // side**2], dim=1)
    padded: Dict[str, torch.Tensor] = dict(data)
    padded["positions"] = torch.cat([positions, grid.to(dtype) * PADDING_SPACING + 1.0])
    padded["node_attrs"] = torch.cat(
        [data["node_attrs"], data["node_attrs"][:1].expand(padding_atoms, -1)]
    )
    padded["batch"] = torch.cat(
        [data["batch"], data["batch"].new_full((padding_atoms,), num_graphs)]
    )
    padded["ptr"] = torch.cat([data["ptr"], data["ptr"][-1:] + padding_atoms])
    if "density_coefficients" in data:
        coefficients = data["density_coefficients"]
        padded["density_coefficients"] = torch.cat(
            [coefficients, coefficients.new_zeros(padding_atoms, coefficients.shape[1])]
        )
    sender = torch.arange(padding_edges, device=device) % padding_atoms
    ring = torch.stack([sender, (sender + 1) % padding_atoms]) + num_atoms
    padded["edge_index"] = torch.cat([data["edge_index"], ring], dim=1)
    for key in ("shifts", "unit_shifts"):
        padded[key] = torch.cat([data[key], data[key].new_zeros(padding_edges, 3)])
    eye = torch.eye(3, device=device, dtype=dtype)
    cell = data["cell"]
    padded["cell"] = torch.cat(
        [cell.reshape(-1, 3, 3), (eye * PADDING_CELL_LENGTH).unsqueeze(0)]
    ).reshape(-1, *cell.shape[1:])
    if "rcell" in data:
        rcell = data["rcell"]
        padded["rcell"] = torch.cat(
            [rcell.reshape(-1, 3, 3), (eye * 2 * math.pi / PADDING_CELL_LENGTH)[None]]
        ).reshape(-1, *rcell.shape[1:])
    if "volume" in data:
        padded["volume"] = torch.cat(
            [data["volume"], data["volume"].new_full((1,), PADDING_CELL_LENGTH**3)]
        )
    if "head" in data:
        padded["head"] = torch.cat([data["head"], data["head"][:1]])
    if "total_spin" in data:
        padded["total_spin"] = torch.cat(
            [data["total_spin"], data["total_spin"].new_ones(1)]
        )
    for key in _PER_GRAPH_ZEROS:
        if key in data:
            value = data[key]
            padded[key] = torch.cat([value, value.new_zeros(1, *value.shape[1:])])
    pbc = data.get("pbc")
    if pbc is not None and pbc.dim() == 2 and pbc.shape[0] == num_graphs:
        padded["pbc"] = torch.cat([pbc, pbc.new_ones(1, 3)])
    if "kspace_miller" in data:
        padding_rows = budgets.kspace_rows - int(data["kspace_miller"].shape[0])
        if padding_rows < 0:
            raise ValueError(f"batch does not fit the budgets {budgets}")
        miller = data["kspace_miller"].new_zeros(padding_rows, 3)
        miller[:, 0] = 1.0
        padded["kspace_miller"] = torch.cat([data["kspace_miller"], miller])
        padded["kspace_structure"] = torch.cat(
            [
                data["kspace_structure"],
                data["kspace_structure"].new_full((padding_rows,), num_graphs),
            ]
        )
        padded["kspace_padding"] = torch.cat(
            [
                data["kspace_padding"],
                data["kspace_padding"].new_ones(padding_rows),
            ]
        )
    return padded


class _ForceGraph(torch.nn.Module):
    """Forward, forces and stress of ``model`` (the compiled region)."""

    def __init__(
        self, model: torch.nn.Module, compute_stress: bool, outputs: Sequence[str]
    ):
        super().__init__()
        self.model = model
        self.compute_stress = compute_stress
        self.outputs = tuple(outputs)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        positions: torch.Tensor,
        displacement: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        inputs = {**data, "positions": positions}
        if displacement is not None:
            inputs["displacement"] = displacement
        # Grad enabled here even when the caller disables it (frozen weights compile
        # with grad mode off): the forces and the stress are gradients of the energy.
        with torch.enable_grad():
            output = self.model(
                inputs,
                training=False,
                compute_force=True,
                compute_stress=self.compute_stress,
            )
        # Detached: nothing returned requires grad, so AOTAutograd compiles an inference
        # graph (no joint graph, no activations saved for a backward). With the energy
        # attached it would differentiate the outputs again, through the freed force graph.
        return {key: output[key].detach() for key in self.outputs}


class CompiledForceEvaluation:
    """Energy, forces and (optionally) stress of one batch from one compiled graph.

    ``outputs = evaluation(data)`` with a model input dict (``AtomicData`` keys, plus
    ``kspace_*`` for periodic POLAR-M). The model must have been built without e3nn's
    code generation (``mace.tools.compile.disable_e3nn_codegen``) and, for POLAR-M, use
    ``pbc_handling`` 'pbc' (lattice indices in the batch) or 'realspace' (pair list in the
    batch); 'auto' reads the pbc flags on the host. The graph runs on a simplified view
    sharing the model's parameters.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        compute_stress: bool = True,
        static_shapes: bool = False,
        outputs: Sequence[str] = DEFAULT_OUTPUTS,
        autotune_gemms: bool = False,
        freeze_weights: bool = False,
    ):
        if getattr(model, "pbc_handling", None) == "auto":
            raise ValueError(
                "compiled inference needs pbc_handling 'pbc' or 'realspace', not 'auto'"
            )
        self.model = model
        self.view = mace_compile.simplified_view(model)
        self.compute_stress = compute_stress
        self.static_shapes = static_shapes
        self.freeze_weights = freeze_weights
        self.outputs = tuple(
            key for key in outputs if compute_stress or key not in ("stress", "virials")
        )
        # The graph is compiled as a module that owns the view, so its parameters are
        # module attributes: Inductor's freezing folds only those (passed as an
        # argument, they stay graph inputs).
        self.graph_module = _ForceGraph(self.view, compute_stress, self.outputs)
        self._compiled = torch.compile(
            self.graph_module,
            fullgraph=True,
            dynamic=not static_shapes,
            options=inference_inductor_options(
                static_shapes, autotune_gemms, freeze_weights
            ),
        )

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        positions = data["positions"].detach().requires_grad_(True)
        displacement = None
        if self.compute_stress:
            num_graphs = int(data["ptr"].numel()) - 1
            displacement = torch.zeros(
                (num_graphs, 3, 3), dtype=positions.dtype, device=positions.device
            ).requires_grad_(True)
        if not self.static_shapes:
            for value in data.values():
                if torch.is_tensor(value) and value.dim() > 0:
                    torch._dynamo.maybe_mark_dynamic(value, 0)
            torch._dynamo.maybe_mark_dynamic(positions, 0)
        else:
            torch.compiler.cudagraph_mark_step_begin()
        if self.freeze_weights:
            with torch.no_grad():
                return self._compiled(data, positions, displacement)
        return self._compiled(data, positions, displacement)


__all__ = [
    "CompiledForceEvaluation",
    "InferenceBudgets",
    "fits_budgets",
    "inference_budgets",
    "pad_inference_batch",
    "DEFAULT_OUTPUTS",
    "configure_dynamo_for_inference",
    "inference_inductor_options",
    "kspace_lattice_inputs",
]
