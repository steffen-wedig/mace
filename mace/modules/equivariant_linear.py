"""Equivariant linear maps of irreps features, one plain torch product per irrep block.

An equivariant linear from ``irreps_in`` to ``irreps_out`` multiplies, for every pair of an
input block (mul_in x ir) and an output block (mul_out x ir) of the same irrep, the
[rows, 2l+1, mul_in] channels by one [mul_in, mul_out] matrix, the same for all 2l+1
components. Scalar (0e) outputs may carry a bias.

The parametrisation is e3nn's ``o3.Linear`` with its default normalisation: a flat
``weight`` in e3nn's order (blocks ordered by input block, then output block), each matrix
scaled by 1/sqrt(sum of the input multiplicities of the blocks writing the same output
block); biases (initialised to zero) added unscaled; output blocks without an input block
of their irrep are zero. State dicts match ``o3.Linear``'s; cuequivariance's ``Linear``
stores the same weight vector as [1, n], which loads too.

Two evaluations, as two classes with one forward each:

- :class:`EquivariantLinear`: one matrix product per block (a cuBLAS call);
- :class:`NarrowEquivariantLinear`: per block ``(x[..., :, None] * matrix).sum(-2)``, a
  reduction that ``torch.compile`` fuses with the producer of ``x`` and the consumer of the
  output. For linears to one or two channels per output block (readouts), where a matrix
  product does too little work per launch and cannot be fused (experiment 18).

Features are [..., irreps.dim] in the given layout: "mul_ir" (e3nn: [mul, 2l+1] per block)
or "ir_mul" (cuequivariance: [2l+1, mul] per block). The forward reads only integers and
floats precomputed at construction (:class:`LinearPlan`), no e3nn object.
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional

import torch
from e3nn import o3

LAYOUTS = ("mul_ir", "ir_mul")

#: Most output channels per block that NarrowEquivariantLinear accepts. Experiment 18
#: (GPU kernel time of a compiled float64 force-training step, B200, 160 to 61408 rows):
#: with one output channel the multiply-and-sum takes 9-47 % less kernel time than the
#: matrix product, with two about the same (0.71-1.12) in half the kernels, from four on
#: mostly more. Few input channels give no consistent gain (matrix product kept).
NARROW_CHANNEL_LIMIT = 2


class LinearBlock(NamedTuple):
    """One [input multiplicity, output multiplicity] matrix of an equivariant linear."""

    weight_start: int
    weight_stop: int
    input_multiplicity: int
    output_multiplicity: int
    dimension: int  # 2l+1
    path_weight: float
    input_start: int
    input_stop: int
    output_start: int
    output_stop: int


class LinearBias(NamedTuple):
    """The biases of one scalar output block."""

    bias_start: int
    bias_stop: int
    output_start: int
    output_stop: int


class LinearPlan(NamedTuple):
    """Where the matrices and biases of an equivariant linear sit in its parameters and
    in its input and output features."""

    blocks: List[LinearBlock]
    biases: List[LinearBias]
    input_dimension: int
    output_dimension: int
    weight_numel: int
    bias_numel: int


def linear_plan(
    irreps_in: o3.Irreps, irreps_out: o3.Irreps, biases: bool
) -> LinearPlan:
    """The plan of the equivariant linear from ``irreps_in`` to ``irreps_out`` (e3nn's
    ``o3.Linear(irreps_in, irreps_out, biases=biases)``)."""
    irreps_in = o3.Irreps(irreps_in)
    irreps_out = o3.Irreps(irreps_out)
    input_slices = irreps_in.slices()
    output_slices = irreps_out.slices()
    pairs = [
        (input_index, output_index)
        for input_index, (_, input_irrep) in enumerate(irreps_in)
        for output_index, (_, output_irrep) in enumerate(irreps_out)
        if input_irrep == output_irrep
    ]
    fan_in: Dict[int, int] = {}
    for input_index, output_index in pairs:
        fan_in[output_index] = fan_in.get(output_index, 0) + irreps_in[input_index].mul
    blocks: List[LinearBlock] = []
    weight_offset = 0
    for input_index, output_index in pairs:
        input_multiplicity = irreps_in[input_index].mul
        output_multiplicity = irreps_out[output_index].mul
        size = input_multiplicity * output_multiplicity
        if size == 0:
            continue
        blocks.append(
            LinearBlock(
                weight_start=weight_offset,
                weight_stop=weight_offset + size,
                input_multiplicity=input_multiplicity,
                output_multiplicity=output_multiplicity,
                dimension=irreps_in[input_index].ir.dim,
                path_weight=1.0 / math.sqrt(fan_in[output_index] or 1),
                input_start=input_slices[input_index].start,
                input_stop=input_slices[input_index].stop,
                output_start=output_slices[output_index].start,
                output_stop=output_slices[output_index].stop,
            )
        )
        weight_offset += size
    bias_blocks: List[LinearBias] = []
    bias_offset = 0
    if biases:
        for output_index, (multiplicity, irrep) in enumerate(irreps_out):
            if not irrep.is_scalar() or multiplicity == 0:
                continue
            bias_blocks.append(
                LinearBias(
                    bias_start=bias_offset,
                    bias_stop=bias_offset + multiplicity,
                    output_start=output_slices[output_index].start,
                    output_stop=output_slices[output_index].stop,
                )
            )
            bias_offset += multiplicity
    return LinearPlan(
        blocks=blocks,
        biases=bias_blocks,
        input_dimension=irreps_in.dim,
        output_dimension=irreps_out.dim,
        weight_numel=weight_offset,
        bias_numel=bias_offset,
    )


def block_channels(
    features: torch.Tensor, block: LinearBlock, layout: str
) -> torch.Tensor:
    """The block's input channels of [rows, input dimension] features as [rows, dimension,
    input multiplicity] (a view)."""
    channels = features[:, block.input_start : block.input_stop]
    if layout == "ir_mul":
        return channels.view(
            channels.shape[0], block.dimension, block.input_multiplicity
        )
    return channels.view(
        channels.shape[0], block.input_multiplicity, block.dimension
    ).transpose(1, 2)


def flatten_block(channels: torch.Tensor, layout: str) -> torch.Tensor:
    """[rows, dimension, multiplicity] back to the flat block in the layout."""
    if layout == "ir_mul":
        return channels.reshape(channels.shape[0], -1)
    return channels.transpose(1, 2).reshape(channels.shape[0], -1)


def block_matrix(weight: torch.Tensor, block: LinearBlock) -> torch.Tensor:
    """The block's [input multiplicity, output multiplicity] matrix, normalisation
    included."""
    return (
        weight.reshape(-1)[block.weight_start : block.weight_stop].view(
            block.input_multiplicity, block.output_multiplicity
        )
        * block.path_weight
    )


def assemble_output(
    block_outputs: List[torch.Tensor],
    plan: LinearPlan,
    layout: str,
    bias: Optional[torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    """The [rows, output dimension] output in the layout from the [rows, dimension, output
    multiplicity] products of the plan's blocks, plus the biases (zeros where nothing
    writes). ``reference`` gives rows, dtype and device."""
    rows = reference.shape[0]
    by_start: Dict[int, torch.Tensor] = {}
    for block, output in zip(plan.blocks, block_outputs):
        flat = flatten_block(output, layout)
        previous = by_start.get(block.output_start)
        by_start[block.output_start] = flat if previous is None else previous + flat
    if bias is not None:
        for bias_block in plan.biases:
            values = bias[bias_block.bias_start : bias_block.bias_stop]
            previous = by_start.get(bias_block.output_start)
            by_start[bias_block.output_start] = (
                values.expand(rows, -1) if previous is None else previous + values
            )
    pieces: List[torch.Tensor] = []
    position = 0
    for start in sorted(by_start):
        if start > position:
            pieces.append(reference.new_zeros(rows, start - position))
        piece = by_start[start]
        pieces.append(piece)
        position = start + piece.shape[1]
    if position < plan.output_dimension or not pieces:
        pieces.append(reference.new_zeros(rows, plan.output_dimension - position))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)


class _EquivariantLinearParameters(torch.nn.Module):
    """Parameters, plan and layout of an equivariant linear; no forward (each subclass
    has its own)."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        layout: str = "mul_ir",
        biases: bool = False,
    ):
        super().__init__()
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.layout = layout
        self.plan = linear_plan(self.irreps_in, self.irreps_out, biases)
        self.weight = torch.nn.Parameter(torch.randn(self.plan.weight_numel))
        if self.plan.bias_numel > 0:
            self.bias = torch.nn.Parameter(torch.zeros(self.plan.bias_numel))
        else:
            self.register_parameter("bias", None)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # o3.Linear's mask of output blocks without a path: implied by the plan here.
        state_dict.pop(prefix + "output_mask", None)
        # o3.Linear registers an empty "bias" buffer when it has no biases.
        bias = state_dict.get(prefix + "bias")
        if self.bias is None and bias is not None and bias.numel() == 0:
            state_dict.pop(prefix + "bias")
        # cuequivariance's Linear stores the weight vector as [1, n].
        weight = state_dict.get(prefix + "weight")
        if weight is not None and weight.dim() == 2 and weight.shape[0] == 1:
            state_dict[prefix + "weight"] = weight.reshape(-1)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def extra_repr(self) -> str:
        return (
            f"{self.irreps_in} -> {self.irreps_out}, layout={self.layout}, "
            f"biases={self.plan.bias_numel > 0}"
        )


class EquivariantLinear(_EquivariantLinearParameters):
    """Equivariant linear with one matrix product per irrep block.

    The block's [rows, dimension, input multiplicity] channels are multiplied as one 2D
    [rows * dimension, input multiplicity] matrix (a view for scalar blocks; a copy for
    l > 0 blocks in the mul_ir layout).
    """

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        leading = features.shape[:-1]
        # Rows from the leading dimensions (also for a [rows, 0] input).
        flat = features.reshape(leading.numel(), features.shape[-1])
        block_outputs: List[torch.Tensor] = []
        for block in self.plan.blocks:
            channels = block_channels(flat, block, self.layout)
            product = torch.matmul(
                channels.reshape(-1, block.input_multiplicity),
                block_matrix(self.weight, block),
            )
            block_outputs.append(
                product.view(flat.shape[0], block.dimension, block.output_multiplicity)
            )
        output = assemble_output(block_outputs, self.plan, self.layout, self.bias, flat)
        return output.reshape(leading + (self.plan.output_dimension,))


class NarrowEquivariantLinear(_EquivariantLinearParameters):
    """Equivariant linear with every irrep block as a broadcast multiply and a sum.

    For linears whose blocks have at most ``NARROW_CHANNEL_LIMIT`` output channels
    (readouts to one or two channels per irrep).
    Compiled, each block is one reduction fused with its neighbours, and its derivatives
    of any order are again multiplies and sums. Evaluated eagerly, it materialises the
    [rows, dimension, input multiplicity, output multiplicity] product.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        layout: str = "mul_ir",
        biases: bool = False,
    ):
        super().__init__(irreps_in, irreps_out, layout=layout, biases=biases)
        wide = [
            block.output_multiplicity
            for block in self.plan.blocks
            if block.output_multiplicity > NARROW_CHANNEL_LIMIT
        ]
        if wide:
            raise ValueError(
                f"{self.irreps_in} -> {self.irreps_out}: blocks with {wide} output "
                f"channels, more than {NARROW_CHANNEL_LIMIT}, are faster as matrix "
                "products: use EquivariantLinear"
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        leading = features.shape[:-1]
        # Rows from the leading dimensions (also for a [rows, 0] input).
        flat = features.reshape(leading.numel(), features.shape[-1])
        block_outputs: List[torch.Tensor] = []
        for block in self.plan.blocks:
            channels = block_channels(flat, block, self.layout)
            matrix = block_matrix(self.weight, block)
            block_outputs.append((channels[..., :, None] * matrix).sum(dim=-2))
        output = assemble_output(block_outputs, self.plan, self.layout, self.bias, flat)
        return output.reshape(leading + (self.plan.output_dimension,))


__all__ = [
    "LAYOUTS",
    "NARROW_CHANNEL_LIMIT",
    "EquivariantLinear",
    "LinearBias",
    "LinearBlock",
    "LinearPlan",
    "NarrowEquivariantLinear",
    "assemble_output",
    "block_channels",
    "block_matrix",
    "flatten_block",
    "linear_plan",
]
