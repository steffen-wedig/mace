"""Equivariant linears as matrix products per irrep segment.

An e3nn ``o3.Linear``, a ``cuet.Linear`` and a ``LayoutNativeLinear`` with the same irreps
compute the same map: for every path from an input segment (``mul_in x l``) to an output
segment (``mul_out x l``) the [n, 2l+1, mul_in] channels are multiplied by
``path_weight * W[mul_in, mul_out]`` (e3nn's flat weight order, cueq stores it the same).
A :class:`LinearFold` records where those matrices sit in the flat weight, with integers
and floats only, so a traced forward reads no e3nn object.

Two uses:

- fold a linear into the next product (``SparseUvuTensorProduct.forward_folded``);
- evaluate several linears that read the same tensor as one product per input segment
  (:func:`evaluate_linears_sharing_input`): the weights are concatenated along the
  output channels, the input is read once, and the backward has one GEMM per segment for
  the input gradient and one for the weight gradients instead of one per linear. Only
  weights are concatenated (no activation copies); exact up to the order of summation.
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
from e3nn import o3


class LinearFoldPath(NamedTuple):
    """One weight block of an equivariant linear (e3nn's flat weight order)."""

    weight_start: int
    weight_stop: int
    input_multiplicity: int
    output_multiplicity: int
    path_weight: float
    input_start: int
    input_stop: int
    output_start: int


class LinearFoldBias(NamedTuple):
    bias_start: int
    bias_stop: int
    output_start: int


class LinearFold(NamedTuple):
    """Where the per-irrep matrices and biases of a linear sit in its parameters.

    Valid for o3.Linear, cuet.Linear and LayoutNativeLinear alike: all store e3nn's flat
    weight order (and the first and last scalar biases).
    """

    paths: List[LinearFoldPath]
    biases: List[LinearFoldBias]
    output_dimension: int


# (input y in the model layout, the linear's weight, its bias or None, its LinearFold)
FoldedLinearInput = Tuple[
    torch.Tensor, torch.Tensor, Optional[torch.Tensor], LinearFold
]


def linear_fold(
    irreps_in: o3.Irreps, irreps_out: o3.Irreps, biases: bool
) -> LinearFold:
    """The LinearFold of ``o3.Linear(irreps_in, irreps_out, biases=biases)``."""
    irreps_in = o3.Irreps(irreps_in)
    irreps_out = o3.Irreps(irreps_out)
    reference = o3.Linear(
        irreps_in,
        irreps_out,
        biases=biases,
        internal_weights=False,
        shared_weights=True,
    )
    input_slices = irreps_in.slices()
    output_slices = irreps_out.slices()
    paths: List[LinearFoldPath] = []
    bias_blocks: List[LinearFoldBias] = []
    weight_offset = 0
    bias_offset = 0
    for instruction in reference.instructions:
        size = math.prod(instruction.path_shape)
        if size == 0:
            continue
        output_start = output_slices[instruction.i_out].start
        if instruction.i_in == -1:
            bias_blocks.append(
                LinearFoldBias(bias_offset, bias_offset + size, output_start)
            )
            bias_offset += size
            continue
        input_multiplicity, output_multiplicity = instruction.path_shape
        paths.append(
            LinearFoldPath(
                weight_offset,
                weight_offset + size,
                int(input_multiplicity),
                int(output_multiplicity),
                float(instruction.path_weight),
                input_slices[instruction.i_in].start,
                input_slices[instruction.i_in].stop,
                output_start,
            )
        )
        weight_offset += size
    if weight_offset != reference.weight_numel or bias_offset != reference.bias_numel:
        raise RuntimeError("LinearFold does not cover the linear's parameters")
    return LinearFold(paths, bias_blocks, irreps_out.dim)


def linear_fold_of(linear: torch.nn.Module) -> LinearFold:
    """The LinearFold of a built linear (o3.Linear, cuet.Linear or LayoutNativeLinear)."""
    bias = getattr(linear, "bias", None)
    has_biases = isinstance(bias, torch.Tensor) and bias.numel() > 0
    # cuet.Linear holds cue.Irreps; their string form is e3nn's.
    return linear_fold(
        o3.Irreps(str(linear.irreps_in)),
        o3.Irreps(str(linear.irreps_out)),
        biases=has_biases,
    )


def folded_linear_input(
    y: torch.Tensor, linear: torch.nn.Module, fold: LinearFold
) -> FoldedLinearInput:
    bias = linear.bias if fold.biases else None
    return (y, linear.weight, bias, fold)


def path_dimension(path: LinearFoldPath) -> int:
    """2l+1 of the path's irrep."""
    return (path.input_stop - path.input_start) // path.input_multiplicity


def path_matrix(weight: torch.Tensor, path: LinearFoldPath) -> torch.Tensor:
    """The path's [input multiplicity, output multiplicity] matrix, path weight included."""
    block = weight.reshape(-1)[path.weight_start : path.weight_stop]
    return (
        block.view(path.input_multiplicity, path.output_multiplicity) * path.path_weight
    )


def channels_last(
    block: torch.Tensor, multiplicity: int, dimension: int, layout: str
) -> torch.Tensor:
    """An irrep segment [n, multiplicity * dimension] in the layout as [n, dimension,
    multiplicity] (a view)."""
    if layout == "ir_mul":
        return block.view(block.shape[0], dimension, multiplicity)
    return block.view(block.shape[0], multiplicity, dimension).transpose(1, 2)


def from_channels_last(block: torch.Tensor, layout: str) -> torch.Tensor:
    """[n, dimension, multiplicity] back to the flat segment in the layout."""
    if layout == "ir_mul":
        return block.reshape(block.shape[0], -1)
    return block.transpose(1, 2).reshape(block.shape[0], -1)


class SegmentProduct(NamedTuple):
    """An input irrep segment, to be multiplied by a [multiplicity, columns] matrix."""

    input_start: int
    input_stop: int
    multiplicity: int
    dimension: int


def products_sharing_input(
    x: torch.Tensor,
    layout: str,
    products: List[Tuple[SegmentProduct, torch.Tensor]],
) -> List[torch.Tensor]:
    """``channels_last(x[:, segment]) @ matrix`` for every entry, as [n, dimension, columns].

    One matmul per input segment, over the matrices of that segment concatenated along
    the columns.
    """
    by_segment: Dict[Tuple[int, int], List[int]] = {}
    for index, (segment, _) in enumerate(products):
        by_segment.setdefault((segment.input_start, segment.input_stop), []).append(
            index
        )
    results: List[Optional[torch.Tensor]] = [None] * len(products)
    for indices in by_segment.values():
        segment = products[indices[0]][0]
        block = channels_last(
            x[:, segment.input_start : segment.input_stop],
            segment.multiplicity,
            segment.dimension,
            layout,
        )
        matrices = [products[index][1] for index in indices]
        matrix = matrices[0] if len(matrices) == 1 else torch.cat(matrices, dim=1)
        # One 2D GEMM over the n * dimension rows. torch.matmul of the strided
        # [n, dimension, multiplicity] view with a matrix would broadcast the matrix
        # over n and run a per-node bmm (its gradient too): ~10x slower.
        merged = torch.matmul(block.reshape(-1, segment.multiplicity), matrix).view(
            block.shape[0], segment.dimension, matrix.shape[1]
        )
        if len(matrices) == 1:
            results[indices[0]] = merged
            continue
        parts = merged.split([matrix.shape[1] for matrix in matrices], dim=-1)
        for index, part in zip(indices, parts):
            results[index] = part
    return [result for result in results if result is not None]


def linear_products(
    linear: torch.nn.Module, fold: LinearFold
) -> List[Tuple[SegmentProduct, torch.Tensor]]:
    """The linear as segment products, one per path (weights only, no biases)."""
    if fold.biases:
        raise ValueError("linears with biases are not evaluated as segment products")
    return [
        (
            SegmentProduct(
                path.input_start,
                path.input_stop,
                path.input_multiplicity,
                path_dimension(path),
            ),
            path_matrix(linear.weight, path),
        )
        for path in fold.paths
    ]


def assemble_linear_output(
    path_outputs: List[torch.Tensor], fold: LinearFold, layout: str
) -> torch.Tensor:
    """The linear's output [n, output dimension] in the layout from the [n, dimension,
    output multiplicity] products of its paths (zeros where no path writes)."""
    by_start: Dict[int, torch.Tensor] = {}
    for path, output in zip(fold.paths, path_outputs):
        flat = from_channels_last(output, layout)
        previous = by_start.get(path.output_start)
        by_start[path.output_start] = flat if previous is None else previous + flat
    pieces: List[torch.Tensor] = []
    position = 0
    reference = path_outputs[0]
    for start in sorted(by_start):
        if start > position:
            pieces.append(reference.new_zeros(reference.shape[0], start - position))
        piece = by_start[start]
        pieces.append(piece)
        position = start + piece.shape[1]
    if position < fold.output_dimension:
        pieces.append(
            reference.new_zeros(reference.shape[0], fold.output_dimension - position)
        )
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)


def evaluate_linears_sharing_input(
    x: torch.Tensor,
    linears: List[torch.nn.Module],
    folds: Optional[List[LinearFold]],
    layout: Optional[str],
) -> List[torch.Tensor]:
    """``[linear(x) for linear in linears]`` with one matmul per input irrep segment.

    The linears must have no biases and read ``x`` in ``layout`` ("ir_mul" for cueq
    linears, "mul_ir" for e3nn linears; a linear that reads and writes scalars only is
    the same in both). Without folds or layout (e3nn linears on cueq-layout features)
    every linear is called on its own.
    """
    if folds is None or layout is None:
        return [linear(x) for linear in linears]
    products: List[Tuple[SegmentProduct, torch.Tensor]] = []
    counts: List[int] = []
    for linear, fold in zip(linears, folds):
        linear_terms = linear_products(linear, fold)
        products.extend(linear_terms)
        counts.append(len(linear_terms))
    outputs = products_sharing_input(x, layout, products)
    results: List[torch.Tensor] = []
    offset = 0
    for fold, count in zip(folds, counts):
        results.append(
            assemble_linear_output(outputs[offset : offset + count], fold, layout)
        )
        offset += count
    return results


class LayerOutputLinears(NamedTuple):
    """The linears PolarMACE evaluates on one layer's output: the feature mixer's, the
    next interaction's ``skip_tp`` and ``linear_up`` (``with_next``) and the readout's
    first linear (``with_readout``); their folds in that order (None: evaluated one by
    one)."""

    with_next: bool
    with_readout: bool
    folds: Optional[List[LinearFold]]


class SharedInputPlan(NamedTuple):
    """Per layer output, the linears PolarMACE.forward evaluates together, and the
    layout they read (None: each linear on its own). Built from the module structure."""

    layout: Optional[str]
    layers: List[LayerOutputLinears]


__all__ = [
    "FoldedLinearInput",
    "LayerOutputLinears",
    "LinearFold",
    "LinearFoldBias",
    "LinearFoldPath",
    "SegmentProduct",
    "SharedInputPlan",
    "assemble_linear_output",
    "channels_last",
    "evaluate_linears_sharing_input",
    "folded_linear_input",
    "from_channels_last",
    "linear_fold",
    "linear_fold_of",
    "linear_products",
    "path_dimension",
    "path_matrix",
    "products_sharing_input",
]
