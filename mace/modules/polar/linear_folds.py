"""Equivariant linears folded into the next product, or evaluated together when several
read the same tensor.

The linears are ``mace.modules.equivariant_linear`` linears: their :class:`LinearPlan` says
where each block's [input multiplicity, output multiplicity] matrix sits in the weight and
in the features, with integers and floats only, so a traced forward reads no e3nn object.

Two uses:

- fold a linear into the next product (``SparseUvuTensorProduct.forward_folded``) through
  :func:`folded_linear_input`;
- evaluate several EquivariantLinears that read the same tensor as one product per input
  block (:func:`evaluate_linears_sharing_input`): the weights are concatenated along the
  output channels, the input is read once, and the backward has one GEMM per block for
  the input gradient and one for the weight gradients instead of one per linear. Only
  weights are concatenated (no activation copies); exact up to the order of summation.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

from mace.modules.equivariant_linear import (
    EquivariantLinear,
    LinearBlock,
    LinearPlan,
    assemble_output,
    block_channels,
    block_matrix,
)


class FoldedLinearInput(NamedTuple):
    """A linear and its input, to be folded into the next product instead of evaluated."""

    features: torch.Tensor
    weight: torch.Tensor
    bias: Optional[torch.Tensor]
    plan: LinearPlan


def folded_linear_input(
    features: torch.Tensor, linear: torch.nn.Module
) -> FoldedLinearInput:
    return FoldedLinearInput(features, linear.weight, linear.bias, linear.plan)


def products_sharing_input(
    features: torch.Tensor,
    layout: str,
    products: List[Tuple[LinearBlock, torch.Tensor]],
) -> List[torch.Tensor]:
    """``block_channels(features, block) @ matrix`` for every entry, as [rows, dimension,
    columns].

    One matmul per input block, over the matrices of that block concatenated along the
    columns.
    """
    by_input: Dict[Tuple[int, int], List[int]] = {}
    for index, (block, _) in enumerate(products):
        by_input.setdefault((block.input_start, block.input_stop), []).append(index)
    results: List[Optional[torch.Tensor]] = [None] * len(products)
    for indices in by_input.values():
        block = products[indices[0]][0]
        channels = block_channels(features, block, layout)
        matrices = [products[index][1] for index in indices]
        matrix = matrices[0] if len(matrices) == 1 else torch.cat(matrices, dim=1)
        # One 2D GEMM over the rows * dimension rows. torch.matmul of the strided
        # [rows, dimension, multiplicity] view with a matrix would broadcast the matrix
        # over the rows and run a per-row bmm (its gradient too): ~10x slower.
        merged = torch.matmul(channels.reshape(-1, block.input_multiplicity), matrix).view(
            channels.shape[0], block.dimension, matrix.shape[1]
        )
        if len(matrices) == 1:
            results[indices[0]] = merged
            continue
        parts = merged.split([matrix.shape[1] for matrix in matrices], dim=-1)
        for index, part in zip(indices, parts):
            results[index] = part
    return [result for result in results if result is not None]


def evaluate_linears_sharing_input(
    features: torch.Tensor, linears: List[torch.nn.Module], layout: str
) -> List[torch.Tensor]:
    """``[linear(features) for linear in linears]`` with one matmul per input block.

    The linears must be EquivariantLinears without biases in ``layout``
    (:func:`check_linears_sharing_input`).
    """
    products: List[Tuple[LinearBlock, torch.Tensor]] = []
    counts: List[int] = []
    for linear in linears:
        products.extend(
            (block, block_matrix(linear.weight, block)) for block in linear.plan.blocks
        )
        counts.append(len(linear.plan.blocks))
    outputs = products_sharing_input(features, layout, products)
    results: List[torch.Tensor] = []
    offset = 0
    for linear, count in zip(linears, counts):
        results.append(
            assemble_output(
                outputs[offset : offset + count], linear.plan, layout, None, features
            )
        )
        offset += count
    return results


def check_linears_sharing_input(linears: List[torch.nn.Module], layout: str) -> None:
    """Raise unless every linear can be evaluated by evaluate_linears_sharing_input."""
    for linear in linears:
        if type(linear) is not EquivariantLinear:
            raise TypeError(
                f"{type(linear).__name__}: only EquivariantLinears are evaluated together"
            )
        if linear.bias is not None:
            raise ValueError("linears with biases are not evaluated together")
        if linear.layout != layout:
            raise ValueError(f"linear in layout {linear.layout}, features in {layout}")


class LayerOutputLinears(NamedTuple):
    """The linears PolarMACE evaluates on one layer's output: the feature mixer's, the
    next interaction's ``skip_tp`` and ``linear_up`` (``with_next``) and the readout's
    first linear (``with_readout``)."""

    with_next: bool
    with_readout: bool


class SharedInputPlan(NamedTuple):
    """Per layer output, the linears PolarMACE.forward evaluates together, and the
    layout they read. Built from the module structure."""

    layout: str
    layers: List[LayerOutputLinears]


__all__ = [
    "FoldedLinearInput",
    "LayerOutputLinears",
    "SharedInputPlan",
    "check_linears_sharing_input",
    "evaluate_linears_sharing_input",
    "folded_linear_input",
    "products_sharing_input",
]
