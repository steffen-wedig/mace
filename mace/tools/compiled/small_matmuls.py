"""Small matrix products as fused reductions in compiled training (Inductor post-grad pass).

A product with at most a few outputs per row (a readout to one or two channels, a source map
to the 2 spin channels, the backward of a fold over the 6 field and charge channels) or a
contraction over a few elements (one-hot element embeddings) does too little arithmetic per
byte for a GEMM: as a cuBLAS kernel it costs a launch (and
often a split-K reduce) for microseconds of work, and it cannot fuse with its neighbours. As
``(a[..., :, :, None] * b[..., None, :, :]).sum(-2)`` Inductor generates a reduction that
fuses with the producer of ``a`` or the consumer of the result.

Inductor itself does this only for matrix-vector products under ``coordinate_descent_tuning``
(which also benchmarks Triton kernels at their first launch and broke CUDA-graph capture in
experiment 08) and in ``decompose_mem_bound_mm`` only when both K and N are below 32. This
pass covers the shapes of force training: ``mm``, ``addmm`` and ``bmm`` whose output has at
most ``max_outputs`` rows or columns, or whose contraction is at most
``max_contraction``, with static shapes. Exact up to the order of summation.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any, Dict, Optional

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)

aten = torch.ops.aten

MAX_OUTPUTS = 8
MAX_CONTRACTION = 8

_PATTERNS = PatternMatcherPass(pass_name="small_matmuls")


def _static_shape(node: Any) -> Optional[tuple]:
    value = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    if value is None or not all(isinstance(size, int) for size in value.shape):
        return None
    return tuple(value.shape)


def is_small_product(rows: int, columns: int, contraction: int) -> bool:
    """Whether a [rows, contraction] x [contraction, columns] product is decomposed."""
    if contraction == 1:
        return False  # Inductor already writes outer products as pointwise multiplies.
    return min(rows, columns) <= MAX_OUTPUTS or contraction <= MAX_CONTRACTION


def _matrix_shapes(first: Any, second: Any) -> Optional[tuple[int, int, int]]:
    first_shape = _static_shape(first)
    second_shape = _static_shape(second)
    if first_shape is None or second_shape is None:
        return None
    return first_shape[-2], second_shape[-1], first_shape[-1]


@register_graph_pattern(CallFunction(aten.mm, Arg(), Arg()), pass_dict=_PATTERNS)
def _small_mm(match: Match, first: torch.fx.Node, second: torch.fx.Node) -> None:
    shapes = _matrix_shapes(first, second)
    if shapes is None or not is_small_product(*shapes):
        return

    def replacement(first, second):
        return (first[:, :, None] * second[None, :, :]).sum(dim=1)

    match.replace_by_example(replacement, [first, second])


@register_graph_pattern(CallFunction(aten.bmm, Arg(), Arg()), pass_dict=_PATTERNS)
def _small_bmm(match: Match, first: torch.fx.Node, second: torch.fx.Node) -> None:
    shapes = _matrix_shapes(first, second)
    if shapes is None or not is_small_product(*shapes):
        return

    def replacement(first, second):
        return (first[:, :, :, None] * second[:, None, :, :]).sum(dim=2)

    match.replace_by_example(replacement, [first, second])


@register_graph_pattern(
    CallFunction(aten.addmm, Arg(), Arg(), Arg()), pass_dict=_PATTERNS
)
def _small_addmm(
    match: Match, bias: torch.fx.Node, first: torch.fx.Node, second: torch.fx.Node
) -> None:
    # Only the plain form bias + first @ second (beta = alpha = 1).
    node = match.nodes[-1]
    if node.kwargs.get("beta", 1) != 1 or node.kwargs.get("alpha", 1) != 1:
        return
    shapes = _matrix_shapes(first, second)
    if shapes is None or not is_small_product(*shapes):
        return

    def replacement(bias, first, second):
        return bias + (first[:, :, None] * second[None, :, :]).sum(dim=1)

    match.replace_by_example(replacement, [bias, first, second])


class SmallMatmulsAsReductions(CustomGraphPass):
    """Inductor post-grad pass (``post_grad_custom_pre_pass``); cacheable by its source."""

    def __call__(self, graph: torch.fx.Graph) -> None:
        _PATTERNS.apply(graph)
        graph.lint()

    def uuid(self) -> Optional[Any]:
        source = inspect.getsource(inspect.getmodule(SmallMatmulsAsReductions))
        return hashlib.sha256(source.encode()).hexdigest()


def inductor_options() -> Dict[str, Any]:
    """Inductor options of one ``torch.compile`` call that run the pass on its graphs.

    Passed per compile (``torch.compile(..., options=inductor_options())``), not set in
    Inductor's global config: it applies to the graphs that ask for it (compiled
    training, a compiled inference step) and to nothing else in the process.
    """
    return {"post_grad_custom_pre_pass": SmallMatmulsAsReductions()}


__all__ = [
    "MAX_CONTRACTION",
    "MAX_OUTPUTS",
    "SmallMatmulsAsReductions",
    "inductor_options",
    "is_small_product",
]
