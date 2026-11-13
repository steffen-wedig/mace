"""Linears that read the same tensor, evaluated as one product per irrep, give the same model.

PolarMACE evaluates the linears of each layer's output (feature mixer, the next layer's
``skip_tp`` and ``linear_up``, a nonlinear readout's first linear) as one matmul per input
irrep over the concatenated weights (``prepare_linear_folds``,
``evaluate_linears_sharing_input``).
Values and every gradient (through the force, as in training) must not change.
"""

from __future__ import annotations

import pytest
import torch
from e3nn import o3

from mace.modules.polar.linear_folds import (
    evaluate_linears_sharing_input,
    linear_fold_of,
)
from mace.modules.wrapper_ops import CUET_AVAILABLE, CuEquivarianceConfig, Linear
from tests.extensions.polar.test_polar_field_linear_fold import (
    separate_linears,
)
from tests.extensions.polar.test_polar_shared_geometry import (
    _batch,
    _molecules,
    _outputs_and_gradients,
    _polar_model,
)
from tests.helpers import GRAPH_LONGRANGE_AVAILABLE

pytestmark = pytest.mark.polar

CUEQ_CONFIG = CuEquivarianceConfig(
    enabled=True, layout="ir_mul", group="O3_e3nn", optimize_all=True
)
needs_cueq_gpu = pytest.mark.skipif(
    not (CUET_AVAILABLE and torch.cuda.is_available()),
    reason="needs cuequivariance and a GPU",
)

# Two scalar segments (both write the one 0e output: summed), an output segment no
# input reaches (2e: zeros) and outputs of different widths.
INPUT_IRREPS = o3.Irreps("3x0e + 2x0e + 4x1o")
OUTPUT_IRREPS = [
    o3.Irreps("5x0e + 3x1o"),
    o3.Irreps("2x0e + 2x2e + 6x1o"),
    o3.Irreps("7x0e"),
]


def _through_force_gradient(outputs, x, leaves) -> list:
    # As in force training: a loss on the input gradient, differentiated again.
    energy = sum(torch.tanh(output).sum() for output in outputs)
    (input_gradient,) = torch.autograd.grad(energy, x, create_graph=True)
    loss = energy + (input_gradient**2).sum()
    gradients = torch.autograd.grad(loss, leaves, allow_unused=True)
    return [output.detach() for output in outputs] + list(gradients)


def _compare_shared_input_linears(linears, layout, device) -> None:
    dtype = torch.float64
    x = torch.randn(7, INPUT_IRREPS.dim, dtype=dtype, device=device, requires_grad=True)
    leaves = [x] + [linear.weight for linear in linears]
    folds = [linear_fold_of(linear) for linear in linears]
    separately = _through_force_gradient([linear(x) for linear in linears], x, leaves)
    together = _through_force_gradient(
        evaluate_linears_sharing_input(x, linears, folds, layout), x, leaves
    )
    for expected, actual in zip(separately, together):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)


def test_e3nn_linears_sharing_input_match_separate_linears():
    torch.manual_seed(0)
    linears = [
        o3.Linear(INPUT_IRREPS, irreps_out).to(torch.float64)
        for irreps_out in OUTPUT_IRREPS
    ]
    _compare_shared_input_linears(linears, "mul_ir", torch.device("cpu"))


@needs_cueq_gpu
def test_cueq_linears_sharing_input_match_separate_linears():
    torch.manual_seed(0)
    linears = [
        Linear(INPUT_IRREPS, irreps_out, cueq_config=CUEQ_CONFIG).to(
            device="cuda", dtype=torch.float64
        )
        for irreps_out in OUTPUT_IRREPS
    ]
    _compare_shared_input_linears(linears, "ir_mul", torch.device("cuda"))


# --------------------------------------------------------------------------- model level
def _compare_with_and_without_plan(cueq_config, device) -> None:
    model = _polar_model(cueq_config=cueq_config).to(device)
    plan = model.shared_input_plan
    assert plan.layout is not None
    assert plan.layers[0].with_next and plan.layers[-1].with_readout
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in _batch(_molecules()).items()
    }
    shared = _outputs_and_gradients(model, batch)
    # Reference: each linear on its own.
    separate_linears(model)
    separate = _outputs_and_gradients(model, batch)
    assert shared.keys() == separate.keys()
    for key, value in separate.items():
        scale = max(value.abs().max().item(), 1.0)
        assert (shared[key] - value).abs().max().item() <= 1e-10 * scale, key


@pytest.mark.skipif(
    not GRAPH_LONGRANGE_AVAILABLE, reason="graph_longrange is not installed"
)
def test_shared_input_linears_match_separate_linears_in_model():
    _compare_with_and_without_plan(cueq_config=None, device=torch.device("cpu"))


@pytest.mark.skipif(
    not GRAPH_LONGRANGE_AVAILABLE, reason="graph_longrange is not installed"
)
@needs_cueq_gpu
def test_cueq_shared_input_linears_match_separate_linears_in_model():
    _compare_with_and_without_plan(cueq_config=CUEQ_CONFIG, device=torch.device("cuda"))
