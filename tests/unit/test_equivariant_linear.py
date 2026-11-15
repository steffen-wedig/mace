"""EquivariantLinear and NarrowEquivariantLinear compute e3nn's o3.Linear in both layouts."""

from __future__ import annotations

import pytest
import torch
from e3nn import o3

from mace.modules.equivariant_linear import (
    NARROW_CHANNEL_LIMIT,
    EquivariantLinear,
    NarrowEquivariantLinear,
    linear_plan,
)

# (irreps_in, irreps_out, biases): unsimplified and repeated irreps (several input blocks
# write one output block, which changes the normalisation), an output irrep without an
# input block (zeros), odd parity, scalar biases.
WIDE_CASES = [
    ("16x0e + 8x1o", "12x0e + 4x1o", False),
    ("8x0e + 4x1o + 6x0e + 2x1o + 3x2e", "5x0e + 7x1o + 2x2e", True),
    ("10x0e + 3x1e", "4x0e + 6x1e + 2x1o", True),
    ("2x0e + 1x1o + 2x0e + 1x1o", "32x0e + 16x1o", False),
    ("2x0e + 2x1o", "64x0e + 64x1o", False),
]
# Every output block has at most NARROW_CHANNEL_LIMIT channels: readouts to one or two
# scalars, per-spin multipoles (charges and dipoles of two spins).
NARROW_CASES = [
    ("32x0e", "1x0e", True),
    ("16x0e", "2x0e", True),
    ("32x0e + 32x1o", "0e + 1o + 0e + 1o + 2x0e", True),
    ("64x0e + 64x1o", "0e + 1o + 0e + 1o", False),
]
LAYOUTS = ("mul_ir", "ir_mul")


def to_layout(features: torch.Tensor, irreps: o3.Irreps, layout: str) -> torch.Tensor:
    """mul_ir features in ``layout``."""
    if layout == "mul_ir":
        return features
    blocks = []
    for (multiplicity, irrep), block in zip(irreps, irreps.slices()):
        blocks.append(
            features[..., block]
            .reshape(*features.shape[:-1], multiplicity, irrep.dim)
            .transpose(-1, -2)
            .reshape(*features.shape[:-1], -1)
        )
    return torch.cat(blocks, dim=-1)


def reference_and_linear(linear_class, irreps_in, irreps_out, biases, layout):
    torch.manual_seed(0)
    reference = o3.Linear(irreps_in, irreps_out, biases=biases)
    if biases:
        with torch.no_grad():
            reference.bias.normal_()
    linear = linear_class(irreps_in, irreps_out, layout=layout, biases=biases)
    linear.load_state_dict(reference.state_dict())
    return reference, linear


def force_like_terms(module, features):
    """Value, input gradient and the gradient of the squared input gradient (the second
    derivative force training takes) to the input and the weights."""
    output = module(features)
    energy = torch.tanh(output).sum()
    (gradient,) = torch.autograd.grad(energy, features, create_graph=True)
    second = torch.autograd.grad(
        (gradient**2).sum(), [features] + list(module.parameters())
    )
    return output, gradient, second


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    ("linear_class", "case"),
    [(EquivariantLinear, case) for case in WIDE_CASES + NARROW_CASES]
    + [(NarrowEquivariantLinear, case) for case in NARROW_CASES],
)
def test_matches_e3nn_with_first_and_second_derivatives(linear_class, case, layout):
    irreps_in, irreps_out = o3.Irreps(case[0]), o3.Irreps(case[1])
    reference, linear = reference_and_linear(
        linear_class, irreps_in, irreps_out, case[2], layout
    )
    torch.manual_seed(1)
    features = torch.randn(7, irreps_in.dim, dtype=torch.float64)
    reference = reference.double()
    linear = linear.double()

    reference_features = features.clone().requires_grad_(True)
    layout_features = to_layout(features, irreps_in, layout).requires_grad_(True)
    expected_output, expected_gradient, expected_second = force_like_terms(
        reference, reference_features
    )
    output, gradient, second = force_like_terms(linear, layout_features)

    torch.testing.assert_close(
        output, to_layout(expected_output, irreps_out, layout), rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        gradient,
        to_layout(expected_gradient, irreps_in, layout),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        second[0], to_layout(expected_second[0], irreps_in, layout), rtol=1e-12, atol=1e-12
    )
    for value, expected in zip(second[1:], expected_second[1:]):
        torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("linear_class", [EquivariantLinear, NarrowEquivariantLinear])
def test_an_empty_output_has_zero_columns(linear_class):
    # Dipole-only models build readouts to no irreps at all.
    linear = linear_class("4x0e + 2x1o", o3.Irreps(""), biases=True)
    assert linear(torch.randn(5, 10)).shape == (5, 0)


def test_a_linear_without_blocks_keeps_the_rows_of_an_empty_input():
    # As e3nn's: a gate that passes no channel feeds a [rows, 0] tensor to a linear
    # without paths (AtomicDipolesMACE's readout with scalar-only hidden irreps).
    linear = EquivariantLinear("8x0e", "1x1o")
    assert torch.equal(linear(torch.randn(3, 0)), torch.zeros(3, 3))


def test_leading_dimensions_are_kept():
    linear = EquivariantLinear("4x0e + 2x1o", "3x0e + 1x1o")
    assert linear(torch.randn(5, 2, 10)).shape == (5, 2, 6)


@pytest.mark.parametrize("case", WIDE_CASES + NARROW_CASES)
def test_parameter_counts_match_e3nn(case):
    reference = o3.Linear(case[0], case[1], biases=case[2])
    plan = linear_plan(o3.Irreps(case[0]), o3.Irreps(case[1]), case[2])
    assert plan.weight_numel == reference.weight_numel
    assert plan.bias_numel == reference.bias_numel


def test_loads_a_cueq_shaped_weight():
    reference = o3.Linear("8x0e + 4x1o", "3x0e + 2x1o")
    linear = EquivariantLinear("8x0e + 4x1o", "3x0e + 2x1o", layout="ir_mul")
    linear.load_state_dict({"weight": reference.weight.detach().reshape(1, -1)})
    torch.testing.assert_close(linear.weight, reference.weight)


def test_narrow_linear_refuses_wider_output_blocks():
    wide = NARROW_CHANNEL_LIMIT + 1
    with pytest.raises(ValueError, match="use EquivariantLinear"):
        NarrowEquivariantLinear("64x0e", f"{wide}x0e")
    # Few input channels do not make a linear narrow (experiment 18).
    with pytest.raises(ValueError, match="use EquivariantLinear"):
        NarrowEquivariantLinear("1x0e", "64x0e")
    NarrowEquivariantLinear("512x0e", f"{NARROW_CHANNEL_LIMIT}x0e")


def test_unknown_layout_is_refused():
    with pytest.raises(ValueError, match="layout"):
        EquivariantLinear("2x0e", "2x0e", layout="channels_first")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled check runs on a GPU")
@pytest.mark.parametrize("linear_class", [EquivariantLinear, NarrowEquivariantLinear])
def test_compiles_without_graph_breaks(linear_class):
    linear = linear_class("32x0e + 32x1o", "0e + 1o + 0e + 1o + 2x0e", biases=True).to(
        device="cuda", dtype=torch.float64
    )
    features = torch.randn(50, 128, device="cuda", dtype=torch.float64)
    compiled = torch.compile(linear, fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(features), linear(features))
