"""The Taylor-series branch of the smeared-Coulomb kernels as one contraction.

_smeared_coulomb_kernels_series evaluates sum_k c[k, n] x^k for every order n as a
contraction of the stacked powers [1, x, ..., x^12] with a coefficient table. These
tests pin it against the term-by-term loop it replaced (values, first and second
derivatives, both width conventions), check that the dispatched kernels have finite
derivatives at zero distance and around the crossover, and that the series traces into
a single graph under torch.compile.
"""

import shutil

import pytest
import torch

from mace.modules.polar.electrostatics.realspace_electrostatics import (
    SERIES_CROSSOVER,
    SERIES_NUM_TERMS,
    _smeared_coulomb_kernels_series,
    smeared_coulomb_kernels,
)

SCALAR_WIDTH = 1.5
WIDTH_VALUES = [0.7, 1.0, 1.5, 2.3]

# Float64 rounding level. The partial sums are O(1) and every term is at most
# crossover^k / k! <= 0.5, so the two summation orders differ by a few ulp.
VALUE_TOLERANCE = {"rtol": 1e-13, "atol": 1e-15}


def reference_series(scaled_distance_squared, combined_smearing_width, highest_order):
    """The term-by-term loop _smeared_coulomb_kernels_series used before the contraction."""
    accumulators = [
        torch.zeros_like(scaled_distance_squared) for _ in range(highest_order + 1)
    ]
    term = torch.ones_like(scaled_distance_squared)
    for k in range(SERIES_NUM_TERMS):
        for order in range(highest_order + 1):
            accumulators[order] = accumulators[order] + term / (2 * order + 2 * k + 1)
        term = term * (-scaled_distance_squared) / (k + 1)

    kernels = []
    width_power = 1.0 / combined_smearing_width
    sqrt_two_over_pi = (2.0 / torch.pi) ** 0.5
    for order in range(highest_order + 1):
        kernels.append(sqrt_two_over_pi * width_power * accumulators[order])
        width_power = width_power / combined_smearing_width**2
    return kernels


def scaled_distance_grid():
    """0 .. SERIES_CROSSOVER including both ends and points just next to them."""
    grid = torch.linspace(0.0, SERIES_CROSSOVER, 101, dtype=torch.float64)
    extra = torch.tensor(
        [1e-300, 1e-12, 1e-6, SERIES_CROSSOVER - 1e-9], dtype=torch.float64
    )
    return torch.sort(torch.cat([grid, extra])).values


def width_cases():
    """(x shape transform, width): scalar width on [n], tensor width on [n, 1]."""
    return [
        pytest.param(lambda x: x, SCALAR_WIDTH, id="scalar_width"),
        pytest.param(
            lambda x: x.unsqueeze(-1),
            torch.tensor(WIDTH_VALUES, dtype=torch.float64),
            id="broadcast_width",
        ),
    ]


@pytest.mark.parametrize("highest_order", [0, 1, 2])
@pytest.mark.parametrize("reshape,width", width_cases())
def test_series_matches_reference_loop(highest_order, reshape, width):
    scaled = reshape(scaled_distance_grid())
    kernels = _smeared_coulomb_kernels_series(scaled, width, highest_order)
    expected = reference_series(scaled, width, highest_order)
    assert len(kernels) == highest_order + 1
    for kernel, expected_kernel in zip(kernels, expected, strict=True):
        assert kernel.shape == expected_kernel.shape
        assert kernel.dtype == torch.float64
        torch.testing.assert_close(kernel, expected_kernel, **VALUE_TOLERANCE)


@pytest.mark.parametrize("highest_order", [0, 1, 2])
@pytest.mark.parametrize("reshape,width", width_cases())
def test_series_derivatives_match_reference_loop(highest_order, reshape, width):
    """First and second derivatives in x, through create_graph=True as in force training."""
    grid = scaled_distance_grid()
    scaled = reshape(grid.clone()).requires_grad_(True)
    scaled_reference = reshape(grid.clone()).requires_grad_(True)
    kernels = _smeared_coulomb_kernels_series(scaled, width, highest_order)
    expected = reference_series(scaled_reference, width, highest_order)

    for kernel, expected_kernel in zip(kernels, expected, strict=True):
        derivatives = []
        for inputs, output in ((scaled, kernel), (scaled_reference, expected_kernel)):
            first = torch.autograd.grad(
                output.sum(), inputs, create_graph=True, retain_graph=True
            )[0]
            second = torch.autograd.grad(first.sum(), inputs, retain_graph=True)[0]
            derivatives.append((first, second))
        (first, second), (expected_first, expected_second) = derivatives
        assert torch.isfinite(first).all() and torch.isfinite(second).all()
        # Derivatives sum the same terms times k (and k (k - 1)), so the
        # rounding stays at the level of the values.
        torch.testing.assert_close(first, expected_first, **VALUE_TOLERANCE)
        torch.testing.assert_close(second, expected_second, **VALUE_TOLERANCE)


def test_series_derivatives_at_zero_are_the_leading_coefficients():
    """d/dx and d^2/dx^2 of the accumulators at x = 0 are c[1, n] and 2 c[2, n]."""
    scaled = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    kernels = _smeared_coulomb_kernels_series(scaled, 1.0, 2)
    sqrt_two_over_pi = (2.0 / torch.pi) ** 0.5
    for order, kernel in enumerate(kernels):
        first = torch.autograd.grad(
            kernel.sum(), scaled, create_graph=True, retain_graph=True
        )[0]
        second = torch.autograd.grad(first.sum(), scaled, retain_graph=True)[0]
        expected_first = -sqrt_two_over_pi / (2 * order + 3)
        expected_second = 2.0 * sqrt_two_over_pi / (2.0 * (2 * order + 5))
        torch.testing.assert_close(
            first, torch.full_like(first, expected_first), rtol=1e-15, atol=0.0
        )
        torch.testing.assert_close(
            second, torch.full_like(second, expected_second), rtol=1e-15, atol=0.0
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_series_keeps_dtype_and_scalar_input_shape(dtype):
    scaled = torch.tensor(0.3, dtype=dtype)
    kernels = _smeared_coulomb_kernels_series(scaled, SCALAR_WIDTH, 2)
    expected = reference_series(scaled.double(), SCALAR_WIDTH, 2)
    for kernel, expected_kernel in zip(kernels, expected, strict=True):
        assert kernel.shape == torch.Size([])
        assert kernel.dtype == dtype
        torch.testing.assert_close(
            kernel.double(),
            expected_kernel,
            rtol=1e-6 if dtype == torch.float32 else 1e-13,
            atol=0.0,
        )


@pytest.mark.parametrize("reshape,width", width_cases())
def test_dispatched_kernel_derivatives_finite_at_zero_and_across_crossover(
    reshape, width
):
    scaled_values = torch.tensor(
        [
            0.0,
            1e-12,
            0.25,
            SERIES_CROSSOVER - 1e-9,
            SERIES_CROSSOVER,
            SERIES_CROSSOVER + 1e-9,
            0.75,
            5.0,
        ],
        dtype=torch.float64,
    )
    width_tensor = torch.as_tensor(width, dtype=torch.float64)
    distance_squared = (
        reshape(scaled_values) * 2.0 * width_tensor.min() ** 2
    ).requires_grad_(True)
    kernels = smeared_coulomb_kernels(distance_squared, width, highest_order=2)
    for kernel in kernels:
        assert torch.isfinite(kernel).all()
        first = torch.autograd.grad(
            kernel.sum(), distance_squared, create_graph=True, retain_graph=True
        )[0]
        second = torch.autograd.grad(first.sum(), distance_squared, retain_graph=True)[
            0
        ]
        assert torch.isfinite(first).all()
        assert torch.isfinite(second).all()


def _inductor_cpu_available():
    """Whether Inductor can build a trivial CPU kernel (needs a C++ compiler)."""
    if shutil.which("g++") is None and shutil.which("c++") is None:
        return False
    torch._dynamo.reset()
    try:
        torch.compile(lambda tensor: tensor * 2.0 + 1.0, fullgraph=True)(
            torch.ones(4, dtype=torch.float64)
        )
    except Exception:  # noqa: BLE001 - any toolchain failure means unavailable
        return False
    finally:
        torch._dynamo.reset()
    return True


def test_series_traces_as_one_small_graph():
    """fullgraph tracing succeeds and the contraction keeps the graph small."""
    graphs = []

    def counting_backend(graph_module, example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    torch._dynamo.reset()
    compiled = torch.compile(
        _smeared_coulomb_kernels_series, backend=counting_backend, fullgraph=True
    )
    scaled = scaled_distance_grid().unsqueeze(-1)
    width = torch.tensor(WIDTH_VALUES, dtype=torch.float64)
    kernels = compiled(scaled, width, 2)
    torch._dynamo.reset()

    assert len(graphs) == 1
    node_count = len(graphs[0].graph.nodes)
    # 13 powers + stack + table + contraction + 3 orders with width powers;
    # the unrolled loop it replaced traced to well over a hundred nodes.
    assert node_count < 60, node_count
    for kernel, expected in zip(
        kernels, reference_series(scaled, width, 2), strict=True
    ):
        torch.testing.assert_close(kernel, expected, **VALUE_TOLERANCE)


def test_series_compiled_with_inductor_on_cpu():
    if not _inductor_cpu_available():
        pytest.skip(
            "Inductor cannot build CPU kernels here (no working C++ compiler; "
            "on DAIS load gcc/15 and export CXX)"
        )
    torch._dynamo.reset()
    compiled = torch.compile(_smeared_coulomb_kernels_series, fullgraph=True)
    scaled = scaled_distance_grid().unsqueeze(-1)
    width = torch.tensor(WIDTH_VALUES, dtype=torch.float64)
    try:
        kernels = compiled(scaled, width, 2)
    finally:
        torch._dynamo.reset()
    for kernel, expected in zip(
        kernels, reference_series(scaled, width, 2), strict=True
    ):
        torch.testing.assert_close(kernel, expected, **VALUE_TOLERANCE)
