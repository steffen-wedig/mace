"""Small matrix products become fused reductions under compile, without changing values."""

from __future__ import annotations

import torch
from torch._inductor.utils import run_and_get_code

from mace.tools.compiled.small_matmuls import inductor_options, is_small_product

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_small_products_are_the_narrow_or_short_ones():
    assert is_small_product(rows=260, columns=1, contraction=512)
    assert is_small_product(rows=260, columns=2, contraction=32)
    assert is_small_product(rows=260, columns=512, contraction=4)
    assert is_small_product(rows=4, columns=512, contraction=260)
    assert is_small_product(rows=6, columns=780, contraction=512)
    assert not is_small_product(rows=260, columns=16, contraction=512)
    assert not is_small_product(rows=260, columns=512, contraction=512)
    assert not is_small_product(rows=260, columns=512, contraction=1)


def _force_loss(positions, lift, wide, narrow, table):
    features = torch.tanh(positions @ lift)  # [n, 64] from 3 coordinates
    hidden = torch.tanh(features @ wide)  # regular GEMM
    readout = torch.einsum("nk,ko->no", hidden, narrow)  # 2 outputs: small
    elements = torch.nn.functional.one_hot(
        torch.arange(positions.shape[0], device=positions.device) % 4, 4
    )
    embedding = elements.to(positions.dtype) @ table  # contraction over 4: small
    energy = (readout.sum(-1) * embedding.sum(-1)).sum()
    (gradient,) = torch.autograd.grad(energy, positions, create_graph=True)
    return energy + (gradient**2).sum()


def _extern_matmuls(codes) -> int:
    generated = "\n".join(codes)
    return sum(
        generated.count(f"extern_kernels.{name}(") for name in ("mm", "bmm", "addmm")
    )


def test_compiled_force_loss_matches_eager_with_small_products_fused():
    import torch._inductor.config as inductor_config  # noqa: PLC0415

    torch.manual_seed(0)
    dtype = torch.float64
    lift = torch.randn(3, 64, dtype=dtype, device=DEVICE)
    parameters = [
        torch.randn(64, 64, dtype=dtype, device=DEVICE, requires_grad=True),
        torch.randn(64, 2, dtype=dtype, device=DEVICE, requires_grad=True),
        torch.randn(4, 64, dtype=dtype, device=DEVICE, requires_grad=True),
    ]
    positions = torch.randn(40, 3, dtype=dtype, device=DEVICE)

    def gradients(function):
        leaf = positions.clone().requires_grad_(True)
        loss = function(leaf, lift, *parameters)
        return [loss.detach(), *torch.autograd.grad(loss, parameters)]

    def compiled_run(options):
        torch._dynamo.reset()
        compiled = torch.compile(
            _force_loss, fullgraph=True, dynamic=False, options=options
        )
        return run_and_get_code(gradients, compiled)

    expected = gradients(_force_loss)
    previous_trace = torch._dynamo.config.trace_autograd_ops
    torch._dynamo.config.trace_autograd_ops = True
    try:
        _, plain_codes = compiled_run({})
        actual, fused_codes = compiled_run(inductor_options())
    finally:
        torch._dynamo.config.trace_autograd_ops = previous_trace
        torch._dynamo.reset()
    # Per compile only: Inductor's global config is untouched.
    assert inductor_config.post_grad_custom_pre_pass is None
    for want, have in zip(expected, actual):
        torch.testing.assert_close(have, want, rtol=1e-10, atol=1e-12)
    # The 64 x 64 products stay GEMMs; the 2-output readout, the 4-element table and
    # their gradients do not.
    assert 0 < _extern_matmuls(fused_codes) < _extern_matmuls(plain_codes)
