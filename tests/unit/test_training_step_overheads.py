"""Host-overhead reductions of the training step: foreach EMA, fused Adam on CUDA, and the
parameter list of gradient clipping built once.

Each must give the update of the implementation it replaces.
"""

from __future__ import annotations

import argparse

import pytest
import torch
from torch_ema import ExponentialMovingAverage

from mace.tools.checkpoint import optimizer_state_for
from mace.tools.ema import ForeachExponentialMovingAverage
from mace.tools.scripts_utils import get_optimizer
from mace.tools.train import parameter_list


def _module(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(5, 7), torch.nn.SiLU(), torch.nn.Linear(7, 3)
    ).double()


def _perturb(module: torch.nn.Module, step: int) -> None:
    generator = torch.Generator().manual_seed(100 + step)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(
                torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
            )


def test_foreach_ema_matches_torch_ema_and_shares_its_state():
    reference_module, foreach_module = _module(0), _module(0)
    reference = ExponentialMovingAverage(reference_module.parameters(), decay=0.99)
    foreach = ForeachExponentialMovingAverage(foreach_module.parameters(), decay=0.99)
    for step in range(25):
        _perturb(reference_module, step)
        _perturb(foreach_module, step)
        reference.update()
        foreach.update()
    assert foreach.num_updates == reference.num_updates
    for expected, actual in zip(reference.shadow_params, foreach.shadow_params):
        assert torch.allclose(actual, expected, rtol=1e-14, atol=1e-14)
    # Checkpoints are interchangeable.
    restored = ForeachExponentialMovingAverage(_module(1).parameters(), decay=0.5)
    restored.load_state_dict(reference.state_dict())
    for expected, actual in zip(reference.shadow_params, restored.shadow_params):
        assert torch.equal(actual, expected)


def _amsgrad_options(module: torch.nn.Module) -> dict:
    return dict(
        params=[
            {"name": "first", "params": module[0].parameters(), "weight_decay": 5e-7},
            {"name": "last", "params": module[2].parameters(), "weight_decay": 0.0},
        ],
        lr=1e-2,
        amsgrad=True,
        betas=(0.9, 0.999),
    )


def test_optimizer_is_fused_on_cuda_only():
    module = _module(0)
    cpu_optimizer = get_optimizer(
        argparse.Namespace(optimizer="adam", device="cpu"), _amsgrad_options(module)
    )
    assert not cpu_optimizer.defaults.get("fused")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused Adam needs CUDA")
@pytest.mark.parametrize("optimizer_name", ["adam", "adamw"])
def test_fused_amsgrad_matches_foreach(optimizer_name):
    fused_module, foreach_module = _module(0).cuda(), _module(0).cuda()
    fused = get_optimizer(
        argparse.Namespace(optimizer=optimizer_name, device="cuda"),
        _amsgrad_options(fused_module),
    )
    assert fused.defaults["fused"]
    optimizer_class = type(fused)
    foreach = optimizer_class(**_amsgrad_options(foreach_module), foreach=True)
    for step in range(10):
        generator = torch.Generator(device="cuda").manual_seed(step)
        for module, optimizer in ((fused_module, fused), (foreach_module, foreach)):
            optimizer.zero_grad(set_to_none=True)
            for parameter in module.parameters():
                parameter.grad = torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
            optimizer.step()
            generator = torch.Generator(device="cuda").manual_seed(step)
    for expected, actual in zip(foreach_module.parameters(), fused_module.parameters()):
        assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused Adam needs CUDA")
def test_fused_optimizer_loads_a_foreach_checkpoint():
    module = _module(0).cuda()
    foreach = torch.optim.Adam(**_amsgrad_options(module), foreach=True)
    for parameter in module.parameters():
        parameter.grad = torch.ones_like(parameter)
    foreach.step()
    fused = get_optimizer(
        argparse.Namespace(optimizer="adam", device="cuda"), _amsgrad_options(module)
    )
    fused.load_state_dict(optimizer_state_for(fused, foreach.state_dict()))
    assert all(group["fused"] for group in fused.param_groups)
    fused.step()
    state = fused.state[next(module.parameters())]
    assert state["step"].device.type == "cuda"
    assert float(state["step"]) == 2.0
    # And back: a fused checkpoint restores into a CPU (foreach) optimizer.
    cpu_module = _module(0)
    cpu_optimizer = get_optimizer(
        argparse.Namespace(optimizer="adam", device="cpu"), _amsgrad_options(cpu_module)
    )
    cpu_optimizer.load_state_dict(
        optimizer_state_for(cpu_optimizer, fused.state_dict())
    )
    for parameter in cpu_module.parameters():
        parameter.grad = torch.ones_like(parameter)
    cpu_optimizer.step()
    assert not any(group.get("fused") for group in cpu_optimizer.param_groups)


def test_gradient_clipping_over_the_cached_parameter_list():
    """clip_grad_norm_ over parameter_list(model) (built once per model) clips exactly as
    over model.parameters(), and later calls reuse the same list (no module-tree walk).
    """
    reference_module, cached_module = _module(0), _module(0)
    inputs = torch.randn(11, 5, dtype=torch.float64)
    for module in (reference_module, cached_module):
        (module(inputs) ** 2).sum().backward()
    reference_norm = torch.nn.utils.clip_grad_norm_(reference_module.parameters(), 0.1)
    cached_norm = torch.nn.utils.clip_grad_norm_(parameter_list(cached_module), 0.1)
    assert torch.equal(cached_norm, reference_norm)
    for reference, cached in zip(
        reference_module.parameters(), cached_module.parameters()
    ):
        assert torch.equal(cached.grad, reference.grad)
    listed = parameter_list(cached_module)
    assert listed is parameter_list(cached_module)
    assert all(a is b for a, b in zip(listed, cached_module.parameters()))
    # Moving the model replaces parameter data, not the parameter objects.
    cached_module.float()
    assert all(
        a is b
        for a, b in zip(parameter_list(cached_module), cached_module.parameters())
    )
