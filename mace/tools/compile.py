import copy
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Tuple

try:
    import torch._dynamo as dynamo
except ImportError:
    dynamo = None
from e3nn import get_optimization_defaults, set_optimization_defaults
import torch
from torch import autograd, nn
from torch.fx import symbolic_trace

ModuleFactory = Callable[..., nn.Module]
TypeTuple = Tuple[type, ...]


def configure_autograd_for_compile(allow_autograd: bool = True) -> None:
    """Align Dynamo autograd settings with MACE's compile strategy.

    Recent PyTorch releases require `trace_autograd_ops=True` when
    `torch.autograd.grad` is traced into the compiled graph.
    """
    if dynamo is None:
        return

    if allow_autograd:
        dynamo.allow_in_graph(autograd.grad)
        if hasattr(dynamo.config, "trace_autograd_ops"):
            dynamo.config.trace_autograd_ops = True
    else:
        dynamo.disallow_in_graph(autograd.grad)


@contextmanager
def disable_e3nn_codegen():
    """Context manager that disables the legacy PyTorch code generation used in e3nn."""
    init_val = get_optimization_defaults()["jit_script_fx"]
    set_optimization_defaults(jit_script_fx=False)
    yield
    set_optimization_defaults(jit_script_fx=init_val)


def prepare(func: ModuleFactory, allow_autograd: bool = True) -> ModuleFactory:
    """Function transform that prepares a MACE module for torch.compile

    Args:
        func (ModuleFactory): A function that creates an nn.Module
        allow_autograd (bool, optional): Force inductor compiler to inline call to
            `torch.autograd.grad`. Defaults to True.

    Returns:
        ModuleFactory: Decorated function that creates a torch.compile compatible module
    """
    configure_autograd_for_compile(allow_autograd=allow_autograd)

    @wraps(func)
    def wrapper(*args, **kwargs):
        with disable_e3nn_codegen():
            model = func(*args, **kwargs)

        model = simplify(model)
        return model

    return wrapper


_SIMPLIFY_REGISTRY = set()


def simplify_if_compile(module: nn.Module) -> nn.Module:
    """Decorator to register a module for symbolic simplification

    The decorated module will be simplifed using `torch.fx.symbolic_trace`.
    This constrains the module to not have any dynamic control flow, see:

    https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing

    Args:
        module (nn.Module): the module to register

    Returns:
        nn.Module: registered module
    """
    _SIMPLIFY_REGISTRY.add(module)
    return module


def simplify(module: nn.Module) -> nn.Module:
    """Recursively searches for registered modules to simplify with
    `torch.fx.symbolic_trace` to support compiling with the PyTorch Dynamo compiler.

    Modules are registered with the `simplify_if_compile` decorator and

    Args:
        module (nn.Module): the module to simplify

    Returns:
        nn.Module: the simplified module
    """
    simplify_types = tuple(_SIMPLIFY_REGISTRY)

    for name, child in module.named_children():
        if isinstance(child, simplify_types):
            traced = symbolic_trace(child)
            setattr(module, name, traced)
        else:
            simplify(child)

    return module


def simplified_view(module: nn.Module) -> nn.Module:
    """A view of ``module`` for torch.compile that leaves ``module`` untouched.

    Like :func:`simplify`, registered modules are replaced by their
    ``torch.fx.symbolic_trace``; but the replacement happens on shallow copies,
    and the traced graphs hold the original parameters and buffers by reference.
    Training through the view therefore updates the original model, which keeps
    its own module types for checkpoints, saved models and everything that
    inspects them (config extraction, calculators, warm starts).
    """
    if isinstance(module, tuple(_SIMPLIFY_REGISTRY)):
        return symbolic_trace(module)
    view = copy.copy(module)
    view._modules = {  # pylint: disable=protected-access
        name: None if child is None else simplified_view(child)
        for name, child in module._modules.items()  # pylint: disable=protected-access
    }
    return view


class CompiledTrainingModel(nn.Module):
    """Routes training forwards through torch.compile, everything else to the model.

    Takes the place of the distributed model in ``tools.train``: training steps
    call it with ``training=True`` and get the compiled forward of a
    :func:`simplified_view` (one graph, dynamic shapes); validation calls it with
    ``training=False`` and gets the eager model. The model is built without e3nn
    codegen (:func:`disable_e3nn_codegen`) and stays unsimplified, so it is what
    is checkpointed and saved. A compiled graph cannot turn its inputs into
    differentiable leaves, so the positions get ``requires_grad`` here, before
    the compiled call.
    """

    def __init__(self, model: nn.Module, backend: str = "inductor") -> None:
        super().__init__()
        configure_autograd_for_compile(allow_autograd=True)
        self.model = model
        self.compiled = torch.compile(
            simplified_view(model), fullgraph=True, dynamic=True, backend=backend
        )

    def forward(self, data, training: bool = False, **kwargs):
        if not training:
            return self.model(data, training=False, **kwargs)
        data["positions"].requires_grad_(True)
        return self.compiled(data, training=True, **kwargs)
