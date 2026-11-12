import copy
import logging
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Tuple

try:
    import torch._dynamo as dynamo
except ImportError:
    dynamo = None
from e3nn import get_optimization_defaults, set_optimization_defaults
from e3nn import nn as e3nn_nn
from torch import autograd, nn
from torch.fx import symbolic_trace
from torch.fx.proxy import TraceError

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


# e3nn's Activation loops over its Irreps in Python, which Dynamo cannot guard on. It was
# registered only as a side effect of building a NonLinearReadoutBlock, so models without
# one (PolarMACE with bias readouts) kept it unsimplified.
simplify_if_compile(e3nn_nn.Activation)


def simplified_view(module: nn.Module) -> nn.Module:
    """A copy of the module tree for Dynamo that shares every parameter and buffer with
    ``module``, with the registered modules replaced by their fx-traced graphs.

    Unlike :func:`simplify`, ``module`` itself is left unchanged: optimizers, EMA, config
    extraction, conversion and saving keep working on it, and the view sees every update
    of its parameters (same objects; fx's GraphModule references its root's parameters).
    Only the containers on the path to a registered module are copied, shallowly (their
    parameter and buffer dictionaries are the original's); every other subtree is the
    original module object. Build the view after moving the model to its device and
    dtype.
    """
    simplify_types = tuple(_SIMPLIFY_REGISTRY)
    if not any(isinstance(inner, simplify_types) for inner in module.modules()):
        return module
    if isinstance(module, simplify_types):
        try:
            return symbolic_trace(module)
        except TraceError as error:
            # Python control flow on the inputs, e.g. a cueq linear inside a readout
            # (cueq forwards check their inputs): simplify its children instead. Dynamo
            # traces the cueq modules itself; the e3nn activation is what it needs traced.
            logging.info(
                "simplified_view: %s not traceable as a whole (%s); simplifying its "
                "children",
                type(module).__name__,
                error,
            )
    view = copy.copy(module)
    view._modules = {  # pylint: disable=protected-access
        name: None if child is None else simplified_view(child)
        for name, child in module._modules.items()  # pylint: disable=protected-access
    }
    return view


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
