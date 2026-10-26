"""Plugins, and building objects from config sections.

A plugin is a distribution with a module in the ``mace.plugins`` entry point
group. Importing it defines its config kinds and registers their builders.

Builder modules must be cheap to import: a backend imports its heavy library
(vesin, CUDA, ...) when it is built, not at module level. Registration then
needs no lazy loading, and a backend that is not installed fails when it is
built, naming what to install.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Concatenate, Generic, ParamSpec, TypeVar

if TYPE_CHECKING:
    from mace_core.config.kinds import KindedSection

__all__ = ["Builders", "NotAvailableError", "load_plugins"]

ConfigT = TypeVar("ConfigT", bound="KindedSection")
KindT = TypeVar("KindT", bound="KindedSection")
ProductT = TypeVar("ProductT")
ContextP = ParamSpec("ContextP")


class NotAvailableError(LookupError):
    """A kind has no builder in this framework."""


@cache
def load_plugins() -> dict[str, str]:
    """Import every ``mace.plugins`` module, once per process.

    Returns:
        The plugins that failed to import, by name, with the reason. A broken
        plugin is skipped, not fatal: its kinds are simply missing.
    """
    failed = {}
    for entry in entry_points(group="mace.plugins"):
        try:
            entry.load()
        except Exception as failure:
            failed[entry.name] = repr(failure)
    return failed


class Builders(Generic[ConfigT, ProductT, ContextP]):
    """How each kind of one config family is built, in one framework.

    ``Builders[NeighborListConfig, NeighborListBackend, []]`` maps every
    neighbour-list config class to what builds its backend. ``ContextP`` is
    what a builder takes besides its config: ``[float]`` for a radial basis
    that needs ``r_max``, ``[]`` for nothing.
    """

    def __init__(self, family: type[ConfigT]) -> None:
        self.family = family
        self._builders: dict[type[Any], Callable[..., ProductT]] = {}

    def register(
        self,
        config: type[KindT],
        builder: Callable[Concatenate[KindT, ContextP], ProductT],
    ) -> None:
        """Build ``config`` with ``builder``. Type checkers check the pair."""
        if not issubclass(config, self.family):
            raise TypeError(f"{config.__name__} is not a {self.family.__name__}.")
        if self._builders.setdefault(config, builder) is not builder:
            raise TypeError(f"{config.__name__} already has a builder.")

    def build(
        self, config: ConfigT, *args: ContextP.args, **kwargs: ContextP.kwargs
    ) -> ProductT:
        """The object ``config`` describes.

        Raises:
            NotAvailableError: if its kind has no builder here.
        """
        if type(config) not in self._builders:
            load_plugins()
        builder = self._builders.get(type(config))
        if builder is None:
            raise NotAvailableError(
                f"{self.family.__name__} kind {config.kind!r} has no builder "
                f"here; the kinds with one are "
                f"{sorted(c.model_fields['kind'].default for c in self._builders)}."
            )
        return builder(config, *args, **kwargs)
