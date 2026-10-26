"""Finding what installed distributions add, and building objects from configs.

Two things vary independently and are registered apart:

* **Which kinds exist**, and what each one's settings are: a config class, a
  :class:`~mace_core.config.kinds.KindedSection` subclass, found when a config
  is validated. Config classes are light, so validating a config imports no
  backend.
* **How a kind is built** in one framework: a builder, registered on a
  :class:`Builders` with :meth:`Builders.register` and imported only when that
  kind is built, in the process that builds it. One config class can have a
  numpy builder, a torch builder and a jax builder.

Both are found through entry points named after the kind, so a plugin adds a
kind without anything here naming it. In-tree kinds use the same groups.

**Discovery records failures; building raises.** A builder whose import fails
because a library is missing is a fact about this machine, recorded and
listed. Building that kind raises with the recorded reason. Nothing is ever
substituted, since another backend is another set of numbers.

**A name is registered once.** Two entry points registering different objects
under one name raise wherever the group is read: resolving by install order
would let a third-party distribution shadow a built-in kind with nothing said.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Any, Concatenate, Generic, ParamSpec, TypeVar

if TYPE_CHECKING:
    from mace_core.config.kinds import KindedSection

__all__ = [
    "Builders",
    "DuplicateRegistrationError",
    "NotAvailableError",
    "RegisteredBuilder",
    "clear_caches",
    "registered_entry_points",
]


class NotAvailableError(RuntimeError):
    """Something was asked for by name and cannot be delivered here."""


class DuplicateRegistrationError(RuntimeError):
    """Two registrations claim one name for different objects."""


def _describe(entry: EntryPoint) -> str:
    """An entry point as ``name = value`` plus the distribution declaring it."""
    distribution = getattr(entry, "dist", None)
    origin = getattr(distribution, "name", None) or "an unknown distribution"
    return f"{entry.name} = {entry.value} (from {origin})"


@cache
def registered_entry_points(group: str) -> dict[str, EntryPoint]:
    """Every entry point in ``group`` by name, read from installed metadata.

    Nothing is imported. Read once per process: the metadata does not change
    while it runs (:func:`clear_caches` forgets it, for tests that install a
    distribution).

    Raises:
        DuplicateRegistrationError: if two entry points register different
            objects under one name.
    """
    by_name: dict[str, EntryPoint] = {}
    for entry in entry_points(group=group):
        seen = by_name.get(entry.name)
        if seen is None:
            by_name[entry.name] = entry
        elif seen.value != entry.value:
            raise DuplicateRegistrationError(
                f"two entry points in {group!r} register {entry.name!r}: "
                f"{_describe(seen)} and {_describe(entry)}. Neither is chosen, "
                f"because the choice would follow install order. Uninstall one "
                f"of the two distributions, or register one under another name."
            )
    return by_name


ConfigT = TypeVar("ConfigT", bound="KindedSection")
KindT = TypeVar("KindT", bound="KindedSection")
ProductT = TypeVar("ProductT")
ContextP = ParamSpec("ContextP")

_REGISTRIES: list[Builders[Any, Any, ...]] = []


def clear_caches() -> None:
    """Forget the entry points read and the import failures recorded.

    For tests that install a distribution while the process runs. Builders
    and config classes already registered stay registered.
    """
    registered_entry_points.cache_clear()
    for registry in _REGISTRIES:
        registry._failures.clear()


@dataclass(frozen=True)
class RegisteredBuilder:
    """One kind a :class:`Builders` knows about, and whether it imported.

    Attributes:
        kind: The registered name.
        loaded: Whether its builder imported on this machine.
        reason: Why it did not, when it did not.
    """

    kind: str
    loaded: bool
    reason: str = ""


class Builders(Generic[ConfigT, ProductT, ContextP]):
    """The builders of one family of config sections, in one framework.

    ``Builders[NeighborListConfig, NeighborListBackend, []]`` maps each
    neighbour-list config class to the callable that turns it into a backend.
    ``ContextP`` is what a builder needs beyond its config, such as ``[float]``
    for a radial basis that takes ``r_max``.

    A builder is registered by the module that defines it, with
    :meth:`register`. That module is found through the entry point named after
    the kind in ``group`` and imported the first time the kind is built, so
    importing it costs nothing to a process that builds another kind.

    Args:
        family: The config base every registered config class subclasses.
        product: What every builder returns. A ``runtime_checkable`` protocol
            is checked on each product.
        group: The entry point group whose entry named after a kind is the
            module that registers that kind's builder.
    """

    def __init__(
        self, family: type[ConfigT], product: type[ProductT], group: str
    ) -> None:
        self.family = family
        self.product = product
        self.group = group
        self._builders: dict[type[ConfigT], Callable[..., ProductT]] = {}
        self._failures: dict[str, NotAvailableError] = {}
        _REGISTRIES.append(self)

    def register(
        self,
        config: type[KindT],
        builder: Callable[Concatenate[KindT, ContextP], ProductT],
    ) -> None:
        """Register ``builder`` as the way to build ``config``.

        Raises:
            TypeError: if ``config`` is not in this registry's family.
            DuplicateRegistrationError: if another builder is registered for
                ``config``. Registering the same one again is a no-op.
        """
        if not issubclass(config, self.family):
            raise TypeError(
                f"{config.__name__} is not a {self.family.__name__}, so it "
                f"cannot be registered in {self.group!r}."
            )
        seen = self._builders.get(config)  # type: ignore[arg-type]
        if seen is not None and seen is not builder:
            raise DuplicateRegistrationError(
                f"{config.__name__} has two builders: {_name(seen)} and "
                f"{_name(builder)}. A module imported twice under different "
                f"names, or reloaded, registers a second copy."
            )
        self._builders[config] = builder  # type: ignore[index]

    def build(
        self, config: ConfigT, *args: ContextP.args, **kwargs: ContextP.kwargs
    ) -> ProductT:
        """The object ``config`` describes.

        Raises:
            NotAvailableError: if no builder is registered for the kind, if
                its module did not import (with the reason), or if it built
                something that is not a ``product``. No other kind is tried.
            DuplicateRegistrationError: as :func:`registered_entry_points`.
        """
        builder = self._builders.get(type(config)) or self._load(config.kind)
        product = builder(config, *args, **kwargs)
        if _is_runtime_protocol(self.product) and not isinstance(product, self.product):
            raise NotAvailableError(
                f"the builder of {config.kind!r} returned a "
                f"{type(product).__name__}, which is not a "
                f"{self.product.__name__}."
            )
        return product

    def available(self) -> list[RegisteredBuilder]:
        """Every kind with a builder here, sorted, imported to see which fail.

        Use :meth:`build` to build one; it imports only that one.
        """
        kinds = {config.kind_name() for config in self._builders}
        kinds |= set(registered_entry_points(self.group))
        found = []
        for kind in sorted(kinds):
            try:
                self._load(kind)
            except NotAvailableError as failure:
                found.append(RegisteredBuilder(kind, False, str(failure)))
            else:
                found.append(RegisteredBuilder(kind, True))
        return found

    def _load(self, kind: str) -> Callable[..., ProductT]:
        """The builder of ``kind``, importing its module on first use."""
        if (builder := self._registered(kind)) is not None:
            return builder
        if kind in self._failures:
            raise self._failures[kind]
        entry = registered_entry_points(self.group).get(kind)
        try:
            if entry is None:
                raise NotAvailableError(
                    f"no builder is registered for {kind!r} in {self.group!r}. "
                    f"Built here: {sorted(self._kinds())}."
                )
            try:
                entry.load()
            except Exception as failure:
                raise NotAvailableError(
                    f"{kind!r} is registered in {self.group!r} but did not import "
                    f"on this machine: {failure!r}. No other kind is substituted."
                ) from failure
            builder = self._registered(kind)
            if builder is None:
                raise NotAvailableError(
                    f"{_describe(entry)} imported but registered no builder for "
                    f"{kind!r} on {self.group!r}."
                )
        except NotAvailableError as failure:
            # A module that failed to import is not cached by Python, so
            # without this every build would retry it.
            self._failures[kind] = failure
            raise
        return builder

    def _registered(self, kind: str) -> Callable[..., ProductT] | None:
        for config, builder in self._builders.items():
            if config.kind_name() == kind:
                return builder
        return None

    def _kinds(self) -> set[str]:
        return {config.kind_name() for config in self._builders} | set(
            registered_entry_points(self.group)
        )


def _name(builder: Callable[..., Any]) -> str:
    return f"{builder.__module__}.{getattr(builder, '__qualname__', builder)!s}"


def _is_runtime_protocol(product: type) -> bool:
    return bool(getattr(product, "_is_runtime_protocol", False))
