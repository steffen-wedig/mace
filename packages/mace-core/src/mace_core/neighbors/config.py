"""The ``neighbor_list`` config section: one class per backend.

These classes import nothing beyond pydantic, so validating a config that
names vesin imports neither vesin nor torch; the backend is imported when it
is built (:data:`~mace_core.neighbors.backends.NEIGHBOR_BACKENDS`).

A backend from another distribution adds its config class here through the
``mace.neighbor_list`` entry point group, named after its kind.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from mace_core.config.kinds import KindedSection

__all__ = [
    "BruteForceConfig",
    "MatscipyConfig",
    "NeighborListConfig",
    "VesinConfig",
]


class NeighborListConfig(KindedSection):
    """Which neighbour-list backend builds the graphs, and its settings.

    A field typed ``NeighborListConfig | None`` accepts any registered kind;
    ``None`` leaves the choice to the framework building the graphs (see
    :func:`~mace_core.neighbors.backends.resolve_neighbor_backend`). The
    choice never changes a number, since every backend returns the same edge
    set, so it is run configuration and not part of a model's state.
    """

    entry_point_group: ClassVar[str | None] = "mace.neighbor_list"


class BruteForceConfig(NeighborListConfig):
    """Every pair, every reachable image. Needs only numpy."""

    kind: Literal["brute_force"] = "brute_force"


class MatscipyConfig(NeighborListConfig):
    """matscipy's linked-cell search, the one the legacy list uses: the
    ``mace-core[matscipy]`` extra."""

    kind: Literal["matscipy"] = "matscipy"


class VesinConfig(NeighborListConfig):
    """vesin's cell-list search: the ``mace-core[vesin]`` extra."""

    kind: Literal["vesin"] = "vesin"
    #: CPU threads per search; ``0`` lets vesin choose (``OMP_NUM_THREADS``,
    #: else every core). One by default, because graphs are built in several
    #: data-loader workers at once. Changes the speed, never the edges.
    n_threads: int = Field(default=1, ge=0)
