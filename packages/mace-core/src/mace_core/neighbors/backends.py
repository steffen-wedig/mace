"""Neighbour-list backends: the protocol and the registry that builds them.

A backend finds the edges and nothing else. It receives the
:class:`~mace_core.neighbors.geometry.SearchGeometry` the graph builder already
computed and returns ``(edge_index, unit_shifts)``; the effective cell and the
shifts are filled in afterwards by
:func:`~mace_core.neighbors.neighborhood.get_neighborhood`, the same way
whichever backend ran. Swapping backends is therefore a choice about speed and
robustness and never about the numbers: every registered backend has to
reproduce the brute-force edge set.

Which backend, with which settings, is a
:class:`~mace_core.neighbors.config.NeighborListConfig` kind; how it is built
is registered here, on :data:`NEIGHBOR_BACKENDS`, by the module that defines
it. Those modules import their library only when a backend is built, so
importing them all costs nothing. The providers registered here return numpy
arrays and import no framework.

**One default per framework; anything else is asked for by name.** A config
that names no backend gets the default of the framework building the graphs,
which it passes to :func:`resolve_neighbor_backend`; ``brute_force`` when no
framework is involved. A named backend is used as named. Either is built or
refused, never replaced by another. Only ``brute_force`` is guaranteed by
``mace-core``; ``vesin`` and ``matscipy`` are its extras, and each framework
package requires the one it defaults to.

**Resolve once.** A graph builder resolves its backend when it is set up and
hands the object to every
:func:`~mace_core.neighbors.neighborhood.get_neighborhood` call.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from mace_core.neighbors.config import NeighborListConfig
from mace_core.neighbors.geometry import SearchGeometry
from mace_core.plugins import Builders

__all__ = [
    "DEFAULT_NEIGHBOR_BACKEND",
    "NEIGHBOR_BACKENDS",
    "NeighborListBackend",
    "resolve_neighbor_backend",
]

#: The backend used when the config names none and no framework passes its
#: own default. It ships with ``mace-core`` and needs nothing beyond numpy.
DEFAULT_NEIGHBOR_BACKEND = "brute_force"


@runtime_checkable
class NeighborListBackend(Protocol):
    """Finds every directed pair of atoms closer than the cutoff.

    Built from its config section by :data:`NEIGHBOR_BACKENDS`; the module
    defining a backend registers its builder there.
    """

    def search(
        self, positions: np.ndarray, cutoff: float, geometry: SearchGeometry
    ) -> tuple[np.ndarray, np.ndarray]:
        """The edge set of one structure.

        Args:
            positions: ``[n_atoms, 3]`` float64 Cartesian positions, Angstrom.
                Read-only to the backend.
            cutoff: Strict bound: a pair is an edge when its float64 squared
                length is ``< cutoff * cutoff``. Comparing a square-rooted
                length instead drops a pair whose length rounds onto the
                cutoff, such as the face diagonals of a unit cube against
                ``cutoff = np.sqrt(2)``.
            geometry: The periodicity and the search box to bin into. A
                backend that does not bin may ignore ``search_cell``, but
                images exist only along axes where ``geometry.pbc`` is true.

        Returns:
            ``edge_index``, ``[2, n_edges]`` integers, row 0 the sender ``i``
            and row 1 the receiver ``j``; and ``unit_shifts``, ``[n_edges, 3]``
            integers ``n`` such that ``positions[j] - positions[i] +
            n @ cell`` is the edge vector. Every pair appears once per
            direction. The zero-shift self pair may be included or not;
            :func:`~mace_core.neighbors.neighborhood.get_neighborhood` drops
            it either way. Edge order is unspecified.
        """
        ...


#: Builds a numpy neighbour-list backend from its config section.
NEIGHBOR_BACKENDS: Builders[NeighborListConfig, NeighborListBackend, []] = Builders(
    NeighborListConfig
)


def resolve_neighbor_backend(
    config: NeighborListConfig | None = None, default: str = DEFAULT_NEIGHBOR_BACKEND
) -> NeighborListBackend:
    """The backend a graph builder uses: the one the config names, or ``default``.

    Args:
        config: The ``neighbor_list`` section. ``None`` names no backend.
        default: The framework's default kind, used when the config names none.

    Raises:
        mace_core.plugins.NotAvailableError: if the kind has no builder.
        ImportError: if its library is not installed, naming the extra.
            No other backend is tried.
    """
    config = config or NeighborListConfig.model_validate({"kind": default})
    return NEIGHBOR_BACKENDS.build(config)
