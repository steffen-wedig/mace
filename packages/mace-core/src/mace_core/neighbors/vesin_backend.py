"""vesin's cell-list search, mace-torch's default.

vesin bins atoms into cells of the box it is given, so it searches the
:attr:`~mace_core.neighbors.geometry.SearchGeometry.search_cell`, as matscipy
does, and takes the periodicity per axis. It compares squared lengths against
the cutoff and never reports the zero-shift self pair.

Its edges come out in an order of its own, which differs from matscipy's. The
order does not depend on ``n_threads``, so the thread count changes the speed
and never a number. matscipy stays registered because the legacy list searches
with it, and a comparison against legacy that includes the edge order has to
name it.

vesin is the ``mace-core[vesin]`` extra. It is imported when the backend is
built rather than with this module, because importing vesin also imports torch
or cupy whenever either is installed, and ``mace-core`` reaches no framework.
Without vesin, building the backend raises and says how to install it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mace_core.neighbors.backends import NEIGHBOR_BACKENDS
from mace_core.neighbors.config import VesinConfig
from mace_core.neighbors.geometry import SearchGeometry

__all__ = ["VesinBackend"]


class VesinBackend:
    """Neighbour search through :class:`vesin.NeighborList`.

    Args:
        config: The thread count; see :class:`VesinConfig`, which checks it.
    """

    def __init__(self, config: VesinConfig | None = None) -> None:
        config = config or VesinConfig()
        try:
            from vesin import NeighborList
        except ImportError as missing:
            raise ImportError(
                "the vesin neighbour-list backend needs vesin: "
                "pip install 'mace-core[vesin]'"
            ) from missing

        self._neighbor_list: Any = NeighborList
        self.n_threads = config.n_threads

    def search(
        self, positions: np.ndarray, cutoff: float, geometry: SearchGeometry
    ) -> tuple[np.ndarray, np.ndarray]:
        """See :meth:`~mace_core.neighbors.backends.NeighborListBackend.search`.

        Edges come out in vesin's order.
        """
        neighbor_list = self._neighbor_list(
            cutoff=cutoff, full_list=True, n_threads=self.n_threads
        )
        senders, receivers, unit_shifts = neighbor_list.compute(
            points=positions,
            box=geometry.search_cell,
            periodic=list(geometry.pbc),
            quantities="ijS",
        )
        return np.stack((senders, receivers)), unit_shifts


NEIGHBOR_BACKENDS.register(VesinConfig, VesinBackend)
