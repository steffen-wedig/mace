"""The neighbour list: the pinned edge set and the three-regime effective cell.

Every registered backend is held to the brute-force backend's edge set on the
characterization cases below (triclinic, a cell smaller than the cutoff, an
aperiodic cluster, mixed periodicity with and without a zero vacuum row, and a
structure with no edges). The brute-force backend itself is checked against
counts known from crystallography and against an independent oracle on the
committed fixtures in ``tests/parity/test_graph_parity.py``.
"""

from __future__ import annotations

import sys
import textwrap
from typing import Any, Literal

import numpy as np
import pytest
from mace_core import plugins
from mace_core.config import BaseConfig, KindedSection
from mace_core.neighbors import (
    DEFAULT_NEIGHBOR_BACKEND,
    NEIGHBOR_BACKENDS,
    BruteForceBackend,
    MatscipyConfig,
    NeighborListBackend,
    NeighborListConfig,
    VesinConfig,
    get_neighborhood,
    resolve_neighbor_backend,
    search_geometry,
)
from mace_core.neighbors import geometry as geometry_module
from mace_core.neighbors.matscipy_backend import MatscipyBackend
from mace_core.neighbors.vesin_backend import VesinBackend
from mace_core.plugins import DuplicateRegistrationError, NotAvailableError
from pydantic import ValidationError

MATSCIPY = MatscipyBackend()


def _backend(kind, **settings):
    """The backend a config section of ``kind`` describes."""
    return NEIGHBOR_BACKENDS.build(NeighborListConfig.of(kind, **settings))


def _edges(neighbors):
    """The edge set as sorted ``(sender, receiver, n0, n1, n2)`` integers."""
    rows = np.concatenate((neighbors.edge_index.T, neighbors.unit_shifts), axis=1)
    return sorted(map(tuple, rows.tolist()))


def _physical(cell):
    """The case's cell, for the cases that have one."""
    assert cell is not None
    return cell


def _cluster(n_atoms=8, spread=1.8, seed=20260810):
    return np.random.default_rng(seed).uniform(0.0, spread, size=(n_atoms, 3))


TRICLINIC = np.array([[3.1, 0.2, 0.1], [0.9, 2.8, -0.2], [0.4, -0.3, 3.3]])
HEXAGONAL_SLAB = np.array([[3.0, 0.0, 0.0], [1.5, 2.598, 0.0], [0.0, 0.0, 0.0]])

CASES = {
    "triclinic": (_cluster(5, 2.5, 1), 3.2, (True, True, True), TRICLINIC),
    "cell_smaller_than_cutoff": (
        np.zeros((1, 3)),
        3.0,
        (True, True, True),
        np.eye(3) * 1.5,
    ),
    "aperiodic_cluster": (_cluster(), 2.0, (False, False, False), None),
    "mixed_pbc_with_vacuum": (
        _cluster(6, 2.0, 2),
        3.5,
        (True, True, False),
        np.array([[3.0, 0.0, 0.0], [1.5, 2.598, 0.0], [0.0, 0.0, 12.0]]),
    ),
    "mixed_pbc_zero_vacuum_row": (
        _cluster(6, 2.0, 3),
        3.5,
        (True, True, False),
        HEXAGONAL_SLAB,
    ),
    "mixed_pbc_one_axis": (
        _cluster(4, 2.0, 4),
        3.0,
        (True, False, False),
        np.eye(3) * 2.7,
    ),
    "zero_edges": (np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), 2.0, None, None),
}


@pytest.fixture(params=sorted(CASES), name="case")
def fixture_case(request):
    return CASES[request.param]


# --- the brute-force backend is falsifiable ---------------------------------


@pytest.mark.parametrize(
    ("cell", "neighbours"),
    [
        # Simple cubic, a = 1, cutoff between the first and second shell: 6.
        (np.eye(3), 6),
        # Face-centred cubic primitive cell, nearest-neighbour distance 1: 12.
        (
            np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * np.sqrt(2),
            12,
        ),
    ],
)
def test_brute_force_reproduces_known_coordination_numbers(cell, neighbours):
    neighbors = get_neighborhood(
        np.zeros((1, 3)), 1.2, (True, True, True), cell, backend=BruteForceBackend()
    )
    assert neighbors.edge_index.shape[1] == neighbours


def test_brute_force_reaches_images_of_a_strongly_skewed_cell():
    """The image bound comes from the reciprocal vectors, so a cell whose
    second vector nearly lies along the first still finds every image."""
    cell = np.array([[1.0, 0.0, 0.0], [3.9, 0.3, 0.0], [0.0, 0.0, 1.0]])
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.1, 0.5]])
    brute = get_neighborhood(
        positions, 2.5, (True, True, True), cell, backend=BruteForceBackend()
    )
    matscipy = get_neighborhood(
        positions, 2.5, (True, True, True), cell, backend=MATSCIPY
    )
    assert _edges(brute) == _edges(matscipy)
    assert brute.edge_index.shape[1] > 0


# --- the pinned edge set ------------------------------------------------------


def test_every_registered_backend_reproduces_the_brute_force_edge_set(case):
    positions, cutoff, pbc, cell = case
    expected = _edges(
        get_neighborhood(positions, cutoff, pbc, cell, backend=BruteForceBackend())
    )
    for kind in NeighborListConfig.kinds():
        backend = _backend(kind)
        neighbors = get_neighborhood(positions, cutoff, pbc, cell, backend=backend)
        assert _edges(neighbors) == expected, kind


def test_the_returned_cell_does_not_depend_on_the_backend(case):
    positions, cutoff, pbc, cell = case
    cells = [
        get_neighborhood(positions, cutoff, pbc, cell, backend=_backend(kind)).cell
        for kind in NeighborListConfig.kinds()
    ]
    for returned in cells[1:]:
        assert np.array_equal(returned, cells[0])


def test_the_list_is_full_and_directed(case):
    positions, cutoff, pbc, cell = case
    edges = _edges(get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY))
    reversed_edges = sorted((j, i, -a, -b, -c) for i, j, a, b, c in edges)
    assert edges == reversed_edges


def test_every_edge_is_strictly_shorter_than_the_cutoff(case):
    positions, cutoff, pbc, cell = case
    neighbors = get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY)
    sender, receiver = neighbors.edge_index
    vectors = positions[receiver] - positions[sender] + neighbors.shifts
    assert np.all(np.sum(vectors * vectors, axis=1) < cutoff * cutoff)


def test_a_length_that_rounds_onto_the_cutoff_is_inside_for_every_backend():
    """The face diagonals of a unit cube are shorter than ``np.sqrt(2)``, whose
    float64 value is above the exact root, but their norm rounds onto it."""
    positions, cell, cutoff = np.zeros((1, 3)), np.eye(3), np.sqrt(2)
    diagonal = np.array([1.0, 1.0, 0.0])
    assert not np.linalg.norm(diagonal) < cutoff
    for kind in NeighborListConfig.kinds():
        neighbors = get_neighborhood(
            positions, cutoff, (True, True, True), cell, backend=_backend(kind)
        )
        assert neighbors.edge_index.shape[1] == 6 + 12, kind


def test_shifts_are_unit_shifts_times_the_returned_cell(case):
    positions, cutoff, pbc, cell = case
    neighbors = get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY)
    assert np.array_equal(
        neighbors.shifts, np.dot(neighbors.unit_shifts, neighbors.cell)
    )
    assert neighbors.edge_index.dtype == np.int64
    assert neighbors.unit_shifts.dtype == np.int64


def test_periodic_self_images_survive_and_the_zero_shift_self_pair_does_not():
    positions, cutoff, pbc, cell = CASES["cell_smaller_than_cutoff"]
    neighbors = get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY)
    assert neighbors.edge_index.shape[1] > 0
    assert np.all(neighbors.edge_index == 0)
    assert not np.any(np.all(neighbors.unit_shifts == 0, axis=1))


def test_a_structure_with_no_edges_is_legal():
    positions, cutoff, pbc, cell = CASES["zero_edges"]
    neighbors = get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY)
    assert neighbors.edge_index.shape == (2, 0)
    assert neighbors.unit_shifts.shape == (0, 3)
    assert neighbors.shifts.shape == (0, 3)


def test_a_single_aperiodic_atom_has_no_edges():
    neighbors = get_neighborhood(np.zeros((1, 3)), 5.0, backend=MATSCIPY)
    assert neighbors.edge_index.shape == (2, 0)


def test_nothing_passed_in_is_modified(case):
    positions, cutoff, pbc, cell = case
    positions_before = positions.copy()
    cell_before = None if cell is None else cell.copy()
    get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY)
    assert np.array_equal(positions, positions_before)
    if cell is not None:
        assert np.array_equal(cell, _physical(cell_before))


def test_the_returned_arrays_share_no_memory_with_the_inputs_and_are_read_only():
    positions, cutoff, pbc, cell = CASES["mixed_pbc_with_vacuum"]
    cell = _physical(cell)
    neighbors = get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY)
    assert not np.shares_memory(neighbors.cell, cell)
    for array in (
        neighbors.edge_index,
        neighbors.unit_shifts,
        neighbors.shifts,
        neighbors.cell,
    ):
        assert not array.flags.writeable


# --- the three effective-cell regimes -----------------------------------------


def test_a_fully_aperiodic_cluster_returns_the_extended_search_cell():
    positions = _cluster()
    neighbors = get_neighborhood(positions, 2.0, backend=MATSCIPY)
    extent = positions.max(axis=0) - positions.min(axis=0)
    assert np.array_equal(neighbors.cell, np.diag(extent + 2 * 2.0 + 1))
    assert np.linalg.det(neighbors.cell) > 0


def test_the_extended_cell_depends_on_the_extent_not_the_origin():
    positions = _cluster()
    near = get_neighborhood(positions, 2.0, backend=MATSCIPY).cell
    far = get_neighborhood(positions + 100.0, 2.0, backend=MATSCIPY).cell
    np.testing.assert_allclose(near, far, rtol=0, atol=1e-12)


def test_a_periodic_structure_returns_its_physical_cell():
    positions, cutoff, pbc, cell = CASES["triclinic"]
    cell = _physical(cell)
    assert np.array_equal(
        get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY).cell, cell
    )


def test_a_slab_with_vacuum_returns_its_physical_cell():
    positions, cutoff, pbc, cell = CASES["mixed_pbc_with_vacuum"]
    cell = _physical(cell)
    assert np.array_equal(
        get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY).cell, cell
    )


def test_a_zero_vacuum_row_is_patched_from_the_search_cell():
    positions, cutoff, pbc, cell = CASES["mixed_pbc_zero_vacuum_row"]
    cell = _physical(cell)
    returned = get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY).cell
    extent = positions[:, 2].max() - positions[:, 2].min()
    assert np.array_equal(returned[:2], cell[:2])
    assert np.array_equal(returned[2], [0.0, 0.0, extent + 2 * cutoff + 1])
    assert np.linalg.det(returned) != 0


def test_a_slab_stress_from_the_returned_cell_uses_the_physical_volume():
    """stress = -virials / volume: the vacuum box the search used must not
    enter the volume."""
    positions, cutoff, pbc, cell = CASES["mixed_pbc_with_vacuum"]
    cell = _physical(cell)
    virials = np.diag([0.3, -0.1, 0.2])
    geometry = search_geometry(positions, cutoff, pbc, cell)
    returned = get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY).cell
    stress = -virials / np.linalg.det(returned)
    assert np.array_equal(stress, -virials / np.linalg.det(cell))
    assert np.linalg.det(geometry.search_cell) != np.linalg.det(cell)


def test_the_three_regimes_are_three_different_matrices():
    positions = _cluster(6, 2.0, 3)
    physical = HEXAGONAL_SLAB.copy()
    physical[2] = [0.0, 0.0, 12.0]
    aperiodic = get_neighborhood(positions, 3.5, None, physical, backend=MATSCIPY).cell
    periodic = get_neighborhood(
        positions, 3.5, (True, True, False), physical, backend=MATSCIPY
    ).cell
    patched = get_neighborhood(
        positions, 3.5, (True, True, False), HEXAGONAL_SLAB, backend=MATSCIPY
    ).cell
    assert not np.array_equal(aperiodic, periodic)
    assert not np.array_equal(periodic, patched)
    assert not np.array_equal(aperiodic, patched)


@pytest.mark.parametrize("cell", [None, np.zeros((3, 3))])
def test_a_missing_or_all_zero_cell_is_the_identity_before_extension(cell):
    geometry = search_geometry(np.zeros((1, 3)), 2.0, (True, False, False), cell)
    assert np.array_equal(geometry.cell[0], [1.0, 0.0, 0.0])
    assert np.array_equal(geometry.search_cell[1], [0.0, 2 * 2.0 + 1, 0.0])


def test_pbc_none_is_fully_aperiodic():
    assert search_geometry(np.zeros((2, 3)), 1.0).pbc == (False, False, False)


def test_a_periodic_axis_without_a_lattice_vector_is_refused():
    with pytest.raises(ValueError, match="axis 2 is periodic"):
        search_geometry(np.zeros((1, 3)), 1.0, (True, True, True), HEXAGONAL_SLAB)


@pytest.mark.parametrize("cutoff", [0.0, -1.0, float("inf"), float("nan"), True])
def test_a_cutoff_that_is_not_a_positive_distance_is_refused(cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        search_geometry(np.zeros((1, 3)), cutoff)


def test_only_search_geometry_decides_a_regime():
    """The graph builder and the brute-force backend read the box from it; no
    other function in the package computes one."""
    import inspect

    from mace_core.neighbors import (
        brute_force,
        matscipy_backend,
        neighborhood,
        vesin_backend,
    )

    assert "search_geometry(" in inspect.getsource(neighborhood.get_neighborhood)
    for module in (brute_force, matscipy_backend, neighborhood, vesin_backend):
        source = inspect.getsource(module)
        assert "2 * cutoff" not in source, module.__name__
        assert "search_cell" not in source or module is not neighborhood
    assert "2 * cutoff + 1" in inspect.getsource(geometry_module)


# --- the config section and the registry --------------------------------------


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Kinds and builders a test installs are forgotten after it."""
    kinds = dict(NeighborListConfig._kinds)
    builders = dict(NEIGHBOR_BACKENDS._builders)
    plugins.clear_caches()
    yield
    NeighborListConfig._kinds.clear()
    NeighborListConfig._kinds.update(kinds)
    NEIGHBOR_BACKENDS._builders.clear()
    NEIGHBOR_BACKENDS._builders.update(builders)
    plugins.clear_caches()


class _Run(BaseConfig):
    neighbor_list: NeighborListConfig | None = None


def test_vesin_matscipy_and_brute_force_are_kinds_with_builders():
    assert {"vesin", "matscipy", "brute_force"} <= set(NeighborListConfig.kinds())
    listed = {entry.kind: entry for entry in NEIGHBOR_BACKENDS.available()}
    assert all(listed[kind].loaded for kind in ("vesin", "matscipy", "brute_force"))


def test_the_kind_picks_the_class_and_the_class_checks_the_rest():
    run = _Run.from_dict({"neighbor_list": {"kind": "vesin", "n_threads": 4}})
    assert isinstance(run.neighbor_list, VesinConfig)
    assert run.neighbor_list.n_threads == 4
    with pytest.raises(ValidationError, match=r"neighbor_list\.n_thread\b"):
        _Run.from_dict({"neighbor_list": {"kind": "vesin", "n_thread": 4}})
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        _Run.from_dict({"neighbor_list": {"kind": "vesin", "n_threads": -1}})
    with pytest.raises(ValidationError, match="Extra inputs"):
        _Run.from_dict({"neighbor_list": {"kind": "matscipy", "n_threads": 2}})


def test_an_unknown_or_missing_kind_is_refused_with_the_kinds():
    for section in ({"kind": "cell_list_on_the_moon"}, {"n_threads": 2}):
        with pytest.raises(ValidationError, match=r"brute_force.*matscipy.*vesin"):
            _Run.from_dict({"neighbor_list": section})


def test_the_resolved_config_keeps_the_fields_of_the_kind():
    """The field is annotated with the family, whose dump alone would drop
    ``n_threads``; the saved config has to load back to the same one."""
    run = _Run.from_dict({"neighbor_list": {"kind": "vesin", "n_threads": 3}})
    resolved = run.to_resolved_dict()
    assert resolved == {"neighbor_list": {"kind": "vesin", "n_threads": 3}}
    assert _Run.from_dict(resolved) == run
    assert run.to_user_dict() == resolved
    assert '"n_threads":3' in run.model_dump_json()


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_the_json_schema_lists_every_kind_with_its_fields(mode):
    family = _Run.model_json_schema(mode=mode)["$defs"]["NeighborListConfig"]
    by_title = {kind["title"]: kind for kind in family["oneOf"]}
    assert set(by_title) >= {"BruteForceConfig", "MatscipyConfig", "VesinConfig"}
    assert set(by_title["VesinConfig"]["properties"]) == {"kind", "n_threads"}
    assert by_title["VesinConfig"]["additionalProperties"] is False


def test_no_section_leaves_the_choice_to_the_framework():
    assert _Run.from_dict({}).neighbor_list is None
    assert DEFAULT_NEIGHBOR_BACKEND == "brute_force"
    assert isinstance(resolve_neighbor_backend(), BruteForceBackend)
    assert isinstance(resolve_neighbor_backend(None, "vesin"), VesinBackend)
    assert isinstance(resolve_neighbor_backend(None, "matscipy"), MatscipyBackend)


def test_a_named_backend_wins_over_the_default():
    named: Any = resolve_neighbor_backend(VesinConfig(n_threads=3), "matscipy")
    assert named.n_threads == 3
    assert isinstance(
        resolve_neighbor_backend(MatscipyConfig(), "vesin"), MatscipyBackend
    )


def test_vesin_takes_its_thread_count_from_the_config():
    """The thread count changes the speed and never the edges or their order."""
    positions, cutoff, pbc, cell = CASES["triclinic"]
    threaded: Any = _backend("vesin", n_threads=4)
    single: Any = _backend("vesin")
    assert (threaded.n_threads, single.n_threads) == (4, 1)
    many = get_neighborhood(positions, cutoff, pbc, cell, backend=threaded)
    one = get_neighborhood(positions, cutoff, pbc, cell, backend=single)
    assert np.array_equal(many.edge_index, one.edge_index)
    assert np.array_equal(many.unit_shifts, one.unit_shifts)


def test_validating_a_config_imports_no_backend():
    """Checking a config that names vesin imports neither vesin nor torch, so
    a parent process stays clean before its data-loader workers fork."""
    import subprocess

    probe = (
        "import sys\n"
        "from mace_core.neighbors import NeighborListConfig\n"
        "NeighborListConfig.of('vesin', n_threads=2)\n"
        "NeighborListConfig.of('matscipy')\n"
        "print(*(m in sys.modules for m in ('vesin', 'torch', 'matscipy',"
        " 'mace_core.neighbors.vesin_backend')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.split() == ["False"] * 4


def test_importing_the_vesin_backend_module_does_not_import_vesin():
    import subprocess

    probe = (
        "import sys; import mace_core.neighbors.vesin_backend; "
        "print('vesin' in sys.modules, 'torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.split() == ["False", "False"]


def _build_in_worker(config):
    backend = NEIGHBOR_BACKENDS.build(config)
    neighbors = get_neighborhood(*CASES["triclinic"], backend=backend)
    return type(backend).__name__, _edges(neighbors)


def test_a_config_built_in_a_spawned_worker_gives_the_same_edges():
    """A worker receives the config, not the backend, and builds it itself:
    the builder registers in that process the first time it is needed."""
    import multiprocessing

    expected = _edges(get_neighborhood(*CASES["triclinic"], backend=_backend("vesin")))
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        name, edges = pool.apply(_build_in_worker, (VesinConfig(n_threads=1),))
    assert (name, edges) == ("VesinBackend", expected)


def test_a_built_backend_pickles():
    import pickle

    for kind in NeighborListConfig.kinds():
        backend = pickle.loads(pickle.dumps(_backend(kind)))
        neighbors = get_neighborhood(*CASES["triclinic"], backend=backend)
        assert _edges(neighbors) == _edges(
            get_neighborhood(*CASES["triclinic"], backend=BruteForceBackend())
        ), kind


def test_a_backend_returning_malformed_edges_is_refused():
    class Broken:
        def search(self, positions, cutoff, geometry):
            return np.zeros((3, 1), np.int64), np.zeros((1, 3), np.int64)

    with pytest.raises(ValueError, match="edge_index of shape"):
        get_neighborhood(_cluster(), 2.0, backend=Broken())


# --- a backend from another distribution --------------------------------------


def _install(path, distribution, **groups):
    """A distribution whose metadata declares entry points, one group per
    keyword (``neighbor_list`` and ``neighbor_backends``)."""
    info = path / f"{distribution}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 1.0\n"
    )
    sections = "".join(f"[mace.{group}]\n{line}\n" for group, line in groups.items())
    (info / "entry_points.txt").write_text(sections)


def _module(path, monkeypatch, name, source):
    (path / f"{name}.py").write_text(textwrap.dedent(source))
    monkeypatch.syspath_prepend(str(path))
    monkeypatch.delitem(sys.modules, name, raising=False)


SKIN_CONFIG = """
    from typing import Literal
    from pydantic import Field
    from mace_core.neighbors import NeighborListConfig

    class SkinConfig(NeighborListConfig):
        kind: Literal["skin"] = "skin"
        skin: float = Field(0.0, ge=0.0)
"""

SKIN_BACKEND = """
    import numpy as np
    from mace_core.neighbors import NEIGHBOR_BACKENDS
    from skin_config import SkinConfig

    class SkinBackend:
        def __init__(self, config: SkinConfig) -> None:
            self.skin = config.skin

        def search(self, positions, cutoff, geometry):
            return np.zeros((2, 0), np.int64), np.zeros((0, 3), np.int64)

    NEIGHBOR_BACKENDS.register(SkinConfig, SkinBackend)
"""


def _install_skin(tmp_path, monkeypatch, backend_source=SKIN_BACKEND):
    _module(tmp_path, monkeypatch, "skin_config", SKIN_CONFIG)
    _module(tmp_path, monkeypatch, "skin_backend", backend_source)
    _install(
        tmp_path,
        "skin_neighbors",
        neighbor_list="skin = skin_config:SkinConfig",
        neighbor_backends="skin = skin_backend",
    )


def test_a_kind_from_another_distribution_is_validated_and_built(tmp_path, monkeypatch):
    """Nothing in mace_core names it. Validating imports only its config
    module; building imports its backend module."""
    _install_skin(tmp_path, monkeypatch)
    assert "skin" in NeighborListConfig.kinds()
    run = _Run.from_dict({"neighbor_list": {"kind": "skin", "skin": 0.5}})
    assert "skin_config" in sys.modules
    assert "skin_backend" not in sys.modules
    with pytest.raises(ValidationError, match=r"neighbor_list\.skn"):
        _Run.from_dict({"neighbor_list": {"kind": "skin", "skn": 0.5}})
    assert run.neighbor_list is not None
    backend: Any = NEIGHBOR_BACKENDS.build(run.neighbor_list)
    assert backend.skin == 0.5
    assert _Run.from_dict(run.to_resolved_dict()) == run


def test_a_backend_that_fails_to_import_is_reported_and_not_replaced(
    tmp_path, monkeypatch
):
    """The reason is kept, so the second build does not import it again."""
    _install_skin(
        tmp_path,
        monkeypatch,
        'raise RuntimeError("importing this backend sets up a device")',
    )
    config = NeighborListConfig.of("skin")
    for _ in range(2):
        with pytest.raises(NotAvailableError, match="sets up a device"):
            NEIGHBOR_BACKENDS.build(config)
    listed = {entry.kind: entry for entry in NEIGHBOR_BACKENDS.available()}
    assert not listed["skin"].loaded
    assert "sets up a device" in listed["skin"].reason
    assert listed["brute_force"].loaded


def test_building_one_kind_imports_no_other_backend(tmp_path, monkeypatch):
    _install_skin(
        tmp_path,
        monkeypatch,
        'raise RuntimeError("importing this backend sets up a device")',
    )
    assert isinstance(_backend("brute_force"), BruteForceBackend)
    assert "skin_backend" not in sys.modules


def test_a_kind_without_a_builder_is_refused_when_built(tmp_path, monkeypatch):
    _module(tmp_path, monkeypatch, "skin_config", SKIN_CONFIG)
    _install(tmp_path, "skin_neighbors", neighbor_list="skin = skin_config:SkinConfig")
    config = NeighborListConfig.of("skin")
    with pytest.raises(NotAvailableError, match="no builder is registered for 'skin'"):
        NEIGHBOR_BACKENDS.build(config)


def test_a_builder_module_that_registers_nothing_is_refused(tmp_path, monkeypatch):
    _install_skin(tmp_path, monkeypatch, "x = 1")
    with pytest.raises(NotAvailableError, match="registered no builder"):
        NEIGHBOR_BACKENDS.build(NeighborListConfig.of("skin"))


def test_a_builder_whose_product_is_no_backend_is_refused(tmp_path, monkeypatch):
    source = SKIN_BACKEND.replace("def search(", "def find(")
    _install_skin(tmp_path, monkeypatch, source)
    with pytest.raises(NotAvailableError, match="not a NeighborListBackend"):
        NEIGHBOR_BACKENDS.build(NeighborListConfig.of("skin"))


def test_two_distributions_registering_one_kind_differently_raise(
    tmp_path, monkeypatch
):
    _install(tmp_path, "first", neighbor_backends="duplicate = first")
    _install(tmp_path, "second", neighbor_backends="duplicate = second")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(DuplicateRegistrationError, match="duplicate"):
        NEIGHBOR_BACKENDS.available()


def test_two_identical_registrations_are_one(tmp_path, monkeypatch):
    target = "brute_force = mace_core.neighbors.brute_force"
    _install(tmp_path, "first", neighbor_backends=target)
    _install(tmp_path, "second", neighbor_backends=target)
    monkeypatch.syspath_prepend(str(tmp_path))
    kinds = [entry.kind for entry in NEIGHBOR_BACKENDS.available()]
    assert kinds.count("brute_force") == 1


def test_a_kind_defined_twice_is_refused_when_the_class_is_defined():
    with pytest.raises(DuplicateRegistrationError, match="'vesin' is defined twice"):

        class AnotherVesin(NeighborListConfig):
            kind: Literal["vesin"] = "vesin"


@pytest.mark.parametrize("annotation", ["str", 'Literal["a", "b"]', "Literal[1]"])
def test_a_kind_that_is_not_one_literal_string_is_refused(annotation):
    namespace = {"Literal": Literal, "NeighborListConfig": NeighborListConfig}
    with pytest.raises(TypeError, match="one-string Literal"):
        exec(
            f"class Odd(NeighborListConfig):\n    kind: {annotation} = 'a'\n",
            namespace,
        )


def test_a_builder_for_another_family_is_refused():
    class Other(KindedSection):
        pass

    class OtherKind(Other):
        kind: Literal["other"] = "other"

    with pytest.raises(TypeError, match="not a NeighborListConfig"):
        NEIGHBOR_BACKENDS.register(OtherKind, lambda config: BruteForceBackend())  # type: ignore[arg-type]


def test_registering_a_second_builder_for_a_kind_is_refused():
    NEIGHBOR_BACKENDS.register(VesinConfig, VesinBackend)
    with pytest.raises(DuplicateRegistrationError, match="two builders"):
        NEIGHBOR_BACKENDS.register(VesinConfig, lambda config: VesinBackend(config))


def test_the_protocol_is_the_product():
    assert NEIGHBOR_BACKENDS.product is NeighborListBackend


@pytest.mark.parametrize(
    ("module", "library"),
    [("vesin_backend", "vesin"), ("matscipy_backend", "matscipy")],
)
def test_a_missing_library_names_the_extra_to_install(monkeypatch, module, library):
    import importlib
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        return None if name == library else real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setitem(sys.modules, f"{library}.neighbours", None)
    monkeypatch.delitem(sys.modules, f"mace_core.neighbors.{module}", raising=False)
    with pytest.raises(ImportError, match=rf"mace-core\[{library}\]"):
        importlib.import_module(f"mace_core.neighbors.{module}")
