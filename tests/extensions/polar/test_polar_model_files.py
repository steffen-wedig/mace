"""Model files written before PolarMACE moved into ``mace.modules.polar`` still load.

Earlier code pickled PolarMACE as ``mace.modules.extensions.PolarMACE``, its field blocks
under ``mace.modules.field_blocks`` and the electrostatics under ``graph_longrange``.
``load_polar_model`` remaps those class paths; parameters and outputs must not change.
"""

from __future__ import annotations

import sys
import types

import torch

from mace.modules.polar import PolarMACE
from mace.modules.polar.conversion import MOVED_MODULES, load_polar_model, main
from tests.extensions.polar.test_polar_shared_geometry import (
    _batch,
    _clone,
    _molecules,
    _polar_model,
)

OLD_MODULE_OF = {new: old for old, new in MOVED_MODULES.items()}


def _old_module(module: str) -> str | None:
    for new, old in OLD_MODULE_OF.items():
        if module == new or module.startswith(new + "."):
            return old + module[len(new) :]
    return None


class MonopoleDipoleCorrectionBlock(torch.nn.Module):
    """Stand-in for the energy block's molecule-in-box correction of earlier code."""


class GTOInternalFieldtoFeaturesBlock(torch.nn.Module):
    """Stand-in for the corrections' displaced-source block (a removed class of a module
    that still exists)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("factors", torch.ones(3))


class NonPeriodicFeatureCorrections(torch.nn.Module):
    """Stand-in for the feature block's slab / molecule-in-box corrections, with its
    displaced-source child as in the real pickles."""

    def __init__(self):
        super().__init__()
        self.displaced_interactions = GTOInternalFieldtoFeaturesBlock()


REMOVED = [
    (MonopoleDipoleCorrectionBlock, "graph_longrange.slabs"),
    (NonPeriodicFeatureCorrections, "graph_longrange.features"),
    (GTOInternalFieldtoFeaturesBlock, "graph_longrange.gto_utils"),
]


def _save_under_old_paths(model: torch.nn.Module, path) -> None:
    """Pickle the model as the code before the move did: every class of the polar
    package under its old module path (aliases exist only while pickling), with the
    correction submodules of the removed slab and molecule-in-box modes."""
    model.coulomb_energy.monopole_dipole_correction = MonopoleDipoleCorrectionBlock()
    model.electric_potential_descriptor.non_periodic_correction_terms = (
        NonPeriodicFeatureCorrections()
    )
    moved_classes = [(cls, cls.__module__, old) for cls, old in REMOVED]
    for name, module in list(sys.modules.items()):
        old = _old_module(name)
        if old is None or module is None:
            continue
        for value in vars(module).values():
            if isinstance(value, type) and value.__module__ == name:
                moved_classes.append((value, name, old))
    # Old module paths that no longer exist (graph_longrange.*, mace.modules.field_blocks)
    # get stand-in modules, their parent packages included (pickle imports the module);
    # a path that still exists (mace.modules.extensions) gets the class set on it.
    aliases: dict[str, types.ModuleType] = {}
    added_attributes: list[tuple[types.ModuleType, str]] = []
    try:
        for cls, _, old in moved_classes:
            module = sys.modules.get(old) or aliases.get(old)
            if module is None:
                parts = old.split(".")
                for depth in range(1, len(parts) + 1):
                    package = ".".join(parts[:depth])
                    if package not in sys.modules and package not in aliases:
                        aliases[package] = types.ModuleType(package)
                module = aliases[old]
            if not hasattr(module, cls.__qualname__):
                setattr(module, cls.__qualname__, cls)
                added_attributes.append((module, cls.__qualname__))
            cls.__module__ = old
        sys.modules.update(aliases)
        torch.save(model, path)
    finally:
        for cls, name, _ in moved_classes:
            cls.__module__ = name
        for module, attribute in added_attributes:
            delattr(module, attribute)
        for package in aliases:
            sys.modules.pop(package, None)
        del model.coulomb_energy.monopole_dipole_correction
        del model.electric_potential_descriptor.non_periodic_correction_terms


def _energy_and_forces(model) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(_clone(_batch(_molecules())), training=False, compute_force=True)
    return output["energy"].detach(), output["forces"].detach()


def test_model_pickled_under_the_old_class_paths_loads(tmp_path):
    model = _polar_model()
    old_file = tmp_path / "old.model"
    _save_under_old_paths(model, old_file)
    pickled = old_file.read_bytes()
    assert b"graph_longrange" in pickled and b"mace.modules.extensions" in pickled

    loaded = load_polar_model(old_file)
    assert type(loaded) is PolarMACE
    # The removed corrections are dropped on load.
    assert "monopole_dipole_correction" not in loaded.coulomb_energy._modules
    assert (
        "non_periodic_correction_terms"
        not in loaded.electric_potential_descriptor._modules
    )
    assert loaded.state_dict().keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key
    for expected, actual in zip(_energy_and_forces(model), _energy_and_forces(loaded)):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    # The command line writes the model again under the current paths.
    new_file = tmp_path / "new.model"
    main([str(old_file), str(new_file)])
    assert b"graph_longrange" not in new_file.read_bytes()
    reloaded = torch.load(new_file, weights_only=False)
    torch.testing.assert_close(
        _energy_and_forces(reloaded)[0], _energy_and_forces(model)[0], rtol=0, atol=0
    )
