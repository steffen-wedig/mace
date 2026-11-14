"""Loading PolarMACE whole-model files written by earlier code, and their repair.

- ``load_polar_model``: unpickles a model saved before PolarMACE, its field blocks and
  the electrostatics (formerly the ``graph_longrange`` package) moved into
  ``mace.modules.polar`` (class paths remapped by ``MOVED_MODULES``). Parameters and
  buffers are kept as they are.
- ``convert_polar_model``: repairs derived state of a restored model in place (integer
  sizes, field linears in the cueq layout, the plans of evaluation); PolarMACE runs it on
  every unpickling.

Command line: ``python -m mace.modules.polar.conversion old.model new.model`` writes the
model again under the new class paths. Load trusted files only (``weights_only=False``).
"""

import argparse
import pickle
import types
from typing import Optional, Sequence

import torch

# Module paths of pickles written before the polar package, and where they live now.
MOVED_MODULES = {
    "mace.modules.extensions": "mace.modules.polar.model",
    "mace.modules.field_blocks": "mace.modules.polar.field_blocks",
    "mace.modules.linear_folds": "mace.modules.polar.linear_folds",
    "mace.modules.boundary_split": "mace.modules.polar.boundary_split",
    "mace.tools.polar_conversion": "mace.modules.polar.conversion",
    "graph_longrange": "mace.modules.polar.electrostatics",
}
# Classes that stayed in mace.modules.extensions (the LES and magnetic models).
_STAYED_IN_EXTENSIONS = {
    "MACELES",
    "MagneticMACE",
    "MagneticScaleShiftMACE",
    "MagneticSCFMACE",
    "TimeReversalSymmetrizedMACE",
}


def moved_module(module: str, name: str) -> str:
    """The current module of class ``name`` pickled under ``module``."""
    if module == "mace.modules.extensions" and name in _STAYED_IN_EXTENSIONS:
        return module
    for old, new in MOVED_MODULES.items():
        if module == old or module.startswith(old + "."):
            return new + module[len(old) :]
    return module


# Modules of the electrostatics that were removed (slabs, molecule-in-box, jellium and
# external point sources): their objects in old pickles load as placeholders.
REMOVED_MODULES = {
    "mace.modules.polar.electrostatics.slabs",
    "mace.modules.polar.electrostatics.jellium_energy",
    "mace.modules.polar.electrostatics.jellium_features",
    "mace.modules.polar.electrostatics.external_source_energy",
    "mace.modules.polar.electrostatics.external_source_features",
    "mace.modules.polar.electrostatics.external_source_realspace",
    "mace.modules.polar.electrostatics.realspace_grid_integrals",
}
REMOVED_CLASSES = {
    ("mace.modules.polar.electrostatics.features", "NonPeriodicFeatureCorrections"),
    # The corrections' displaced-source block (a child of NonPeriodicFeatureCorrections).
    ("mace.modules.polar.electrostatics.gto_utils", "GTOInternalFieldtoFeaturesBlock"),
}


class RemovedModule(torch.nn.Module):
    """Placeholder for a module of a removed class in an old pickle (dropped by
    ``convert_polar_model``)."""

    def forward(self, *args, **kwargs):
        raise RuntimeError("a module of a removed class was called")


class _RemappingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module = moved_module(module, name)
        if module in REMOVED_MODULES or (module, name) in REMOVED_CLASSES:
            return RemovedModule
        return super().find_class(module, name)


# torch.load takes a module with an ``Unpickler`` (and ``load``) as its pickle module.
remapping_pickle_module = types.SimpleNamespace(
    Unpickler=_RemappingUnpickler,
    load=lambda file, **kwargs: _RemappingUnpickler(file, **kwargs).load(),
    __name__="remapping_pickle_module",
)


def load_model_file(path, map_location=None) -> torch.nn.Module:
    """A whole-model file of any MACE model. PolarMACE files from before the move into
    ``mace.modules.polar`` (class paths under ``mace.modules.extensions``,
    ``graph_longrange``, ...) load too: their class paths are remapped and the submodules
    of removed classes dropped; every other pickle loads as with ``torch.load``. The
    calculators load files through this, so models trained with earlier code keep working
    without a conversion step."""
    return torch.load(
        path,
        map_location=map_location,
        weights_only=False,
        pickle_module=remapping_pickle_module,
    )


def load_polar_model(path, map_location: Optional[str] = None) -> torch.nn.Module:
    """A PolarMACE whole-model file, from before or after the move into the package."""
    return load_model_file(path, map_location=map_location)


def main(arguments: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite a PolarMACE model file under the current class paths."
    )
    parser.add_argument("input", help="model file written by earlier code")
    parser.add_argument("output", help="model file to write")
    parsed = parser.parse_args(arguments)
    model = load_polar_model(parsed.input, map_location="cpu")
    torch.save(model, parsed.output)


# realspace: open structures; pbc: fully periodic cells; auto: one of the two per batch
# (PolarMACE splits a batch mixing open and periodic structures, boundary_split).
PBC_HANDLING_MODES = ("auto", "realspace", "pbc")
# Modes of earlier code, removed with the slab and molecule-in-box corrections. A model
# saved in one of them loads in 'auto' (the same for open and fully periodic batches).
REMOVED_PBC_HANDLING_MODES = ("slab", "molecule_in_box", "mixed_periodic")


def validate_pbc_handling(mode: str) -> None:
    if mode not in PBC_HANDLING_MODES:
        raise ValueError(
            f"Unsupported pbc_handling {mode!r}; choose {PBC_HANDLING_MODES}"
        )


def ensure_polar_compatibility(model):
    """Repair restored state without overriding an existing evaluator choice."""
    pbc_handling = getattr(model, "pbc_handling", "auto")
    if pbc_handling in REMOVED_PBC_HANDLING_MODES:
        pbc_handling = "auto"
    return convert_polar_model(model, pbc_handling)


def convert_polar_model(model, pbc_handling: str = "auto"):
    """Repair a legacy model in place and return it, preserving all tensors.

    Idempotent for compatible models. All self-interaction blocks, including
    nested real-space blocks, need their derived Python dimension restored.
    Dispatch is rebound on this instance through the public upstream setters.
    """
    from mace.modules.polar.electrostatics.energy import (  # pylint: disable=import-error
        GTOElectrostaticEnergy,
    )
    from mace.modules.polar.electrostatics.features import (  # pylint: disable=import-error
        GTOElectrostaticFeatures,
    )
    from mace.modules.polar.electrostatics.gto_utils import (  # pylint: disable=import-error
        GTOSelfInteractionBlock,
    )

    validate_pbc_handling(pbc_handling)
    # PolarMACE imports this helper for __setstate__; importing the class back
    # here would make every mace.modules import cyclic. The block checks below
    # still provide the concrete graph_longrange type contract.
    if not any(cls.__name__ == "PolarMACE" for cls in type(model).__mro__):
        raise TypeError("Expected a PolarMACE whole model, not a checkpoint state dict")
    blocks = (
        (model.electric_potential_descriptor, GTOElectrostaticFeatures),
        (model.coulomb_energy, GTOElectrostaticEnergy),
    )
    repairs = []
    for block, expected_type in blocks:
        if not isinstance(block, expected_type) or not hasattr(
            block, "set_pbc_handling"
        ):
            raise TypeError(
                "Conversion requires a more recent version of graph_longrange, please refer to the mace docs"
            )
        for module in block.modules():
            if isinstance(module, GTOSelfInteractionBlock):
                dimension = module.features_irreps.dim
                if getattr(module, "features_dim", dimension) != dimension:
                    raise ValueError(
                        "Conflicting self-interaction features_dim in model"
                    )
                repairs.append((module, dimension))
    descriptor = model.electric_potential_descriptor
    expected_permutation = (
        descriptor._build_output_permutation(  # pylint: disable=protected-access
            model.field_feature_max_l, len(model.field_feature_widths)
        )
    )
    permutation = descriptor._buffers.get(  # pylint: disable=protected-access
        "output_permutation"
    )
    if permutation is None or not permutation.cpu().equal(expected_permutation):
        raise ValueError("Unsupported electrostatic output_permutation buffer schema")
    for module, dimension in repairs:
        module.features_dim = dimension
    # graph_longrange no longer keeps a geometry cache on the block (the geometry is
    # a returned value); pickles from before carry the last batch's cache and its
    # autograd graph. Nothing reads it: drop it.
    descriptor.static_quantities = None
    # Slab and molecule-in-box corrections of earlier code (loaded as placeholders by
    # load_polar_model): no longer evaluated, dropped.
    for block, name in (
        (descriptor, "non_periodic_correction_terms"),
        (model.coulomb_energy, "monopole_dipole_correction"),
    ):
        if name in block._modules:  # pylint: disable=protected-access
            del block._modules[name]  # pylint: disable=protected-access
    # Integer sizes the forwards read (models built before they existed).
    model.charges_dimension = model.charges_irreps.dim
    model.potential_dimension = model.potential_irreps.dim
    from mace.modules.polar.field_blocks import (  # noqa: PLC0415  # import cycle at module level
        SparseUvuTensorProduct,
    )

    for module in model.modules():
        if isinstance(module, SparseUvuTensorProduct):
            module.output_dimension = module.irreps_out.dim
    # Field blocks of models built before their linears followed the cueq layout: same
    # parameters, transplanted into cueq-layout linears; the layout transposes go.
    for module in model.modules():
        upgrade = getattr(module, "use_layout_native_linears", None)
        if upgrade is not None:
            upgrade()
        # Fold metadata of the field linears (models built before the folds).
        prepare = getattr(module, "prepare_linear_folds", None)
        if prepare is not None:
            prepare()
    model.set_electrostatic_pbcs(pbc_handling)
    return model


if __name__ == "__main__":
    main()
