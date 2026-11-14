"""Padding of compiled MD inference (``mace.tools.compiled.inference``).

The padding structure must leave the real structures' outputs unchanged and stay finite.
With POLAR-M's ``apply_cutoff=False`` it is inert: its edges lie beyond the cutoff and carry
nothing, whatever their number. Padding atoms inside the cutoff with ~47 edges each
overflowed in float32 and turned the real structures' energies into NaN through the dense
k-space products (experiment 10).
"""

from __future__ import annotations

import pytest
import torch

from mace.tools.compiled.inference import (
    InferenceBudgets,
    inference_budgets,
    kspace_lattice_inputs,
    pad_inference_batch,
)
from tests.extensions.polar.test_polar_reciprocal_lattice import _periodic_batch
from tests.extensions.polar.test_polar_shared_geometry import _clone, _polar_model

KEYS = ("energy", "forces", "stress")


def _outputs(model, batch: dict) -> dict:
    output = model(_clone(batch), training=False, compute_force=True, compute_stress=True)
    return {key: output[key].detach() for key in KEYS}


@pytest.fixture(name="default_dtype")
def _default_dtype(request):
    """The default dtype of the evaluation (MD sets it to the model's): a float32 periodic
    forward with stress under a float64 default fails in the backward of
    ``torch.linalg.inv_ex`` (LU of a float64 identity), as it did before experiment 10."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    yield request.param
    torch.set_default_dtype(previous)


@pytest.mark.parametrize("apply_cutoff", [False, True])
@pytest.mark.parametrize(
    "default_dtype", [torch.float32, torch.float64], indirect=True, ids=["float32", "float64"]
)
def test_padding_structure_is_inert(default_dtype, apply_cutoff):
    dtype = default_dtype
    model = _polar_model(dtype=dtype)
    model.set_electrostatic_pbcs("pbc")
    # POLAR-M's setting (the cutoff multiplies the edge harmonics) or the cutoff applied
    # inside the radial network's input only.
    model.radial_embedding.apply_cutoff = apply_cutoff
    batch = {
        key: value.to(dtype) if torch.is_tensor(value) and value.is_floating_point() else value
        for key, value in _periodic_batch().items()
    }
    batch.update(kspace_lattice_inputs(model, batch["cell"].view(-1, 3, 3), margin=0.1))
    expected = _outputs(model, batch)
    budgets = inference_budgets(batch)
    # Many more padding edges than real ones: the padding must not care.
    budgets = InferenceBudgets(
        budgets.atoms, budgets.edges + 20 * int(batch["edge_index"].shape[1]), budgets.kspace_rows
    )
    padded = pad_inference_batch(batch, budgets, padding_spacing=2.0 * float(model.r_max))
    assert padded["positions"].shape[0] == budgets.atoms
    assert padded["edge_index"].shape[1] == budgets.edges
    actual = _outputs(model, padded)
    num_graphs = int(batch["ptr"].numel()) - 1
    num_atoms = int(batch["positions"].shape[0])
    for key in KEYS:
        assert bool(torch.isfinite(actual[key]).all()), key
    tolerance = {torch.float32: 1e-5, torch.float64: 1e-11}[dtype]
    torch.testing.assert_close(
        actual["energy"][:num_graphs], expected["energy"], rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        actual["forces"][:num_atoms], expected["forces"], rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        actual["stress"][:num_graphs], expected["stress"], rtol=tolerance, atol=tolerance
    )
    if apply_cutoff:
        return
    # Inert: the padding structure's energy does not depend on its number of edges (every
    # padding edge lies beyond the cutoff), so no edge count can make it overflow.
    fewer = InferenceBudgets(budgets.atoms, budgets.edges - 7, budgets.kspace_rows)
    padding_energy = _outputs(
        model, pad_inference_batch(batch, fewer, padding_spacing=2.0 * float(model.r_max))
    )["energy"][num_graphs]
    torch.testing.assert_close(actual["energy"][num_graphs], padding_energy, rtol=0, atol=0)
