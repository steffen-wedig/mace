"""PolarMACE's linears follow the model layout; models built with e3nn linears load.

The field linears, the Fukui source map and the charge-update readout are
``mace.modules.equivariant_linear`` linears in the model layout: no layout transposes
around them. Models saved before (e3nn ``o3.Linear``, mul_ir) get the same parameters in
EquivariantLinears on load, without changing a value, a force or a gradient.
"""

from __future__ import annotations

import pytest
import torch
from e3nn import o3

from mace.modules.equivariant_linear import EquivariantLinear, NarrowEquivariantLinear
from mace.modules.polar.conversion import convert_polar_model, replace_e3nn_linears
from mace.modules.polar.field_blocks import (
    AgnosticChargeBiasedLinearPotentialEmbedding,
    AgnosticEmbeddedOneBodyVariableUpdate,
    OneBodyMLPFieldReadout,
)
from mace.modules.wrapper_ops import CUET_AVAILABLE, CuEquivarianceConfig
from tests.extensions.polar.test_polar_shared_geometry import (
    _batch,
    _molecules,
    _outputs_and_gradients,
    _polar_model,
)
from tests.helpers import GRAPH_LONGRANGE_AVAILABLE

pytestmark = pytest.mark.polar
needs_graph_longrange = pytest.mark.skipif(
    not GRAPH_LONGRANGE_AVAILABLE, reason="graph_longrange is not installed"
)


def _as_built_before(model) -> int:
    """Every equivariant linear replaced by an e3nn o3.Linear with the same parameters,
    as models saved before had them. Returns how many were replaced."""
    replacements = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, (EquivariantLinear, NarrowEquivariantLinear)):
                replacements.append((parent, name, child))
    for parent, name, linear in replacements:
        biases = linear.bias is not None
        e3nn_linear = o3.Linear(linear.irreps_in, linear.irreps_out, biases=biases).to(
            device=linear.weight.device, dtype=linear.weight.dtype
        )
        with torch.no_grad():
            e3nn_linear.weight.copy_(linear.weight)
            if biases:
                e3nn_linear.bias.copy_(linear.bias)
        setattr(parent, name, e3nn_linear)
    return len(replacements)


@needs_graph_longrange
def test_field_linears_are_chosen_explicitly_in_the_model_layout():
    model = _polar_model(cueq_config=None)
    update = next(
        module
        for module in model.modules()
        if isinstance(module, AgnosticEmbeddedOneBodyVariableUpdate)
    )
    embedding = update.potential_embedding
    assert isinstance(embedding, AgnosticChargeBiasedLinearPotentialEmbedding)
    for linear in (
        embedding.potential_linear,
        embedding.charge_embedding,
        embedding.node_feats_linear,
    ):
        assert type(linear) is EquivariantLinear
    assert type(update.readout.linear_2) is NarrowEquivariantLinear
    assert type(model.fukui_source_map.linear_2) is NarrowEquivariantLinear
    for module in model.modules():
        if isinstance(module, OneBodyMLPFieldReadout):
            assert type(module.linear_up_q) is EquivariantLinear
            assert type(module.linear_up_v) is EquivariantLinear
        if isinstance(module, (EquivariantLinear, NarrowEquivariantLinear)):
            assert module.layout == "mul_ir"
    assert not hasattr(model, "_fukui_to_mul_ir")
    assert not hasattr(update, "_readout_to_mul_ir")


@needs_graph_longrange
def test_a_model_built_with_e3nn_linears_loads_with_the_same_outputs():
    torch.manual_seed(0)
    model = _polar_model(cueq_config=None)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.normal_()
    batch = _batch(_molecules())
    current = _outputs_and_gradients(model, batch)
    # The forward reads the linears' plans: a model with e3nn linears runs only after
    # the conversion that every unpickling applies.
    assert _as_built_before(model) > 0
    convert_polar_model(model, model.pbc_handling)
    assert not any(isinstance(module, o3.Linear) for module in model.modules())
    upgraded = _outputs_and_gradients(model, batch)
    assert upgraded.keys() == current.keys()
    for key, value in current.items():
        scale = max(value.abs().max().item(), 1.0)
        assert (upgraded[key] - value).abs().max().item() <= 1e-10 * scale, key


@pytest.mark.skipif(not CUET_AVAILABLE, reason="needs cuequivariance")
def test_e3nn_linears_in_a_cueq_layout_model_are_refused():
    model = torch.nn.Module()
    model.cueq_config = CuEquivarianceConfig(
        enabled=True, layout="ir_mul", group="O3_e3nn", optimize_all=True
    )
    model.readout = o3.Linear("4x0e + 2x1o", "1x0e")
    with pytest.raises(ValueError, match="convert_e3nn_cueq"):
        replace_e3nn_linears(model)
