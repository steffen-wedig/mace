"""The field linears folded into the dot-product weights give the same model.

PolarMACE's field blocks contract node features with a sum of linears of few-channel field
inputs (and, in the update block, of the node features themselves). The folded forward
multiplies each linear's per-irrep weight into the dot-product weight instead of forming the
linears' 512-wide outputs. It must not change a value, a force or a gradient (including the
gradients through the force, as in force training).
"""

from __future__ import annotations

import types

import pytest
import torch
from e3nn import o3

from mace.modules.polar.field_blocks import (
    AgnosticEmbeddedOneBodyVariableUpdate,
    OneBodyMLPFieldReadout,
    SparseUvuTensorProduct,
    instructions_for_sparse_tp,
)
from mace.modules.polar.linear_folds import folded_linear_input, linear_fold
from mace.modules.wrapper_ops import CUET_AVAILABLE, CuEquivarianceConfig
from tests.extensions.polar.test_polar_shared_geometry import (
    _batch,
    _molecules,
    _outputs_and_gradients,
    _polar_model,
)
from tests.helpers import GRAPH_LONGRANGE_AVAILABLE


def separate_linears(model) -> None:
    """Every linear that reads a layer's output evaluated on its own: the plan without a
    layout (as for e3nn linears on cueq-layout features)."""
    plan = model.shared_input_plan
    model.shared_input_plan = plan._replace(
        layout=None, layers=[layer._replace(folds=None) for layer in plan.layers]
    )


pytestmark = pytest.mark.polar

NODE_IRREPS = o3.Irreps("6x0e + 6x1o")


def _to_layout(features: torch.Tensor, irreps: o3.Irreps, layout: str) -> torch.Tensor:
    """mul_ir features in the given layout."""
    if layout == "mul_ir":
        return features
    blocks = []
    for (mul, ir), block_slice in zip(irreps, irreps.slices()):
        block = features[:, block_slice].reshape(-1, mul, ir.dim)
        blocks.append(block.transpose(1, 2).reshape(-1, mul * ir.dim))
    return torch.cat(blocks, dim=-1)


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
@pytest.mark.parametrize("biases", [False, True])
def test_folded_dot_matches_dot_of_linear_sum(layout, biases):
    torch.manual_seed(0)
    dtype = torch.float64
    invariant_irreps = o3.Irreps(f"{NODE_IRREPS.count(o3.Irrep(0, 1))}x0e")
    dot = SparseUvuTensorProduct(
        NODE_IRREPS,
        NODE_IRREPS,
        invariant_irreps,
        instructions_for_sparse_tp(NODE_IRREPS, NODE_IRREPS, invariant_irreps),
        layout=layout,
    ).to(dtype)
    small_irreps = o3.Irreps("2x0e + 2x1o")
    linears = [
        o3.Linear(small_irreps, NODE_IRREPS, biases=biases).to(dtype),
        o3.Linear(NODE_IRREPS, NODE_IRREPS).to(dtype),
    ]
    with torch.no_grad():
        for linear in linears:
            if linear.bias.numel():
                linear.bias.normal_()
    folds = [
        linear_fold(small_irreps, NODE_IRREPS, biases=biases),
        linear_fold(NODE_IRREPS, NODE_IRREPS, biases=False),
    ]
    nodes = 5
    x1 = torch.randn(nodes, NODE_IRREPS.dim, dtype=dtype, requires_grad=True)
    inputs = [
        torch.randn(nodes, small_irreps.dim, dtype=dtype, requires_grad=True),
        torch.randn(nodes, NODE_IRREPS.dim, dtype=dtype, requires_grad=True),
    ]
    parameters = [dot.weight] + [
        parameter for linear in linears for parameter in linear.parameters()
    ]
    leaves = [x1, *inputs, *parameters]

    def through_force_gradient(output: torch.Tensor) -> list:
        # As in force training: a loss on the input gradient, differentiated again.
        energy = torch.tanh(output).sum()
        (input_gradient,) = torch.autograd.grad(energy, x1, create_graph=True)
        loss = energy + (input_gradient**2).sum()
        gradients = torch.autograd.grad(loss, leaves, allow_unused=True)
        return [output.detach(), *gradients]

    x1_layout = _to_layout(x1, NODE_IRREPS, layout)
    summed = sum(linear(y) for linear, y in zip(linears, inputs))
    unfolded = dot(x1_layout, _to_layout(summed, NODE_IRREPS, layout))
    folded = dot.forward_folded(
        x1_layout,
        [
            folded_linear_input(_to_layout(y, irreps, layout), linear, fold)
            for y, irreps, linear, fold in zip(
                inputs, [small_irreps, NODE_IRREPS], linears, folds
            )
        ],
    )
    for expected, actual in zip(
        through_force_gradient(unfolded), through_force_gradient(folded)
    ):
        if expected is None:
            assert actual is None
            continue
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)


def test_linear_fold_covers_the_parameters():
    fold = linear_fold(o3.Irreps("2x0e + 2x1o"), o3.Irreps("8x0e + 8x1o"), biases=True)
    assert [path.input_multiplicity for path in fold.paths] == [2, 2]
    assert fold.paths[-1].weight_stop == 2 * 8 + 2 * 8
    assert [(bias.bias_start, bias.bias_stop) for bias in fold.biases] == [(0, 8)]


# --------------------------------------------------------------------------- model level
def _unfolded_update(
    self,
    node_attrs,
    node_feats,
    edge_attrs,
    edge_feats,
    edge_index,
    potential_features,
    local_charges,
):
    # Reference: the embedding evaluated (its linears not folded).
    mixed = self.potential_embedding(
        potential_features, node_feats, node_attrs, local_charges
    )
    invariants = self.dot_products(node_feats, mixed)
    embedded = torch.cat([invariants, self.source_embedding(node_attrs)], dim=-1)
    new_feats = self.tp_out(node_feats, self.nonlinearity(embedded))
    readout_to_mul_ir = getattr(self, "_readout_to_mul_ir", None)
    readout_from_mul_ir = getattr(self, "_readout_from_mul_ir", None)
    if readout_to_mul_ir is not None:
        new_feats = readout_to_mul_ir(new_feats)
    multipoles = self.readout(new_feats)
    if readout_from_mul_ir is not None:
        multipoles = readout_from_mul_ir(multipoles)
    return multipoles


def _unfolded_readout(
    self,
    node_attrs,
    node_feats,
    edge_attrs,
    edge_feats,
    edge_index,
    field_feats,
    charges_0,
    charges_induced,
):
    # The linears follow the model layout here (mul_ir model, or cueq-layout linears).
    charges_up = self.linear_up_q(charges_induced + charges_0)
    potential_up = self.linear_up_v(field_feats)
    invariants = torch.cat(
        [
            self.dot_products_q(node_feats, charges_up),
            self.dot_products_v(node_feats, potential_up),
        ],
        dim=-1,
    )
    return self.mlp(invariants).squeeze(-1)


def _unfold(model) -> None:
    # The reference evaluates every linear on its own (no shared-input products either).
    separate_linears(model)
    for module in model.modules():
        if isinstance(module, AgnosticEmbeddedOneBodyVariableUpdate):
            module.forward = types.MethodType(_unfolded_update, module)
        elif isinstance(module, OneBodyMLPFieldReadout):
            module.forward = types.MethodType(_unfolded_readout, module)


def _compare_folded_and_unfolded(cueq_config, device) -> None:
    model = _polar_model(cueq_config=cueq_config).to(device)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.normal_()
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in _batch(_molecules()).items()
    }
    folded = _outputs_and_gradients(model, batch)
    _unfold(model)
    unfolded = _outputs_and_gradients(model, batch)
    assert folded.keys() == unfolded.keys()
    for key, value in unfolded.items():
        scale = max(value.abs().max().item(), 1.0)
        assert (folded[key] - value).abs().max().item() <= 1e-10 * scale, key


@pytest.mark.skipif(
    not GRAPH_LONGRANGE_AVAILABLE, reason="graph_longrange is not installed"
)
def test_model_with_folded_field_linears_matches_unfolded():
    _compare_folded_and_unfolded(cueq_config=None, device=torch.device("cpu"))


@pytest.mark.skipif(
    not (CUET_AVAILABLE and torch.cuda.is_available() and GRAPH_LONGRANGE_AVAILABLE),
    reason="needs cuequivariance, a GPU and graph_longrange",
)
def test_cueq_model_with_folded_field_linears_matches_unfolded():
    cueq_config = CuEquivarianceConfig(
        enabled=True, layout="ir_mul", group="O3_e3nn", optimize_all=True
    )
    _compare_folded_and_unfolded(cueq_config=cueq_config, device=torch.device("cuda"))
