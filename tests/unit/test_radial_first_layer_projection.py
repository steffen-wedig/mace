"""The radial MLPs' first layer from per-element projection tables gives the same block.

RealAgnosticResidualNonLinearInteractionBlock feeds conv_tp_weights and density_fn the edge
input [radial features, source_embedding[sender], target_embedding[receiver]]. The block
now multiplies the embedding columns of the first Linear per node and gathers the
products, instead of multiplying the concatenated [edges, radial + 2 x channels] input.
Values and every gradient (through the radial features twice, as in force training) must
not change.
"""

from __future__ import annotations

import torch
from e3nn import o3

from mace.modules.blocks import RealAgnosticResidualNonLinearInteractionBlock


def _block() -> RealAgnosticResidualNonLinearInteractionBlock:
    torch.manual_seed(0)
    return RealAgnosticResidualNonLinearInteractionBlock(
        node_attrs_irreps=o3.Irreps("3x0e"),
        node_feats_irreps=o3.Irreps("8x0e + 8x1o"),
        edge_attrs_irreps=o3.Irreps.spherical_harmonics(2),
        edge_feats_irreps=o3.Irreps("5x0e"),
        target_irreps=o3.Irreps("8x0e + 8x1o + 8x2e"),
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        avg_num_neighbors=3.0,
        edge_irreps=o3.Irreps("4x0e + 4x1o"),
        radial_MLP=[16, 16],
    ).to(torch.float64)


def _concatenated(block, radial_feats, node_attrs, edge_index):
    source_embedding = block.source_embedding(node_attrs)
    target_embedding = block.target_embedding(node_attrs)
    edge_input = torch.cat(
        [
            radial_feats,
            source_embedding[edge_index[0]],
            target_embedding[edge_index[1]],
        ],
        dim=-1,
    )
    return block.conv_tp_weights(edge_input), block.density_fn(edge_input)


def _projected(block, radial_feats, node_attrs, edge_index):
    tp_hidden, density_hidden = (
        block._radial_first_layers(  # pylint: disable=protected-access
            radial_feats, node_attrs, edge_index
        )
    )
    return block.conv_tp_weights.net[1:](tp_hidden), block.density_fn.net[1:](
        density_hidden
    )


def test_projected_first_layers_match_the_concatenated_input():
    block = _block()
    nodes, edges = 6, 20
    generator = torch.Generator().manual_seed(1)
    edge_index = torch.randint(0, nodes, (2, edges), generator=generator)
    radial_feats = torch.randn(edges, 5, dtype=torch.float64, requires_grad=True)
    # Not one-hot: the table form must hold for any attributes (it uses linearity only).
    node_attrs = torch.rand(nodes, 3, dtype=torch.float64, generator=generator)
    parameters = [
        parameter for parameter in block.parameters() if parameter.requires_grad
    ]

    def gradients(evaluate) -> list:
        tp_weights, density = evaluate(block, radial_feats, node_attrs, edge_index)
        energy = (torch.tanh(tp_weights) ** 2).sum() + (density**2).sum()
        (radial_gradient,) = torch.autograd.grad(
            energy, radial_feats, create_graph=True
        )
        loss = energy + (radial_gradient**2).sum()
        return [tp_weights.detach(), density.detach()] + list(
            torch.autograd.grad(loss, [radial_feats, *parameters], allow_unused=True)
        )

    for expected, actual in zip(gradients(_concatenated), gradients(_projected)):
        if expected is None:
            assert actual is None
            continue
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-13)


def test_block_forward_runs_with_projected_first_layers():
    radial_width = 5
    block = _block()
    nodes, edges = 4, 10
    edge_index = torch.randint(0, nodes, (2, edges))
    node_attrs = torch.nn.functional.one_hot(torch.randint(0, 3, (nodes,)), 3).to(
        torch.float64
    )
    message, skip = block(
        node_attrs=node_attrs,
        node_feats=torch.randn(nodes, block.node_feats_irreps.dim, dtype=torch.float64),
        edge_attrs=torch.randn(edges, block.edge_attrs_irreps.dim, dtype=torch.float64),
        edge_feats=torch.randn(edges, radial_width, dtype=torch.float64),
        edge_index=edge_index,
    )
    assert message.shape[0] == nodes and skip.shape[0] == nodes
