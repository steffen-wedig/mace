"""The first layer evaluated on the element-embedding table gives the same model.

PolarMACE's first interaction receives node_feats = node_embedding(node_attrs). Its first
linears (skip_tp, linear_up, linear_res) have no biases, so they are linear in node_attrs:
the block evaluates them on the table (one row per element) and expands the rows per node.
Values and every gradient (through the force, as in training) must not change.
"""

from __future__ import annotations

import pytest
import torch

from mace.modules.wrapper_ops import CUET_AVAILABLE, CuEquivarianceConfig
from tests.extensions.polar.test_polar_shared_geometry import (
    _batch,
    _molecules,
    _outputs_and_gradients,
    _polar_model,
)
from tests.helpers import GRAPH_LONGRANGE_AVAILABLE

pytestmark = [
    pytest.mark.polar,
    pytest.mark.skipif(
        not GRAPH_LONGRANGE_AVAILABLE, reason="graph_longrange is not installed"
    ),
]


def _compare_table_and_per_node(cueq_config, device) -> None:
    model = _polar_model(cueq_config=cueq_config).to(device)
    assert model.interactions[0].accepts_node_feats_table
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in _batch(_molecules()).items()
    }
    from_table = _outputs_and_gradients(model, batch)
    model.interactions[0].accepts_node_feats_table = False
    per_node = _outputs_and_gradients(model, batch)
    assert from_table.keys() == per_node.keys()
    for key, value in per_node.items():
        scale = max(value.abs().max().item(), 1.0)
        assert (from_table[key] - value).abs().max().item() <= 1e-10 * scale, key


def test_first_layer_from_element_table_matches_per_node():
    _compare_table_and_per_node(cueq_config=None, device=torch.device("cpu"))


@pytest.mark.skipif(
    not (CUET_AVAILABLE and torch.cuda.is_available()),
    reason="needs cuequivariance and a GPU",
)
def test_cueq_first_layer_from_element_table_matches_per_node():
    cueq_config = CuEquivarianceConfig(
        enabled=True, layout="ir_mul", group="O3_e3nn", optimize_all=True
    )
    _compare_table_and_per_node(cueq_config=cueq_config, device=torch.device("cuda"))
