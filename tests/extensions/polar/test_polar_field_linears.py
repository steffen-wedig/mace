"""PolarMACE's field blocks use linears in the cueq layout under cueq; same map as e3nn's.

Under a cueq config the potential embedding and the field readout used to run e3nn
linears in mul_ir between layout transposes. They now build LayoutNativeLinear (cueq
layout, the fused kernel for few input channels); older models are upgraded on load by
transplanting the e3nn parameters. Neither may change a value, a force or a gradient.
"""

from __future__ import annotations

import pytest
import torch
from e3nn import o3

from mace.modules.wrapper_ops import (
    CUET_AVAILABLE,
    CuEquivarianceConfig,
    TransposeIrrepsLayoutWrapper,
)
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
        not (CUET_AVAILABLE and torch.cuda.is_available()),
        reason="needs cuequivariance and a GPU (fused kernels)",
    ),
    pytest.mark.skipif(
        not GRAPH_LONGRANGE_AVAILABLE, reason="graph_longrange is not installed"
    ),
]

DEVICE = torch.device("cuda")


def _cueq_config() -> CuEquivarianceConfig:
    return CuEquivarianceConfig(
        enabled=True, layout="ir_mul", group="O3_e3nn", optimize_all=True
    )


def _transpose(irreps, source, target, cueq_config):
    return TransposeIrrepsLayoutWrapper(
        irreps=irreps, source=source, target=target, cueq_config=cueq_config
    )


@pytest.mark.parametrize(
    "irreps_in,irreps_out,biases",
    [
        ("2x0e+2x1o+2x0e+2x1o", "16x0e+16x1o", False),  # fused kernel
        ("1x0e+1x1o+1x0e+1x1o", "16x0e+16x1o", True),  # fused kernel, scalar biases
        ("16x0e+16x1o", "16x0e+16x1o", False),  # naive kernel
    ],
)
def test_transplanted_linear_equals_e3nn_between_transposes(
    irreps_in, irreps_out, biases
):
    from mace.modules.wrapper_ops import LayoutNativeLinear  # noqa: PLC0415

    cueq_config = _cueq_config()
    torch.manual_seed(0)
    e3nn_linear = o3.Linear(irreps_in, irreps_out, biases=biases).to(
        DEVICE, torch.float64
    )
    with torch.no_grad():
        if biases:
            e3nn_linear.bias.normal_()
    native = LayoutNativeLinear.from_e3nn(e3nn_linear, cueq_config)
    to_mul_ir = _transpose(o3.Irreps(irreps_in), "ir_mul", "mul_ir", cueq_config).to(
        DEVICE
    )
    from_mul_ir = _transpose(
        o3.Irreps(irreps_out), "mul_ir", "ir_mul", cueq_config
    ).to(DEVICE)

    def reference(x):
        return from_mul_ir(e3nn_linear(to_mul_ir(x)))

    features = torch.randn(
        7, o3.Irreps(irreps_in).dim, device=DEVICE, dtype=torch.float64
    )
    results = []
    for linear, parameters in (
        (reference, list(e3nn_linear.parameters())),
        (native, list(native.parameters())),
    ):
        inputs = features.clone().requires_grad_(True)
        output = linear(inputs)
        (input_gradient,) = torch.autograd.grad(
            output.sin().sum(), inputs, create_graph=True
        )
        parameter_gradients = torch.autograd.grad(
            (input_gradient**2).sum(), parameters
        )
        results.append(
            [output, input_gradient]
            + [gradient.reshape(-1) for gradient in parameter_gradients]
        )
    for expected, actual in zip(*results):
        assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_e3nn_shaped_weights_load_into_the_native_linear():
    from mace.modules.wrapper_ops import LayoutNativeLinear  # noqa: PLC0415

    e3nn_linear = o3.Linear("1x0e+1x1o", "4x0e+4x1o", biases=True)
    native = LayoutNativeLinear(
        e3nn_linear.irreps_in, e3nn_linear.irreps_out, _cueq_config(), biases=True
    )
    native.load_state_dict(e3nn_linear.state_dict())
    assert torch.equal(native.weight.reshape(-1), e3nn_linear.weight)
    assert torch.equal(native.bias, e3nn_linear.bias)


def _as_built_before(model) -> None:
    """Rebuild the field blocks as models built before this change had them: e3nn
    linears with the same parameters, between layout transposes."""
    from mace.modules.polar.field_blocks import (  # noqa: PLC0415
        AgnosticChargeBiasedLinearPotentialEmbedding,
        OneBodyMLPFieldReadout,
    )

    def e3nn_copy(native, biases):
        linear = o3.Linear(
            o3.Irreps(str(native.irreps_in)),
            o3.Irreps(str(native.irreps_out)),
            biases=biases,
        ).to(
            native.weight.device, native.weight.dtype
        )
        with torch.no_grad():
            linear.weight.copy_(native.weight.reshape(-1))
            if biases:
                linear.bias.copy_(native.bias)
        return linear

    for block in model.modules():
        config = getattr(block, "cueq_config", None)
        if isinstance(block, AgnosticChargeBiasedLinearPotentialEmbedding):
            for name in ("potential_linear", "node_feats_linear", "charge_embedding"):
                setattr(block, name, e3nn_copy(getattr(block, name), biases=False))
            block._potential_to_mul_ir = _transpose(
                block.potential_irreps, "ir_mul", "mul_ir", config
            )
            block._node_feats_to_mul_ir = _transpose(
                block.node_feats_irreps, "ir_mul", "mul_ir", config
            )
            block._charges_to_mul_ir = _transpose(
                block.charges_irreps, "ir_mul", "mul_ir", config
            )
            block._node_feats_from_mul_ir = _transpose(
                block.node_feats_irreps, "mul_ir", "ir_mul", config
            )
        elif isinstance(block, OneBodyMLPFieldReadout):
            for name in ("linear_up_q", "linear_up_v"):
                setattr(block, name, e3nn_copy(getattr(block, name), biases=True))
            block._q_to_mul_ir = _transpose(
                block.charges_irreps, "ir_mul", "mul_ir", config
            )
            block._v_to_mul_ir = _transpose(
                block.potential_irreps, "ir_mul", "mul_ir", config
            )
            block._up_from_mul_ir = _transpose(
                block.node_feats_irreps, "mul_ir", "ir_mul", config
            )
    model.to(DEVICE)


def test_upgrading_a_model_built_before_keeps_every_output():
    from mace.modules.wrapper_ops import LayoutNativeLinear  # noqa: PLC0415
    from mace.modules.polar.conversion import convert_polar_model  # noqa: PLC0415

    model = _polar_model(cueq_config=_cueq_config()).to(DEVICE)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.normal_()
    batch = {
        key: value.to(DEVICE) if torch.is_tensor(value) else value
        for key, value in _batch(_molecules()).items()
    }
    native = _outputs_and_gradients(model, batch)
    _as_built_before(model)
    assert isinstance(
        model.field_dependent_charges_maps[0].potential_embedding.potential_linear,
        o3.Linear,
    )
    before = _outputs_and_gradients(model, batch)
    convert_polar_model(model, model.pbc_handling)
    embedding = model.field_dependent_charges_maps[0].potential_embedding
    assert isinstance(embedding.potential_linear, LayoutNativeLinear)
    assert embedding._potential_to_mul_ir is None
    upgraded = _outputs_and_gradients(model, batch)
    assert before.keys() == upgraded.keys() == native.keys()
    for key, value in before.items():
        scale = max(value.abs().max().item(), 1.0)
        assert (upgraded[key] - value).abs().max().item() <= 1e-10 * scale, key
        assert (native[key] - value).abs().max().item() <= 1e-10 * scale, key
