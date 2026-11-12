"""
Wrapper class for o3.Linear that optionally uses cuet.Linear
"""

import dataclasses
import types
from typing import Any, List, Optional

import torch
from e3nn import o3

from mace.modules.symmetric_contraction import SymmetricContraction
from mace.tools.cg import O3_e3nn
from mace.tools.scatter import scatter_sum

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    CUET_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CUET_AVAILABLE = False

try:
    import openequivariance as oeq

    OEQ_AVAILABLE = True
except ImportError:
    OEQ_AVAILABLE = False


@dataclasses.dataclass
class CuEquivarianceConfig:
    """Configuration for cuequivariance acceleration"""

    enabled: bool = False
    layout: str = "mul_ir"  # One of: mul_ir, ir_mul
    layout_str: str = "mul_ir"
    group: str = "O3"
    optimize_all: bool = False  # Set to True to enable all optimizations
    optimize_linear: bool = False
    optimize_channelwise: bool = False
    optimize_symmetric: bool = False
    optimize_fctp: bool = False
    conv_fusion: bool = False  # Set to True to enable conv fusion

    def __post_init__(self):
        if self.enabled and CUET_AVAILABLE:
            self.layout_str = self.layout
            self.layout = getattr(cue, self.layout)
            self.group = (
                O3_e3nn if self.group == "O3_e3nn" else getattr(cue, self.group)
            )
        if not CUET_AVAILABLE:
            self.enabled = False


def get_layout(cueq_config: Optional["CuEquivarianceConfig"] = None) -> str:
    """Return the irreps layout string for the active backend."""
    if cueq_config is not None and cueq_config.enabled:
        return getattr(cueq_config, "layout_str", "mul_ir")
    return "mul_ir"


@dataclasses.dataclass
class OEQConfig:
    """Configuration for cuequivariance acceleration"""

    enabled: bool = False
    optimize_all: bool = False
    optimize_channelwise: bool = False
    conv_fusion: Optional[str] = "atomic"

    def __post_init__(self):
        if not OEQ_AVAILABLE:
            self.enabled = False


class Linear:
    """Returns either a cuet.Linear or o3.Linear based on config"""

    def __new__(
        cls,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_linear)
        ):
            return cuet.Linear(
                cue.Irreps(cueq_config.group, irreps_in),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                method="naive",
            )

        return o3.Linear(
            irreps_in,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )


def cueq_handles_linears(cueq_config: Optional[CuEquivarianceConfig]) -> bool:
    """Whether Linear builds cuet.Linear (in the cueq layout) for this config."""
    return (
        CUET_AVAILABLE
        and cueq_config is not None
        and cueq_config.enabled
        and (cueq_config.optimize_all or cueq_config.optimize_linear)
    )


# Largest input multiplicity for which cuet.Linear's fused_tp method beats naive.
# Measured (RTX 5080, cuequivariance 0.10, forward + force gradient + backward through it,
# 350 nodes): 2x0e+2x1o+2x0e+2x1o -> 512x0e+512x1o takes 24 instead of 126 kernel launches
# and 0.7 instead of 1.3 ms; 512x0e+512x1o -> 512x0e+512x1o is 18x slower with fused_tp.
FUSED_LINEAR_MAX_INPUT_MULTIPLICITY = 4


if CUET_AVAILABLE:

    class LayoutNativeLinear(cuet.Linear):
        """cuet.Linear in the cueq layout, optionally with e3nn's scalar biases.

        Same parameters as o3.Linear(irreps_in, irreps_out, biases=biases): "weight" in
        e3nn's flat order (stored [1, n] instead of [n]; e3nn-shaped weights load by
        reshaping) and "bias" over the scalar output channels, added with weight 1 as in
        e3nn. Inputs with few channels per irrep use the fused kernel (one launch per
        pass instead of one per segment); see FUSED_LINEAR_MAX_INPUT_MULTIPLICITY.
        """

        def __init__(
            self,
            irreps_in: o3.Irreps,
            irreps_out: o3.Irreps,
            cueq_config: CuEquivarianceConfig,
            biases: bool = False,
        ):
            irreps_in = o3.Irreps(irreps_in)
            irreps_out = o3.Irreps(irreps_out)
            largest_input_multiplicity = max((mul for mul, _ in irreps_in), default=0)
            super().__init__(
                cue.Irreps(cueq_config.group, irreps_in),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=True,
                method=(
                    "fused_tp"
                    if largest_input_multiplicity <= FUSED_LINEAR_MAX_INPUT_MULTIPLICITY
                    else "naive"
                ),
            )
            # The fused kernel fixes its math dtype when built; _apply rebuilds it when
            # the parameters change dtype (float64 inputs need float64 math).
            self.linear_polynomial = cue.descriptors.linear(
                cue.Irreps(cueq_config.group, irreps_in),
                cue.Irreps(cueq_config.group, irreps_out),
            ).polynomial
            self.math_dtype = torch.get_default_dtype()
            self.output_dimension = irreps_out.dim
            bias_positions: List[int] = []
            if biases:
                start = 0
                for mul, ir in irreps_out:
                    # A scalar segment has the same channels in both layouts.
                    if ir.l == 0 and ir.p == 1:
                        bias_positions.extend(range(start, start + mul))
                    start += mul * ir.dim
            if bias_positions:
                self.bias = torch.nn.Parameter(torch.zeros(len(bias_positions)))
                self.register_buffer(
                    "bias_positions",
                    torch.tensor(bias_positions, dtype=torch.long),
                    persistent=False,
                )
            else:
                self.bias = None
                self.bias_positions = None

        def forward(  # pylint: disable=arguments-differ
            self, x: torch.Tensor
        ) -> torch.Tensor:
            output = super().forward(x)
            if self.bias is None:
                return output
            full_bias = output.new_zeros(self.output_dimension).index_copy(
                0, self.bias_positions, self.bias.to(output.dtype)
            )
            return output + full_bias

        def _apply(self, fn, recurse=True):
            super()._apply(fn, recurse)
            if self.method == "fused_tp" and self.weight.dtype != self.math_dtype:
                self.math_dtype = self.weight.dtype
                self.f = cuet.SegmentedPolynomial(
                    self.linear_polynomial, method="fused_tp", math_dtype=self.math_dtype
                ).to(self.weight.device)
            return self

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            # o3.Linear's mask of output channels without a path: not a parameter here.
            state_dict.pop(prefix + "output_mask", None)
            key = prefix + "weight"
            weight = state_dict.get(key)
            if (
                weight is not None
                and weight.shape != self.weight.shape
                and weight.numel() == self.weight.numel()
            ):
                state_dict[key] = weight.reshape(self.weight.shape)
            super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

        @classmethod
        def from_e3nn(
            cls, linear: o3.Linear, cueq_config: CuEquivarianceConfig
        ) -> "LayoutNativeLinear":
            """The same linear map with the parameters of an o3.Linear (transplanted)."""
            biases = bool(getattr(linear, "bias_numel", 0))
            native = cls(linear.irreps_in, linear.irreps_out, cueq_config, biases=biases)
            reference = linear.weight
            native = native.to(device=reference.device, dtype=reference.dtype)
            with torch.no_grad():
                native.weight.copy_(reference.reshape(native.weight.shape))
                if biases:
                    native.bias.copy_(linear.bias)
            return native


def field_linear(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    cueq_config: Optional[CuEquivarianceConfig] = None,
    biases: bool = False,
) -> torch.nn.Module:
    """Linear in the cueq layout when cueq handles linears, else e3nn's o3.Linear (mul_ir).

    The caller transposes layouts around it only in the second case.
    """
    if cueq_handles_linears(cueq_config):
        return LayoutNativeLinear(irreps_in, irreps_out, cueq_config, biases=biases)
    return o3.Linear(
        irreps_in,
        irreps_out,
        internal_weights=True,
        shared_weights=True,
        biases=biases,
    )


def with_scatter_sum(conv_tp: torch.nn.Module) -> torch.nn.Module:
    conv_tp.original_forward = conv_tp.forward

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        tp_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        mji = self.original_forward(node_feats[sender], edge_attrs, tp_weights)
        message = scatter_sum(src=mji, index=receiver, dim=0, dim_size=num_nodes)
        return message

    conv_tp.forward = types.MethodType(forward, conv_tp)
    return conv_tp


class CueqConvFusionWrapper(torch.nn.Module):
    """A serializable wrapper around cuet.SegmentedPolynomial / mptp.ff.

    It adapts MACE conv-style forward:

        forward(node_feats, edge_attrs, tp_weights, edge_index)

    to cuEq SegmentedPolynomial forward:

        forward([tp_weights, node_feats, edge_attrs], input_indices, output_shapes, output_indices)
    """

    def __init__(self, conv_tp: torch.nn.Module, polynomial: Any):
        super().__init__()

        # cuet.SegmentedPolynomial silently downgrades to SegmentedPolynomialNaive
        # when cuequivariance_ops_torch cannot be imported: it warns and sets
        # .method to "naive" even though uniform_1d was requested. The naive path
        # has no fused conv, so refuse here instead of letting it surface later,
        # far from the cause, as a missing attribute on the implementation object.
        method = getattr(conv_tp, "method", None)
        if method != "uniform_1d":
            raise RuntimeError(
                "cuEquivariance conv fusion needs the uniform_1d kernels, but "
                f"cuequivariance selected method={method!r} "
                f"({type(getattr(conv_tp, 'm', None)).__name__}). That means "
                "cuequivariance_ops_torch could not be imported. Check that the "
                "installed cueq-cuda-* extra matches torch.version.cuda, and note "
                "the ops wheel links a specific cuBLAS while declaring only a "
                "floor, so it can resolve against a cuBLAS it was not built for."
            )

        self.conv_tp = conv_tp
        # operands[0] is the weight operand; its size is num_segments * extent,
        # the same product the fused implementation exposes through
        # buffer_num_segments/operand_extent. Reading it from the descriptor keeps
        # this independent of which backend cuequivariance selected.
        self.weight_numel = polynomial.operands[0].size

    @property
    def m(self):
        return self.conv_tp.m

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        tp_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]

        return self.conv_tp(
            [tp_weights, node_feats, edge_attrs],
            {1: sender},
            {0: node_feats},
            {0: receiver},
        )[0]


def with_cueq_conv_fusion(conv_tp: torch.nn.Module, polynomial: Any) -> torch.nn.Module:
    """Wraps a cuet.SegmentedPolynomial / ConvTensorProduct to use conv fusion."""
    return CueqConvFusionWrapper(conv_tp, polynomial)


def with_oeq_conv_fusion(
    conv_tp: torch.nn.Module,
    transpose_in: Optional[torch.nn.Module] = None,
    transpose_out: Optional[torch.nn.Module] = None,
) -> torch.nn.Module:
    """Wraps an oeq.TensorProductConv to match MACE's conv_tp calling convention.

    oeq.TensorProductConv.forward(X, Y, W, rows, cols) performs a fused
    tensor-product + scatter.  MACE interaction blocks call
    conv_tp(node_feats, edge_attrs, tp_weights, edge_index) where
    edge_index[0]=sender, edge_index[1]=receiver.

    When cueq is active with ir_mul layout, optional transpose modules
    convert node_feats from ir_mul → mul_ir before the oeq forward and
    the output from mul_ir → ir_mul after, since oeq always uses mul_ir.
    """
    conv_tp.original_forward = conv_tp.forward
    if not hasattr(conv_tp, "weight_numel"):
        conv_tp.weight_numel = conv_tp.input_args["problem"].weight_numel
    conv_tp.layout_transpose_in = transpose_in
    conv_tp.layout_transpose_out = transpose_out

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        tp_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        if self.layout_transpose_in is not None:
            node_feats = self.layout_transpose_in(node_feats)
        out = self.original_forward(
            node_feats,
            edge_attrs,
            tp_weights,
            receiver,
            sender,
        )
        if self.layout_transpose_out is not None:
            out = self.layout_transpose_out(out)
        return out

    conv_tp.forward = types.MethodType(forward, conv_tp)
    return conv_tp


def with_oeq_scatter_sum(conv_tp: torch.nn.Module) -> torch.nn.Module:
    """Wraps an oeq.TensorProduct (non-fused) to add scatter like e3nn path."""
    conv_tp.original_forward = conv_tp.forward

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        tp_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        mji = self.original_forward(node_feats[sender], edge_attrs, tp_weights)
        message = scatter_sum(src=mji, index=receiver, dim=0, dim_size=num_nodes)
        return message

    conv_tp.forward = types.MethodType(forward, conv_tp)
    return conv_tp


class TensorProduct:
    """Wrapper around o3.TensorProduct/cuet.ChannelwiseTensorProduct/oeq.TensorProduct followed by a scatter sum"""

    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: Optional[List] = None,
        shared_weights: bool = False,
        internal_weights: bool = False,
        use_conv_fusion: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_channelwise)
        ):
            if cueq_config.conv_fusion and use_conv_fusion:
                polynomial = (
                    cue.descriptors.channelwise_tensor_product(
                        cue.Irreps(cueq_config.group, irreps_in1),
                        cue.Irreps(cueq_config.group, irreps_in2),
                        cue.Irreps(cueq_config.group, irreps_out),
                    )
                    .flatten_coefficient_modes()
                    .squeeze_modes()
                    .polynomial
                )
                return with_cueq_conv_fusion(
                    cuet.SegmentedPolynomial(
                        polynomial,
                        math_dtype=torch.get_default_dtype(),
                        method="uniform_1d",
                    ),
                    polynomial,
                )
            return cuet.ChannelWiseTensorProduct(
                cue.Irreps(cueq_config.group, irreps_in1),
                cue.Irreps(cueq_config.group, irreps_in2),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                dtype=torch.get_default_dtype(),
                math_dtype=torch.get_default_dtype(),
            )
        if (
            OEQ_AVAILABLE
            and oeq_config is not None
            and oeq_config.enabled
            and (oeq_config.optimize_all or oeq_config.optimize_channelwise)
        ):
            dtype = oeq.torch_to_oeq_dtype(torch.get_default_dtype())
            tpp = oeq.TPProblem(
                irreps_in1,
                irreps_in2,
                irreps_out,
                instructions,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                irrep_dtype=dtype,
                weight_dtype=dtype,
            )

            if oeq_config.conv_fusion is None:
                return with_oeq_scatter_sum(oeq.TensorProduct(tpp))
            if oeq_config.conv_fusion == "atomic":
                t_in, t_out = None, None
                if (
                    CUET_AVAILABLE
                    and cueq_config is not None
                    and cueq_config.enabled
                    and cueq_config.layout_str == "ir_mul"
                ):
                    t_in = cuet.TransposeIrrepsLayout(
                        cue.Irreps(cueq_config.group, irreps_in1),
                        source=cue.ir_mul,
                        target=cue.mul_ir,
                        use_fallback=True,
                    )
                    t_out = cuet.TransposeIrrepsLayout(
                        cue.Irreps(cueq_config.group, irreps_out),
                        source=cue.mul_ir,
                        target=cue.ir_mul,
                        use_fallback=True,
                    )
                return with_oeq_conv_fusion(
                    oeq.TensorProductConv(tpp, deterministic=False),
                    transpose_in=t_in,
                    transpose_out=t_out,
                )

            raise ValueError(f"Unknown conv_fusion option: {oeq_config.conv_fusion}")

        return o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )


class FullyConnectedTensorProduct:
    """Wrapper around o3.FullyConnectedTensorProduct/cuet.FullyConnectedTensorProduct"""

    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_fctp)
        ):
            return cuet.FullyConnectedTensorProduct(
                cue.Irreps(cueq_config.group, irreps_in1),
                cue.Irreps(cueq_config.group, irreps_in2),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                method="naive",
            )

        return o3.FullyConnectedTensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )


class SymmetricContractionWrapper:
    """Wrapper around SymmetricContraction/cuet.SymmetricContraction"""

    def __new__(
        cls,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        num_elements: Optional[int] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
        use_reduced_cg: bool = True,
    ):
        use_reduced_cg = use_reduced_cg and CUET_AVAILABLE
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_symmetric)
        ):
            return cuet.SymmetricContraction(
                cue.Irreps(cueq_config.group, irreps_in),
                cue.Irreps(cueq_config.group, irreps_out),
                layout_in=cue.ir_mul,
                layout_out=cueq_config.layout,
                contraction_degree=correlation,
                num_elements=num_elements,
                original_mace=(not use_reduced_cg),
                dtype=torch.get_default_dtype(),
                math_dtype=torch.get_default_dtype(),
            )

        return SymmetricContraction(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=use_reduced_cg,
        )


class TransposeIrrepsLayoutWrapper:
    """Wrapper around cuet.TransposeIrrepsLayout"""

    def __new__(
        cls,
        irreps: o3.Irreps,
        source: str,
        target: str,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if CUET_AVAILABLE and cueq_config is not None and cueq_config.enabled:
            # If layouts are the same, no-op
            if source == target:
                return None
            return cuet.TransposeIrrepsLayout(
                cue.Irreps(cueq_config.group, irreps),
                source=getattr(cue, source),
                target=getattr(cue, target),
                use_fallback=True,
            )

        return None
