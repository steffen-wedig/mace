"""`--readout_block`: the last-layer readout of a model built from scratch.

MACE-POLAR-1 checkpoints end in a `NonLinearBiasReadoutBlock`, while the from-scratch
path builds the constructor default `NonLinearReadoutBlock`; without this option a
model trained from scratch cannot match a foundation's architecture. Assertions are on
the model `_build_model` returns -- the call `configure_model` makes.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import modules
from mace.modules.blocks import NonLinearBiasReadoutBlock, NonLinearReadoutBlock
from mace.tools import build_default_arg_parser
from mace.tools.model_script_utils import _build_model
from mace.tools.scripts_utils import extract_config_mace_model
from mace.tools.torch_tools import default_dtype

ATOMIC_NUMBERS = [1, 8]


def arguments(model, readout_block=None, extra=()):
    argv = [
        "--name", "readout",
        "--train_file", "train.xyz",
        "--model", model,
        "--MLP_irreps", "8x0e",
        "--radial_MLP", "[16]",
        *extra,
    ]
    if readout_block is not None:
        argv += ["--readout_block", readout_block]
    args = build_default_arg_parser().parse_args(argv)
    args.std, args.mean = 1.0, 0.0
    return args


def model_config(args):
    return dict(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=2,
        interaction_cls=modules.interaction_classes[args.interaction],
        num_interactions=2,
        num_elements=len(ATOMIC_NUMBERS),
        hidden_irreps=o3.Irreps("4x0e + 4x1o"),
        edge_irreps=None,
        atomic_energies=np.zeros(len(ATOMIC_NUMBERS)),
        apply_cutoff=args.apply_cutoff,
        avg_num_neighbors=3.0,
        atomic_numbers=ATOMIC_NUMBERS,
        use_reduced_cg=args.use_reduced_cg,
        use_so3=args.use_so3,
        use_edge_irreps_first=args.use_edge_irreps_first,
        cueq_config=None,
    )


POLAR = ("--num_recursion_steps", "1", "--atomic_multipoles_max_l", "1", "--field_feature_max_l", "1")


def build(model, readout_block=None, extra=()):
    args = arguments(model, readout_block, extra)
    with default_dtype(torch.float64):
        return _build_model(args, model_config(args), None, ["Default"])


@pytest.mark.parametrize(("model", "extra"), [("ScaleShiftMACE", ()), ("PolarMACE", POLAR)])
def test_default_keeps_the_nonlinear_readout(model, extra):
    readout = build(model, extra=extra).readouts[-1]
    assert type(readout) is NonLinearReadoutBlock


@pytest.mark.parametrize(("model", "extra"), [("ScaleShiftMACE", ()), ("PolarMACE", POLAR)])
def test_bias_readout_is_built_on_request(model, extra):
    built = build(model, "NonLinearBiasReadoutBlock", extra)
    assert type(built.readouts[-1]) is NonLinearBiasReadoutBlock
    # Only the last readout changes; the earlier layers keep their linear readout.
    assert all(type(readout) is not NonLinearBiasReadoutBlock for readout in built.readouts[:-1])
    # The choice survives config extraction (what a later finetune or conversion reads).
    assert extract_config_mace_model(built)["readout_cls"] is NonLinearBiasReadoutBlock


def test_bias_readout_model_evaluates():
    built = build("ScaleShiftMACE", "NonLinearBiasReadoutBlock")
    parameters = sum(p.numel() for p in built.readouts[-1].parameters())
    reference = sum(p.numel() for p in build("ScaleShiftMACE").readouts[-1].parameters())
    assert parameters > reference  # the bias terms


def test_readout_block_with_a_foundation_is_refused():
    args = arguments("PolarMACE", "NonLinearBiasReadoutBlock", POLAR)
    config = model_config(args)
    with pytest.raises(ValueError, match="foundation model brings its own readout"):
        _build_model(args, config, config, ["Default"])


def test_readout_block_for_an_unsupported_model_is_refused():
    args = arguments("AtomicDipolesMACE", "NonLinearBiasReadoutBlock")
    with pytest.raises(ValueError, match="not supported for --model AtomicDipolesMACE"):
        _build_model(args, model_config(args), None, ["Default"])


def test_parser_rejects_an_unknown_readout_block():
    with pytest.raises(SystemExit):
        build_default_arg_parser().parse_args(
            ["--name", "x", "--train_file", "t.xyz", "--readout_block", "LinearReadoutBlock"]
        )
