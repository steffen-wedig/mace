"""Warm-start a MultiLevelScaleShiftMACE from a trained single-head model.

The residual multi-level architecture claims that the base level shapes the
shared trunk and the correction levels read it out. Training everything
from scratch with a strongly weighted sparse correction level inverts that:
the random trunk is pulled to serve the small expensive-level set while the
base stalls. Warm-starting takes the claim literally -- the trunk and the
base readout path are copied from an already CONVERGED single-head base
model, the delta paths start fresh (zero-initialised deltas then start
every level exactly at the base prediction), and training only has to fit
the corrections.

Mapping:

* trunk (embeddings, radial, interactions, products): copied one-to-one by
  state-dict name -- the residual model shares its trunk structure with the
  stock single-head model;
* per-layer linear readouts: the single-head ``linear`` becomes the
  residual block's ``base_linear``;
* the final non-linear readout: ``linear_1`` maps onto ``linear_1`` (shared
  hidden layer of the base path) and ``linear_2`` onto ``base_linear``;
* per-head buffers (E0 table, scale/shift) are NOT copied: the multi-level
  model's own buffers come from the export and must already agree with the
  base model's on the base head -- checked, not overwritten.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from mace.modules.blocks import (
    ResidualLevelLinearReadoutBlock,
    ResidualLevelNonLinearReadoutBlock,
)


class WarmStartError(RuntimeError):
    """Raised when the base model cannot be mapped onto the target model."""


def _copy_named_tensor(
    target_state: Dict[str, torch.Tensor],
    target_name: str,
    source_state: Dict[str, torch.Tensor],
    source_name: str,
) -> None:
    if source_name not in source_state:
        raise WarmStartError(f"base model has no parameter {source_name!r}")
    if target_name not in target_state:
        raise WarmStartError(f"target model has no parameter {target_name!r}")
    source = source_state[source_name]
    target = target_state[target_name]
    if source.shape != target.shape:
        raise WarmStartError(
            f"shape mismatch for {source_name!r} -> {target_name!r}: "
            f"{tuple(source.shape)} vs {tuple(target.shape)}; the base model "
            "and the multi-level model must share their trunk hyperparameters"
        )
    with torch.no_grad():
        target.copy_(source.to(target.dtype))


def _readout_name_pairs(model: torch.nn.Module) -> List[Tuple[str, str]]:
    """(source name, target name) pairs for every readout parameter."""
    pairs: List[Tuple[str, str]] = []
    for index, readout in enumerate(model.readouts):
        prefix = f"readouts.{index}."
        if isinstance(readout, ResidualLevelLinearReadoutBlock):
            for parameter_name, _ in readout.base_linear.named_parameters():
                pairs.append(
                    (
                        prefix + "linear." + parameter_name,
                        prefix + "base_linear." + parameter_name,
                    )
                )
        elif isinstance(readout, ResidualLevelNonLinearReadoutBlock):
            for parameter_name, _ in readout.linear_1.named_parameters():
                pairs.append(
                    (
                        prefix + "linear_1." + parameter_name,
                        prefix + "linear_1." + parameter_name,
                    )
                )
            for parameter_name, _ in readout.base_linear.named_parameters():
                pairs.append(
                    (
                        prefix + "linear_2." + parameter_name,
                        prefix + "base_linear." + parameter_name,
                    )
                )
        else:
            raise WarmStartError(
                f"readout {index} is a {type(readout).__name__}, not a "
                "residual multi-level readout; warm-starting expects a "
                "MultiLevelScaleShiftMACE"
            )
    return pairs


def initialise_from_base_model(
    model: torch.nn.Module,
    base_model_path: Path,
    map_location: str = "cpu",
) -> Dict[str, int]:
    """Copy the trained base model's trunk and base readout path into ``model``.

    Returns counts of copied and kept (delta/own) parameters, for logging.
    The base model must be a saved single-head model whose trunk
    hyperparameters match; per-head buffers are checked for agreement on
    the base head rather than copied.
    """
    base_model = torch.load(
        Path(base_model_path), map_location=map_location, weights_only=False
    )
    source_state = base_model.state_dict()
    target_state = model.state_dict()

    readout_pairs = _readout_name_pairs(model)

    copied = 0
    for source_name, source_tensor in source_state.items():
        if source_name.startswith("readouts."):
            continue
        if source_name not in target_state:
            continue  # per-head buffers handled below; anything else is structural
        if target_state[source_name].shape != source_tensor.shape:
            # Per-head buffers (E0 table, scale/shift) legitimately differ in
            # shape; everything trainable must match.
            if any(
                parameter_name == source_name
                for parameter_name, _ in base_model.named_parameters()
            ):
                raise WarmStartError(
                    f"trainable parameter {source_name!r} has shape "
                    f"{tuple(source_tensor.shape)} in the base model but "
                    f"{tuple(target_state[source_name].shape)} in the target"
                )
            continue
        _copy_named_tensor(target_state, source_name, source_state, source_name)
        copied += 1
    for source_name, target_name in readout_pairs:
        _copy_named_tensor(target_state, target_name, source_state, source_name)
        copied += 1

    kept = len(target_state) - copied

    _check_base_head_buffers(model, base_model)
    logging.info(
        "Warm start from %s: %d tensors copied (trunk + base readout path), "
        "delta paths and per-head buffers kept",
        base_model_path,
        copied,
    )
    return {"copied": copied, "kept": kept}


def _check_base_head_buffers(
    model: torch.nn.Module, base_model: torch.nn.Module
) -> None:
    """The base head's E0s and scale/shift must agree with the base model.

    They are set from the export's statistics on both sides, so any
    disagreement means the models were trained against different exports --
    the warm start would then silently change the base prediction.
    """
    base_e0s = base_model.atomic_energies_fn.atomic_energies.flatten()
    target_e0s = model.atomic_energies_fn.atomic_energies
    if target_e0s.dim() == 1:
        target_base_e0s = target_e0s
    else:
        target_base_e0s = target_e0s[0]
    if not torch.allclose(
        base_e0s.to(target_base_e0s.dtype), target_base_e0s, atol=1e-8
    ):
        raise WarmStartError(
            "the base head's atomic reference energies differ from the base "
            "model's; both must come from the same export"
        )
    for label in ("scale", "shift"):
        source_value = float(getattr(base_model.scale_shift, label).flatten()[0])
        target_value = float(getattr(model.scale_shift, label).flatten()[0])
        if abs(source_value - target_value) > 1e-8:
            raise WarmStartError(
                f"the base head's {label} ({target_value:.6f}) differs from "
                f"the base model's ({source_value:.6f}); both must come from "
                "the same export"
            )


def freeze_non_delta_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Freeze the trunk and base readout path; only delta parameters train.

    A parameter belongs to a delta path exactly when its name contains
    ``"delta"`` (the residual readout blocks name their correction paths
    ``delta_linear`` / ``delta_readouts``). Returns counts for logging.
    """
    frozen = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        if "delta" in name:
            parameter.requires_grad_(True)
            trainable += 1
        else:
            parameter.requires_grad_(False)
            frozen += 1
    if trainable == 0:
        raise WarmStartError(
            "freezing left no trainable parameters; the model has no delta "
            "readout paths"
        )
    logging.info(
        "Froze %d non-delta parameters; %d delta parameters stay trainable",
        frozen,
        trainable,
    )
    return {"frozen": frozen, "trainable": trainable}
