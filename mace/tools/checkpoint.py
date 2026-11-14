###########################################################################################
# Checkpointing
# Authors: Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from .torch_tools import TensorDict

Checkpoint = Dict[str, TensorDict]


@dataclasses.dataclass
class CheckpointState:
    """What a checkpoint saves and restores.

    With an ``ema`` (torch_ema ExponentialMovingAverage over ``model.parameters()``), a
    checkpoint holds the EMA weights as ``model`` (what the exports load) and the raw
    weights as ``raw_model``, plus the EMA's update count, so that a resumed run continues
    both the raw trajectory and the average. Without one, ``model`` is the raw weights.
    Pass the state with the raw weights in the model (not inside ``average_parameters``).
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR
    ema: Optional[Any] = None


# How an optimizer executes, not what it has learnt: taken from the optimizer being
# restored, never from the checkpoint (a checkpoint written by a foreach run restores
# into a fused one and the reverse, e.g. on another device).
OPTIMIZER_EXECUTION_KEYS = ("fused", "foreach", "capturable", "differentiable")


def optimizer_state_for(
    optimizer: torch.optim.Optimizer, saved_state: Dict
) -> Dict:
    """``saved_state`` with the execution keys of ``optimizer``'s parameter groups."""
    if len(saved_state["param_groups"]) != len(optimizer.param_groups):
        return saved_state  # load_state_dict reports the mismatch
    param_groups = []
    for saved_group, current_group in zip(
        saved_state["param_groups"], optimizer.param_groups
    ):
        group = dict(saved_group)
        for key in OPTIMIZER_EXECUTION_KEYS:
            if key in current_group:
                group[key] = current_group[key]
        param_groups.append(group)
    return {**saved_state, "param_groups": param_groups}


def _copied_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def reshape_state_to_parameters(optimizer: torch.optim.Optimizer) -> None:
    """Give restored per-element state (Adam's moments) its parameter's shape.

    A parameter stored with another shape of the same size (e3nn ``(n,)`` against cueq
    ``(1, n)`` linears across fork versions) loads into the model, whose load hooks reshape
    it, but the optimizer restores its moments as saved; the fused Adam kernel then refuses
    the mismatch. A size mismatch is left alone (the optimizer reports it on its step).
    """
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter_state = optimizer.state.get(parameter)
            if not parameter_state:
                continue
            for key, value in parameter_state.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.dim() > 0
                    and value.shape != parameter.shape
                    and value.numel() == parameter.numel()
                ):
                    parameter_state[key] = value.reshape(parameter.shape)


class CheckpointBuilder:
    @staticmethod
    def create_checkpoint(state: CheckpointState) -> Checkpoint:
        checkpoint: Checkpoint = {
            "optimizer": state.optimizer.state_dict(),
            "lr_scheduler": state.lr_scheduler.state_dict(),
        }
        if state.ema is None:
            checkpoint["model"] = state.model.state_dict()
            return checkpoint
        # Copies: average_parameters swaps the parameter values in place.
        checkpoint["raw_model"] = _copied_state(state.model)
        with state.ema.average_parameters():
            checkpoint["model"] = _copied_state(state.model)
        checkpoint["ema_updates"] = state.ema.num_updates
        return checkpoint

    @staticmethod
    def load_checkpoint(
        state: CheckpointState, checkpoint: Checkpoint, strict: bool
    ) -> None:
        """Without ``state.ema`` the model gets ``model`` (the EMA weights of a checkpoint
        written with an EMA: what exports want). With it, the model gets the raw weights
        and the EMA the averaged ones; a checkpoint without raw weights (written before
        they were saved) restarts the EMA from its weights."""
        state.model.load_state_dict(checkpoint["model"], strict=strict)  # type: ignore
        if state.ema is not None:
            with torch.no_grad():
                for shadow, parameter in zip(
                    state.ema.shadow_params, state.model.parameters()
                ):
                    shadow.copy_(parameter.detach())
            if "raw_model" in checkpoint:
                state.model.load_state_dict(checkpoint["raw_model"], strict=strict)  # type: ignore
                state.ema.num_updates = checkpoint["ema_updates"]
            else:
                logging.warning(
                    "Checkpoint holds no raw weights (written before they were saved): "
                    "training resumes from its EMA weights, the EMA restarts from them"
                )
        try:
            state.optimizer.load_state_dict(
                optimizer_state_for(state.optimizer, checkpoint["optimizer"])
            )
            reshape_state_to_parameters(state.optimizer)
        except ValueError as exc:
            logging.warning(
                "Restored model weights but could not restore optimizer/scheduler "
                f"state from checkpoint ({exc}). Resuming at the checkpoint's epoch "
                "with a freshly initialized optimizer. This is expected if the "
                "optimizer parameter groups changed since the checkpoint was written."
            )
        restore_scheduler_state(state.lr_scheduler, checkpoint["lr_scheduler"])


def restore_scheduler_state(scheduler, scheduler_state: Dict[str, Any]) -> None:
    """Restore the epoch scheduler's state unless the checkpoint's came from another kind.

    A run may resume with another ``--scheduler`` than the one that wrote the checkpoint
    (e.g. ``Piecewise``, whose placeholder ``LambdaLR`` holds no state that matters: the
    per-step schedule is a function of the epoch). Another kind's state cannot be loaded
    (``LambdaLR`` needs ``lr_lambdas``, ``ReduceLROnPlateau`` has none), so it is dropped,
    recognised by its keys.
    """
    expected_keys = set(scheduler.state_dict())
    if expected_keys != set(scheduler_state):
        logging.warning(
            "Checkpoint scheduler state does not match the scheduler of this run "
            "(--scheduler changed since it was written): not restored"
        )
        return
    scheduler.load_state_dict(scheduler_state)


@dataclasses.dataclass
class CheckpointPathInfo:
    path: str
    tag: str
    epochs: int
    swa: bool


class CheckpointIO:
    def __init__(
        self, directory: str, tag: str, keep: bool = False, swa_start: int = None
    ) -> None:
        self.directory = directory
        self.tag = tag
        self.keep = keep
        self.old_path: Optional[str] = None
        self.swa_start = swa_start

        self._epochs_string = "_epoch-"
        self._filename_extension = "pt"

    def _get_checkpoint_filename(self, epochs: int, swa_start=None) -> str:
        if swa_start is not None and epochs >= swa_start:
            return (
                self.tag
                + self._epochs_string
                + str(epochs)
                + "_swa"
                + "."
                + self._filename_extension
            )
        return (
            self.tag
            + self._epochs_string
            + str(epochs)
            + "."
            + self._filename_extension
        )

    def _list_file_paths(self) -> List[str]:
        if not os.path.isdir(self.directory):
            return []
        all_paths = [
            os.path.join(self.directory, f) for f in os.listdir(self.directory)
        ]
        return [path for path in all_paths if os.path.isfile(path)]

    def _parse_checkpoint_path(self, path: str) -> Optional[CheckpointPathInfo]:
        filename = os.path.basename(path)
        regex = re.compile(
            rf"^(?P<tag>.+){self._epochs_string}(?P<epochs>\d+)\.{self._filename_extension}$"
        )
        regex2 = re.compile(
            rf"^(?P<tag>.+){self._epochs_string}(?P<epochs>\d+)_swa\.{self._filename_extension}$"
        )
        match = regex.match(filename)
        match2 = regex2.match(filename)
        swa = False
        if not match:
            if not match2:
                return None
            match = match2
            swa = True

        return CheckpointPathInfo(
            path=path,
            tag=match.group("tag"),
            epochs=int(match.group("epochs")),
            swa=swa,
        )

    def _get_latest_checkpoint_path(self, swa) -> Optional[str]:
        all_file_paths = self._list_file_paths()
        checkpoint_info_list = [
            self._parse_checkpoint_path(path) for path in all_file_paths
        ]
        selected_checkpoint_info_list = [
            info for info in checkpoint_info_list if info and info.tag == self.tag
        ]

        if len(selected_checkpoint_info_list) == 0:
            logging.warning(
                f"Cannot find checkpoint with tag '{self.tag}' in '{self.directory}'"
            )
            return None

        candidates = [info for info in selected_checkpoint_info_list if info.swa == bool(swa)]
        if not candidates:
            # Asked for a stage-two checkpoint where there is none (or the reverse): the
            # caller falls back to the other kind.
            logging.info(
                f"No {'stage-two' if swa else 'stage-one'} checkpoint with tag '{self.tag}'"
            )
            return None
        latest_checkpoint_info = max(candidates, key=lambda info: info.epochs)
        return latest_checkpoint_info.path

    def save(
        self, checkpoint: Checkpoint, epochs: int, keep_last: bool = False
    ) -> None:
        if not self.keep and self.old_path and not keep_last:
            logging.debug(f"Deleting old checkpoint file: {self.old_path}")
            os.remove(self.old_path)

        filename = self._get_checkpoint_filename(epochs, self.swa_start)
        path = os.path.join(self.directory, filename)
        logging.debug(f"Saving checkpoint: {path}")
        os.makedirs(self.directory, exist_ok=True)
        torch.save(obj=checkpoint, f=path)
        self.old_path = path

    def load_latest(
        self, swa: Optional[bool] = False, device: Optional[torch.device] = None
    ) -> Optional[Tuple[Checkpoint, int]]:
        path = self._get_latest_checkpoint_path(swa=swa)
        if path is None:
            return None

        return self.load(path, device=device)

    def load(
        self, path: str, device: Optional[torch.device] = None
    ) -> Tuple[Checkpoint, int]:
        checkpoint_info = self._parse_checkpoint_path(path)

        if checkpoint_info is None:
            raise RuntimeError(f"Cannot find path '{path}'")

        logging.info(f"Loading checkpoint: {checkpoint_info.path}")
        return (
            torch.load(f=checkpoint_info.path, map_location=device),
            checkpoint_info.epochs,
        )


class CheckpointHandler:
    def __init__(self, *args, **kwargs) -> None:
        self.io = CheckpointIO(*args, **kwargs)
        self.builder = CheckpointBuilder()

    def save(
        self, state: CheckpointState, epochs: int, keep_last: bool = False
    ) -> None:
        checkpoint = self.builder.create_checkpoint(state)
        self.io.save(checkpoint, epochs, keep_last)

    def load_latest(
        self,
        state: CheckpointState,
        swa: Optional[bool] = False,
        device: Optional[torch.device] = None,
        strict=False,
    ) -> Optional[int]:
        result = self.io.load_latest(swa=swa, device=device)
        if result is None:
            return None

        checkpoint, epochs = result
        self.builder.load_checkpoint(state=state, checkpoint=checkpoint, strict=strict)
        return epochs

    def load(
        self,
        state: CheckpointState,
        path: str,
        strict=False,
        device: Optional[torch.device] = None,
    ) -> int:
        checkpoint, epochs = self.io.load(path, device=device)
        self.builder.load_checkpoint(state=state, checkpoint=checkpoint, strict=strict)
        return epochs
