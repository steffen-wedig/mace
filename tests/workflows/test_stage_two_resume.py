"""Resuming a two-stage run with `--restart_latest`, through real runs.

A full run keeps every checkpoint (`--save_all_checkpoints`); deleting the later ones and
resuming then continues from an earlier epoch. The resumed run must start with the epoch
after the checkpoint's, cross or continue the stage switch without error (it used to fail
with `KeyError: 'swa_lr'` when SWALR met an optimizer state without its key), apply the
stage-two optimizer change exactly once per run, and restore the EMA from the checkpoint.
"""

import re

import ase.io
import pytest
import torch

from tests.helpers import base_mace_params, make_fitting_configs, run_mace_train

EPOCHS = 8
STAGE_TWO = 4


def params(tmp_path, **extra):
    params = base_mace_params()
    params.update(
        {
            "name": "resume",
            "hidden_irreps": "8x0e",
            "checkpoints_dir": str(tmp_path / "checkpoints"),
            "model_dir": str(tmp_path / "model"),
            "results_dir": str(tmp_path / "results"),
            "log_dir": str(tmp_path / "logs"),
            "train_file": str(tmp_path / "fit.xyz"),
            "max_num_epochs": EPOCHS,
            "start_swa": STAGE_TWO,
            "eval_interval": 1,
            "loss": "weighted",
            "save_all_checkpoints": None,
            "swa_lr": 1e-3,
            "swa_warmup_steps": 3,
            "swa_reset_optimizer": True,
        }
    )
    params.pop("restart_latest")
    params.update(extra)
    return params


def log_text(tmp_path):
    return "\n".join(path.read_text() for path in sorted((tmp_path / "logs").glob("*.log")))


def checkpoint_path(tmp_path, epoch):
    suffix = "_swa" if epoch >= STAGE_TWO else ""
    return tmp_path / "checkpoints" / f"resume_run-5_epoch-{epoch}{suffix}.pt"


def full_run_then_truncate(tmp_path, keep_until):
    ase.io.write(tmp_path / "fit.xyz", make_fitting_configs())
    run_mace_train(params(tmp_path))
    for path in (tmp_path / "checkpoints").glob("resume_run-5_epoch-*.pt"):
        epoch = int(re.search(r"epoch-(\d+)", path.name).group(1))
        if epoch > keep_until:
            path.unlink()
    for path in (tmp_path / "logs").glob("*.log"):
        path.unlink()


@pytest.mark.parametrize(
    "keep_until, expect_reset", [(2, True), (5, False)], ids=["stage_one", "stage_two"]
)
def test_a_resumed_two_stage_run_continues_after_its_checkpoint(
    tmp_path, keep_until, expect_reset
):
    full_run_then_truncate(tmp_path, keep_until)
    saved = torch.load(checkpoint_path(tmp_path, keep_until), weights_only=False)
    assert "raw_model" in saved and saved["ema_updates"] > 0

    run_mace_train(params(tmp_path, restart_latest=None))

    log = log_text(tmp_path)
    epochs = [int(epoch) for epoch in re.findall(r"INFO: Epoch (\d+): head", log)]
    assert epochs[0] == keep_until + 1, epochs
    assert epochs[-1] == EPOCHS - 1
    # The stage-two optimizer change belongs to the switch: a run resumed inside stage two
    # already has it in its checkpoint.
    assert ("Stage two: optimizer state reset" in log) is expect_reset
    assert "raw weights" not in log  # no fallback to an EMA-only checkpoint
