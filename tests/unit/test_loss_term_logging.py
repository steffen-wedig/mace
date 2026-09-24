"""Per-term loss logging (`mace.tools.loss_terms` and its hooks in `mace.tools.train`).

The real per-term loss (`InteractionHuberLoss`) is not needed here: a small fake loss
follows the same contract (`term_names`, `compute_terms`, `last_statistics`,
`term_weights`), so these tests pin the bookkeeping, not the physics of the loss.
"""

import json
import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pytest
import torch
from e3nn import o3
from torch.optim.swa_utils import SWALR, AveragedModel

from mace import data, modules, tools
from mace.tools import torch_geometric
from mace.tools.loss_terms import (
    LossTermAccumulator,
    LossTermInstrumentation,
    LossTermJsonlWriter,
    probe_term_gradients,
    supports_loss_terms,
)
from mace.tools.train import SWAContainer, evaluate, train

TERM_NAMES = (
    "frame_energy",
    "frame_forces",
    "monomer_energy",
    "monomer_forces",
    "interaction_energy",
    "interaction_forces",
)


@dataclass
class FakeStatistics:
    weight: torch.Tensor
    unweighted_mean: torch.Tensor
    weighted_sum: torch.Tensor
    weight_sum: torch.Tensor
    entry_count: torch.Tensor
    linear_count: torch.Tensor


def statistics(weight, weighted_sum, weight_sum, entry_count, linear_count):
    as_tensor = lambda value: torch.tensor(float(value), dtype=torch.float64)  # noqa: E731
    unweighted_mean = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    return FakeStatistics(
        weight=as_tensor(weight),
        unweighted_mean=as_tensor(unweighted_mean),
        weighted_sum=as_tensor(weighted_sum),
        weight_sum=as_tensor(weight_sum),
        entry_count=as_tensor(entry_count),
        linear_count=as_tensor(linear_count),
    )


class FakeTermLoss(torch.nn.Module):
    """A contract-following loss on a MACE batch.

    frame terms: squared errors; monomer terms: absolute errors (so their gradients
    point elsewhere); interaction terms: no entries (value 0, no graph).
    """

    term_names = TERM_NAMES

    def __init__(self, weights: Dict[str, float], huber_delta: float = 0.01) -> None:
        super().__init__()
        self.weights = dict(weights)
        self.huber_delta = huber_delta
        self.last_statistics: Dict[str, FakeStatistics] = {}

    def term_weights(self) -> Dict[str, float]:
        return dict(self.weights)

    def _term(self, name, losses, errors, entry_weights):
        entry_weights = entry_weights.to(losses.dtype)
        present = entry_weights > 0
        weighted_sum = (entry_weights * losses).sum()
        weight_sum = entry_weights[present].sum()
        unweighted_mean = weighted_sum / weight_sum.clamp(min=1e-30)
        self.last_statistics[name] = FakeStatistics(
            weight=torch.tensor(self.weights[name], dtype=losses.dtype),
            unweighted_mean=unweighted_mean.detach(),
            weighted_sum=weighted_sum.detach(),
            weight_sum=weight_sum.detach(),
            entry_count=present.sum().to(losses.dtype),
            linear_count=(present & (errors.abs() > self.huber_delta))
            .sum()
            .to(losses.dtype),
        )
        return self.weights[name] * unweighted_mean

    def _empty(self, name, reference):
        zero = reference.new_zeros(())
        self.last_statistics[name] = FakeStatistics(
            weight=torch.tensor(self.weights[name], dtype=reference.dtype),
            unweighted_mean=zero,
            weighted_sum=zero,
            weight_sum=zero,
            entry_count=zero,
            linear_count=zero,
        )
        return self.weights[name] * zero

    def compute_terms(self, ref, pred):
        atom_counts = (ref.ptr[1:] - ref.ptr[:-1]).to(pred["energy"].dtype)
        energy_error = (ref.energy - pred["energy"]) / atom_counts
        force_error = ref.forces - pred["forces"]
        configuration_weights = ref.weight * ref.energy_weight
        atom_weights = torch.ones(force_error.shape[0], dtype=force_error.dtype)
        return {
            "frame_energy": self._term(
                "frame_energy", energy_error**2, energy_error, configuration_weights
            ),
            "frame_forces": self._term(
                "frame_forces",
                (force_error**2).mean(dim=-1),
                force_error.abs().amax(dim=-1),
                atom_weights,
            ),
            "monomer_energy": self._term(
                "monomer_energy",
                energy_error.abs(),
                energy_error,
                configuration_weights,
            ),
            "monomer_forces": self._term(
                "monomer_forces",
                force_error.abs().mean(dim=-1),
                force_error.abs().amax(dim=-1),
                atom_weights,
            ),
            "interaction_energy": self._empty("interaction_energy", energy_error),
            "interaction_forces": self._empty("interaction_forces", energy_error),
        }

    def forward(self, ref, pred, ddp=None):  # pylint: disable=unused-argument
        return sum(self.compute_terms(ref, pred).values())


STAGE_ONE_WEIGHTS = {
    "frame_energy": 1.0,
    "frame_forces": 100.0,
    "monomer_energy": 1.0,
    "monomer_forces": 100.0,
    "interaction_energy": 1.0,
    "interaction_forces": 100.0,
}
STAGE_TWO_WEIGHTS = {
    **STAGE_ONE_WEIGHTS,
    "frame_energy": 1000.0,
    "monomer_energy": 1000.0,
}


# Contract detection ---------------------------------------------------------------------


def test_only_losses_with_the_term_contract_are_instrumented():
    assert supports_loss_terms(FakeTermLoss(STAGE_ONE_WEIGHTS))
    assert not supports_loss_terms(modules.WeightedEnergyForcesLoss(1.0, 1.0))


# Accumulator ----------------------------------------------------------------------------


class StatisticsHolder:
    def __init__(self, table):
        self.last_statistics = table


def test_the_summary_pools_several_batches_by_hand():
    names = ("first", "second", "empty")
    accumulator = LossTermAccumulator(names)
    batches = [
        {
            "first": statistics(2.0, 6.0, 3.0, 3, 1),  # mean 2, contribution 4
            "second": statistics(10.0, 1.0, 2.0, 2, 0),  # mean 0.5, contribution 5
            "empty": statistics(5.0, 0.0, 0.0, 0, 0),
        },
        {
            "first": statistics(2.0, 1.0, 1.0, 1, 1),  # mean 1, contribution 2
            "second": statistics(10.0, 3.0, 2.0, 2, 2),  # mean 1.5, contribution 15
            "empty": statistics(5.0, 0.0, 0.0, 0, 0),
        },
    ]
    for table in batches:
        accumulator.update(StatisticsHolder(table))

    summary = accumulator.summary()
    total = (4 + 2) / 2 + (5 + 15) / 2
    assert summary["first"]["weight"] == pytest.approx(2.0)
    assert summary["first"]["mean_contribution"] == pytest.approx(3.0)
    assert summary["first"]["share_of_total"] == pytest.approx(3.0 / total)
    assert summary["first"]["pooled_unweighted_mean"] == pytest.approx(7.0 / 4.0)
    assert summary["first"]["entry_count"] == pytest.approx(4.0)
    assert summary["first"]["linear_fraction"] == pytest.approx(2.0 / 4.0)
    assert summary["first"]["batch_count"] == 2
    assert summary["second"]["mean_contribution"] == pytest.approx(10.0)
    assert summary["second"]["share_of_total"] == pytest.approx(10.0 / total)
    assert summary["second"]["pooled_unweighted_mean"] == pytest.approx(4.0 / 4.0)
    assert summary["second"]["linear_fraction"] == pytest.approx(2.0 / 4.0)
    assert summary["empty"]["mean_contribution"] == 0.0
    assert summary["empty"]["share_of_total"] == 0.0
    assert summary["empty"]["pooled_unweighted_mean"] is None
    assert summary["empty"]["linear_fraction"] is None


def test_a_summary_without_batches_is_all_null():
    summary = LossTermAccumulator(("first",)).summary()
    assert summary["first"]["mean_contribution"] is None
    assert summary["first"]["batch_count"] == 0


def test_share_of_total_is_null_when_every_term_is_zero():
    accumulator = LossTermAccumulator(("first",))
    accumulator.update(StatisticsHolder({"first": statistics(1.0, 0.0, 1.0, 1, 0)}))
    assert accumulator.summary()["first"]["share_of_total"] is None


def test_a_missing_term_is_an_error():
    accumulator = LossTermAccumulator(("first", "second"))
    with pytest.raises(KeyError, match="second"):
        accumulator.update(StatisticsHolder({"first": statistics(1, 1, 1, 1, 0)}))


def test_reset_starts_a_fresh_epoch():
    accumulator = LossTermAccumulator(("first",))
    accumulator.update(StatisticsHolder({"first": statistics(1.0, 4.0, 1.0, 1, 0)}))
    accumulator.reset()
    accumulator.update(StatisticsHolder({"first": statistics(1.0, 2.0, 1.0, 1, 0)}))
    assert accumulator.summary()["first"]["mean_contribution"] == pytest.approx(2.0)


# Gradient probe -------------------------------------------------------------------------


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, dtype=torch.float64)
        self.frozen = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        self.frozen.requires_grad_(False)

    def forward(self, data, training=False, **kwargs):  # pylint: disable=unused-argument
        return {"output": self.linear(data["x"]) * self.frozen}


class TinyBatch:
    def __init__(self, x):
        self.x = x

    def to_dict(self):
        return {"x": self.x}


class LambdaTermLoss(torch.nn.Module):
    """Terms are functions of the tiny model's output."""

    def __init__(self, functions):
        super().__init__()
        self.functions = functions
        self.term_names = tuple(functions)
        self.last_statistics = {}

    def compute_terms(self, ref, pred):  # pylint: disable=unused-argument
        return {
            name: function(pred["output"]) for name, function in self.functions.items()
        }

    def forward(self, ref, pred, ddp=None):  # pylint: disable=unused-argument
        return sum(self.compute_terms(ref, pred).values())


OUTPUT_ARGUMENTS = {"forces": False, "virials": False, "stress": False}


def tiny_setup():
    torch.manual_seed(0)
    model = TinyModel()
    batch = TinyBatch(torch.randn(5, 3, dtype=torch.float64))
    return model, batch


def flat_gradient(model, value):
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    parts = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(parameter) if part is None else part).reshape(-1)
            for parameter, part in zip(parameters, parts)
        ]
    )


def test_per_term_gradients_sum_to_the_gradient_of_the_total():
    model, batch = tiny_setup()
    loss_fn = LambdaTermLoss(
        {
            "frame_energy": lambda output: (output[:, 0] ** 2).mean(),
            "frame_forces": lambda output: 3.0 * output[:, 1].abs().mean(),
            "monomer_energy": lambda output: (output.sum(dim=1) ** 2).mean(),
        }
    )
    result = probe_term_gradients(model, loss_fn, batch, OUTPUT_ARGUMENTS)

    terms = loss_fn.compute_terms(None, model(batch.to_dict()))
    per_term = {name: flat_gradient(model, value) for name, value in terms.items()}
    total = flat_gradient(model, loss_fn(None, model(batch.to_dict())))
    assert torch.allclose(sum(per_term.values()), total)
    assert result["total_gradient_norm"] == pytest.approx(float(total.norm()))
    for name, value in terms.items():
        assert result["gradient_norms"][name] == pytest.approx(
            float(per_term[name].norm())
        )
        assert result["term_values"][name] == pytest.approx(float(value))
    assert result["trainable_parameter_count"] == 3 * 2 + 2
    assert result["wall_time_seconds"] >= 0.0


def test_cosines_of_constructed_gradients():
    model, batch = tiny_setup()
    # Gradients with respect to the linear layer: d/dW of output[:, i].sum() is a
    # row pattern, so row 0 and row 1 are orthogonal, and scaling keeps the direction.
    loss_fn = LambdaTermLoss(
        {
            "frame_energy": lambda output: output[:, 0].sum(),
            "interaction_energy": lambda output: 5.0 * output[:, 0].sum(),
            "frame_forces": lambda output: output[:, 0].sum(),
            "interaction_forces": lambda output: -2.0 * output[:, 0].sum(),
            "monomer_energy": lambda output: output[:, 1].sum(),
            "monomer_forces": lambda output: 0.0 * output[:, 1].sum(),
        }
    )
    result = probe_term_gradients(model, loss_fn, batch, OUTPUT_ARGUMENTS)
    cosines = result["cosine_similarities"]
    assert cosines["frame_energy_vs_interaction_energy"] == pytest.approx(1.0)
    assert cosines["frame_forces_vs_interaction_forces"] == pytest.approx(-1.0)
    assert cosines["frame_energy_vs_monomer_energy"] == pytest.approx(0.0, abs=1e-12)
    assert cosines["frame_forces_vs_monomer_forces"] is None  # zero term, zero gradient
    assert result["gradient_norms"]["monomer_forces"] == 0.0


def test_pairs_with_an_absent_term_are_skipped():
    model, batch = tiny_setup()
    loss_fn = LambdaTermLoss({"frame_energy": lambda output: output.sum()})
    result = probe_term_gradients(model, loss_fn, batch, OUTPUT_ARGUMENTS)
    assert result["cosine_similarities"] == {}


def test_the_probe_leaves_parameter_gradients_and_mode_alone():
    model, batch = tiny_setup()
    loss_fn = LambdaTermLoss({"frame_energy": lambda output: (output**2).sum()})
    model.eval()
    probe_term_gradients(model, loss_fn, batch, OUTPUT_ARGUMENTS)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.training

    model.train()
    model(batch.to_dict())["output"].sum().backward()
    before = [parameter.grad.clone() for parameter in model.linear.parameters()]
    probe_term_gradients(model, loss_fn, batch, OUTPUT_ARGUMENTS)
    after = [parameter.grad for parameter in model.linear.parameters()]
    assert all(torch.equal(first, second) for first, second in zip(before, after))
    assert model.training


def test_the_probe_does_not_advance_the_random_number_generator():
    model, batch = tiny_setup()
    loss_fn = LambdaTermLoss(
        {"frame_energy": lambda output: (output * torch.rand_like(output)).sum()}
    )
    torch.manual_seed(1)
    expected = torch.rand(3)
    torch.manual_seed(1)
    probe_term_gradients(model, loss_fn, batch, OUTPUT_ARGUMENTS)
    assert torch.equal(torch.rand(3), expected)


# JSONL writer ---------------------------------------------------------------------------


def test_the_writer_writes_one_valid_json_object_per_line(tmp_path):
    path = tmp_path / "nested" / "run_loss_terms.jsonl"
    writer = LossTermJsonlWriter(str(path))
    writer.write({"record": "first", "value": 1.5, "nested": {"list": [1.0, 2.0]}})
    # Readable before anything is closed: every record is flushed.
    assert json.loads(path.read_text().splitlines()[0])["value"] == 1.5
    writer.write(
        {
            "record": "second",
            "not_a_number": math.nan,
            "nested": {"infinite": math.inf, "list": [-math.inf, 3.0]},
        }
    )
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    second = json.loads(lines[1])
    assert second["not_a_number"] is None
    assert second["nested"] == {"infinite": None, "list": [None, 3.0]}


def test_a_record_needs_a_record_field(tmp_path):
    writer = LossTermJsonlWriter(str(tmp_path / "log.jsonl"))
    with pytest.raises(ValueError, match="record"):
        writer.write({"value": 1.0})


def test_a_negative_probe_interval_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="probe_interval"):
        LossTermInstrumentation(
            LossTermJsonlWriter(str(tmp_path / "log.jsonl")),
            TERM_NAMES,
            probe_interval=-1,
            device=torch.device("cpu"),
        )


# Integration with `mace.tools.train` ----------------------------------------------------


@pytest.fixture(name="float64")
def fixture_float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def water_configuration(shift: float, energy: float) -> data.Configuration:
    return data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array([[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        + shift,
        properties={
            "energy": energy,
            "forces": np.array([[0.0, -1.3, 0.0], [1.0, 0.2, 0.0], [0.0, 1.1, 0.3]])
            * (1.0 + shift),
        },
        property_weights={"energy": 1.0, "forces": 1.0},
    )


def build_model():
    torch.manual_seed(0)
    table = tools.AtomicNumberTable([1, 8])
    return modules.MACE(
        r_max=5,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=1,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=1,
        num_elements=2,
        hidden_irreps=o3.Irreps("4x0e"),
        MLP_irreps=o3.Irreps("4x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([1.0, 3.0]),
        avg_num_neighbors=3,
        atomic_numbers=table.zs,
        correlation=2,
        radial_type="bessel",
    )


def build_loaders():
    table = tools.AtomicNumberTable([1, 8])
    graphs = [
        data.AtomicData.from_config(
            water_configuration(shift, energy), z_table=table, cutoff=5.0
        )
        for shift, energy in [(0.0, -1.5), (0.1, -1.2), (0.2, -1.8), (0.3, -1.0)]
    ]
    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=graphs,
        batch_size=2,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(3),
    )
    valid_loader = torch_geometric.dataloader.DataLoader(
        dataset=graphs[:3], batch_size=2, shuffle=False, drop_last=False
    )
    return train_loader, {"Default": valid_loader}


def run_training(
    tmp_path, instrumentation_interval, max_num_epochs=3, loss_factory=FakeTermLoss
):
    """Two optimizer steps per epoch; stage two from epoch 1."""
    model = build_model()
    train_loader, valid_loaders = build_loaders()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    stage_one_loss = loss_factory(STAGE_ONE_WEIGHTS)
    stage_two_loss = loss_factory(STAGE_TWO_WEIGHTS)
    swa = SWAContainer(
        model=AveragedModel(model),
        scheduler=SWALR(optimizer, swa_lr=1e-3),
        start=1,
        loss_fn=stage_two_loss,
    )
    instrumentation = None
    if instrumentation_interval is not None:
        instrumentation = LossTermInstrumentation.for_run(
            results_dir=str(tmp_path / "results"),
            tag="tiny_run",
            loss_fn=stage_one_loss,
            probe_interval=instrumentation_interval,
            device=torch.device("cpu"),
        )
    train(
        model=model,
        loss_fn=stage_one_loss,
        train_loader=train_loader,
        valid_loaders=valid_loaders,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        start_epoch=0,
        max_num_epochs=max_num_epochs,
        patience=100,
        checkpoint_handler=tools.CheckpointHandler(
            directory=str(tmp_path / "checkpoints"), tag="tiny_run"
        ),
        logger=tools.MetricsLogger(
            directory=str(tmp_path / "results"), tag="tiny_run_train"
        ),
        eval_interval=1,
        output_args={"forces": True, "virials": False, "stress": False},
        device=torch.device("cpu"),
        log_errors="PerAtomRMSE",
        swa=swa,
        loss_term_instrumentation=instrumentation,
    )
    return model


def read_records(tmp_path):
    path = tmp_path / "results" / "tiny_run_loss_terms.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_training_writes_every_record_type(tmp_path, float64):  # pylint: disable=unused-argument
    run_training(tmp_path, instrumentation_interval=2)
    records = read_records(tmp_path)
    kinds = [record["record"] for record in records]

    assert kinds[0] == "run_start"
    assert records[0]["term_names"] == list(TERM_NAMES)
    assert records[0]["probe_interval"] == 2

    train_summaries = [
        record
        for record in records
        if record["record"] == "term_summary" and record["split"] == "train"
    ]
    valid_summaries = [
        record
        for record in records
        if record["record"] == "term_summary" and record["split"] == "valid"
    ]
    assert [record["epoch"] for record in train_summaries] == [0, 1, 2]
    assert [record["stage"] for record in train_summaries] == ["one", "two", "two"]
    assert [record["epoch"] for record in valid_summaries] == [None, 0, 1, 2]
    assert {record["head"] for record in valid_summaries} == {"Default"}
    for summary in train_summaries:
        terms = summary["terms"]
        assert terms["frame_forces"]["batch_count"] == 2
        assert terms["frame_forces"]["entry_count"] == 12.0  # 4 waters, 3 atoms
        assert terms["interaction_energy"]["pooled_unweighted_mean"] is None
        shares = [term["share_of_total"] for term in terms.values()]
        assert sum(shares) == pytest.approx(1.0)
    assert valid_summaries[0]["terms"]["frame_energy"]["entry_count"] == 3.0
    assert train_summaries[0]["terms"]["frame_energy"]["weight"] == 1.0
    assert train_summaries[1]["terms"]["frame_energy"]["weight"] == 1000.0

    switches = [record for record in records if record["record"] == "stage_switch"]
    assert len(switches) == 1
    assert switches[0]["epoch"] == 1
    assert switches[0]["old_weights"] == STAGE_ONE_WEIGHTS
    assert switches[0]["new_weights"] == STAGE_TWO_WEIGHTS

    probes = [record for record in records if record["record"] == "gradient_probe"]
    assert [probe["step"] for probe in probes] == [0, 2, 4]
    assert [probe["stage"] for probe in probes] == ["one", "two", "two"]
    for probe in probes:
        assert probe["gradient_norms"]["frame_energy"] > 0.0
        assert probe["gradient_norms"]["interaction_energy"] == 0.0
        assert (
            probe["cosine_similarities"]["frame_energy_vs_interaction_energy"] is None
        )
        cosine = probe["cosine_similarities"]["frame_energy_vs_monomer_energy"]
        assert abs(cosine) <= 1.0 + 1e-9
        assert probe["wall_time_seconds"] >= 0.0


def test_the_mean_contribution_is_what_the_optimizer_saw(tmp_path, float64):  # pylint: disable=unused-argument
    """The epoch's summed mean contributions equal the mean logged training loss."""
    run_training(tmp_path, instrumentation_interval=0, max_num_epochs=1)
    records = read_records(tmp_path)
    assert not [record for record in records if record["record"] == "gradient_probe"]
    summary = next(
        record
        for record in records
        if record["record"] == "term_summary" and record["split"] == "train"
    )
    steps = [
        json.loads(line)
        for line in (tmp_path / "results" / "tiny_run_train.txt")
        .read_text()
        .splitlines()
    ]
    step_losses = [float(step["loss"]) for step in steps if step["mode"] == "opt"]
    total = sum(term["mean_contribution"] for term in summary["terms"].values())
    assert total == pytest.approx(np.mean(step_losses))


def test_instrumentation_does_not_change_the_trajectory(tmp_path, float64):  # pylint: disable=unused-argument
    plain = run_training(tmp_path / "plain", instrumentation_interval=None)
    probed = run_training(tmp_path / "probed", instrumentation_interval=1)
    for first, second in zip(plain.parameters(), probed.parameters()):
        assert torch.equal(first, second)


def test_evaluate_fills_a_term_accumulator(float64):  # pylint: disable=unused-argument
    model = build_model()
    _, valid_loaders = build_loaders()
    loss_fn = FakeTermLoss(STAGE_ONE_WEIGHTS)
    accumulator = LossTermAccumulator(TERM_NAMES)
    valid_loss, _ = evaluate(
        model=model,
        loss_fn=loss_fn,
        data_loader=valid_loaders["Default"],
        output_args={"forces": True, "virials": False, "stress": False},
        device=torch.device("cpu"),
        term_accumulator=accumulator,
    )
    summary = accumulator.summary()
    assert summary["frame_energy"]["batch_count"] == 2
    assert summary["frame_forces"]["entry_count"] == 9.0
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert np.isfinite(valid_loss)


def interaction_huber_loss(weights):
    return modules.InteractionHuberLoss(
        energy_weight=weights["frame_energy"],
        forces_weight=weights["frame_forces"],
        monomer_energy_weight=weights["monomer_energy"],
        monomer_forces_weight=weights["monomer_forces"],
        interaction_energy_weight=weights["interaction_energy"],
        interaction_forces_weight=weights["interaction_forces"],
    )


@pytest.mark.skipif(
    not hasattr(modules, "InteractionHuberLoss"), reason="needs InteractionHuberLoss"
)
def test_training_with_the_interaction_huber_loss(tmp_path, float64):  # pylint: disable=unused-argument
    """The real contract loss on extxyz-style batches: aperiodic waters are monomers."""
    run_training(
        tmp_path, instrumentation_interval=2, loss_factory=interaction_huber_loss
    )
    records = read_records(tmp_path)
    assert records[0]["term_names"] == list(TERM_NAMES)
    train_summary = next(
        record
        for record in records
        if record["record"] == "term_summary" and record["split"] == "train"
    )
    terms = train_summary["terms"]
    assert terms["monomer_energy"]["entry_count"] == 4.0
    assert terms["monomer_forces"]["entry_count"] == 36.0  # 12 atoms, 3 components
    assert terms["frame_energy"]["entry_count"] == 0.0
    assert terms["interaction_energy"]["pooled_unweighted_mean"] is None
    switch = next(record for record in records if record["record"] == "stage_switch")
    assert switch["new_weights"]["frame_energy"] == 1000.0
    probe = next(record for record in records if record["record"] == "gradient_probe")
    assert probe["gradient_norms"]["monomer_forces"] > 0.0
    assert probe["gradient_norms"]["frame_energy"] == 0.0
    assert probe["cosine_similarities"]["frame_forces_vs_monomer_forces"] is None
