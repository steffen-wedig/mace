###########################################################################################
# Per-term loss instrumentation: epoch summaries, gradient probe and a JSONL writer
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
"""Per-term bookkeeping for losses that expose their terms.

A loss takes part when it follows the term contract (duck typed, see
`supports_loss_terms`):

- ``term_names``: the names of its terms, in a fixed order;
- ``compute_terms(ref, pred)``: a dict term name -> ``weight * unweighted_mean``, with
  the autograd graph attached, whose values sum to ``forward(ref, pred)``;
- ``last_statistics``: a dict term name -> statistics of the most recent
  ``compute_terms``/``forward`` call (detached scalar tensors ``weight``,
  ``unweighted_mean``, ``weighted_sum``, ``weight_sum``, ``entry_count``,
  ``linear_count``);
- ``term_weights()``: a dict term name -> the global weight of that term.

Every other loss runs exactly as before and nothing is written.

The records go to ``<results_dir>/<tag>_loss_terms.jsonl``, one JSON object per line:

- ``run_start``: tag, term names, probe interval;
- ``term_summary``: per epoch for the train split, per validation evaluation and head
  for the valid split;
- ``gradient_probe``: per-term gradient norms and cosine similarities on one fixed
  training batch every ``probe_interval`` optimizer steps;
- ``stage_switch``: the old and new term weights at the start of stage two.

Summaries accumulate on the device and move to the host once per summary. The probe
uses ``torch.autograd.grad``, so it never touches ``param.grad``, the optimizer, the EMA
or the learning-rate scheduler, and it forks the random number generator state so the
training trajectory is the same with or without it.
"""

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

STATISTIC_FIELDS: Tuple[str, ...] = (
    "weight",
    "contribution",
    "weighted_sum",
    "weight_sum",
    "entry_count",
    "linear_count",
)

# Term pairs whose gradient directions the probe compares.
COSINE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("frame_energy", "interaction_energy"),
    ("frame_forces", "interaction_forces"),
    ("frame_energy", "monomer_energy"),
    ("frame_forces", "monomer_forces"),
)


def supports_loss_terms(loss_fn: Any) -> bool:
    """Whether a loss follows the term contract described in the module docstring."""
    return all(
        hasattr(loss_fn, attribute)
        for attribute in ("compute_terms", "last_statistics", "term_names")
    )


def training_forward_kwargs(output_args: Dict[str, bool]) -> Dict[str, bool]:
    """The model keyword arguments of a training step (shared with `take_step`)."""
    kwargs = dict(
        training=True,
        compute_force=output_args["forces"],
        compute_virials=output_args["virials"],
        compute_stress=output_args["stress"],
    )
    if output_args.get("magforces", False):
        kwargs["compute_magforces"] = True
    return kwargs


def _finite_or_none(value: Any) -> Any:
    """Replace non-finite floats by None, recursively through dicts and lists."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    return value


def _ratio_or_none(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0.0 or not math.isfinite(denominator):
        return None
    return numerator / denominator


class LossTermJsonlWriter:
    """Appends one JSON object per line and flushes after every record.

    Non-finite floats are written as null, so a diverging run still logs.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        if "record" not in record:
            raise ValueError(f"a loss-term record needs a 'record' field: {record}")
        line = json.dumps(_finite_or_none(record), allow_nan=False)
        with open(self.path, mode="a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


class LossTermAccumulator:
    """Sums the per-term statistics of many loss calls on the device.

    `update` only adds tensors (no host transfer); `summary` moves everything to the
    host in one transfer.
    """

    def __init__(self, term_names: Sequence[str]) -> None:
        self.term_names = tuple(term_names)
        self.reset()

    def reset(self) -> None:
        self.sums: Optional[torch.Tensor] = None  # [term, statistic field]
        self.batch_count = 0

    def update(self, loss_fn: Any) -> None:
        statistics = loss_fn.last_statistics
        missing = [name for name in self.term_names if name not in statistics]
        if missing:
            raise KeyError(
                f"the loss did not report statistics for the terms {missing}; "
                f"reported: {sorted(statistics)}"
            )
        reference = statistics[self.term_names[0]].unweighted_mean
        rows = []
        for name in self.term_names:
            term = statistics[name]
            weight = torch.as_tensor(
                term.weight, dtype=reference.dtype, device=reference.device
            ).detach()
            unweighted_mean = term.unweighted_mean.detach()
            rows.append(
                torch.stack(
                    [
                        weight,
                        weight * unweighted_mean,
                        term.weighted_sum.detach().to(reference.dtype),
                        term.weight_sum.detach().to(reference.dtype),
                        term.entry_count.detach().to(reference.dtype),
                        term.linear_count.detach().to(reference.dtype),
                    ]
                )
            )
        batch_sums = torch.stack(rows)
        self.sums = batch_sums if self.sums is None else self.sums + batch_sums
        self.batch_count += 1

    def summary(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Per-term summary; one host transfer."""
        if self.sums is None or self.batch_count == 0:
            return {
                name: {
                    "weight": None,
                    "mean_contribution": None,
                    "share_of_total": None,
                    "pooled_unweighted_mean": None,
                    "entry_count": 0.0,
                    "linear_fraction": None,
                    "batch_count": 0,
                }
                for name in self.term_names
            }
        table: List[List[float]] = self.sums.cpu().tolist()
        column = {field: index for index, field in enumerate(STATISTIC_FIELDS)}
        mean_contributions = [
            row[column["contribution"]] / self.batch_count for row in table
        ]
        total_contribution = sum(mean_contributions)
        result = {}
        for name, row, mean_contribution in zip(
            self.term_names, table, mean_contributions
        ):
            entry_count = row[column["entry_count"]]
            result[name] = {
                "weight": row[column["weight"]] / self.batch_count,
                "mean_contribution": mean_contribution,
                "share_of_total": _ratio_or_none(mean_contribution, total_contribution),
                "pooled_unweighted_mean": _ratio_or_none(
                    row[column["weighted_sum"]], row[column["weight_sum"]]
                ),
                "entry_count": entry_count,
                "linear_fraction": _ratio_or_none(
                    row[column["linear_count"]], entry_count
                ),
                "batch_count": self.batch_count,
            }
        return result


def _pair_key(first: str, second: str) -> str:
    return f"{first}_vs_{second}"


def probe_term_gradients(
    model: torch.nn.Module,
    loss_fn: Any,
    batch: Any,
    output_args: Dict[str, bool],
    cosine_pairs: Sequence[Tuple[str, str]] = COSINE_PAIRS,
) -> Dict[str, Any]:
    """Per-term gradients of the loss on one batch with respect to the trainable parameters.

    The forward pass is the training forward of `take_step`. Gradients come from
    `torch.autograd.grad`, so `param.grad` is left alone; the model's train/eval mode
    and the random number generator state are restored afterwards.
    """
    start_time = time.perf_counter()
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("the gradient probe found no trainable parameters")
    device = parameters[0].device
    was_training = model.training
    rng_devices = [device] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=rng_devices):
            model.train()
            output = model(batch.to_dict(), **training_forward_kwargs(output_args))
            terms = loss_fn.compute_terms(ref=batch, pred=output)
            names = list(terms.keys())
            term_values = torch.stack(
                [torch.as_tensor(terms[name]).detach().reshape(()) for name in names]
            )
            nonzero = term_values.ne(0).cpu().tolist()  # one transfer per probe
            gradients: Dict[str, torch.Tensor] = {}
            for name, is_nonzero in zip(names, nonzero):
                value = terms[name]
                if not is_nonzero or not value.requires_grad:
                    continue
                parts = torch.autograd.grad(
                    value, parameters, retain_graph=True, allow_unused=True
                )
                gradients[name] = torch.cat(
                    [
                        (
                            torch.zeros_like(parameter)
                            if part is None
                            else part.detach()
                        ).reshape(-1)
                        for parameter, part in zip(parameters, parts)
                    ]
                )
            del output, terms
    finally:
        model.train(was_training)

    parameter_count = sum(parameter.numel() for parameter in parameters)
    zeros = torch.zeros(parameter_count, dtype=parameters[0].dtype, device=device)
    total_gradient = zeros.clone()
    for gradient in gradients.values():
        total_gradient = total_gradient + gradient
    norms = torch.stack(
        [gradients.get(name, zeros).norm() for name in names] + [total_gradient.norm()]
    )
    pairs = [
        (first, second)
        for first, second in cosine_pairs
        if first in names and second in names
    ]
    dot_products = torch.stack(
        [
            torch.dot(gradients.get(first, zeros), gradients.get(second, zeros))
            for first, second in pairs
        ]
        or [zeros.new_zeros(())]
    )
    host_values = torch.cat([term_values.to(norms.dtype), norms, dot_products])
    host = host_values.cpu().tolist()  # one transfer per probe
    value_list = host[: len(names)]
    norm_list = host[len(names) : 2 * len(names) + 1]
    dot_list = host[2 * len(names) + 1 :]
    norm_by_name = dict(zip(names, norm_list[:-1]))
    cosines = {}
    for (first, second), dot_product in zip(pairs, dot_list):
        denominator = norm_by_name[first] * norm_by_name[second]
        cosines[_pair_key(first, second)] = (
            None if denominator == 0.0 else dot_product / denominator
        )
    return {
        "term_values": dict(zip(names, value_list)),
        "gradient_norms": norm_by_name,
        "total_gradient_norm": norm_list[-1],
        "cosine_similarities": cosines,
        "trainable_parameter_count": parameter_count,
        "wall_time_seconds": time.perf_counter() - start_time,
    }


class LossTermInstrumentation:
    """Collects per-term statistics during training and writes the JSONL records.

    `train` owns one of these (None = off). The hooks it calls, in order:
    `start_train_epoch`, per batch `before_train_step` / `after_train_step`,
    `finish_train_epoch`; per validation loader a fresh `validation_accumulator` that
    `evaluate` updates, then `write_validation_summary`; `stage_switch` once.
    """

    def __init__(
        self,
        writer: LossTermJsonlWriter,
        term_names: Sequence[str],
        probe_interval: int,
        device: torch.device,
    ) -> None:
        if probe_interval < 0:
            raise ValueError(f"probe_interval must be >= 0, got {probe_interval}")
        self.writer = writer
        self.term_names = tuple(term_names)
        self.probe_interval = probe_interval
        self.device = device
        self.stage = "one"
        self.step = 0
        self.probe_batch: Any = None
        self.train_accumulator = LossTermAccumulator(self.term_names)

    @classmethod
    def for_run(
        cls,
        results_dir: str,
        tag: str,
        loss_fn: Any,
        probe_interval: int,
        device: torch.device,
    ) -> "LossTermInstrumentation":
        path = os.path.join(results_dir, f"{tag}_loss_terms.jsonl")
        instrumentation = cls(
            writer=LossTermJsonlWriter(path),
            term_names=loss_fn.term_names,
            probe_interval=probe_interval,
            device=device,
        )
        instrumentation.write_run_start(tag, loss_fn)
        return instrumentation

    def write_run_start(self, tag: str, loss_fn: Any) -> None:
        self.writer.write(
            {
                "record": "run_start",
                "tag": tag,
                "term_names": list(self.term_names),
                "probe_interval": self.probe_interval,
                "weights": _term_weights_or_none(loss_fn),
            }
        )

    # Training split -------------------------------------------------------------------

    def start_train_epoch(self) -> None:
        self.train_accumulator.reset()

    def before_train_step(
        self,
        model: torch.nn.Module,
        loss_fn: Any,
        batch: Any,
        output_args: Dict[str, bool],
        epoch: int,
    ) -> None:
        if self.probe_interval == 0:
            return
        if self.probe_batch is None:
            # The first training batch seen, copied so the step cannot alter it. Taking
            # it from the running loop (not a fresh `iter(loader)`) leaves the shuffling
            # random number generator untouched.
            self.probe_batch = batch.clone().to(self.device)
        if self.step % self.probe_interval == 0:
            self.write_gradient_probe(model, loss_fn, output_args, epoch)

    def write_gradient_probe(
        self,
        model: torch.nn.Module,
        loss_fn: Any,
        output_args: Dict[str, bool],
        epoch: int,
    ) -> None:
        result = probe_term_gradients(model, loss_fn, self.probe_batch, output_args)
        self.writer.write(
            {
                "record": "gradient_probe",
                "step": self.step,
                "epoch": epoch,
                "stage": self.stage,
                **result,
            }
        )

    def after_train_step(self, loss_fn: Any) -> None:
        """Reads the statistics of the loss call inside the step that just ran."""
        self.train_accumulator.update(loss_fn)
        self.step += 1

    def finish_train_epoch(self, epoch: int) -> None:
        self._write_summary(
            self.train_accumulator, split="train", head=None, epoch=epoch
        )

    # Validation split -----------------------------------------------------------------

    def validation_accumulator(self) -> LossTermAccumulator:
        return LossTermAccumulator(self.term_names)

    def write_validation_summary(
        self, accumulator: LossTermAccumulator, head: str, epoch: Optional[int]
    ) -> None:
        self._write_summary(accumulator, split="valid", head=head, epoch=epoch)

    # Stage two ------------------------------------------------------------------------

    def stage_switch(self, epoch: int, old_loss_fn: Any, new_loss_fn: Any) -> None:
        self.writer.write(
            {
                "record": "stage_switch",
                "epoch": epoch,
                "step": self.step,
                "old_weights": _term_weights_or_none(old_loss_fn),
                "new_weights": _term_weights_or_none(new_loss_fn),
            }
        )
        if not supports_loss_terms(new_loss_fn):
            raise TypeError(
                "the stage-two loss does not follow the loss-term contract "
                f"({type(new_loss_fn).__name__}); per-term logging cannot continue"
            )
        if tuple(new_loss_fn.term_names) != self.term_names:
            raise ValueError(
                f"the stage-two loss has terms {tuple(new_loss_fn.term_names)}, "
                f"the stage-one loss {self.term_names}"
            )
        self.stage = "two"

    def _write_summary(
        self,
        accumulator: LossTermAccumulator,
        split: str,
        head: Optional[str],
        epoch: Optional[int],
    ) -> None:
        self.writer.write(
            {
                "record": "term_summary",
                "split": split,
                "head": head,
                "epoch": epoch,
                "stage": self.stage,
                "step": self.step,
                "terms": accumulator.summary(),
            }
        )


def _term_weights_or_none(loss_fn: Any) -> Optional[Dict[str, float]]:
    if not hasattr(loss_fn, "term_weights"):
        return None
    return {name: float(weight) for name, weight in loss_fn.term_weights().items()}
