"""Tests for MultiLevelScaleShiftMACE: residual per-level readouts.

Pins the properties the residual multi-level design depends on:
one forward pass produces every level, rotation invariance per level,
zeroed delta projections return the base model-wide, a delta only moves
its own level, gradients reach every parameter, the stop-gradient blocks
the base readout path, and the masked loss divides by labelled entries
only.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional
from e3nn import o3
from scipy.spatial.transform import Rotation as R

from mace import data, modules, tools
from mace.tools import torch_geometric

torch.set_default_dtype(torch.float64)

TABLE = tools.AtomicNumberTable([1, 8])
LEVELS = ["revpbe", "delta_cc"]
BASE_LEVEL = 0

POSITIONS = np.array(
    [
        [0.0, -2.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
FORCES = np.array(
    [
        [0.0, -1.3, 0.0],
        [1.0, 0.2, 0.0],
        [0.0, 1.1, 0.3],
    ]
)


def make_configuration(positions, head, forces_weight=1.0):
    return data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=positions,
        properties={"forces": FORCES, "energy": -1.5},
        property_weights={"forces": forces_weight, "energy": 1.0},
        head=head,
    )


def make_model(
    atomic_energies=None,
    atomic_inter_scale=(1.0, 1.0),
    atomic_inter_shift=(0.0, 0.0),
    detach_base_for_deltas=False,
    zero_init_delta_readouts=False,
    num_interactions=2,
):
    if atomic_energies is None:
        atomic_energies = np.zeros((len(LEVELS), len(TABLE)))
    return modules.MultiLevelScaleShiftMACE(
        r_max=5,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=num_interactions,
        num_elements=len(TABLE),
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.asarray(atomic_energies, dtype=float),
        avg_num_neighbors=8,
        atomic_numbers=TABLE.zs,
        correlation=3,
        heads=list(LEVELS),
        atomic_inter_scale=(
            list(atomic_inter_scale)
            if isinstance(atomic_inter_scale, (list, tuple))
            else atomic_inter_scale
        ),
        atomic_inter_shift=(
            list(atomic_inter_shift)
            if isinstance(atomic_inter_shift, (list, tuple))
            else atomic_inter_shift
        ),
        base_level=BASE_LEVEL,
        detach_base_for_deltas=detach_base_for_deltas,
        zero_init_delta_readouts=zero_init_delta_readouts,
    )


def make_batch(configurations):
    atomic_data = [
        data.AtomicData.from_config(
            configuration, z_table=TABLE, cutoff=5.0, heads=list(LEVELS)
        )
        for configuration in configurations
    ]
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=atomic_data,
        batch_size=len(atomic_data),
        shuffle=False,
        drop_last=False,
    )
    return next(iter(data_loader))


def test_all_levels_in_one_call_and_head_gather_consistency():
    model = make_model(
        atomic_energies=np.array([[1.0, 3.0], [1.5, 3.5]]),
        atomic_inter_scale=(1.3, 0.01),
        atomic_inter_shift=(-3.0, -0.08),
    )
    batch = make_batch(
        [
            make_configuration(POSITIONS, head="revpbe"),
            make_configuration(POSITIONS, head="delta_cc"),
        ]
    )
    output = model(batch.to_dict(), training=True)
    energy_all_levels = output["energy_all_levels"]
    assert energy_all_levels.shape == (2, len(LEVELS))
    # The head-gathered energy of each graph equals its own column.
    assert torch.allclose(output["energy"][0], energy_all_levels[0, 0])
    assert torch.allclose(output["energy"][1], energy_all_levels[1, 1])
    # Identical geometries: the level columns agree across the two graphs.
    assert torch.allclose(energy_all_levels[0], energy_all_levels[1])


def test_rotation_invariance_of_every_level():
    model = make_model(
        atomic_inter_scale=(1.3, 0.01), atomic_inter_shift=(-3.0, -0.08)
    )
    rotation = R.from_euler("z", 60, degrees=True).as_matrix()
    positions_rotated = np.array(rotation @ POSITIONS.T).T
    batch = make_batch(
        [
            make_configuration(POSITIONS, head="revpbe"),
            make_configuration(positions_rotated, head="revpbe"),
        ]
    )
    output = model(batch.to_dict(), training=False, compute_force=False)
    energy_all_levels = output["energy_all_levels"]
    assert torch.allclose(energy_all_levels[0], energy_all_levels[1], atol=1e-10)


def test_zeroed_deltas_return_base_model_wide():
    # zero_init zeroes the delta projections of EVERY readout layer, so with
    # equal E0s and zero shifts every level must return exactly the base.
    model = make_model(
        atomic_inter_scale=(1.3, 1.3),
        atomic_inter_shift=(0.0, 0.0),
        zero_init_delta_readouts=True,
    )
    batch = make_batch([make_configuration(POSITIONS, head="revpbe")])
    output = model(batch.to_dict(), training=False, compute_force=False)
    energy_all_levels = output["energy_all_levels"]
    assert torch.equal(energy_all_levels[:, 0], energy_all_levels[:, 1])


def test_zeroed_deltas_start_at_base_plus_shift_and_e0():
    # With fitted delta shift and per-level E0s, a zero-initialised level
    # starts at the base plus its mean correction: the difference between
    # the columns is exactly N_atoms * shift_delta plus the E0 difference.
    delta_shift = -0.0871822347815128
    e0s = np.array([[1.0, 3.0], [1.2, 3.4]])
    model = make_model(
        atomic_energies=e0s,
        atomic_inter_scale=(1.3, 0.011),
        atomic_inter_shift=(0.0, delta_shift),
        zero_init_delta_readouts=True,
    )
    batch = make_batch([make_configuration(POSITIONS, head="revpbe")])
    output = model(batch.to_dict(), training=False, compute_force=False)
    energy_all_levels = output["energy_all_levels"]
    # One O and two H atoms.
    e0_difference = (e0s[1, 1] - e0s[0, 1]) + 2 * (e0s[1, 0] - e0s[0, 0])
    expected_difference = 3 * delta_shift + e0_difference
    actual_difference = (energy_all_levels[0, 1] - energy_all_levels[0, 0]).item()
    assert actual_difference == pytest.approx(expected_difference, abs=1e-12)


def test_delta_moves_only_its_own_level():
    model = make_model(zero_init_delta_readouts=True)
    batch = make_batch([make_configuration(POSITIONS, head="revpbe")])
    before = model(batch.to_dict(), training=False, compute_force=False)[
        "energy_all_levels"
    ].detach()
    # Perturb the final readout's delta projection only.
    with torch.no_grad():
        for parameter in model.readouts[-1].delta_linear.parameters():
            parameter.add_(0.5)
    after = model(batch.to_dict(), training=False, compute_force=False)[
        "energy_all_levels"
    ].detach()
    assert torch.equal(before[:, 0], after[:, 0])  # base column untouched
    assert not torch.allclose(before[:, 1], after[:, 1])


def test_gradient_reaches_every_parameter():
    model = make_model()
    batch = make_batch(
        [
            make_configuration(POSITIONS, head="revpbe"),
            make_configuration(POSITIONS, head="delta_cc"),
        ]
    )
    output = model(batch.to_dict(), training=True, compute_force=False)
    output["energy_all_levels"].sum().backward()
    parameters_without_gradient = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or torch.all(parameter.grad == 0)
    ]
    assert not parameters_without_gradient, parameters_without_gradient


def test_stop_gradient_blocks_base_readout_path():
    model = make_model(detach_base_for_deltas=True)
    batch = make_batch([make_configuration(POSITIONS, head="revpbe")])
    output = model(batch.to_dict(), training=True, compute_force=False)
    # Backward only from the delta level's energy.
    output["energy_all_levels"][:, 1].sum().backward()
    for readout in model.readouts:
        for parameter in readout.base_linear.parameters():
            assert parameter.grad is None or torch.all(parameter.grad == 0)
        for parameter in readout.delta_linear.parameters():
            assert parameter.grad is not None
            assert not torch.all(parameter.grad == 0)
    # The trunk still receives gradient through the delta projections.
    trunk_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("interactions.") and parameter.grad is not None
    ]
    assert any(torch.any(gradient != 0) for gradient in trunk_gradients)


def test_forces_of_gathered_head_are_computed():
    model = make_model(
        atomic_inter_scale=(1.3, 0.01), atomic_inter_shift=(-3.0, -0.08)
    )
    batch = make_batch([make_configuration(POSITIONS, head="delta_cc")])
    output = model(batch.to_dict(), training=True, compute_force=True)
    assert output["forces"] is not None
    assert output["forces"].shape == (3, 3)
    assert torch.all(torch.isfinite(output["forces"]))


def test_scalar_scale_shift_is_rejected():
    with pytest.raises(ValueError, match="per level"):
        make_model(atomic_inter_scale=1.0, atomic_inter_shift=0.0)


def test_non_scalar_hidden_irreps_are_rejected():
    with pytest.raises(ValueError, match="scalars"):
        modules.ResidualLevelNonLinearReadoutBlock(
            irreps_in=o3.Irreps("16x0e + 16x1o"),
            MLP_irreps=o3.Irreps("8x0e + 4x1o"),
            gate=torch.nn.functional.silu,
            num_levels=2,
        )


def write_multilevel_xyz(path, entries):
    """entries: list of (positions, base_energy, base_forces, cc_energy_or_None)."""
    import ase.io
    from ase import Atoms

    atoms_list = []
    for positions, base_energy, base_forces, cc_energy in entries:
        atoms = Atoms("OHH", positions=positions)
        atoms.info["REF_energy_revpbe"] = base_energy
        atoms.arrays["REF_forces_revpbe"] = base_forces
        if cc_energy is not None:
            atoms.info["REF_energy_delta_cc"] = cc_energy
        atoms_list.append(atoms)
    ase.io.write(str(path), atoms_list)


def make_fused_batch(tmp_path, entries):
    from mace.data.multilevel import load_multilevel_dataset

    file_path = tmp_path / "multilevel.xyz"
    write_multilevel_xyz(file_path, entries)
    dataset, labelled_counts = load_multilevel_dataset(
        file_path=str(file_path),
        r_max=5.0,
        z_table=TABLE,
        heads=list(LEVELS),
        force_carrying_heads=[LEVELS[BASE_LEVEL]],
        energy_key="REF_energy",
        forces_key="REF_forces",
    )
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=len(dataset), shuffle=False, drop_last=False
    )
    return next(iter(data_loader)), labelled_counts


def test_fused_dataset_masks_missing_levels(tmp_path):
    batch, labelled_counts = make_fused_batch(
        tmp_path,
        [
            (POSITIONS, -1.5, FORCES, -1.6),
            (POSITIONS + 0.01, -1.4, FORCES, None),  # no CC label
        ],
    )
    assert labelled_counts == {"revpbe": 2, "delta_cc": 1}
    assert batch.energy_levels.shape == (2, 2)
    assert batch.energy_levels_weight.tolist() == [[1.0, 1.0], [1.0, 0.0]]
    assert batch.forces_levels.shape == (6, 1, 3)
    assert batch.forces_levels_weight.tolist() == [[1.0], [1.0]]


def test_fused_forward_produces_per_level_forces(tmp_path):
    model = make_model(
        atomic_inter_scale=(1.3, 0.01), atomic_inter_shift=(-3.0, -0.08)
    )
    model.force_carrying_levels = [BASE_LEVEL]
    model.base_force_column = 0
    batch, _ = make_fused_batch(
        tmp_path, [(POSITIONS, -1.5, FORCES, -1.6), (POSITIONS + 0.01, -1.4, FORCES, None)]
    )
    training_output = model(batch.to_dict(), training=True, compute_force=True)
    assert training_output["forces_all_levels"] is not None
    assert training_output["forces_all_levels"].shape == (6, 1, 3)
    # The gathered forces are the base column without an extra backward.
    assert torch.equal(
        training_output["forces"],
        training_output["forces_all_levels"][:, 0, :],
    )
    # And they match the stock gathered-head force path on the same batch
    # (fused configs all carry the base head).
    stock_output = model(batch.to_dict(), training=False, compute_force=True)
    assert torch.allclose(
        training_output["forces"], stock_output["forces"], atol=1e-10
    )


def test_multilevel_loss_counts_only_labelled_entries(tmp_path):
    model = make_model(
        atomic_inter_scale=(1.3, 0.01), atomic_inter_shift=(-3.0, -0.08)
    )
    model.force_carrying_levels = [BASE_LEVEL]
    model.base_force_column = 0
    loss_fn = modules.MultiLevelWeightedEnergyForcesLoss(
        energy_weight=1.0, forces_weight=0.0
    )

    both_labelled = (POSITIONS, -1.5, FORCES, -1.6)
    base_only = (POSITIONS + 0.01, -1.4, FORCES, None)

    mixed_batch, _ = make_fused_batch(tmp_path, [both_labelled, base_only])
    mixed_output = model(mixed_batch.to_dict(), training=True)
    mixed_loss = loss_fn(mixed_batch, mixed_output)

    # Reference: compute the same masked mean by hand from the outputs.
    num_atoms = mixed_batch.ptr[1:] - mixed_batch.ptr[:-1]
    residuals = (
        mixed_batch.energy_levels - mixed_output["energy_all_levels"]
    ) / num_atoms.unsqueeze(-1)
    weights = mixed_batch.energy_levels_weight
    expected = (weights * residuals**2).sum() / (weights != 0).sum()
    assert mixed_loss.item() == pytest.approx(expected.item(), rel=1e-12)
    # Three labelled energies, not four: the denominator is 3.
    assert (weights != 0).sum().item() == 3


def test_multilevel_loss_falls_back_on_per_head_batches():
    model = make_model(
        atomic_inter_scale=(1.3, 0.01), atomic_inter_shift=(-3.0, -0.08)
    )
    batch = make_batch(
        [
            make_configuration(POSITIONS, head="revpbe"),
            make_configuration(POSITIONS, head="delta_cc", forces_weight=0.0),
        ]
    )
    output = model(batch.to_dict(), training=True)
    multilevel_loss = modules.MultiLevelWeightedEnergyForcesLoss(
        energy_weight=3.0, forces_weight=7.0
    )
    masked_loss = modules.WeightedEnergyForcesMaskedLoss(
        energy_weight=3.0, forces_weight=7.0
    )
    assert multilevel_loss(batch, output).item() == pytest.approx(
        masked_loss(batch, output).item(), rel=1e-12
    )


def test_masked_loss_denominator_ignores_unlabelled_entries():
    # Two identical configurations, one of which carries no force label.
    # The masked forces term must equal the plain forces term computed on
    # the labelled configuration alone, instead of being diluted by the
    # zero-weighted entries.
    labelled = make_configuration(POSITIONS, head="revpbe", forces_weight=1.0)
    unlabelled = make_configuration(POSITIONS, head="delta_cc", forces_weight=0.0)
    mixed_batch = make_batch([labelled, unlabelled])
    labelled_batch = make_batch([labelled])

    model = make_model(
        atomic_inter_scale=(1.3, 0.01), atomic_inter_shift=(-3.0, -0.08)
    )
    masked_loss = modules.WeightedEnergyForcesMaskedLoss(
        energy_weight=0.0, forces_weight=1.0
    )
    diluted_loss = modules.WeightedEnergyForcesLoss(
        energy_weight=0.0, forces_weight=1.0
    )

    mixed_output = model(mixed_batch.to_dict(), training=True)
    labelled_output = model(labelled_batch.to_dict(), training=True)

    masked_value = masked_loss(mixed_batch, mixed_output)
    reference_value = diluted_loss(labelled_batch, labelled_output)
    diluted_value = diluted_loss(mixed_batch, mixed_output)

    assert masked_value.item() == pytest.approx(reference_value.item(), rel=1e-10)
    # The stock loss halves the term because half the batch is unlabelled.
    assert diluted_value.item() == pytest.approx(
        reference_value.item() / 2, rel=1e-10
    )
