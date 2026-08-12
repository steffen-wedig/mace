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
        for parameter in model.readouts[-1].delta_readouts[0].projection.parameters():
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


def test_stop_gradient_blocks_base_readout_parameters():
    """The parameter detach: delta-level losses cannot move the base readout."""
    model = make_model(detach_base_for_deltas=True)
    batch = make_batch([make_configuration(POSITIONS, head="revpbe")])
    output = model(batch.to_dict(), training=True, compute_force=False)
    # Backward only from the delta level's energy.
    output["energy_all_levels"][:, 1].sum().backward()
    for readout in model.readouts:
        base_parameters = list(readout.base_linear.parameters())
        if hasattr(readout, "delta_readouts"):
            # Final block: the base hidden layer feeds only the base column.
            base_parameters += list(readout.linear_1.parameters())
            delta_parameters = [
                parameter
                for delta_readout in readout.delta_readouts
                for parameter in delta_readout.parameters()
            ]
        else:
            delta_parameters = list(readout.delta_linear.parameters())
        for parameter in base_parameters:
            assert parameter.grad is None or torch.all(parameter.grad == 0)
        for parameter in delta_parameters:
            assert parameter.grad is not None
            assert not torch.all(parameter.grad == 0)
    # The trunk still receives gradient -- through the delta projections AND
    # (unlike a tensor detach) through the base column's activations.
    trunk_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("interactions.") and parameter.grad is not None
    ]
    assert any(torch.any(gradient != 0) for gradient in trunk_gradients)


def test_stop_gradient_leaves_energies_and_forces_exact():
    """The detach changes gradients only: outputs match the plain model."""
    model_plain = make_model(
        atomic_inter_scale=(1.3, 0.01), atomic_inter_shift=(-3.0, -0.08)
    )
    model_detached = make_model(
        atomic_inter_scale=(1.3, 0.01),
        atomic_inter_shift=(-3.0, -0.08),
        detach_base_for_deltas=True,
    )
    model_detached.load_state_dict(model_plain.state_dict())

    batch = make_batch([make_configuration(POSITIONS, head="delta_cc")])
    output_plain = model_plain(batch.to_dict(), training=True, compute_force=True)
    batch = make_batch([make_configuration(POSITIONS, head="delta_cc")])
    output_detached = model_detached(
        batch.to_dict(), training=True, compute_force=True
    )
    assert torch.allclose(
        output_plain["energy_all_levels"], output_detached["energy_all_levels"]
    )
    # A tensor detach would strip the base contribution from the delta
    # level's forces; the parameter detach keeps them exact.
    assert torch.allclose(output_plain["forces"], output_detached["forces"])


def test_stop_gradient_blocks_base_readout_through_force_loss():
    """Second order: a delta-level FORCE loss cannot move the base readout."""
    model = make_model(
        atomic_inter_scale=(1.3, 0.01),
        atomic_inter_shift=(-3.0, -0.08),
        detach_base_for_deltas=True,
    )
    batch = make_batch([make_configuration(POSITIONS, head="delta_cc")])
    output = model(batch.to_dict(), training=True, compute_force=True)
    force_loss = torch.square(output["forces"]).sum()
    base_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("readouts") and "delta" not in name
    ]
    gradients = torch.autograd.grad(
        force_loss,
        [parameter for _, parameter in base_parameters],
        allow_unused=True,
    )
    for (name, _), gradient in zip(base_parameters, gradients):
        assert gradient is None or torch.all(gradient == 0), name


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


def make_single_head_base_model(atomic_energies, scale, shift):
    """A stock single-head ScaleShiftMACE with the same trunk hyperparameters."""
    return modules.ScaleShiftMACE(
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
        num_interactions=2,
        num_elements=len(TABLE),
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.asarray(atomic_energies, dtype=float),
        avg_num_neighbors=8,
        atomic_numbers=TABLE.zs,
        correlation=3,
        heads=[LEVELS[BASE_LEVEL]],
        atomic_inter_scale=scale,
        atomic_inter_shift=shift,
    )


def test_warm_start_reproduces_the_base_model_on_the_base_head(tmp_path):
    from mace.tools.multilevel_init import initialise_from_base_model

    base_e0s = np.array([[-13.7, -2042.8]])
    base_model = make_single_head_base_model(base_e0s, scale=1.3, shift=-3.0)
    base_model_path = tmp_path / "base.model"
    torch.save(base_model, base_model_path)

    model = make_model(
        atomic_energies=np.vstack([base_e0s, np.array([[0.11, 1.85]])]),
        atomic_inter_scale=(1.3, 0.01),
        atomic_inter_shift=(-3.0, -0.08),
        zero_init_delta_readouts=True,
    )
    counts = initialise_from_base_model(model, base_model_path)
    assert counts["copied"] > 0

    batch = make_batch(
        [
            make_configuration(POSITIONS, head=LEVELS[BASE_LEVEL]),
            make_configuration(POSITIONS + 0.05, head=LEVELS[BASE_LEVEL]),
        ]
    )
    warm_output = model(batch.to_dict(), training=False, compute_force=True)
    base_output = base_model(batch.to_dict(), training=False, compute_force=True)
    assert torch.allclose(warm_output["energy"], base_output["energy"], atol=1e-10)
    assert torch.allclose(warm_output["forces"], base_output["forces"], atol=1e-10)
    # Zero-initialised deltas: every level starts at base plus its shift/E0,
    # so the fused per-level energies exist and are finite.
    fused_output = model(batch.to_dict(), training=True)
    assert torch.isfinite(fused_output["energy_all_levels"]).all()


def test_warm_start_rejects_a_mismatched_trunk(tmp_path):
    from mace.tools.multilevel_init import WarmStartError, initialise_from_base_model

    base_e0s = np.array([[-13.7, -2042.8]])
    small_base = make_single_head_base_model(base_e0s, scale=1.3, shift=-3.0)
    # Shrink the trunk after the fact by rebuilding with fewer interactions.
    mismatched = make_model(
        atomic_energies=np.vstack([base_e0s, np.array([[0.11, 1.85]])]),
        atomic_inter_scale=(1.3, 0.01),
        atomic_inter_shift=(-3.0, -0.08),
        num_interactions=1,
    )
    base_model_path = tmp_path / "base.model"
    torch.save(small_base, base_model_path)
    with pytest.raises(WarmStartError):
        initialise_from_base_model(mismatched, base_model_path)


def test_warm_start_rejects_disagreeing_export_statistics(tmp_path):
    from mace.tools.multilevel_init import WarmStartError, initialise_from_base_model

    base_e0s = np.array([[-13.7, -2042.8]])
    base_model = make_single_head_base_model(base_e0s, scale=1.3, shift=-3.0)
    base_model_path = tmp_path / "base.model"
    torch.save(base_model, base_model_path)
    model = make_model(
        atomic_energies=np.vstack([base_e0s, np.array([[0.11, 1.85]])]),
        atomic_inter_scale=(1.4, 0.01),  # different base scale
        atomic_inter_shift=(-3.0, -0.08),
    )
    with pytest.raises(WarmStartError, match="scale"):
        initialise_from_base_model(model, base_model_path)


def test_freezing_leaves_only_delta_parameters_trainable():
    from mace.tools.multilevel_init import freeze_non_delta_parameters

    model = make_model()
    counts = freeze_non_delta_parameters(model)
    assert counts["trainable"] > 0 and counts["frozen"] > 0
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == ("delta" in name), name

    # A forward/backward through every level still works: position gradients
    # flow through frozen parameters even though they receive none, and the
    # delta paths (the only trainable ones) receive gradients.
    batch = make_batch([make_configuration(POSITIONS, head=LEVELS[BASE_LEVEL])])
    output = model(batch.to_dict(), training=True, compute_force=True)
    output["energy_all_levels"].sum().backward()
    for name, parameter in model.named_parameters():
        if "delta" in name and parameter.grad is not None:
            break
    else:
        pytest.fail("no delta parameter received a gradient")


def test_per_level_energy_weights_normalise_each_level(tmp_path):
    model = make_model(
        atomic_inter_scale=(1.3, 0.01), atomic_inter_shift=(-3.0, -0.08)
    )
    model.force_carrying_levels = [BASE_LEVEL]
    model.base_force_column = 0
    loss_fn = modules.MultiLevelWeightedEnergyForcesLoss(
        energy_weight=999.0,  # must be ignored on fused batches in this mode
        forces_weight=0.0,
        energy_weights_per_level=[10.0, 1000.0],
    )

    batch, _ = make_fused_batch(
        tmp_path,
        [(POSITIONS, -1.5, FORCES, -1.6), (POSITIONS + 0.01, -1.4, FORCES, None)],
    )
    output = model(batch.to_dict(), training=True)
    loss = loss_fn(batch, output)

    num_atoms = batch.ptr[1:] - batch.ptr[:-1]
    residuals = (batch.energy_levels - output["energy_all_levels"]) / num_atoms.unsqueeze(-1)
    weights = batch.energy_levels_weight
    squared = weights * residuals**2
    base_mean = squared[:, 0].sum() / 2  # both structures carry the base label
    cc_mean = squared[:, 1].sum() / 1  # only the first carries the CC label
    assert loss.item() == pytest.approx(
        (10.0 * base_mean + 1000.0 * cc_mean).item(), rel=1e-12
    )


def test_per_level_weights_make_the_sparse_level_coverage_invariant(tmp_path):
    """The CC term must be a mean over CC labels, however many base labels ride along."""
    model = make_model(
        atomic_inter_scale=(1.3, 0.01), atomic_inter_shift=(-3.0, -0.08)
    )
    model.force_carrying_levels = [BASE_LEVEL]
    model.base_force_column = 0
    cc_only_loss = modules.MultiLevelWeightedEnergyForcesLoss(
        forces_weight=0.0, energy_weights_per_level=[0.0, 1.0]
    )

    labelled = (POSITIONS, -1.5, FORCES, -1.6)
    (tmp_path / "lonely").mkdir()
    (tmp_path / "crowded").mkdir()
    lonely_batch, _ = make_fused_batch(tmp_path / "lonely", [labelled])
    crowded_batch, _ = make_fused_batch(
        tmp_path / "crowded",
        [labelled] + [(POSITIONS + 0.01 * i, -1.4, FORCES, None) for i in (1, 2, 3)],
    )
    lonely = cc_only_loss(lonely_batch, model(lonely_batch.to_dict(), training=True))
    crowded = cc_only_loss(crowded_batch, model(crowded_batch.to_dict(), training=True))
    assert crowded.item() == pytest.approx(lonely.item(), rel=1e-10)


def test_per_level_weights_survive_a_batch_without_the_sparse_level(tmp_path):
    model = make_model(
        atomic_inter_scale=(1.3, 0.01), atomic_inter_shift=(-3.0, -0.08)
    )
    model.force_carrying_levels = [BASE_LEVEL]
    model.base_force_column = 0
    loss_fn = modules.MultiLevelWeightedEnergyForcesLoss(
        forces_weight=0.0, energy_weights_per_level=[10.0, 1000.0]
    )
    batch, _ = make_fused_batch(
        tmp_path, [(POSITIONS, -1.5, FORCES, None), (POSITIONS + 0.01, -1.4, FORCES, None)]
    )
    output = model(batch.to_dict(), training=True)
    loss = loss_fn(batch, output)
    assert torch.isfinite(loss)
    num_atoms = batch.ptr[1:] - batch.ptr[:-1]
    residuals = (batch.energy_levels - output["energy_all_levels"]) / num_atoms.unsqueeze(-1)
    squared = batch.energy_levels_weight * residuals**2
    assert loss.item() == pytest.approx((10.0 * squared[:, 0].sum() / 2).item(), rel=1e-12)


def test_per_level_weight_count_mismatch_is_rejected(tmp_path):
    model = make_model(
        atomic_inter_scale=(1.3, 0.01), atomic_inter_shift=(-3.0, -0.08)
    )
    model.force_carrying_levels = [BASE_LEVEL]
    model.base_force_column = 0
    loss_fn = modules.MultiLevelWeightedEnergyForcesLoss(
        forces_weight=0.0, energy_weights_per_level=[1.0, 2.0, 3.0]
    )
    batch, _ = make_fused_batch(tmp_path, [(POSITIONS, -1.5, FORCES, -1.6)])
    output = model(batch.to_dict(), training=True)
    with pytest.raises(ValueError, match="per-level energy"):
        loss_fn(batch, output)


def test_parse_multilevel_energy_weights():
    from mace.tools.scripts_utils import parse_multilevel_energy_weights

    heads = ["revpbe", "delta_cc"]
    assert parse_multilevel_energy_weights(None, heads) is None
    assert parse_multilevel_energy_weights(
        "delta_cc:1000, revpbe:10", heads
    ) == [10.0, 1000.0]
    with pytest.raises(ValueError, match="every head needs exactly one"):
        parse_multilevel_energy_weights("revpbe:10", heads)
    with pytest.raises(ValueError, match="every head needs exactly one"):
        parse_multilevel_energy_weights("revpbe:10,delta_cc:1,mp2:5", heads)
    with pytest.raises(ValueError, match="malformed"):
        parse_multilevel_energy_weights("revpbe=10,delta_cc=1000", heads)
    with pytest.raises(ValueError, match="duplicate"):
        parse_multilevel_energy_weights("revpbe:10,revpbe:20", heads)


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


def test_stage_two_swap_puts_the_per_level_weights_on_the_training_device(tmp_path):
    """Regression: Stage Two swapped in a loss that was still on the CPU.

    The Stage One loss reaches the device as a submodule of the evaluation
    metric; the Stage Two loss built by ``get_swa`` never was moved. Scalar
    weights hide that (0-dim CPU tensors promote against CUDA operands), the
    per-level weight VECTOR does not -- it raised "Expected all tensors to
    be on the same device" exactly at the transition.
    """
    from mace.tools.train import MACELoss, SWAContainer, stage_two_loss_fn

    stage_one_loss = modules.MultiLevelWeightedEnergyForcesLoss(
        forces_weight=0.0, energy_weights_per_level=[10.0, 1000.0]
    )
    stage_two_loss = modules.MultiLevelWeightedEnergyForcesLoss(
        forces_weight=0.0, energy_weights_per_level=[1000.0, 1000.0]
    )
    # The per-level weights must be a registered buffer, or nothing that
    # moves the module would move them.
    assert "energy_weights_per_level" in dict(stage_two_loss.named_buffers())

    device = torch.device("cpu")
    MACELoss(loss_fn=stage_one_loss).to(device)  # what evaluate() does in stage one
    swa = SWAContainer(model=None, scheduler=None, start=0, loss_fn=stage_two_loss)
    swapped_loss = stage_two_loss_fn(swa, device)
    assert swapped_loss.energy_weights_per_level.device == device

    # The move must be a real move, which a CPU-only run cannot show: the
    # meta device stands in for the training device.
    meta_loss = stage_two_loss_fn(
        SWAContainer(
            model=None,
            scheduler=None,
            start=0,
            loss_fn=modules.MultiLevelWeightedEnergyForcesLoss(
                forces_weight=0.0, energy_weights_per_level=[1.0, 2.0]
            ),
        ),
        torch.device("meta"),
    )
    assert meta_loss.energy_weights_per_level.device.type == "meta"

    # And the swapped-in loss still evaluates a fused batch.
    model = make_model(atomic_inter_scale=(1.3, 0.01), atomic_inter_shift=(-3.0, -0.08))
    model.force_carrying_levels = [BASE_LEVEL]
    model.base_force_column = 0
    batch, _ = make_fused_batch(
        tmp_path,
        [(POSITIONS, -1.5, FORCES, -1.6), (POSITIONS + 0.01, -1.4, FORCES, None)],
    )
    output = model(batch.to_dict(), training=True)
    assert torch.isfinite(swapped_loss(batch, output))


def test_a_loss_without_per_level_weights_still_moves_between_devices():
    """Both construction paths register the same buffer, so ``.to`` is safe."""
    loss_fn = modules.MultiLevelWeightedEnergyForcesLoss(
        energy_weight=3.0, forces_weight=7.0
    )
    assert loss_fn.energy_weights_per_level is None
    moved_loss = loss_fn.to(torch.device("meta"))
    assert moved_loss.energy_weights_per_level is None
    assert moved_loss.energy_weight.device.type == "meta"
