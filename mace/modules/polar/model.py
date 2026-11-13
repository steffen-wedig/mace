"""PolarMACE: MACE with field-dependent charges and long-range electrostatics (MACE-POLAR)."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.modules.blocks import (
    NonLinearBiasReadoutBlock,
    NonLinearReadoutBlock,
)
from mace.modules.models import ScaleShiftMACE
from mace.modules.polar.electrostatics.energy import GTOElectrostaticEnergy
from mace.modules.polar.electrostatics.features import GTOElectrostaticFeatures
from mace.modules.polar.electrostatics.gto_utils import (
    DisplacedGTOExternalFieldBlock,
    gto_basis_kspace_cutoff,
)
from mace.modules.polar.electrostatics.kspace import compute_k_vectors_flat
from mace.modules.polar.electrostatics.realspace_electrostatics import (
    RealSpaceAnalyticalElectrostaticFeatures,
    RealSpaceAnalyticalEnergy,
    RealSpaceGeometry,
    merge_kernel_widths,
    prepare_real_space_geometry,
)
from mace.modules.utils import compute_polarizability as _compute_polarizability
from mace.modules.utils import (
    compute_total_charge_dipole_permuted,
    get_outputs,
    prepare_graph,
    safe_double,
)
from mace.modules.wrapper_ops import (
    CuEquivarianceConfig,
    OEQConfig,
    TransposeIrrepsLayoutWrapper,
    cueq_handles_linears,
)
from mace.tools.scatter import scatter_mean, scatter_sum

from .boundary_split import forward_by_boundary, graph_periodicity
from .conversion import (
    ensure_polar_compatibility,
    validate_pbc_handling,
)
from .field_blocks import (
    EnvironmentDependentSpinSourceBlock,
    MultiLayerFeatureMixer,
    field_readout_blocks,
    field_update_blocks,
)
from .linear_folds import (
    LayerOutputLinears,
    SharedInputPlan,
    evaluate_linears_sharing_input,
    linear_fold_of,
)


def _permute_to_e3nn_convention(x: torch.Tensor) -> torch.Tensor:
    return x[..., torch.LongTensor([1, 2, 0]).to(x.device)]


@compile_mode("script")
class PolarMACE(ScaleShiftMACE):
    def __init__(
        self,
        kspace_cutoff_factor: float = 1.5,
        atomic_multipoles_max_l: int = 0,
        atomic_multipoles_smearing_width: float = 1.0,
        field_feature_max_l: int = 0,
        field_feature_widths: List[float] = (1.0,),
        num_recursion_steps: int = 1,
        field_si: bool = False,
        include_electrostatic_self_interaction: bool = False,
        add_local_electron_energy: bool = False,
        quadrupole_feature_corrections: bool = False,
        return_electrostatic_potentials: bool = False,
        field_feature_norms: Optional[List[float]] = None,
        field_norm_factor: Optional[float] = 0.02,
        fixedpoint_update_config: Optional[Dict[str, Any]] = None,
        field_readout_config: Optional[Dict[str, Any]] = None,
        pbc_handling: str = "auto",
        **kwargs,
    ):
        try:
            hidden_irreps: o3.Irreps = kwargs["hidden_irreps"]
            MLP_irreps_raw = kwargs["MLP_irreps"]
            if isinstance(MLP_irreps_raw, int):
                MLP_irreps = o3.Irreps(f"{int(MLP_irreps_raw)}x0e")
            else:
                MLP_irreps = o3.Irreps(str(MLP_irreps_raw))
            gate = kwargs["gate"]
            avg_num_neighbors: float = kwargs["avg_num_neighbors"]
            num_interactions: int = kwargs["num_interactions"]
            num_elements: int = kwargs["num_elements"]
        except KeyError as exc:
            missing = str(exc).strip("'")
            raise KeyError(
                f"Missing required argument '{missing}' in kwargs for PolarMACE. "
                "Pass all ScaleShiftMACE/MACE constructor args as keyword arguments."
            ) from exc

        cueq_config: Optional[CuEquivarianceConfig] = kwargs.get("cueq_config", None)
        oeq_config: Optional[OEQConfig] = kwargs.get("oeq_config", None)
        # Keep a reference for config extraction tools
        self.cueq_config = cueq_config
        self.oeq_config = oeq_config

        # Defaults to mirror previous behavior
        heads = kwargs.get("heads", ["Default"]) or ["Default"]
        kwargs.setdefault("heads", heads)
        kwargs.setdefault("readout_cls", NonLinearReadoutBlock)
        kwargs.setdefault("use_agnostic_product", True)
        kwargs.setdefault("apply_cutoff", True)
        # Provide default atomic_inter_scale/shift if not passed
        kwargs.setdefault("atomic_inter_scale", [1.0] * len(heads))
        kwargs.setdefault("atomic_inter_shift", [0.0] * len(heads))

        # Swallow optional convenience args that shouldn't flow to super().__init__
        kwargs.pop("field_dependence_type", None)
        kwargs.pop("final_field_readout_type", None)
        kwargs.pop("keep_last_layer_irreps", None)

        # Initialize the MACE backbone (interactions, products, readouts, embeddings)
        super().__init__(**kwargs, keep_last_layer_irreps=True)

        self.kspace_cutoff_factor = float(kspace_cutoff_factor)
        self.num_recursion_steps = int(num_recursion_steps)
        self.atomic_multipoles_max_l = int(atomic_multipoles_max_l)
        self.atomic_multipoles_smearing_width = float(atomic_multipoles_smearing_width)
        self.field_feature_max_l = int(field_feature_max_l)
        self.field_feature_widths = list(field_feature_widths)
        self.field_norm_factor = float(field_norm_factor)
        self._field_feature_norms = field_feature_norms
        self.include_electrostatic_self_interaction = (
            include_electrostatic_self_interaction
        )
        self.atomic_multipoles_smearing_width = float(atomic_multipoles_smearing_width)
        self.add_local_electron_energy = add_local_electron_energy
        if quadrupole_feature_corrections:
            raise ValueError(
                "quadrupole_feature_corrections belonged to the molecule-in-box "
                "corrections, which were removed"
            )
        self.quadrupole_feature_corrections = False
        self.field_si = field_si
        self.keep_last_layer_irreps = True

        # k-space cutoff heuristic
        kspace_cutoff = kspace_cutoff_factor * gto_basis_kspace_cutoff(
            [atomic_multipoles_smearing_width] + list(field_feature_widths),
            max(atomic_multipoles_max_l, field_feature_max_l),
        )
        self.register_buffer(
            "kspace_cutoff",
            torch.tensor(kspace_cutoff, dtype=torch.get_default_dtype()),
        )

        # Normalization for field features
        if field_feature_norms is not None:
            assert len(field_feature_norms) == len(field_feature_widths) * (
                field_feature_max_l + 1
            ), f"{len(field_feature_widths) * (field_feature_max_l+1)}, {len(field_feature_norms)}"
        else:
            field_feature_norms = (
                [1.0] * len(field_feature_widths) * (field_feature_max_l + 1)
            )
        expanded: List[float] = []
        for l in range(field_feature_max_l + 1):
            for j in range(len(field_feature_widths)):
                expanded += [field_feature_norms[l * len(field_feature_widths) + j]] * (
                    2 * l + 1
                )
        self.register_buffer(
            "field_feature_norms",
            torch.tensor(expanded, dtype=torch.get_default_dtype()),
        )

        self.lr_source_maps = torch.nn.ModuleList(
            EnvironmentDependentSpinSourceBlock(
                irreps_in=hidden_irreps,
                max_l=atomic_multipoles_max_l,
                cueq_config=cueq_config,
            )
            for _ in range(num_interactions)
        )

        # Field-dependent components
        self.charges_irreps = 2 * o3.Irreps.spherical_harmonics(atomic_multipoles_max_l)
        charges_layout = (
            cueq_config.layout_str
            if (cueq_config is not None and cueq_config.enabled)
            else "mul_ir"
        )
        self._charges_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.charges_irreps,
            source=charges_layout,
            target="mul_ir",
            cueq_config=cueq_config,
        )
        self._charges_from_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.charges_irreps,
            source="mul_ir",
            target=charges_layout,
            cueq_config=cueq_config,
        )
        lr_sh_irreps = o3.Irreps.spherical_harmonics(field_feature_max_l)
        self.field_irreps = (
            (lr_sh_irreps * len(field_feature_widths)).sort()[0].simplify()
        )
        self.potential_irreps = (
            self.field_irreps * 2
        )  # 2 spin channels for the potential irreps
        # Sizes the forward needs, as integers: the forward must not touch e3nn Irreps
        # (Dynamo cannot guard on them, so torch.compile fails).
        self.charges_dimension = self.charges_irreps.dim
        self.potential_dimension = self.potential_irreps.dim

        self.electric_potential_descriptor = GTOElectrostaticFeatures(
            density_max_l=atomic_multipoles_max_l,
            density_smearing_width=atomic_multipoles_smearing_width,
            feature_max_l=field_feature_max_l,
            feature_smearing_widths=list(field_feature_widths),
            kspace_cutoff=kspace_cutoff,
            include_self_interaction=field_si,
            integral_normalization="receiver",
            pbc_handling=pbc_handling,
        )
        field_layout_target = (
            cueq_config.layout_str
            if (cueq_config is not None and cueq_config.enabled)
            else "mul_ir"
        )
        self._field_from_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.field_irreps,
            source="mul_ir",
            target=field_layout_target,
            cueq_config=cueq_config,
        )

        self.fukui_source_map = NonLinearBiasReadoutBlock(
            hidden_irreps,
            MLP_irreps.simplify(),
            gate,
            o3.Irreps("2x0e"),
            cueq_config=None,
            oeq_config=oeq_config,
        )
        fukui_layout = (
            cueq_config.layout_str
            if (cueq_config is not None and cueq_config.enabled)
            else "mul_ir"
        )
        self._fukui_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=hidden_irreps,
            source=fukui_layout,
            target="mul_ir",
            cueq_config=cueq_config,
        )
        if fixedpoint_update_config is None:
            fixedpoint_update_config = {}
        fixedpoint_update_config = fixedpoint_update_config.copy()
        self._fixedpoint_update_config = fixedpoint_update_config.copy()
        lr_source_cls = fixedpoint_update_config.pop(
            "type", "AgnosticEmbeddedOneBodyVariableUpdate"
        )
        if isinstance(lr_source_cls, str):
            lr_source_cls = field_update_blocks[lr_source_cls]
        # Map optional class names to implementations
        pe_cls = fixedpoint_update_config.get("potential_embedding_cls", None)
        if isinstance(pe_cls, str):
            # currently only AgnosticChargeBiasedLinearPotentialEmbedding is required
            from .field_blocks import (
                AgnosticChargeBiasedLinearPotentialEmbedding as _PE,
            )

            fixedpoint_update_config["potential_embedding_cls"] = _PE
        nl_cls = fixedpoint_update_config.get("nonlinearity_cls", None)
        if isinstance(nl_cls, str):
            from .field_blocks import MLPNonLinearity as _NL

            fixedpoint_update_config["nonlinearity_cls"] = _NL
        # Reconstruct irreps needed for field update maps
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        max_ell_field_update = 2
        field_update_sh_irreps = o3.Irreps.spherical_harmonics(max_ell_field_update)
        self.from_ell_max_field_update = (max_ell_field_update + 1) ** 2
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        field_interaction_irreps = (
            (field_update_sh_irreps * num_features).sort()[0].simplify()
        )
        self.field_dependent_charges_maps = torch.nn.ModuleList()
        for _ in range(num_recursion_steps):
            self.field_dependent_charges_maps.append(
                lr_source_cls(
                    node_attrs_irreps=node_attr_irreps,
                    node_feats_irreps=hidden_irreps,
                    edge_attrs_irreps=field_update_sh_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    target_irreps=field_interaction_irreps,
                    hidden_irreps=hidden_irreps,
                    avg_num_neighbors=avg_num_neighbors,
                    potential_irreps=self.potential_irreps,
                    charges_irreps=self.charges_irreps,
                    num_elements=num_elements,
                    field_norm_factor=float(field_norm_factor or 1.0),
                    cueq_config=cueq_config,
                    oeq_config=oeq_config,
                    **fixedpoint_update_config,
                )
            )

        # Post-SCF readout
        self.add_local_electron_energy = add_local_electron_energy
        if field_readout_config is None:
            field_readout_config = {}
        field_readout_config = field_readout_config.copy()
        self._field_readout_config = field_readout_config.copy()
        field_readout_cls = field_readout_config.pop("type", "OneBodyMLPFieldReadout")
        if isinstance(field_readout_cls, str):
            field_readout_cls = field_readout_blocks[field_readout_cls]
        self.local_electron_energy = field_readout_cls(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=hidden_irreps,
            edge_attrs_irreps=field_update_sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=field_interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            potential_irreps=self.potential_irreps,
            charges_irreps=self.charges_irreps,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
            **field_readout_config,
        )

        self.external_field_contribution = DisplacedGTOExternalFieldBlock(
            field_feature_max_l, list(field_feature_widths), "receiver"
        )
        self.coulomb_energy = GTOElectrostaticEnergy(
            density_max_l=atomic_multipoles_max_l,
            density_smearing_width=atomic_multipoles_smearing_width,
            kspace_cutoff=float(kspace_cutoff),
            include_self_interaction=include_electrostatic_self_interaction,
            pbc_handling=pbc_handling,
        )
        self.set_electrostatic_pbcs(pbc_handling)
        self.split_mixed_boundary_batches = True
        self.return_electrostatic_potentials = return_electrostatic_potentials
        self.layer_feature_mixer = MultiLayerFeatureMixer(
            node_feats_irreps=hidden_irreps,
            num_interactions=num_interactions,
            cueq_config=cueq_config,
        )
        self.prepare_linear_folds()

    def _readout_index(self, layer: int) -> int:
        return -1 if len(self.readouts) == 1 else min(layer, len(self.readouts) - 1)

    def prepare_linear_folds(self) -> None:
        """Plan the linears that read each layer's output (``shared_input_plan``).

        A layer's output is read by the feature mixer's linear of that layer, by the next
        interaction's ``skip_tp`` and ``linear_up`` and by a nonlinear readout's first
        linear. ``forward`` evaluates them together (``evaluate_linears_sharing_input``):
        their weights concatenated per input irrep, one GEMM per irrep instead of one per
        linear in the forward and in each backward. Exact up to the order of summation.
        Built from the module structure (integers only, no e3nn object in the forward);
        rebuilt on construction and on unpickling (``ensure_polar_compatibility``).
        """
        cueq_config = getattr(self, "cueq_config", None)
        layout: Optional[str] = None
        if cueq_handles_linears(cueq_config):
            layout = cueq_config.layout_str
        elif cueq_config is None or not cueq_config.enabled:
            layout = "mul_ir"
        # else e3nn linears on cueq-layout features: evaluated one by one.
        num_layers = len(self.interactions)
        layers: List[LayerOutputLinears] = []
        for layer in range(num_layers):
            linears = [self.layer_feature_mixer.linears[layer]]
            with_next = layer + 1 < num_layers and getattr(
                self.interactions[layer + 1], "accepts_first_linears", False
            )
            if with_next:
                next_interaction = self.interactions[layer + 1]
                linears += [next_interaction.skip_tp, next_interaction.linear_up]
            readout = self.readouts[self._readout_index(layer)]
            # A NonLinearBiasReadoutBlock (POLAR-M's last readout) is replaced by an fx
            # graph under compile (simplify_if_compile): its first linear cannot be
            # taken out, so only a NonLinearReadoutBlock's is.
            with_readout = isinstance(readout, NonLinearReadoutBlock)
            if with_readout:
                linears.append(readout.linear_1)
            folds = (
                [linear_fold_of(linear) for linear in linears]
                if layout is not None
                else None
            )
            layers.append(LayerOutputLinears(with_next, with_readout, folds))
        self.shared_input_plan = SharedInputPlan(layout, layers)

    def _layer_output_linears(self, layer: int, node_feats: torch.Tensor) -> Tuple[
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
    ]:
        """The linears that read layer ``layer``'s output, evaluated together: (the
        feature mixer's term, the next interaction's first linears or None, the
        readout's first linear or None)."""
        plan = self.shared_input_plan
        layer_linears = plan.layers[layer]
        linears = [self.layer_feature_mixer.linears[layer]]
        if layer_linears.with_next:
            next_interaction = self.interactions[layer + 1]
            linears += [next_interaction.skip_tp, next_interaction.linear_up]
        readout = self.readouts[self._readout_index(layer)]
        if layer_linears.with_readout:
            linears.append(readout.linear_1)
        outputs = evaluate_linears_sharing_input(
            node_feats, linears, layer_linears.folds, plan.layout
        )
        next_first_linears: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None
        if layer_linears.with_next:
            skip, up = outputs[1], outputs[2]
            residual = self.interactions[layer + 1].linear_res(up)
            next_first_linears = (skip, up, residual)
        readout_hidden = outputs[-1] if layer_linears.with_readout else None
        return outputs[0], next_first_linears, readout_hidden

    def __setstate__(self, state):
        """Restore whole-model pickles, repairing known legacy electrostatic state."""
        super().__setstate__(state)
        ensure_polar_compatibility(self)

    def set_electrostatic_pbcs(self, pbc_handling: str) -> None:
        """Select the same electrostatic evaluator for features and energy.

        ``realspace`` for open structures (clusters, molecules), ``pbc`` for fully
        periodic cells, ``auto`` chooses per batch (a batch mixing the two is split by
        boundary type in ``forward``). Explicit modes must match the input geometry.
        """
        validate_pbc_handling(pbc_handling)
        self.electric_potential_descriptor.set_pbc_handling(pbc_handling)
        self.coulomb_energy.set_pbc_handling(pbc_handling)
        self.pbc_handling = pbc_handling

    def prepare_real_space_geometry(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
        pair_index: Optional[torch.Tensor] = None,
    ) -> Optional["RealSpaceGeometry"]:
        """One real-space geometry for the field features and the Coulomb energy.

        The pair kernels depend only on the positions: they are evaluated once per
        forward for the union of the combined widths of both blocks, and every feature
        evaluation (each recursion step, both spin channels) and the energy read them.
        Autograd then differentiates the kernels once for the forces and once more in
        the backward through the forces, instead of once per evaluation.

        pair_index: directed pairs [2, n_pairs] within each open structure, built on
        the host by the data pipeline ("real_space_pair_index"); without it the pairs
        are built on the device (with host syncs).

        None when either block uses the finite-difference evaluator, which has no
        prepared form; both blocks then evaluate on their own.
        """
        features = self.electric_potential_descriptor.realspace_features
        energy = self.coulomb_energy.realspace_energy
        if not isinstance(
            features, RealSpaceAnalyticalElectrostaticFeatures
        ) or not isinstance(energy, RealSpaceAnalyticalEnergy):
            return None
        return prepare_real_space_geometry(
            positions,
            merge_kernel_widths(features.kernel_widths, energy.kernel_widths),
            max(features.highest_order, energy.highest_order),
            batch=batch,
            pair_index=pair_index,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
        fermi_level: Optional[torch.Tensor] = None,
        external_field: Optional[torch.Tensor] = None,
        compute_polarizability: bool = False,
        polarizability_method: str = "dipole",
    ) -> Dict[str, Optional[torch.Tensor]]:
        # A batch mixing periodic and aperiodic graphs would dispatch ALL of them to the
        # reciprocal-space electrostatics (see below): evaluate the two kinds separately so
        # every structure sees the evaluator it gets on its own (mace.modules.polar.boundary_split).
        if (
            self.pbc_handling == "auto"
            and getattr(self, "split_mixed_boundary_batches", True)
            and not lammps_mliap
            and "pbc" in data
        ):
            graph_is_periodic = graph_periodicity(data)
            if bool(graph_is_periodic.any()) and not bool(graph_is_periodic.all()):
                if compute_hessian:
                    raise NotImplementedError(
                        "hessians are not available for batches mixing periodic and "
                        "aperiodic graphs"
                    )
                return forward_by_boundary(
                    self.forward,
                    data,
                    graph_is_periodic,
                    training=training,
                    compute_force=compute_force,
                    compute_virials=compute_virials,
                    compute_stress=compute_stress,
                    compute_displacement=compute_displacement,
                    compute_hessian=compute_hessian,
                    compute_edge_forces=compute_edge_forces,
                    compute_atomic_stresses=compute_atomic_stresses,
                    lammps_mliap=lammps_mliap,
                    fermi_level=fermi_level,
                    external_field=external_field,
                    compute_polarizability=compute_polarizability,
                    polarizability_method=polarizability_method,
                )
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )

        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        pbc = ctx.pbc
        node_heads = ctx.node_heads
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        # prepare_graph retains the original position leaf in ctx for forces,
        # but puts affinely strained positions in data for the local graph.
        # Use that same geometry for every long-range contribution. Deriving
        # reciprocal cell and volume from the strained cell lets autograd add
        # the cell virial through the existing displacement derivative.
        long_range_positions = data["positions"]
        long_range_cell = cell.view(-1, 3, 3)
        long_range_rcell = data["rcell"].view(-1, 3, 3)
        long_range_volume = data["volume"]
        if compute_virials or compute_stress or compute_displacement:
            assert displacement is not None
            strain = 0.5 * (displacement + displacement.transpose(-1, -2))
            long_range_cell = long_range_cell + torch.matmul(long_range_cell, strain)
            long_range_rcell = (
                2 * torch.pi * torch.linalg.inv_ex(long_range_cell.transpose(-1, -2))[0]
            )
            long_range_volume = torch.linalg.det(long_range_cell).abs()

        if fermi_level is None:
            fermi_level = data["fermi_level"]
        if external_field is None:
            external_field = data["external_field"]
        if compute_polarizability and not external_field.requires_grad:
            external_field = external_field.clone().requires_grad_(True)
        external_potential = torch.hstack(
            (torch.zeros_like(fermi_level).unsqueeze(-1), external_field)
        )
        charges_to_mul_ir = getattr(self, "_charges_to_mul_ir", None)

        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(vectors.dtype)

        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(_permute_to_e3nn_convention(vectors))
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        node_es_list: List[torch.Tensor] = []
        node_feats_list: List[torch.Tensor] = []
        spin_charge_density = torch.zeros(
            (data["batch"].size(-1), self.charges_dimension),
            device=data["batch"].device,
            dtype=vectors.dtype,
        )

        # First linears of each interaction, evaluated outside it: in the first layer on
        # the element-embedding table (its input is linear in node_attrs), in later layers
        # together with the other linears of the previous layer's output.
        first_linears: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        if getattr(self.interactions[0], "accepts_first_linears", False):
            node_feats_table = self.node_embedding(
                torch.eye(
                    data["node_attrs"].shape[1],
                    dtype=data["node_attrs"].dtype,
                    device=data["node_attrs"].device,
                )
            )
            first_linears = self.interactions[0].first_linears_from_element_table(
                data["node_attrs"], node_feats_table
            )
        mixer_terms: List[torch.Tensor] = []
        for i, (interaction, product, lr_src) in enumerate(
            zip(self.interactions, self.products, self.lr_source_maps)
        ):
            node_attrs_slice = data["node_attrs"]
            if is_lammps and i > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            interaction_kwargs: Dict[str, Any] = {}
            if first_linears is not None:
                interaction_kwargs.update(first_linears=first_linears)
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(i == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
                **interaction_kwargs,
            )
            if is_lammps and i == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_list.append(node_feats)

            mixer_term, first_linears, readout_hidden = self._layer_output_linears(
                i, node_feats
            )
            mixer_terms.append(mixer_term)
            readout = self.readouts[self._readout_index(i)]
            if readout_hidden is not None:
                node_es = readout.forward_from_hidden(readout_hidden, node_heads)
            else:
                node_es = readout(node_feats, node_heads)
            node_es_list.append(node_es[num_atoms_arange, node_heads])

            spin_charge_sources = lr_src(node_feats).squeeze(-2)
            spin_charge_density = spin_charge_density + spin_charge_sources

        node_feats_out = torch.cat(node_feats_list, dim=-1)
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)

        # Only the compatibility mode inspects tensor PBCs for dispatch.
        realspace = self.pbc_handling == "realspace"
        if self.pbc_handling == "auto":
            realspace = not bool(torch.any(pbc))
        realspace_geometry: Optional[RealSpaceGeometry] = None
        if realspace:
            k_vectors = positions.new_empty((0, 3))
            kv_norms_squared = positions.new_empty((0,))
            k_vectors_batch = data["batch"].new_empty((0,))
            k_vectors_0mask = positions.new_empty((0,))
            realspace_geometry = self.prepare_real_space_geometry(
                long_range_positions,
                data["batch"],
                pair_index=data.get("real_space_pair_index"),
            )
        else:
            (
                k_vectors,
                kv_norms_squared,
                k_vectors_batch,
                k_vectors_0mask,
            ) = compute_k_vectors_flat(
                self.kspace_cutoff, long_range_cell, long_range_rcell
            )

        field_feature_cache = self.electric_potential_descriptor.precompute_geometry(
            k_vectors=k_vectors,
            k_norm2=kv_norms_squared,
            k_vector_batch=k_vectors_batch,
            k0_mask=k_vectors_0mask,
            node_positions=long_range_positions,
            batch=data["batch"],
            volume=long_range_volume,
            pbc=data["pbc"].view(-1, 3),
            realspace_geometry=realspace_geometry,
        )

        # SCF fixed point. The mixer's sum over layers (MultiLayerFeatureMixer), from the
        # terms evaluated with each layer's output.
        features_mixed = mixer_terms[0]
        for term in mixer_terms[1:]:
            features_mixed = features_mixed + term
        spin_charge_density = spin_charge_density.view(
            spin_charge_density.shape[0], 2, -1
        )
        fukui_input = node_feats
        fukui_to_mul_ir = getattr(self, "_fukui_to_mul_ir", None)
        if fukui_to_mul_ir is not None:
            fukui_input = fukui_to_mul_ir(fukui_input)
        fukui_sources = self.fukui_source_map(fukui_input)
        fukui_norm = scatter_sum(
            src=safe_double(fukui_sources),
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )[data["batch"]].to(vectors.dtype)
        fukui_norm = torch.where(
            fukui_norm == 0, torch.ones_like(fukui_norm), fukui_norm
        )
        fukui_sources = fukui_sources / fukui_norm
        Q_p_S = (data["total_charge"] + (data["total_spin"] - 1))[data["batch"]]
        Q_m_S = (data["total_charge"] - (data["total_spin"] - 1))[data["batch"]]
        pred_total_charges_0 = scatter_sum(
            src=safe_double(spin_charge_density[:, :, 0]),
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )[data["batch"]].to(vectors.dtype)
        spin_charge_density = spin_charge_density.clone()
        spin_charge_density[:, 0, 0] = spin_charge_density[:, 0, 0] + fukui_sources[
            :, 0
        ] * ((Q_p_S / 2) - pred_total_charges_0[:, 0])
        spin_charge_density[:, 1, 0] = spin_charge_density[:, 1, 0] + fukui_sources[
            :, 1
        ] * ((Q_m_S / 2) - pred_total_charges_0[:, 1])
        # print("spin_charge_density", spin_charge_density)

        potential_features = torch.zeros(
            (data["batch"].size(-1), self.potential_dimension),
            device=data["batch"].device,
            dtype=vectors.dtype,
        )
        field_independent_spin_charge_density = spin_charge_density.clone()
        esps: Optional[torch.Tensor] = None

        final_fukui_sources = fukui_sources
        field_from_mul_ir = getattr(self, "_field_from_mul_ir", None)
        # External field potential relative to the barycenter (gauge invariance). It
        # depends on the positions only: the same in every recursion step. Added once
        # per spin channel, hence the half.
        barycenter = scatter_mean(
            src=safe_double(long_range_positions),
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        ).to(positions.dtype)
        half_external_field = 0.5 * self.external_field_contribution(
            data["batch"],
            long_range_positions - barycenter[data["batch"], :],
            external_potential,
        )
        num_nodes = spin_charge_density.shape[0]
        for i in range(self.num_recursion_steps):
            # Both spin channels [n_nodes, 2, C] as one source set axis: one evaluation.
            source_feats = spin_charge_density
            if charges_to_mul_ir is not None:
                source_feats = charges_to_mul_ir(
                    source_feats.reshape(num_nodes * 2, -1)
                ).reshape(num_nodes, 2, -1)
            field_feats = self.electric_potential_descriptor.forward_dynamic(
                cache=field_feature_cache,
                source_feats=source_feats,
            )
            if field_from_mul_ir is not None:
                field_feats = field_from_mul_ir(
                    field_feats.reshape(num_nodes * 2, -1)
                ).reshape(num_nodes, 2, -1)
            esps = None
            field_feats = (
                field_feats + half_external_field.unsqueeze(1)
            ) / self.field_feature_norms
            # [alpha features, beta features] along the last axis.
            potential_features = field_feats.reshape(num_nodes, -1)
            charge_sources_out = self.field_dependent_charges_maps[i](
                node_attrs=data["node_attrs"],
                node_feats=features_mixed,
                edge_attrs=edge_attrs[:, : self.from_ell_max_field_update],
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                potential_features=potential_features,
                local_charges=spin_charge_density.view(
                    spin_charge_density.shape[0], -1
                ),
            )

            current_fukui_sources = charge_sources_out[:, -2:]
            charge_sources = charge_sources_out[:, :-2]
            # print("charge_sources", charge_sources)
            # print("current_fukui_sources", current_fukui_sources)
            spin_charge_density_sources = charge_sources.view(
                spin_charge_density.shape[0], 2, -1
            )
            spin_charge_density = spin_charge_density + spin_charge_density_sources

            fukui_norm2 = scatter_sum(
                src=safe_double(current_fukui_sources),
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )[data["batch"]].to(vectors.dtype)
            fukui_norm2 = torch.where(
                fukui_norm2 == 0, torch.ones_like(fukui_norm2), fukui_norm2
            )
            current_fukui_sources = current_fukui_sources / fukui_norm2
            final_fukui_sources = current_fukui_sources
            pred_total_charges = scatter_sum(
                src=safe_double(spin_charge_density[:, :, 0]),
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )[data["batch"]].to(vectors.dtype)
            spin_charge_density = spin_charge_density.clone()
            spin_charge_density[:, 0, 0] = spin_charge_density[
                :, 0, 0
            ] + current_fukui_sources[:, 0] * ((Q_p_S / 2) - pred_total_charges[:, 0])
            spin_charge_density[:, 1, 0] = spin_charge_density[
                :, 1, 0
            ] + current_fukui_sources[:, 1] * ((Q_m_S / 2) - pred_total_charges[:, 1])

        total_energy = e0 + inter_e
        local_q_e = self.local_electron_energy(
            node_attrs=data["node_attrs"],
            node_feats=node_feats,
            edge_attrs=edge_attrs[:, : self.from_ell_max_field_update],
            edge_feats=edge_feats,
            edge_index=data["edge_index"],
            field_feats=potential_features,
            charges_0=field_independent_spin_charge_density.view(
                field_independent_spin_charge_density.shape[0], -1
            ),
            charges_induced=spin_charge_density.view(spin_charge_density.shape[0], -1),
        )
        le_total = scatter_sum(
            src=local_q_e, index=data["batch"], dim=-1, dim_size=num_graphs
        )
        if getattr(self, "add_local_electron_energy", False):
            total_energy = total_energy + le_total
        else:
            le_total = torch.zeros_like(le_total)

        charge_density = spin_charge_density.sum(dim=1)
        spin_density = spin_charge_density[:, 0, :] - spin_charge_density[:, 1, :]
        charge_density_mul_ir = (
            charges_to_mul_ir(charge_density)
            if charges_to_mul_ir is not None
            else charge_density
        )
        spin_density_mul_ir = (
            charges_to_mul_ir(spin_density)
            if charges_to_mul_ir is not None
            else spin_density
        )
        spin_charge_density_mul_ir = (
            torch.stack(
                [
                    charges_to_mul_ir(spin_charge_density[:, 0, :]),
                    charges_to_mul_ir(spin_charge_density[:, 1, :]),
                ],
                dim=1,
            )
            if charges_to_mul_ir is not None
            else spin_charge_density
        )
        total_charge, total_dipole = compute_total_charge_dipole_permuted(
            charge_density_mul_ir, long_range_positions, data["batch"], num_graphs
        )
        electro_energy = self.coulomb_energy(
            k_vectors=k_vectors,
            k_norm2=kv_norms_squared,
            k_vector_batch=k_vectors_batch,
            k0_mask=k_vectors_0mask,
            source_feats=charge_density_mul_ir,
            node_positions=long_range_positions,
            batch=data["batch"],
            volume=long_range_volume,
            pbc=data["pbc"].view(-1, 3),
            realspace_geometry=realspace_geometry,
        )
        total_energy = (
            total_energy
            + electro_energy
            + torch.sum(external_potential[:, 1:] * total_dipole, dim=-1)
        )

        # Before get_outputs, which frees the graph unless training.
        # "dipole": alpha = -d(dipole)/d(external_field), one autograd pass.
        # "energy": alpha = -d2(energy)/d(external_field)2, two passes; also
        # yields dipole_from_energy. See mace.modules.utils.compute_polarizability.
        polarizability: Optional[torch.Tensor] = None
        dipole_from_energy: Optional[torch.Tensor] = None
        if compute_polarizability:
            polarizability, dipole_from_energy = _compute_polarizability(
                energy=total_energy,
                dipole=total_dipole,
                external_field=external_field,
                method=polarizability_method,
            )

        forces, virials, stress, hessian, edge_forces, _ = get_outputs(
            energy=total_energy,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            pbc=pbc,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces or compute_atomic_stresses,
        )

        atomic_virials: Optional[torch.Tensor] = None
        atomic_stresses: Optional[torch.Tensor] = None
        if compute_atomic_stresses and edge_forces is not None:
            from .utils import get_atomic_virials_stresses as _gav

            atomic_virials, atomic_stresses = _gav(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
                pbc=pbc,
            )

        return {
            "energy": total_energy,
            "node_energy": safe_double(node_e0.clone())
            + safe_double(node_inter_es.clone()),
            "interaction_energy": inter_e,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "hessian": hessian,
            "displacement": displacement,
            "node_feats": node_feats_out,
            "density_coefficients": charge_density_mul_ir,
            "spin_density": spin_density_mul_ir,
            "charges_history": torch.stack(
                [spin_charge_density_mul_ir.clone().detach()], dim=-1
            ),
            "fermi_level": external_potential[:, 0],
            "external_field": external_potential[:, 1:],
            "charges": charge_density_mul_ir[:, 0],
            "spins": spin_density_mul_ir[:, 0],
            "dipole": total_dipole,
            "polarizability": polarizability,
            "dipole_from_energy": dipole_from_energy,
            "total_charge": total_charge,
            "electrostatic_energy": electro_energy,
            "electron_energy": le_total,
            "electrostatic_potentials": esps,
            "spin_charge_density": spin_charge_density_mul_ir,
            "fukui_functions": final_fukui_sources,
        }
