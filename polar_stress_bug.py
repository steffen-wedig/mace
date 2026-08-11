"""Minimal reproducer: PolarMACE stress disagrees with finite difference.

The analytic (autograd) stress is compared against ASE's numerical stress, and
the mismatch is then localised to a single energy term by comparing, term by
term, the hydrostatic strain derivative from autograd against finite difference.

Short-range (`interaction_energy`) agrees. The two reciprocal-space terms do not:
`PolarMACE.forward` builds the k-grid from `data["rcell"]` and passes
`volume=data["volume"]` (both precomputed in `AtomicData.from_config`, hence
strain-independent) and passes the unstrained `ctx.positions`, so the k-space
cell derivative and the 1/V term are missing from the virial.

    python polar_stress_bug.py

Requires the polar extra (`graph_longrange`); downloads MACE-POLAR-1-S on first run.
"""

from __future__ import annotations

import numpy as np
import torch
from ase import Atoms
from ase.build import molecule
from ase.calculators.fd import calculate_numerical_stress

from mace.calculators.foundations_models import mace_polar

VOIGT = ["xx", "yy", "zz", "yz", "xz", "xy"]
TERMS = ["interaction_energy", "electron_energy", "electrostatic_energy", "energy"]
EPS_STRESS = 1e-6
EPS_STRAIN = 1e-5


def dense_water_box() -> Atoms:
    """Eight water molecules in a slightly triclinic ~6.3 A cell (~1 g/cm3)."""
    rng = np.random.default_rng(20260730)
    cell = np.array([[6.30, 0.10, 0.05], [0.08, 6.20, -0.06], [0.04, -0.07, 6.25]])
    box = Atoms(cell=cell, pbc=True)
    for corner in np.ndindex(2, 2, 2):
        water = molecule("H2O")
        water.rotate(float(rng.uniform(0.0, 360.0)), tuple(rng.normal(size=3)))
        water.translate(((np.asarray(corner) + 0.5) / 2.0) @ cell)
        box += water
    box.info["charge"] = 0
    box.info["spin"] = 1
    return box


def energy_terms(calc, atoms: Atoms) -> dict:
    batch = calc._clone_batch(calc._atoms_to_batch(atoms))  # noqa: SLF001
    return calc.models[0](batch.to_dict(), compute_stress=False, training=False)


def strain_derivative_autograd(calc, atoms: Atoms, term: str) -> float:
    """Trace of d<term>/d(strain). The strain leaf is injected via
    data["displacement"], which prepare_graph honours."""
    batch = calc._clone_batch(calc._atoms_to_batch(atoms))  # noqa: SLF001
    data = batch.to_dict()
    displacement = torch.zeros((1, 3, 3), dtype=torch.float64, requires_grad=True)
    data["displacement"] = displacement
    out = calc.models[0](data, compute_stress=True, training=True)
    (grad,) = torch.autograd.grad(out[term].sum(), displacement, allow_unused=True)
    return 0.0 if grad is None else float(torch.diagonal(grad[0]).sum())


def strain_derivative_fd(calc, atoms: Atoms, term: str) -> float:
    """Same quantity by central difference: cell and positions scaled together."""
    cell = atoms.cell.array.copy()
    energies = []
    for scale in (1.0 + EPS_STRAIN, 1.0 - EPS_STRAIN):
        scaled = atoms.copy()
        scaled.info.update(atoms.info)
        scaled.set_cell(cell * scale, scale_atoms=True)
        energies.append(float(energy_terms(calc, scaled)[term].sum().detach()))
    return (energies[0] - energies[1]) / (2.0 * EPS_STRAIN)


def main() -> None:
    atoms = dense_water_box()
    calc = mace_polar(model="polar-1-s", device="cpu", default_dtype="float64")
    atoms.calc = calc

    print(f"\n{len(atoms)} atoms, V = {atoms.get_volume():.3f} A^3")

    print(f"\n[1] stress in eV/A^3 (ASE Voigt order), eps = {EPS_STRESS}\n")
    analytic = atoms.get_stress(voigt=True)
    numerical = calculate_numerical_stress(atoms, eps=EPS_STRESS, voigt=True)
    print(f"{'comp':>5} {'numerical_FD':>14} {'analytic':>14} {'abs_err':>12} {'rel_err':>10}")
    for i, name in enumerate(VOIGT):
        num, ana = float(numerical[i]), float(analytic[i])
        abs_err = abs(num - ana)
        rel_err = abs_err / max(abs(ana), abs(num), 1e-12)
        print(f"{name:>5} {num:>14.6e} {ana:>14.6e} {abs_err:>12.3e} {rel_err * 100:>9.2f}%")

    print("\n[2] hydrostatic strain derivative dE/d(eps) in eV, per energy term\n")
    print(f"{'term':>22} {'numerical_FD':>14} {'autograd':>14} {'rel_err':>10}")
    for term in TERMS:
        fd = strain_derivative_fd(calc, atoms, term)
        ana = strain_derivative_autograd(calc, atoms, term)
        rel_err = abs(fd - ana) / max(abs(fd), abs(ana), 1e-12)
        print(f"{term:>22} {fd:>14.6e} {ana:>14.6e} {rel_err * 100:>9.2f}%")


if __name__ == "__main__":
    main()
