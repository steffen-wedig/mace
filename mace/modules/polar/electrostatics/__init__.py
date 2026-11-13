"""GTO electrostatics of MACE-POLAR (lifted from graph_longrange at f81f242, MIT, see LICENSE.md).

Smeared-Gaussian (GTO) charge and multipole densities, their fields as node features and
their Coulomb energy: real space for open structures (``realspace_electrostatics``, with a
geometry prepared once per forward), reciprocal space for periodic cells (``kspace``,
``features``, ``energy``), slabs (``slabs``), external sources and jellium. Formerly the
separate ``graph_longrange`` package; models pickled against it load through
``mace.modules.polar.conversion.load_polar_model``.
"""
