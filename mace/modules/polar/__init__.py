"""MACE-POLAR: PolarMACE and its field blocks, weight folds and electrostatics glue.

- ``model``: ``PolarMACE`` (the forward and its plans of evaluation).
- ``field_blocks``: field-dependent charge maps, field readout, feature mixer, the folded
  sparse dot products.
- ``linear_folds``: equivariant linears as per-irrep matrix products (folds into products,
  several linears of one input as one product).
- ``boundary_split``: batches mixing periodic and open structures, evaluated per kind.
- ``conversion``: repair of loaded models, and loading pickles written before this package
  existed (``load_polar_model``; ``python -m mace.modules.polar.conversion``).
"""

from .model import PolarMACE

__all__ = ["PolarMACE"]
