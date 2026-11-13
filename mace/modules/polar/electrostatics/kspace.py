"""Reciprocal-space sums: k-vectors and Fourier series of the periodic electrostatics.

Rows of a k-vector set that are flagged in ``excluded_k_mask`` (the origin, rows beyond the
cutoff, padding rows) contribute exactly zero, with zero gradient: their Coulomb factor is
zero and the basis replaces their modulus and direction before any division
(``GTOBasis``, ``compute_coulomb_factor``).

Two ways to the k-vectors of a batch:

- ``compute_k_vectors_flat``: everything from the cell inside the forward (data-dependent
  shapes and a loop over structures; eager only);
- ``reciprocal_lattice_indices`` outside the forward, once per cell change: integer lattice
  indices ``m`` per structure, a superset of the cutoff sphere (optionally padded to a
  fixed row count), and ``lattice_k_vectors`` inside the forward: ``k = m B`` from the
  (strained) reciprocal cell ``B`` with the cutoff applied as ``excluded_k_mask``. Fixed
  shapes, no host reads, and the stress derivative flows through ``B``. The origin is never
  listed (its Coulomb factor is zero; neutral or not, it adds nothing), and only one of each
  ``+/-m`` pair, as in ``compute_k_vectors_flat``.
"""

from typing import NamedTuple, Optional, Union

import numpy as np
import torch
from scipy.constants import pi


class ReciprocalLatticeIndices(NamedTuple):
    """Integer lattice indices of the k-vectors of a batch, one row per k-vector.

    miller: [n_k, 3] in the cell's dtype (integer values), half space, origin excluded;
    structure: [n_k] long, the structure owning each row (rows grouped by structure);
    padding: [n_k] bool, rows that only fill a fixed row count (always excluded).
    """

    miller: torch.Tensor
    structure: torch.Tensor
    padding: torch.Tensor


def _half_space_indices(extent: np.ndarray) -> np.ndarray:
    """All integer ``m`` with ``|m_i| <= extent_i`` in the half space of
    ``compute_k_vectors_flat`` (``m_1 > 0``, or ``m_1 = 0, m_2 > 0``, or ``m_1 = m_2 = 0,
    m_3 > 0``): one of each ``+/-m`` pair, origin excluded."""
    ranges = [np.arange(-int(n), int(n) + 1) for n in extent]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    first, second, third = grid[:, 0], grid[:, 1], grid[:, 2]
    keep = (
        (first > 0)
        | ((first == 0) & (second > 0))
        | ((first == 0) & (second == 0) & (third > 0))
    )
    return grid[keep]


def reciprocal_lattice_indices(
    cell: torch.Tensor,
    cutoff: float,
    margin: float = 0.0,
    rows: Optional[int] = None,
) -> ReciprocalLatticeIndices:
    """Lattice indices of every k-vector within ``cutoff * (1 + margin)`` of each cell.

    cell: [n_graphs, 3, 3], rows are the lattice vectors (read on the host: call outside
    a compiled region). ``margin`` keeps the set valid while the cell changes (NPT): a
    k-vector enters the sum only if ``|k| <= cutoff`` at the cell of the evaluation
    (``lattice_k_vectors``), so the result equals the cutoff sphere's as long as the
    sphere stays inside the listed set. ``rows`` pads to a fixed row count (padding rows
    are excluded, owned by the last structure); fewer rows than needed is an error.
    """
    cells = cell.detach().to("cpu", torch.float64).reshape(-1, 3, 3).numpy()
    reach = cutoff * (1.0 + margin)
    miller_blocks, owner_blocks = [], []
    for index, lattice in enumerate(cells):
        reciprocal = 2 * np.pi * np.linalg.inv(lattice.T)  # rows: reciprocal vectors
        # |m_i| <= reach |a_i| / 2 pi bounds every m with |m B| <= reach.
        extent = np.floor(reach * np.linalg.norm(lattice, axis=1) / (2 * np.pi))
        candidates = _half_space_indices(extent)
        norms = np.linalg.norm(candidates @ reciprocal, axis=1)
        selected = candidates[norms <= reach]
        miller_blocks.append(selected)
        owner_blocks.append(np.full(len(selected), index))
    miller = np.concatenate(miller_blocks).astype(np.float64)
    owner = np.concatenate(owner_blocks)
    padding = np.zeros(len(miller), dtype=bool)
    if rows is not None:
        if rows < len(miller):
            raise ValueError(f"{len(miller)} k-vectors do not fit in {rows} rows")
        extra = rows - len(miller)
        # Padding rows: m = (1, 0, 0) (finite k), always excluded.
        miller = np.concatenate([miller, np.tile([[1.0, 0.0, 0.0]], (extra, 1))])
        owner = np.concatenate([owner, np.full(extra, len(cells) - 1)])
        padding = np.concatenate([padding, np.ones(extra, dtype=bool)])
    device = cell.device
    return ReciprocalLatticeIndices(
        miller=torch.as_tensor(miller, dtype=cell.dtype, device=device),
        structure=torch.as_tensor(owner, dtype=torch.long, device=device),
        padding=torch.as_tensor(padding, device=device),
    )


def lattice_indices_cover(
    listed_cell: torch.Tensor, cell: torch.Tensor, margin: float
) -> torch.Tensor:
    """Whether indices listed at ``listed_cell`` with ``margin`` still hold every k-vector
    within the cutoff at ``cell`` (both [n_graphs, 3, 3], rows are lattice vectors).

    With ``B = 2 pi cell^-T``: ``|m B_listed| <= ||B^-1 B_listed|| |m B|`` and
    ``||B^-1 B_listed|| = ||listed_cell^-1 cell||`` (spectral norm), so the listed sphere of
    radius ``cutoff (1 + margin)`` covers the cutoff sphere at ``cell`` while that norm is at
    most ``1 + margin``. A 0-d bool tensor on the cells' device: reading it is the caller's
    (one host read every few steps of an NPT run, not every step).
    """
    relative = torch.linalg.solve(listed_cell.reshape(-1, 3, 3), cell.reshape(-1, 3, 3))
    stretch = torch.linalg.matrix_norm(relative, ord=2)
    return torch.all(stretch <= 1.0 + margin)


def lattice_k_vectors(
    indices: ReciprocalLatticeIndices,
    r_cell_vectors: torch.Tensor,  # [n_graphs, 3, 3], rows: 2 pi (cell^-1)^T rows
    cutoff: Union[float, torch.Tensor],
):
    """k-vectors of listed lattice indices at the given (strained) reciprocal cells.

    Returns ``(k_vectors [n_k, 3], k_norm2 [n_k], k_vector_batch [n_k],
    excluded_k_mask [n_k])`` as ``compute_k_vectors_flat`` does; rows beyond ``cutoff`` at
    this cell and padding rows are excluded. Fixed shapes, no host reads (``cutoff`` may
    be a tensor, e.g. the model's buffer).
    """
    k_vectors = torch.einsum(
        "ki,kij->kj", indices.miller, r_cell_vectors[indices.structure]
    )
    k_norm2 = torch.einsum("ki,ki->k", k_vectors, k_vectors)
    excluded = torch.logical_or(k_norm2 > cutoff * cutoff, indices.padding)
    return k_vectors, k_norm2, indices.structure, excluded.to(k_vectors.dtype)


def compute_k_vectors_flat(
    cutoff: float,
    cell_vectors: torch.Tensor,  # [n_graphs, 3, 3]
    r_cell_vectors: torch.Tensor,  # [n_graphs, 3, 3]
):
    """
    Given a batch of cells, computes graph-grouped flattened k-vectors.
    The search-space heuristic follows the previous implementation.

    note that this function is meant to run on gpu - so cannot do the matrix inverse internally

        - cutoff
            reciporical space cutoff
        - cell_vectors [n_graph, 3, 3]
            where the -2 dimension indexes the different lattice
            vectors, and the -1 index is the spatial dimension.

    returns:
        - kvectors
            flattened tensor [n_k_total, 3], grouped by graph id
        - k_vectors_normed_squared
            flattened tensor [n_k_total] of squared norms
        - kvector_batch
            tensor [n_k_total] (long) giving graph id for each k-vector
        - excluded_k_mask
            float tensor [n_k_total], 1.0 where k=(0,0,0), else 0.0
    """
    device = cell_vectors.device

    # compute normed real lattice vectors
    norms = torch.norm(cell_vectors, dim=-1)  # [n_graph, 3]
    normed_lattice_vectors = cell_vectors / norms.unsqueeze(-1)  # [n_graph, 3, 3]

    dot_products = torch.einsum(
        "bij,bij->bi", r_cell_vectors, normed_lattice_vectors
    )  # [n_graph, 3]
    max_ns = cutoff * torch.pow(dot_products, -1)  # [n_graph, 3]
    max_ns = torch.ceil(max_ns).type(torch.int64)

    max_max_ns = torch.max(max_ns, dim=0).values  # [3]
    n1max = max_max_ns[..., 0]
    n2max = max_max_ns[..., 1]
    n3max = max_max_ns[..., 2]

    # make a single superset of all coeficients
    origin = torch.cartesian_prod(
        torch.arange(0, 1, 1, device=device),
        torch.arange(0, 1, 1, device=device),
        torch.arange(0, 1, 1, device=device),
    ).type(torch.float32)

    open_half_sphere = torch.cartesian_prod(
        torch.arange(1, n1max, 1, device=device),
        torch.arange(-n2max, n2max, 1, device=device),
        torch.arange(-n3max, n3max, 1, device=device),
    ).type(torch.float32)

    open_half_plane = torch.cartesian_prod(
        torch.arange(0, 1, 1, device=device),
        torch.arange(1, n2max, 1, device=device),
        torch.arange(-n3max, n3max, 1, device=device),
    ).type(torch.float32)

    open_half_line = torch.cartesian_prod(
        torch.arange(0, 1, 1, device=device),
        torch.arange(0, 1, 1, device=device),
        torch.arange(1, n3max, 1, device=device),
    ).type(torch.float32)

    kvecs = torch.cat(
        (origin, open_half_line, open_half_plane, open_half_sphere), dim=-2
    )
    kvecs = kvecs.to(r_cell_vectors.dtype)

    # produce all kvectors for all graphs with the largest superset of coeficients
    k_vectors = torch.einsum(
        "ni,bij->bnj", kvecs, r_cell_vectors
    )  # [n_graphs, n_kvec, 3]

    # find a mask of all of them which are too large
    k_vectors_normed_squared = torch.einsum(
        "bni,bni->bn", k_vectors, k_vectors
    )  # [n_graphs, n_kvec]
    mask = k_vectors_normed_squared.le(cutoff * cutoff)  # [n_graphs, n_kvec]

    # Keep only valid vectors per graph and flatten as [n_k_total, ...], grouped by graph.
    # This also acts on the old TODO by dropping vectors excluded in each graph.
    k_vectors_flat = []
    k_norm2_flat = []
    k_batch_flat = []
    excluded_k_mask_flat = []

    for graph_i in range(k_vectors.size(0)):
        graph_mask = mask[graph_i]

        graph_k = k_vectors[graph_i][graph_mask]
        graph_k_norm2 = k_vectors_normed_squared[graph_i][graph_mask]
        graph_k0_mask = torch.zeros_like(graph_k_norm2)
        graph_k0_mask[0] = 1.0

        k_vectors_flat.append(graph_k)
        k_norm2_flat.append(graph_k_norm2)
        k_batch_flat.append(
            torch.full(
                (graph_k.size(0),),
                graph_i,
                dtype=torch.long,
                device=device,
            )
        )
        excluded_k_mask_flat.append(graph_k0_mask)

    k_vectors_flat = torch.cat(k_vectors_flat, dim=0)  # [n_k_total, 3]
    k_norm2_flat = torch.cat(k_norm2_flat, dim=0)  # [n_k_total]
    k_batch_flat = torch.cat(k_batch_flat, dim=0)  # [n_k_total]
    excluded_k_mask_flat = torch.cat(excluded_k_mask_flat, dim=0)  # [n_k_total]
    return k_vectors_flat, k_norm2_flat, k_batch_flat, excluded_k_mask_flat


def compute_k_vectors(
    cutoff: float,
    cell_vectors: torch.Tensor,  # [n_graphs, 3, 3]
    r_cell_vectors: torch.Tensor,  # [n_graphs, 3, 3]
):
    """
    given a batch of cells, computes a tensor of k-vectors [n_graph, n_max_k, 3], and a mask [n_graph, n_max_k] describing which k vectors should be included.
    The method for determining the initial search space is one I just made up by looking at parallelograms.

    note that this function is meant to run on gpu - so cannot do the matrix inverse internally

        - cutoff
            reciporical space cutoff
        - cell_vectors [n_graph, 3, 3]
            where the -2 dimension indexes the different lattice
            vectors, and the -1 index is the spatial dimension.

    returns:
        - kvectors
            a tensor [n_graph, n_max_k, 3] of the k-vectors associated with each graph
        - k_vectors_normed_squared
            a tensor [n_graph, n_max_k] of the squared norms of the k-vectors
        - mask
            mask describing which of these shuold be included for each graph.
    """
    device = cell_vectors.device

    # compute normed real lattice vectors
    norms = torch.norm(cell_vectors, dim=-1)  # [n_graph, 3]
    normed_lattice_vectors = cell_vectors / norms.unsqueeze(-1)  # [n_graph, 3, 3]

    dot_products = torch.einsum(
        "bij,bij->bi", r_cell_vectors, normed_lattice_vectors
    )  # [n_graph, 3]
    max_ns = cutoff * torch.pow(dot_products, -1)  # [n_graph, 3]
    max_ns = torch.ceil(max_ns).type(torch.int64)

    max_max_ns = torch.max(max_ns, dim=0).values  # [3]
    n1max = max_max_ns[..., 0]
    n2max = max_max_ns[..., 1]
    n3max = max_max_ns[..., 2]

    # make a single superset of all coeficients
    origin = torch.cartesian_prod(
        torch.arange(0, 1, 1, device=device),
        torch.arange(0, 1, 1, device=device),
        torch.arange(0, 1, 1, device=device),
    ).type(torch.float32)

    open_half_sphere = torch.cartesian_prod(
        torch.arange(1, n1max, 1, device=device),
        torch.arange(-n2max, n2max, 1, device=device),
        torch.arange(-n3max, n3max, 1, device=device),
    ).type(torch.float32)

    open_half_plane = torch.cartesian_prod(
        torch.arange(0, 1, 1, device=device),
        torch.arange(1, n2max, 1, device=device),
        torch.arange(-n3max, n3max, 1, device=device),
    ).type(torch.float32)

    open_half_line = torch.cartesian_prod(
        torch.arange(0, 1, 1, device=device),
        torch.arange(0, 1, 1, device=device),
        torch.arange(1, n3max, 1, device=device),
    ).type(torch.float32)

    kvecs = torch.cat(
        (origin, open_half_line, open_half_plane, open_half_sphere), dim=-2
    )
    kvecs = kvecs.to(r_cell_vectors.dtype)

    # produce all kvectors for all graphs with the largest superset of coeficients
    k_vectors = torch.einsum(
        "ni,bij->bnj", kvecs, r_cell_vectors
    )  # [n_graphs, n_kvec, 3]

    # find a mask of all of them which are too large
    k_vectors_normed_squared = torch.einsum(
        "bni,bni->bn", k_vectors, k_vectors
    )  # [n_graphs, n_kvec]
    mask = k_vectors_normed_squared.le(cutoff * cutoff)  # [n_graphs, n_kvec]

    return k_vectors, k_vectors_normed_squared, mask


def evaluate_fourier_series_at_points_flat(
    k_vectors: torch.Tensor,  # [n_k_total, 3]
    k_vector_batch: torch.Tensor,  # [n_k_total]
    fourier_coefficients: torch.Tensor,  # [n_k_total, 2]
    sample_points: torch.Tensor,  # [n_points, 3]
    sample_batch: torch.Tensor,  # [n_points]
    excluded_k_mask: torch.Tensor,  # [n_k_total]
) -> torch.Tensor:
    """
    Evaluate a real-valued field from its Fourier coefficients at arbitrary points.

    \rho(r) = (1/(2\pi)^3) * sum_k [2 Re{rho_k} cos(k·r) - 2 Im{rho_k} sin(k·r)],
    with the k=0 term not doubled.
    """
    inner_products = torch.matmul(k_vectors, sample_points.t())  # [n_k_total, n_points]
    mask = k_vector_batch[:, None] == sample_batch[None, :]
    mask_f = mask.to(dtype=inner_products.dtype)
    cosines = torch.cos(inner_products) * mask_f
    sines = torch.sin(inner_products) * mask_f

    summand = (
        2.0 * fourier_coefficients[:, 0].unsqueeze(-1) * cosines
        - 2.0 * fourier_coefficients[:, 1].unsqueeze(-1) * sines
    )
    k0_factor = torch.ones_like(excluded_k_mask)
    k0_factor[excluded_k_mask > 0.0] = 0.5
    summand = summand * k0_factor.unsqueeze(-1)
    return torch.sum(summand, dim=0) / (2 * pi) ** 3


def fourier_series_dot(
    series_1: torch.Tensor,  # [n_graph, max_n_k, 2]
    series_2: torch.Tensor,  # [n_graph, max_n_k, 2]
    volumes: torch.Tensor,  # [n_graph]
) -> torch.Tensor:
    """Dot product between two real Fourier series, handling k=0."""
    values_at_zero = torch.einsum("bj,bj->b", series_1[:, 0, :], series_2[:, 0, :])
    values = 2 * torch.einsum("bkj,bkj->b", series_1[:, 1:, :], series_2[:, 1:, :])
    return torch.multiply(values + values_at_zero, volumes) / (2 * pi) ** 6
