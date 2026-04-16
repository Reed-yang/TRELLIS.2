"""
Stage 3: GPU Möller-Trumbore Ray-Triangle Intersection for edge weights.

For each of 18 edges per cube, casts a ray along the edge and counts
intersections with the cube's registered mesh triangles. This replaces
the multiprocessing CPU path in custom/feature_edge.py with a fully
vectorized GPU implementation.

Public API:
    s3_edge_weights(batch, mesh) -> CubeBatch
"""
from __future__ import annotations

import torch

from corep_fast.constants import CUBE_EDGE_STARTS, CUBE_EDGE_DIRS
from corep_fast.containers import MeshTensors, CubeBatch, _replace_fields

# Möller-Trumbore epsilon — matches custom/feature_edge.py
_EPSILON = 1e-8

# Maximum number of (cube, triangle) pairs to process in one chunk
# to avoid OOM on large meshes.  Each pair expands to 18 ray tests.
_CHUNK_SIZE = 200_000


def s3_edge_weights(batch: CubeBatch, mesh: MeshTensors) -> CubeBatch:
    """Compute edge_weights (N, 18) via GPU Möller-Trumbore ray-triangle intersection.

    For each of 18 edges per cube, casts a ray and counts intersections
    with the cube's registered mesh triangles.

    Args:
        batch: CubeBatch after s2_components (tri_offsets/tri_values populated).
        mesh:  MeshTensors with triangles (F, 3, 3).

    Returns:
        Updated CubeBatch with edge_weights (N, 18) int32 filled in.
    """
    N = batch.num_cubes
    device = batch.device

    if N == 0:
        return batch

    resolution = batch.resolution
    step = 1.0 / resolution

    # --- Step 1: Build ray origins and directions for all cubes ---
    # base = cube_indices / resolution  -> (N, 3)
    base = batch.cube_indices.float() * step  # (N, 3)

    # Ray origins per cube per edge: base + CUBE_EDGE_STARTS * step
    # CUBE_EDGE_STARTS: (18, 3) float32
    edge_starts = CUBE_EDGE_STARTS.to(device)  # (18, 3)
    edge_dirs = CUBE_EDGE_DIRS.to(device)       # (18, 3)

    # ray_origins[i, e, :] = base[i] + edge_starts[e] * step
    ray_origins = base[:, None, :] + edge_starts[None, :, :] * step  # (N, 18, 3)

    # ray_dirs[e, :] = edge_dirs[e] * step  (same for all cubes)
    ray_dirs = edge_dirs * step  # (18, 3)

    # --- Step 2: Expand CSR -> flat (cube_idx, tri_idx) pairs ---
    tri_offsets = batch.tri_offsets  # (N+1,) int64
    tri_values = batch.tri_values   # (T,) int32

    T_total = tri_values.shape[0]

    if T_total == 0:
        # No triangle registrations at all
        edge_weights = torch.zeros((N, 18), dtype=torch.int32, device=device)
        return _replace_fields(batch, edge_weights=edge_weights)

    # flat_cube_ids[k] = cube index for the k-th (cube, tri) pair
    flat_cube_ids = _csr_to_flat_cube_ids(tri_offsets, T_total, device)  # (T,) int64
    flat_tri_ids = tri_values.long()  # (T,) int64

    # --- Step 3: Batch Möller-Trumbore with chunking ---
    # Accumulator for edge weights
    edge_weights = torch.zeros((N, 18), dtype=torch.int32, device=device)

    for chunk_start in range(0, T_total, _CHUNK_SIZE):
        chunk_end = min(chunk_start + _CHUNK_SIZE, T_total)
        c_cube_ids = flat_cube_ids[chunk_start:chunk_end]  # (C,)
        c_tri_ids = flat_tri_ids[chunk_start:chunk_end]     # (C,)
        C = c_cube_ids.shape[0]

        # Gather ray origins for this chunk: (C, 18, 3)
        c_ray_origins = ray_origins[c_cube_ids]  # (C, 18, 3)

        # Gather triangle vertices: (C, 3, 3) -> V0, V1, V2
        c_triangles = mesh.triangles[c_tri_ids]  # (C, 3, 3)
        V0 = c_triangles[:, 0, :]  # (C, 3)
        V1 = c_triangles[:, 1, :]  # (C, 3)
        V2 = c_triangles[:, 2, :]  # (C, 3)

        # Möller-Trumbore for all C pairs × 18 edges
        # hits: (C, 18) bool
        hits = _moller_trumbore_batch(c_ray_origins, ray_dirs, V0, V1, V2)

        # Scatter-add hits into edge_weights by cube index
        # hits_int: (C, 18) int32
        hits_int = hits.to(torch.int32)
        edge_weights.scatter_add_(0, c_cube_ids[:, None].expand(-1, 18), hits_int)

    return _replace_fields(batch, edge_weights=edge_weights)


def _csr_to_flat_cube_ids(
    offsets: torch.Tensor,  # (N+1,) int64
    total: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert CSR offsets to flat cube IDs.

    For CSR with offsets [0, 3, 5, 9], returns [0,0,0, 1,1, 2,2,2,2].

    Returns:
        flat_cube_ids: (total,) int64
    """
    # Use searchsorted: for each position in [0, total), find which cube it belongs to
    flat_idx = torch.arange(total, dtype=torch.int64, device=device)
    # offsets[1:] contains the exclusive upper bounds for each cube
    # searchsorted(right=True) on offsets[1:] maps flat_idx -> cube_id
    cube_ids = torch.searchsorted(offsets[1:], flat_idx, right=True)
    return cube_ids


def _moller_trumbore_batch(
    ray_origins: torch.Tensor,  # (C, 18, 3)
    ray_dirs: torch.Tensor,     # (18, 3)  -- broadcast over C
    V0: torch.Tensor,           # (C, 3)
    V1: torch.Tensor,           # (C, 3)
    V2: torch.Tensor,           # (C, 3)
) -> torch.Tensor:
    """Vectorized Möller-Trumbore ray-triangle intersection.

    Tests C triangles against 18 rays each, returning a (C, 18) bool
    tensor of intersection results.

    Hit conditions (matching custom/feature_edge.py):
        |det| > EPSILON
        u in [0.0, 1.0]
        v in [0.0, 1.0-u]   (i.e. u + v <= 1.0)
        t in [-EPSILON, 1.0+EPSILON]
    """
    C = ray_origins.shape[0]

    # Triangle edges
    E1 = V1 - V0  # (C, 3)
    E2 = V2 - V0  # (C, 3)

    # Expand to (C, 18, 3) for per-edge computation
    # ray_dirs: (18, 3) -> (1, 18, 3)
    D = ray_dirs.unsqueeze(0).expand(C, -1, -1)  # (C, 18, 3)

    # E1, E2: (C, 3) -> (C, 1, 3) for broadcasting with 18 edges
    E1_exp = E1.unsqueeze(1)  # (C, 1, 3)
    E2_exp = E2.unsqueeze(1)  # (C, 1, 3)

    # P = cross(D, E2)  -> (C, 18, 3)
    P = torch.cross(D, E2_exp.expand(-1, 18, -1), dim=2)

    # det = dot(E1, P) -> (C, 18)
    det = (E1_exp * P).sum(dim=2)

    # valid where |det| > EPSILON
    valid = det.abs() > _EPSILON

    # inv_det: safe reciprocal (zeros where invalid, guarded to avoid div-by-zero)
    safe_det = torch.where(valid, det, torch.ones_like(det))
    inv_det = torch.where(valid, 1.0 / safe_det, torch.zeros_like(det))

    # T_vec = O - V0  -> (C, 18, 3)
    V0_exp = V0.unsqueeze(1)  # (C, 1, 3)
    T_vec = ray_origins - V0_exp  # (C, 18, 3)

    # u = dot(T_vec, P) * inv_det -> (C, 18)
    u = (T_vec * P).sum(dim=2) * inv_det

    # Q = cross(T_vec, E1) -> (C, 18, 3)
    Q = torch.cross(T_vec, E1_exp.expand(-1, 18, -1), dim=2)

    # v = dot(D, Q) * inv_det -> (C, 18)
    v = (D * Q).sum(dim=2) * inv_det

    # t = dot(E2, Q) * inv_det -> (C, 18)
    t = (E2_exp * Q).sum(dim=2) * inv_det

    # Hit = valid & u in [0,1] & v >= 0 & u+v <= 1 & t in [-eps, 1+eps]
    hits = (
        valid
        & (u >= 0.0) & (u <= 1.0)
        & (v >= 0.0) & (u + v <= 1.0)
        & (t >= -_EPSILON) & (t <= 1.0 + _EPSILON)
    )

    return hits  # (C, 18) bool
