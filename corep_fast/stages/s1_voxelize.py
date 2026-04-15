"""
Stage 1: GPU SAT Voxelization -- mesh triangles -> occupied cubes + face CSR.

Replaces custom/voxelize.py with GPU tensor operations.
Uses the Separating Axis Theorem (SAT) with 13 axes to test
triangle-AABB intersection.

Public API:
    s1_voxelize(mesh, resolution, device) -> CubeBatch
"""
from __future__ import annotations

import torch

from corep_fast.containers import MeshTensors, CubeBatch, _replace_fields


def s1_voxelize(
    mesh: MeshTensors,
    resolution: int,
    device: torch.device,
) -> CubeBatch:
    """Voxelize mesh: find occupied cubes and register intersecting faces.

    For each mesh triangle, computes its AABB in grid space, enumerates
    candidate cubes, then applies the 13-axis SAT test to determine
    actual intersection. Results are packed into CubeBatch with CSR
    face registration.

    Args:
        mesh: Normalized mesh tensors (vertices in [0,1]^3).
        resolution: Voxel grid resolution R (grid is R^3).
        device: Target device for output tensors.

    Returns:
        CubeBatch with cube_indices, cube_hash, tri_offsets, tri_values populated.
    """
    R = resolution
    triangles = mesh.triangles  # (F, 3, 3) float32

    # Step 1: compute AABB per triangle in grid coords
    tri_grid = triangles * R  # (F, 3, 3) -- triangle verts in grid space
    tri_min = tri_grid.amin(dim=1)  # (F, 3)
    tri_max = tri_grid.amax(dim=1)  # (F, 3)

    # Integer AABB: candidate cubes for each triangle
    imin = tri_min.floor().to(torch.int32).clamp(min=0, max=R - 1)  # (F, 3)
    imax = tri_max.floor().to(torch.int32).clamp(min=0, max=R - 1)  # (F, 3)
    # Number of candidate cubes per triangle
    spans = (imax - imin + 1).to(torch.int64)  # (F, 3)
    counts_per_tri = spans[:, 0] * spans[:, 1] * spans[:, 2]  # (F,)

    # Step 2: expand -- enumerate all (triangle, candidate_cube) pairs
    offsets = torch.zeros(counts_per_tri.shape[0] + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(counts_per_tri, dim=0)
    total_pairs = int(offsets[-1].item())

    # For each pair, identify triangle_id and cube (ix, iy, iz)
    pair_tri_ids, pair_cube_coords = _expand_candidates(
        imin, imax, spans, offsets, total_pairs, device)

    # Step 3: SAT test -- 13 axes
    hits = _sat_test_batch(tri_grid, pair_tri_ids, pair_cube_coords, R, device)

    # Step 4: compact hits into CSR
    hit_tri_ids = pair_tri_ids[hits]
    hit_cube_coords = pair_cube_coords[hits]

    # Build CSR: sort by cube hash, unique cubes
    hit_cube_hash = (hit_cube_coords[:, 0].long() * R * R
                     + hit_cube_coords[:, 1].long() * R
                     + hit_cube_coords[:, 2].long())

    # Sort by cube hash for grouping
    sort_idx = torch.argsort(hit_cube_hash)
    sorted_hash = hit_cube_hash[sort_idx]
    sorted_tri_ids = hit_tri_ids[sort_idx]

    # Unique cubes and their offsets
    unique_hash, inverse, counts = torch.unique_consecutive(
        sorted_hash, return_inverse=True, return_counts=True)

    N = unique_hash.shape[0]
    tri_offsets = torch.zeros(N + 1, dtype=torch.int64, device=device)
    tri_offsets[1:] = torch.cumsum(counts, dim=0)

    # Recover cube_indices from hash
    cube_iz = (unique_hash % R).to(torch.int32)
    cube_iy = ((unique_hash // R) % R).to(torch.int32)
    cube_ix = (unique_hash // (R * R)).to(torch.int32)
    cube_indices = torch.stack([cube_ix, cube_iy, cube_iz], dim=1)

    # Build CubeBatch
    cb = CubeBatch.empty(num_cubes=N, resolution=R, device=device)
    cb = _replace_fields(
        cb,
        cube_indices=cube_indices,
        cube_hash=unique_hash,
        tri_offsets=tri_offsets,
        tri_values=sorted_tri_ids.to(torch.int32),
    )

    return cb


def _expand_candidates(
    imin: torch.Tensor,  # (F, 3) int32
    imax: torch.Tensor,  # (F, 3) int32
    spans: torch.Tensor,  # (F, 3) int64
    offsets: torch.Tensor,  # (F+1,) int64
    total_pairs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate all (triangle_id, candidate_cube_coord) pairs.

    Returns:
        pair_tri_ids: (total_pairs,) int32
        pair_cube_coords: (total_pairs, 3) int32
    """
    # Use arange + searchsorted to map flat index -> triangle
    flat_idx = torch.arange(total_pairs, dtype=torch.int64, device=device)
    tri_ids = torch.searchsorted(offsets[1:], flat_idx, right=True)  # (total_pairs,)
    local_idx = flat_idx - offsets[tri_ids]  # within-triangle flat index

    sx = spans[tri_ids, 0]
    sy = spans[tri_ids, 1]
    sz = spans[tri_ids, 2]

    lx = local_idx // (sy * sz)
    rem = local_idx % (sy * sz)
    ly = rem // sz
    lz = rem % sz

    pair_tri_ids = tri_ids.to(torch.int32)
    pair_cube_coords = torch.stack([
        (imin[tri_ids, 0].long() + lx).to(torch.int32),
        (imin[tri_ids, 1].long() + ly).to(torch.int32),
        (imin[tri_ids, 2].long() + lz).to(torch.int32),
    ], dim=1)

    return pair_tri_ids, pair_cube_coords


def _sat_test_batch(
    tri_grid: torch.Tensor,    # (F, 3, 3) -- triangle verts in grid space
    pair_tri_ids: torch.Tensor,  # (P,) int32
    pair_cube_coords: torch.Tensor,  # (P, 3) int32
    R: int,
    device: torch.device,
) -> torch.Tensor:
    """Batch SAT test: 13 axes for triangle-AABB intersection.

    Returns:
        hits: (P,) bool -- True if triangle intersects cube.
    """
    P = pair_tri_ids.shape[0]
    if P == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)

    tids = pair_tri_ids.long()

    # Triangle vertices for each pair
    v0 = tri_grid[tids, 0]  # (P, 3)
    v1 = tri_grid[tids, 1]  # (P, 3)
    v2 = tri_grid[tids, 2]  # (P, 3)

    # Cube center and half-extent
    cube_center = pair_cube_coords.float() + 0.5  # (P, 3)
    half = 0.5  # unit cube half-extent

    # Translate triangle to cube-center-origin frame
    f0 = v0 - cube_center  # (P, 3)
    f1 = v1 - cube_center
    f2 = v2 - cube_center

    # Triangle edges
    e0 = f1 - f0  # (P, 3)
    e1 = f2 - f1
    e2 = f0 - f2

    alive = torch.ones(P, dtype=torch.bool, device=device)

    # Helper: test separation along axis a
    def _test_axis(ax: torch.Tensor) -> None:
        """ax: (P, 3). Updates `alive` in place."""
        nonlocal alive
        p0 = (f0 * ax).sum(dim=1)
        p1 = (f1 * ax).sum(dim=1)
        p2 = (f2 * ax).sum(dim=1)
        tri_min = torch.minimum(torch.minimum(p0, p1), p2)
        tri_max = torch.maximum(torch.maximum(p0, p1), p2)
        # Box projection onto axis: |ax_x|*half + |ax_y|*half + |ax_z|*half
        r = ax.abs().sum(dim=1) * half
        alive &= ~((tri_min > r) | (tri_max < -r))

    # --- 9 cross-product axes (3 edges x 3 AABB face normals) ---
    unit_x = torch.tensor([[1., 0., 0.]], device=device).expand(P, -1)
    unit_y = torch.tensor([[0., 1., 0.]], device=device).expand(P, -1)
    unit_z = torch.tensor([[0., 0., 1.]], device=device).expand(P, -1)

    for edge in [e0, e1, e2]:
        _test_axis(torch.cross(edge, unit_x, dim=1))
        _test_axis(torch.cross(edge, unit_y, dim=1))
        _test_axis(torch.cross(edge, unit_z, dim=1))

    # --- 3 AABB face normals (X, Y, Z) ---
    for dim in range(3):
        vals = torch.stack([f0[:, dim], f1[:, dim], f2[:, dim]], dim=1)
        alive &= ~((vals.amin(dim=1) > half) | (vals.amax(dim=1) < -half))

    # --- 1 triangle normal ---
    tri_normal = torch.cross(e0, e1, dim=1)  # (P, 3)
    _test_axis(tri_normal)

    return alive
