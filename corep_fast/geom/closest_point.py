"""
GPU closest-point-on-triangle-mesh kernel.

Implements Ericson's "Real-Time Collision Detection" §5.1.5 region
decomposition for closest-point-on-triangle, fully vectorized with PyTorch
so it runs on both CPU and CUDA tensors.

Two public functions:

- ``closest_point_on_triangles(queries, triangles)``
    Per-(query, triangle) closest point and squared distance.

- ``closest_point_on_mesh(queries, triangles, chunk_size)``
    Closest point on the whole mesh for each query (argmin over faces).
"""

from __future__ import annotations

import torch
from torch import Tensor

# Small epsilon to avoid division by zero on degenerate triangles.
_EPS = 1e-30


def closest_point_on_triangles(
    queries: Tensor,
    triangles: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute the closest point on each triangle for each query point.

    Uses Ericson §5.1.5 region decomposition — seven regions (3 vertex,
    3 edge, 1 interior) each with a closed-form closest-point formula.
    Fully vectorized across P×F via broadcasting.

    Args:
        queries: (P, 3) query points.
        triangles: (F, 3, 3) triangle vertices (v0, v1, v2 per face).

    Returns:
        closest_pts: (P, F, 3) closest point on each triangle for each query.
        sq_dists: (P, F) squared distances.
    """
    P = queries.shape[0]
    F = triangles.shape[0]

    # Handle empty inputs.
    if P == 0 or F == 0:
        dev = queries.device
        return (
            torch.empty(P, F, 3, device=dev, dtype=queries.dtype),
            torch.empty(P, F, device=dev, dtype=queries.dtype),
        )

    # Triangle vertices: each (F, 3).
    a = triangles[:, 0]  # (F, 3)
    b = triangles[:, 1]  # (F, 3)
    c = triangles[:, 2]  # (F, 3)

    # Edge vectors.
    ab = b - a  # (F, 3)
    ac = c - a  # (F, 3)

    # Expand for broadcasting: queries (P, 1, 3), triangle verts (1, F, 3).
    p = queries[:, None, :]     # (P, 1, 3)
    a_ = a[None, :, :]         # (1, F, 3)
    ab_ = ab[None, :, :]       # (1, F, 3)
    ac_ = ac[None, :, :]       # (1, F, 3)

    ap = p - a_                 # (P, F, 3)

    # Dot products — all (P, F).
    d1 = (ab_ * ap).sum(-1)    # dot(ab, ap)
    d2 = (ac_ * ap).sum(-1)    # dot(ac, ap)
    d3_ab_ab = (ab_ * ab_).sum(-1)  # dot(ab, ab)  — broadcast to (P, F) but constant in P
    d4_ac_ac = (ac_ * ac_).sum(-1)  # dot(ac, ac)
    d5_ab_ac = (ab_ * ac_).sum(-1)  # dot(ab, ac)

    b_ = b[None, :, :]        # (1, F, 3)
    c_ = c[None, :, :]        # (1, F, 3)
    bp = p - b_                # (P, F, 3)
    cp = p - c_                # (P, F, 3)

    d3 = (ab_ * bp).sum(-1)   # dot(ab, bp)
    d4 = (ac_ * bp).sum(-1)   # dot(ac, bp)
    d5 = (ab_ * cp).sum(-1)   # dot(ab, cp)
    d6 = (ac_ * cp).sum(-1)   # dot(ac, cp)

    # ---- Region classification (Ericson §5.1.5) ----
    # Region 1: closest to vertex A.
    region_a = (d1 <= 0) & (d2 <= 0)

    # Region 2: closest to vertex B.
    region_b = (d3 >= 0) & (d4 <= d3)

    # Region 3: closest to vertex C.
    region_c = (d5 <= d6) & (d6 >= 0)  # note: d5 here is dot(ab,cp), d6 is dot(ac,cp)
    # Fix: proper vertex-C test uses d5 and d6 from cp.
    # d5 = dot(ab, cp), d6 = dot(ac, cp)
    # Vertex C: d6 >= 0 and d5 <= d6.

    # Region AB edge: point projects onto edge AB.
    vc = d1 * d4 - d3 * d2
    region_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v_ab = d1 / (d1 - d3 + _EPS)

    # Region AC edge: point projects onto edge AC.
    vb = d5 * d2 - d1 * d6
    region_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w_ac = d2 / (d2 - d6 + _EPS)

    # Region BC edge: point projects onto edge BC.
    va = d3 * d6 - d5 * d4
    region_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6) + _EPS)

    # Region interior: point projects inside the triangle.
    denom = va + vb + vc + _EPS
    v_int = vb / denom
    w_int = vc / denom

    # ---- Compute closest points per region ----
    # Start with interior as the default.
    result = a_ + v_int[..., None] * ab_ + w_int[..., None] * ac_  # (P, F, 3)

    # Overwrite with edge / vertex results where the respective region holds.
    # Process from most specific (vertex) to least, so vertex regions win over
    # edge regions when both flags are set (which shouldn't happen for
    # well-formed inputs, but guards against degenerate triangles).

    # Edge BC: a + ab*0 + ac*0 isn't right; BC closest = b + w*(c - b).
    bc_pts = b_ + w_bc[..., None] * (c_ - b_)
    result = torch.where(region_bc[..., None], bc_pts, result)

    # Edge AC: a + w * ac.
    ac_pts = a_ + w_ac[..., None] * ac_
    result = torch.where(region_ac[..., None], ac_pts, result)

    # Edge AB: a + v * ab.
    ab_pts = a_ + v_ab[..., None] * ab_
    result = torch.where(region_ab[..., None], ab_pts, result)

    # Vertex C.
    result = torch.where(region_c[..., None], c_.expand_as(result), result)

    # Vertex B.
    result = torch.where(region_b[..., None], b_.expand_as(result), result)

    # Vertex A.
    result = torch.where(region_a[..., None], a_.expand_as(result), result)

    # Squared distances.
    diff = p - result  # (P, F, 3)
    sq_dists = (diff * diff).sum(-1)  # (P, F)

    return result, sq_dists


def closest_point_on_mesh(
    queries: Tensor,
    triangles: Tensor,
    chunk_size: int = 50_000_000,
) -> tuple[Tensor, Tensor]:
    """Find the closest point on the mesh surface for each query point.

    Wraps :func:`closest_point_on_triangles` with chunked evaluation and
    argmin over all faces.

    Args:
        queries: (P, 3) query points.
        triangles: (F, 3, 3) triangle vertices.
        chunk_size: max P*F elements per chunk (default 50M) to bound VRAM.

    Returns:
        closest_pts: (P, 3) closest surface point per query.
        face_idx: (P,) int64 index of the nearest face.
    """
    P = queries.shape[0]
    F = triangles.shape[0]
    dev = queries.device

    if P == 0:
        return (
            torch.empty(0, 3, device=dev, dtype=queries.dtype),
            torch.empty(0, device=dev, dtype=torch.int64),
        )

    # Determine chunk size along P dimension.
    p_chunk = max(1, chunk_size // max(F, 1))

    best_pts = torch.empty(P, 3, device=dev, dtype=queries.dtype)
    best_idx = torch.empty(P, device=dev, dtype=torch.int64)
    best_dist = torch.full((P,), float("inf"), device=dev, dtype=queries.dtype)

    for start in range(0, P, p_chunk):
        end = min(start + p_chunk, P)
        q_chunk = queries[start:end]  # (chunk, 3)

        pts, sq_d = closest_point_on_triangles(q_chunk, triangles)
        # pts: (chunk, F, 3), sq_d: (chunk, F)

        min_d, min_i = sq_d.min(dim=1)  # (chunk,), (chunk,)

        # Gather the closest point for each query in this chunk.
        # min_i: (chunk,) — need to index pts (chunk, F, 3) at dim=1.
        chunk_pts = pts[torch.arange(pts.shape[0], device=dev), min_i]  # (chunk, 3)

        # Update global best.
        mask = min_d < best_dist[start:end]
        best_dist[start:end] = torch.where(mask, min_d, best_dist[start:end])
        best_idx[start:end] = torch.where(mask, min_i, best_idx[start:end])
        best_pts[start:end] = torch.where(mask[:, None], chunk_pts, best_pts[start:end])

    return best_pts, best_idx
