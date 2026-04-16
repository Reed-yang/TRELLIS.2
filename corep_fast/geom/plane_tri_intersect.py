"""
GPU plane–triangle intersection kernel.

Given K (triangle, plane) pairs, computes the intersection segment
(two endpoints) where the plane cuts through the triangle.  Fully
vectorized — no Python loops over K.

Public function:

- ``plane_triangle_intersect(triangles, plane_normals, plane_points)``
    Returns ``(seg_p1, seg_p2, valid)`` where *valid* marks triangles
    that are strictly cut by the plane (vertices on **both** sides).
"""

from __future__ import annotations

import torch
from torch import Tensor

# Tolerance for classifying a vertex as "on the plane".
_EPS = 1e-14


def plane_triangle_intersect(
    triangles: Tensor,
    plane_normals: Tensor,
    plane_points: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Intersect each triangle with its corresponding plane.

    Args:
        triangles: (K, 3, 3) triangle vertices (v0, v1, v2).
        plane_normals: (K, 3) unit plane normals.
        plane_points: (K, 3) a point on each plane.

    Returns:
        seg_p1: (K, 3) first endpoint of the intersection segment.
        seg_p2: (K, 3) second endpoint of the intersection segment.
        valid: (K,) bool — True iff the triangle is strictly cut by the
            plane (has vertices on **both** sides, i.e. at least one
            strictly positive *and* at least one strictly negative signed
            distance).
    """
    K = triangles.shape[0]
    dev = triangles.device
    dtype = triangles.dtype

    # Vertices: each (K, 3).
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]

    # Signed distances of each vertex to its plane.
    #   d_i = dot(v_i - plane_pt, plane_n)
    d0 = (v0 - plane_points).mul(plane_normals).sum(-1)  # (K,)
    d1 = (v1 - plane_points).mul(plane_normals).sum(-1)
    d2 = (v2 - plane_points).mul(plane_normals).sum(-1)

    # A triangle crosses the plane iff it has vertices on BOTH sides,
    # i.e. at least one d_i > eps AND at least one d_i < -eps.
    has_pos = (d0 > _EPS) | (d1 > _EPS) | (d2 > _EPS)
    has_neg = (d0 < -_EPS) | (d1 < -_EPS) | (d2 < -_EPS)
    valid = has_pos & has_neg  # (K,)

    # ---- Find the two crossing edges and compute intersection points ----
    #
    # For a crossing triangle the plane intersects exactly 2 of the 3 edges.
    # Edge (va, vb) is crossed when da * db < 0  (vertices on opposite sides).
    #
    # Intersection point on edge (va, vb):
    #   t = da / (da - db)          (0 < t < 1 for a true crossing)
    #   p = va + t * (vb - va)      = lerp(va, vb, t)
    #
    # We compute the intersection for all three edges and then pick the
    # first two that actually cross.

    # Edge cross flags.
    cross_01 = d0 * d1 < 0  # (K,)
    cross_12 = d1 * d2 < 0
    cross_20 = d2 * d0 < 0

    # Interpolation parameters (safe even when da == db thanks to clamping;
    # invalid edges will be masked out by `valid` anyway).
    def _lerp_edge(va: Tensor, vb: Tensor, da: Tensor, db: Tensor) -> Tensor:
        """Compute the plane–edge intersection point."""
        t = da / (da - db + 1e-30)  # (K,)
        return va + t[:, None] * (vb - va)  # (K, 3)

    p01 = _lerp_edge(v0, v1, d0, d1)
    p12 = _lerp_edge(v1, v2, d1, d2)
    p20 = _lerp_edge(v2, v0, d2, d0)

    # Assign the two crossing points (seg_p1, seg_p2).
    # Strategy: iterate through the three edges in a fixed order.
    #   - seg_p1 comes from the *first* crossing edge.
    #   - seg_p2 comes from the *second* crossing edge.
    #
    # Because exactly two edges cross for any valid triangle, we can
    # build this without loops using conditional selection.

    # First crossing point: prefer edge 01, then 12, then 20.
    seg_p1 = torch.where(cross_01[:, None], p01, torch.where(cross_12[:, None], p12, p20))

    # Second crossing point: prefer the *last* crossing edge.
    # Last crossing: prefer edge 20, then 12, then 01.
    seg_p2 = torch.where(cross_20[:, None], p20, torch.where(cross_12[:, None], p12, p01))

    # For non-valid triangles the outputs are meaningless; zero them out
    # for cleanliness.
    zeros = torch.zeros(K, 3, device=dev, dtype=dtype)
    seg_p1 = torch.where(valid[:, None], seg_p1, zeros)
    seg_p2 = torch.where(valid[:, None], seg_p2, zeros)

    return seg_p1, seg_p2, valid
