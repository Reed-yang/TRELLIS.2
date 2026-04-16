"""
GPU-batched Sutherland-Hodgman polygon clipping.

Clips C polygons against half-planes in parallel.  The loop over polygon
edges runs for a fixed MAX_VERTS iterations; within each iteration every
polygon in the batch is processed simultaneously via vectorised ops.

Half-plane convention: keep the side where  dot(normal, point) >= d.
"""
from __future__ import annotations

import torch

MAX_VERTS = 12  # triangle clipped by 6 planes -> at most 9 verts; 12 for safety


def sh_clip_against_plane(
    poly: torch.Tensor,
    v_len: torch.Tensor,
    plane_n: torch.Tensor,
    plane_d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip *C* padded polygons against one half-plane each (batched).

    Parameters
    ----------
    poly : (C, MAX_VERTS, 3)  padded polygon vertices.
    v_len : (C,) int32  valid vertex count per polygon.
    plane_n : (C, 3)  outward normal of each clipping plane.
    plane_d : (C,)    plane offset  (keep side where n . p >= d).

    Returns
    -------
    new_poly : (C, MAX_VERTS, 3)  clipped polygon vertices (padded).
    new_len  : (C,) int32         valid vertex count after clipping.
    """
    C = poly.shape[0]
    device = poly.device
    dtype = poly.dtype

    # Output buffers
    out = torch.zeros(C, MAX_VERTS, 3, device=device, dtype=dtype)
    out_count = torch.zeros(C, device=device, dtype=torch.int32)

    # Precompute signed distances for every vertex:  d_i = dot(n, v_i) - d
    # shape (C, MAX_VERTS)
    signed_dist = (poly * plane_n.unsqueeze(1)).sum(-1) - plane_d.unsqueeze(1)

    # Iterate over each possible edge index (fixed loop, bounded by MAX_VERTS).
    for i in range(MAX_VERTS):
        # Edge i -> (i+1) % v_len.  The edge only exists when i < v_len.
        # v_len varies per polygon so the next-index wraps per polygon.
        edge_active = i < v_len  # (C,) bool

        # Next vertex index (wraps around per-polygon length)
        j_idx = (i + 1) % v_len.clamp(min=1)  # avoid mod-by-zero

        # Gather current and next vertices and their signed distances
        vi = poly[:, i, :]              # (C, 3)
        # Gather vj per polygon: poly[c, j_idx[c], :]
        vj = poly[torch.arange(C, device=device), j_idx, :]  # (C, 3)

        di = signed_dist[:, i]          # (C,)
        dj = signed_dist[torch.arange(C, device=device), j_idx]  # (C,)

        inside_i = di >= 0  # (C,) bool
        inside_j = dj >= 0  # (C,) bool

        # ------- Case 1: both inside  -> emit vj -------
        case1 = edge_active & inside_i & inside_j  # (C,)

        # ------- Case 2: inside -> outside  -> emit intersection -------
        case2 = edge_active & inside_i & ~inside_j

        # ------- Case 3: outside -> inside  -> emit intersection + vj -------
        case3 = edge_active & ~inside_i & inside_j

        # ------- Case 4: both outside -> emit nothing -------
        # (no action needed)

        # Compute intersection point for cases 2 & 3
        # t = di / (di - dj), clamped to [0,1]
        denom = di - dj
        denom = denom.where(denom.abs() > 1e-12, torch.ones_like(denom))
        t = (di / denom).clamp(0.0, 1.0)  # (C,)
        intersection = vi + t.unsqueeze(-1) * (vj - vi)  # (C, 3)

        # --- Emit vertices into output buffer via scatter ---
        # We need to write at position out_count[c] for each polygon c,
        # then increment out_count.  Because at most 2 vertices are emitted
        # per iteration and order matters, we do it sequentially per emit.

        # Emit 1: intersection for case3 (outside->inside, intersection first)
        emit1 = case3
        _emit(out, out_count, intersection, emit1)

        # Emit 2: vj for case1 or case3
        emit2 = case1 | case3
        _emit(out, out_count, vj, emit2)

        # Emit 3: intersection for case2 (inside->outside)
        emit3 = case2
        _emit(out, out_count, intersection, emit3)

    return out, out_count


def _emit(
    out: torch.Tensor,
    out_count: torch.Tensor,
    vertex: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Write *vertex* into *out* at position *out_count* for polygons where *mask* is True.

    Mutates *out* and *out_count* in-place.  When *mask* is all-False the
    advanced-index scatter is a no-op (writes nothing), so no explicit guard
    is needed — this avoids a device-to-host sync that ``mask.any()`` would
    trigger on CUDA tensors.

    Parameters
    ----------
    out : (C, MAX_VERTS, 3)
    out_count : (C,) int32  current write cursor per polygon.
    vertex : (C, 3)         vertex to emit.
    mask : (C,) bool        which polygons should emit.
    """
    C = out.shape[0]
    device = out.device

    idx = out_count.long()  # (C,)
    # Clamp index to valid range to avoid OOB (masked entries won't matter)
    idx = idx.clamp(max=MAX_VERTS - 1)

    # Build scatter index: for each polygon c where mask[c], write vertex[c]
    # into out[c, idx[c], :].  In-place is safe — out and out_count are local
    # buffers with no autograd graph.
    c_idx = torch.arange(C, device=device)
    out[c_idx[mask], idx[mask], :] = vertex[mask]
    out_count[mask] += 1


def sh_clip_aabb(
    triangles: torch.Tensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip triangles against axis-aligned bounding boxes (batched).

    Parameters
    ----------
    triangles : (C, 3, 3)  triangle vertices.
    aabb_min  : (C, 3)     AABB minimum corner.
    aabb_max  : (C, 3)     AABB maximum corner.

    Returns
    -------
    poly  : (C, MAX_VERTS, 3)  clipped polygon vertices (padded).
    v_len : (C,) int32         valid vertex count per polygon.
    """
    C = triangles.shape[0]
    device = triangles.device
    dtype = triangles.dtype

    # Initialise polygon buffer from triangles (pad to MAX_VERTS)
    poly = torch.zeros(C, MAX_VERTS, 3, device=device, dtype=dtype)
    poly[:, :3, :] = triangles
    v_len = torch.full((C,), 3, device=device, dtype=torch.int32)

    # Clip against 6 AABB half-planes
    # +axis min planes:  keep  x >= min_x  ->  normal=[1,0,0], d=min_x
    # -axis max planes:  keep  x <= max_x  ->  normal=[-1,0,0], d=-max_x
    for axis in range(3):
        # --- min plane: keep axis >= aabb_min[:, axis] ---
        plane_n = torch.zeros(C, 3, device=device, dtype=dtype)
        plane_n[:, axis] = 1.0
        plane_d = aabb_min[:, axis]
        poly, v_len = sh_clip_against_plane(poly, v_len, plane_n, plane_d)

        # --- max plane: keep axis <= aabb_max[:, axis] ---
        plane_n = torch.zeros(C, 3, device=device, dtype=dtype)
        plane_n[:, axis] = -1.0
        plane_d = -aabb_max[:, axis]
        poly, v_len = sh_clip_against_plane(poly, v_len, plane_n, plane_d)

    return poly, v_len
