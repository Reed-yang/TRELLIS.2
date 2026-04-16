"""GPU fan-triangulation + area-weighted centroid for clipped convex polygons.

Given batched padded convex polygons (C, MAX_VERTS, 3) and their true vertex
counts (C,), computes the area-weighted centroid of each polygon via
fan-triangulation from vertex 0.
"""

import torch
from typing import Tuple

MAX_VERTS = 12


def fan_area_centroid(
    poly: torch.Tensor,
    v_len: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute area-weighted centroid of batched convex polygons.

    Fan-triangulates each polygon from vertex 0 and computes the
    area-weighted centroid across all fan triangles.

    Args:
        poly: (C, MAX_VERTS, 3) padded convex polygon vertices.
        v_len: (C,) int32, number of valid vertices per polygon.

    Returns:
        centroid: (C, 3) area-weighted centroid per polygon.
        total_area: (C,) total area per polygon.
    """
    C = poly.shape[0]
    device = poly.device
    dtype = poly.dtype

    # v0 is the fan pivot for every polygon: (C, 3)
    v0 = poly[:, 0, :]

    # Fan triangle j uses vertices (v0, v_j, v_{j+1}) for j = 1..MAX_VERTS-2
    # Number of potential fan triangles: MAX_VERTS - 2
    num_fan = MAX_VERTS - 2  # 10

    # Gather v_j and v_{j+1} for all fan triangles at once
    # j indices: 1, 2, ..., MAX_VERTS-2  (these are the "left" vertices)
    # j+1 indices: 2, 3, ..., MAX_VERTS-1  (these are the "right" vertices)
    # v_j: (C, num_fan, 3)
    v_j = poly[:, 1:MAX_VERTS - 1, :]      # indices 1..MAX_VERTS-2
    v_j1 = poly[:, 2:MAX_VERTS, :]          # indices 2..MAX_VERTS-1

    # Edge vectors from v0
    # (C, num_fan, 3)
    e1 = v_j - v0.unsqueeze(1)
    e2 = v_j1 - v0.unsqueeze(1)

    # Cross product: (C, num_fan, 3)
    cross = torch.cross(e1, e2, dim=2)

    # Triangle area = 0.5 * ||cross||: (C, num_fan)
    tri_area = 0.5 * torch.norm(cross, dim=2)

    # Triangle centroid = (v0 + v_j + v_{j+1}) / 3: (C, num_fan, 3)
    tri_centroid = (v0.unsqueeze(1) + v_j + v_j1) / 3.0

    # Validity mask: fan triangle j is valid when j+1 < v_len, i.e. index j+1
    # is a valid vertex.  j ranges from 1..MAX_VERTS-2, so j+1 ranges from
    # 2..MAX_VERTS-1.  Valid when j+1 < v_len.
    # j+1 values: (num_fan,) = [2, 3, ..., MAX_VERTS-1]
    j_plus_1 = torch.arange(2, MAX_VERTS, device=device, dtype=torch.int32)
    # (C, num_fan) mask
    valid = j_plus_1.unsqueeze(0) < v_len.unsqueeze(1)

    # Zero out invalid triangles
    tri_area = tri_area * valid.float()

    # Total area per polygon: (C,)
    total_area = tri_area.sum(dim=1)

    # Weighted centroid sum: (C, 3)
    weighted_sum = (tri_area.unsqueeze(2) * tri_centroid).sum(dim=1)

    # Normalize by total area, handling degenerate polygons
    safe_area = total_area.clone()
    degenerate = total_area < 1e-20
    safe_area[degenerate] = 1.0  # avoid division by zero

    centroid = weighted_sum / safe_area.unsqueeze(1)
    centroid[degenerate] = 0.0

    return centroid, total_area
