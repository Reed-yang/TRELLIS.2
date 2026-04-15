"""
Stage 4: Face weights (U-Turn detection) and component points (area-weighted centroids).

Merges custom/feature_face.py (s4a) and custom/feature_point.py (s4b):
  - face_weights (N, 12)  int32  — U-Turn count per triangulated facet
  - point_offsets (N+1,)  int64  + point_values (P, 3) float32
    — area-weighted component centroids packed in CSR

Public API:
    s4_face_point(batch, mesh) -> CubeBatch
"""
from __future__ import annotations

import numpy as np
import torch
import trimesh as _trimesh

from corep_fast.containers import MeshTensors, CubeBatch, _replace_fields

# Facet vertex indices — each facet's 3 cube-vertex indices
# Matches custom/feature_face.py _worker_triangles ordering.
FACET_VERTS = [
    (0, 1, 2),  # T0  — bottom half 1
    (2, 3, 0),  # T1  — bottom half 2
    (4, 5, 6),  # T2  — top half 1
    (6, 7, 4),  # T3  — top half 2
    (0, 1, 4),  # T4  — front half 1
    (4, 1, 5),  # T5  — front half 2
    (1, 2, 6),  # T6  — right half 1
    (5, 1, 6),  # T7  — right half 2
    (2, 3, 7),  # T8  — back half 1
    (6, 2, 7),  # T9  — back half 2
    (3, 0, 7),  # T10 — left half 1
    (7, 0, 4),  # T11 — left half 2
]

# Facet edge indices — each facet's 3 cube-edge indices
# edge_indices[j] is the edge between vertices (vj, v_{j+1 mod 3})
FACET_EDGES = [
    (0, 1, 12),   # T0
    (2, 3, 12),   # T1
    (4, 5, 13),   # T2
    (6, 7, 13),   # T3
    (0, 14, 8),   # T4
    (14, 9, 4),   # T5
    (1, 10, 15),  # T6
    (9, 15, 5),   # T7
    (2, 11, 16),  # T8
    (10, 16, 6),  # T9
    (3, 17, 11),  # T10
    (17, 8, 7),   # T11
]

# Unit cube vertex offsets
_V_OFFSETS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=np.float64)


def s4_face_point(batch: CubeBatch, mesh: MeshTensors) -> CubeBatch:
    """Compute face_weights and component_points via GPU operations.

    Merges custom/ s4a (feature_face) and s4b (feature_point):
    - face_weights (N, 12) — U-Turn count per triangulated facet
    - point_offsets (N+1,) + point_values (P, 3) — area-weighted component centroids
    """
    N = batch.num_cubes
    device = batch.device

    if N == 0:
        return batch

    # ------------------------------------------------------------------
    # Part 1: face_weights via CPU loop (mirrors custom/feature_face.py)
    # ------------------------------------------------------------------
    face_weights = _compute_face_weights_cpu(batch, mesh)

    # ------------------------------------------------------------------
    # Part 2: component_points via per-cube Sutherland-Hodgman clipping
    # ------------------------------------------------------------------
    point_offsets, point_values = _compute_component_points(batch, mesh)

    return _replace_fields(
        batch,
        face_weights=face_weights,
        point_offsets=point_offsets,
        point_values=point_values,
    )


# ======================================================================
# Part 1: Face weights (U-Turn detection) — CPU loop
# ======================================================================

def _compute_face_weights_cpu(batch: CubeBatch, mesh: MeshTensors) -> torch.Tensor:
    """Compute face_weights (N, 12) via per-cube CPU loop.

    For each cube and each of its 12 triangulated facets, intersect
    registered mesh triangles with the facet plane, clip to facet
    boundary, build a segment graph, and count U-Turn pairs.
    """
    N = batch.num_cubes
    device = batch.device
    R = batch.resolution
    step = 1.0 / R

    cube_indices_np = batch.cube_indices.cpu().numpy()
    tri_offsets_np = batch.tri_offsets.cpu().numpy()
    tri_values_np = batch.tri_values.cpu().numpy()
    mesh_verts_np = mesh.vertices.cpu().numpy().astype(np.float64)
    mesh_faces_np = mesh.faces.cpu().numpy()
    mesh_triangles_np = mesh_verts_np[mesh_faces_np]  # (F, 3, 3)

    fw_all = np.zeros((N, 12), dtype=np.int32)

    for ci in range(N):
        ix, iy, iz = cube_indices_np[ci]
        lo = int(tri_offsets_np[ci])
        hi = int(tri_offsets_np[ci + 1])
        if lo >= hi:
            continue

        f_ids = tri_values_np[lo:hi]
        cube_tris = mesh_triangles_np[f_ids]  # (K, 3, 3) float64

        base = np.array([ix, iy, iz], dtype=np.float64) * step
        cube_verts = base + _V_OFFSETS * step  # (8, 3)

        for t_idx, (vert_ids, edge_ids) in enumerate(zip(FACET_VERTS, FACET_EDGES)):
            v0i, v1i, v2i = vert_ids
            V0 = cube_verts[v0i]
            V1 = cube_verts[v1i]
            V2 = cube_verts[v2i]

            segments = _intersect_facet_with_mesh(V0, V1, V2, cube_tris)
            if not segments:
                continue

            fw_all[ci, t_idx] = _count_uturns(
                segments, V0, V1, V2, cube_verts,
                vert_ids, edge_ids,
            )

    return torch.from_numpy(fw_all).to(device=device, dtype=torch.int32)


def _intersect_facet_with_mesh(
    V0: np.ndarray, V1: np.ndarray, V2: np.ndarray,
    mesh_triangles: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute clipped intersection segments between a facet plane and mesh triangles.

    Mirrors custom/feature_face.py intersect_facet_with_mesh exactly.
    """
    E1 = V1 - V0
    E2 = V2 - V0
    Nc = np.cross(E1, E2)
    len_Nc = np.linalg.norm(Nc)
    if len_Nc < 1e-12:
        return []
    Nc = Nc / len_Nc

    M0 = mesh_triangles[:, 0, :]
    M1 = mesh_triangles[:, 1, :]
    M2 = mesh_triangles[:, 2, :]

    d0 = np.sum((M0 - V0) * Nc, axis=1)
    d1 = np.sum((M1 - V0) * Nc, axis=1)
    d2 = np.sum((M2 - V0) * Nc, axis=1)

    has_pos = (d0 > 1e-8) | (d1 > 1e-8) | (d2 > 1e-8)
    has_neg = (d0 < -1e-8) | (d1 < -1e-8) | (d2 < -1e-8)

    d0_zero = np.abs(d0) <= 1e-8
    d1_zero = np.abs(d1) <= 1e-8
    d2_zero = np.abs(d2) <= 1e-8
    coplanar_edge = (d0_zero & d1_zero) | (d1_zero & d2_zero) | (d2_zero & d0_zero)

    intersects_plane = (has_pos & has_neg) | coplanar_edge
    valid_idx = np.where(intersects_plane)[0]
    if len(valid_idx) == 0:
        return []

    segments = []
    for i in valid_idx:
        m0, m1, m2 = M0[i], M1[i], M2[i]
        da, db, dc = d0[i], d1[i], d2[i]

        pts: list[np.ndarray] = []
        if coplanar_edge[i]:
            if d0_zero[i] and d1_zero[i]:
                pts.extend([m0.copy(), m1.copy()])
            if d1_zero[i] and d2_zero[i]:
                pts.extend([m1.copy(), m2.copy()])
            if d2_zero[i] and d0_zero[i]:
                pts.extend([m2.copy(), m0.copy()])
        else:
            for a, b, d_a, d_b in [(m0, m1, da, db), (m1, m2, db, dc), (m2, m0, dc, da)]:
                if d_a * d_b < -1e-14:
                    t = d_a / (d_a - d_b)
                    pts.append(a + t * (b - a))
                elif abs(d_a) <= 1e-8:
                    pts.append(a.copy())

        # Deduplicate
        unique_pts: list[np.ndarray] = []
        for p in pts:
            if not any(np.linalg.norm(p - up) < 1e-8 for up in unique_pts):
                unique_pts.append(p)

        if len(unique_pts) < 2:
            continue

        P1, P2 = unique_pts[0], unique_pts[1]

        # Clip segment to facet boundary (Sutherland-Hodgman style)
        dP = P2 - P1
        t_min, t_max = 0.0, 1.0
        valid = True

        facet_verts_list = [V0, V1, V2]
        for j in range(3):
            A = facet_verts_list[j]
            B = facet_verts_list[(j + 1) % 3]
            edge = B - A
            nk = np.cross(Nc, edge)

            Ck_P1 = np.dot(P1 - A, nk)
            dot = np.dot(dP, nk)

            if dot > 1e-8:
                t_min = max(t_min, -Ck_P1 / dot)
            elif dot < -1e-8:
                t_max = min(t_max, -Ck_P1 / dot)
            else:
                if Ck_P1 < -1e-8:
                    valid = False
                    break

        if valid and t_min <= t_max + 1e-8 and t_max - t_min > 1e-8:
            segments.append((P1 + t_min * dP, P1 + t_max * dP))

    return segments


def _count_uturns(
    segments: list[tuple[np.ndarray, np.ndarray]],
    V0: np.ndarray, V1: np.ndarray, V2: np.ndarray,
    cube_verts: np.ndarray,
    vert_ids: tuple[int, int, int],
    edge_ids: tuple[int, int, int],
) -> int:
    """Build segment graph, find components, count U-Turn pairs.

    Mirrors custom/feature_face.py _process_cube graph analysis.
    """
    # 1. Build topological graph from segments
    nodes: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []

    for p1, p2 in segments:
        idx1 = _find_or_add_node(nodes, p1)
        idx2 = _find_or_add_node(nodes, p2)
        if idx1 != idx2:
            edge_tuple = (min(idx1, idx2), max(idx1, idx2))
            edges.append(edge_tuple)

    if not edges:
        return 0

    unique_edges = set(edges)
    adj: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for u, v in unique_edges:
        adj[u].append(v)
        adj[v].append(u)

    # 2. Extract connected components
    visited: set[int] = set()
    components: list[list[int]] = []
    for i in range(len(nodes)):
        if i not in visited:
            comp: list[int] = []
            q = [i]
            visited.add(i)
            while q:
                curr = q.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            components.append(comp)

    # 3. Analyze endpoint placements and count U-Turns
    v0i, v1i, v2i = vert_ids
    facet_verts_for_edges = [cube_verts[v0i], cube_verts[v1i], cube_verts[v2i]]
    total_uturns = 0

    for comp in components:
        endpoints = [n for n in comp if len(adj[n]) == 1]
        if not endpoints:
            continue  # Closed loops don't contribute

        edge_counts: dict[int, int] = {-1: 0}
        for e_idx in edge_ids:
            edge_counts[e_idx] = 0

        for ep in endpoints:
            P = nodes[ep]
            assigned: list[int] = []

            for j in range(3):
                A = facet_verts_for_edges[j]
                B = facet_verts_for_edges[(j + 1) % 3]
                edge_vec = B - A
                length = np.linalg.norm(edge_vec)
                if length < 1e-12:
                    continue
                t = np.dot(P - A, edge_vec) / (length ** 2)
                if -1e-8 <= t <= 1.0 + 1e-8:
                    proj = A + t * edge_vec
                    if np.linalg.norm(P - proj) < 1e-8:
                        assigned.append(edge_ids[j])

            if not assigned:
                assigned.append(-1)

            for e_id in assigned:
                edge_counts[e_id] += 1

        # 4. Count U-Turns: pairs of endpoints on the same edge
        for e_id, count in edge_counts.items():
            if e_id != -1:
                total_uturns += count // 2

    return total_uturns


def _find_or_add_node(nodes: list[np.ndarray], pt: np.ndarray, tol: float = 1e-8) -> int:
    """Find existing node matching pt, or add a new one."""
    for i, n in enumerate(nodes):
        if np.linalg.norm(pt - n) < tol:
            return i
    nodes.append(pt)
    return len(nodes) - 1


# ======================================================================
# Part 2: Component points — Sutherland-Hodgman AABB clipping
# ======================================================================

def _compute_component_points(
    batch: CubeBatch,
    mesh: MeshTensors,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute area-weighted component centroids packed in CSR.

    For each cube, for each connected component of registered faces:
    1. Clip each mesh triangle to the cube AABB (Sutherland-Hodgman, 6 planes)
    2. Fan-triangulate clipped polygons
    3. Compute area-weighted centroid: sum(area_i * centroid_i) / sum(area_i)

    Returns:
        point_offsets (N+1,) int64 — CSR offsets
        point_values  (P, 3) float32 — one centroid per component
    """
    N = batch.num_cubes
    device = batch.device
    R = batch.resolution

    cube_indices_np = batch.cube_indices.cpu().numpy()
    tri_offsets_np = batch.tri_offsets.cpu().numpy()
    tri_values_np = batch.tri_values.cpu().numpy()
    num_components_np = batch.num_components.cpu().numpy()

    mesh_verts_np = mesh.vertices.cpu().numpy().astype(np.float64)
    mesh_faces_idx_np = mesh.faces.cpu().numpy()  # (F, 3) int32 — vertex indices
    face_adj_np = mesh.face_adj.cpu().numpy()  # (F, 3) int32

    all_points: list[np.ndarray] = []
    offsets = np.zeros(N + 1, dtype=np.int64)

    for ci in range(N):
        ix, iy, iz = cube_indices_np[ci]
        lo = int(tri_offsets_np[ci])
        hi = int(tri_offsets_np[ci + 1])
        nc = int(num_components_np[ci])

        if lo >= hi or nc == 0:
            offsets[ci + 1] = offsets[ci]
            continue

        f_ids = tri_values_np[lo:hi].tolist()

        # AABB bounds for this cube
        min_bound = np.array([ix, iy, iz], dtype=np.float64) / R
        max_bound = np.array([ix + 1, iy + 1, iz + 1], dtype=np.float64) / R

        # Group faces into connected components using Union-Find
        components = _get_local_components(f_ids, mesh_faces_idx_np, face_adj_np)

        cube_points: list[np.ndarray] = []
        for comp_faces in components:
            centroid = _component_centroid(
                comp_faces, mesh_verts_np, mesh_faces_idx_np, min_bound, max_bound,
            )
            cube_points.append(centroid)

        # Ensure we produce exactly nc points (pad with cube center if needed)
        cube_center = (min_bound + max_bound) / 2.0
        while len(cube_points) < nc:
            cube_points.append(cube_center.copy())
        # Trim to nc (shouldn't happen, but safety)
        cube_points = cube_points[:nc]

        all_points.extend(cube_points)
        offsets[ci + 1] = offsets[ci] + nc

    point_offsets = torch.from_numpy(offsets).to(device=device, dtype=torch.int64)
    if all_points:
        point_values = torch.tensor(
            np.stack(all_points, axis=0), dtype=torch.float32, device=device,
        )
    else:
        point_values = torch.zeros((0, 3), dtype=torch.float32, device=device)

    return point_offsets, point_values


def _get_local_components(
    face_ids: list[int],
    mesh_faces: np.ndarray,  # (F, 3) int32
    face_adj: np.ndarray,    # (F, 3) int32
) -> list[list[int]]:
    """Get connected components of face_ids using face_adj (Union-Find).

    Two faces are connected if they share a mesh edge (i.e., one appears
    in the other's face_adj row).
    """
    n = len(face_ids)
    if n == 0:
        return []

    face_set = set(face_ids)
    face_to_idx = {f: i for i, f in enumerate(face_ids)}

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for idx, fid in enumerate(face_ids):
        for e in range(3):
            nbr = int(face_adj[fid, e])
            if nbr >= 0 and nbr in face_set:
                union(idx, face_to_idx[nbr])

    # Group by root
    groups: dict[int, list[int]] = {}
    for idx, fid in enumerate(face_ids):
        root = find(idx)
        groups.setdefault(root, []).append(fid)

    return list(groups.values())


def _component_centroid(
    comp_faces: list[int],
    mesh_verts: np.ndarray,    # (V, 3) float64
    mesh_faces: np.ndarray,    # (F, 3) int32
    min_bound: np.ndarray,     # (3,) float64
    max_bound: np.ndarray,     # (3,) float64
) -> np.ndarray:
    """Compute area-weighted centroid for a component's faces clipped to AABB.

    Uses Sutherland-Hodgman polygon clipping against 6 AABB planes,
    then fan-triangulation to compute area and centroid.
    The centroid is then snapped to the nearest point on the clipped
    surface mesh, matching custom/feature_point.py behaviour.
    """
    # 6 clip planes: (normal, point_on_plane) facing inward
    planes = [
        (np.array([1, 0, 0], dtype=np.float64), min_bound),
        (np.array([-1, 0, 0], dtype=np.float64), max_bound),
        (np.array([0, 1, 0], dtype=np.float64), min_bound),
        (np.array([0, -1, 0], dtype=np.float64), max_bound),
        (np.array([0, 0, 1], dtype=np.float64), min_bound),
        (np.array([0, 0, -1], dtype=np.float64), max_bound),
    ]

    total_area = 0.0
    weighted_centroid = np.zeros(3, dtype=np.float64)
    all_clipped_polys: list[list[np.ndarray]] = []

    for fid in comp_faces:
        vi = mesh_faces[fid]
        poly = [mesh_verts[vi[0]].copy(), mesh_verts[vi[1]].copy(), mesh_verts[vi[2]].copy()]

        # Clip against 6 planes
        for normal, point in planes:
            poly = _clip_polygon_against_plane(poly, normal, point)
            if len(poly) < 3:
                break

        if len(poly) < 3:
            continue

        # Fan-triangulate and accumulate area-weighted centroid
        p0 = poly[0]
        for i in range(1, len(poly) - 1):
            p1 = poly[i]
            p2 = poly[i + 1]
            cross = np.cross(p1 - p0, p2 - p0)
            area = 0.5 * np.linalg.norm(cross)
            centroid = (p0 + p1 + p2) / 3.0
            total_area += area
            weighted_centroid += centroid * area

        all_clipped_polys.append(poly)

    if total_area > 1e-12:
        target_center = weighted_centroid / total_area
        # Snap centroid to nearest point on the clipped surface mesh
        # (matches custom/feature_point.py behaviour)
        return _snap_to_clipped_surface(target_center, all_clipped_polys)
    else:
        # Fallback: snap cube center to the original component mesh surface
        cube_center = (min_bound + max_bound) / 2.0
        return _snap_to_component_surface(
            cube_center, comp_faces, mesh_verts, mesh_faces,
        )


def _snap_to_clipped_surface(
    target: np.ndarray,
    clipped_polys: list[list[np.ndarray]],
) -> np.ndarray:
    """Snap *target* to the nearest point on a mesh built from clipped polygons.

    Mirrors custom/feature_point.py: builds a trimesh from clipped polygons,
    then uses nearest.on_surface to project the centroid back onto the surface.
    """
    verts: list[np.ndarray] = []
    faces: list[list[int]] = []
    for poly in clipped_polys:
        idx_start = len(verts)
        verts.extend(poly)
        for i in range(1, len(poly) - 1):
            faces.append([idx_start, idx_start + i, idx_start + i + 1])

    if not faces:
        return target

    clipped_mesh = _trimesh.Trimesh(
        vertices=verts, faces=faces, process=False,
    )
    closest, _, _ = clipped_mesh.nearest.on_surface([target])
    return closest[0]


def _snap_to_component_surface(
    point: np.ndarray,
    comp_faces: list[int],
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
) -> np.ndarray:
    """Snap *point* to the nearest surface point of the original component mesh.

    Fallback path when clipped area is negligible.
    Mirrors custom/feature_point.py fallback behaviour.
    """
    local_faces = mesh_faces[comp_faces]
    comp_mesh = _trimesh.Trimesh(
        vertices=mesh_verts, faces=local_faces, process=True,
    )
    closest, _, _ = comp_mesh.nearest.on_surface([point])
    return closest[0]


def _clip_polygon_against_plane(
    polygon: list[np.ndarray],
    plane_normal: np.ndarray,
    plane_point: np.ndarray,
) -> list[np.ndarray]:
    """Clip a convex polygon against a half-space using Sutherland-Hodgman.

    Keeps vertices on the side where dot(normal, pt - plane_point) >= 0.
    """
    if not polygon:
        return []

    clipped: list[np.ndarray] = []
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]

        d1 = np.dot(plane_normal, p1 - plane_point)
        d2 = np.dot(plane_normal, p2 - plane_point)

        if d1 >= 0:
            clipped.append(p1)

        if (d1 >= 0 and d2 < 0) or (d1 < 0 and d2 >= 0):
            t = d1 / (d1 - d2)
            p_intersect = p1 + t * (p2 - p1)
            clipped.append(p_intersect)

    return clipped
