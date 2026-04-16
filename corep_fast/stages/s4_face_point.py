"""
Stage 4: Face weights (U-Turn detection) and component points (area-weighted centroids).

Merges custom/feature_face.py (s4a) and custom/feature_point.py (s4b):
  - face_weights (N, 12)  int32  — U-Turn count per triangulated facet
  - point_offsets (N+1,)  int64  + point_values (P, 3) float32
    — area-weighted component centroids packed in CSR

Part A (face_weights): CPU graph BFS per cube, parallelized via multiprocessing.
Part B (component_points): GPU-batched SH clip + fan centroid + closest-point snap.

Public API:
    s4_face_point(batch, mesh, pool=None) -> CubeBatch
"""
from __future__ import annotations

import numpy as np
import torch

from corep_fast.containers import MeshTensors, CubeBatch, _replace_fields
from corep_fast.geom.sh_clip import sh_clip_aabb
from corep_fast.geom.fan_centroid import fan_area_centroid
from corep_fast.geom.closest_point import closest_point_on_mesh

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


def s4_face_point(batch: CubeBatch, mesh: MeshTensors, pool=None,
                  num_workers: int | None = None) -> CubeBatch:
    """Compute face_weights and component_points.

    Part A: face_weights via CPU graph BFS, parallelized with multiprocessing.
    Part B: component_points via GPU-batched SH clip + fan centroid + closest-point snap.

    Args:
        batch: CubeBatch after s3 (with edge_weights, num_components, comp_face_off/val).
        mesh: MeshTensors with vertices, faces, triangles, face_adj.
        pool: Legacy PersistentWorkerPool (only used to derive num_workers if passed).
        num_workers: Worker count for internal multiprocessing. If None, uses (cpu_count - 4).
    """
    N = batch.num_cubes
    device = batch.device

    if N == 0:
        return batch

    # Derive worker count from legacy pool arg if num_workers not explicit
    if num_workers is None and pool is not None:
        num_workers = pool._num_workers

    # ------------------------------------------------------------------
    # Part 1: face_weights via CPU (multiprocessing over cubes)
    # ------------------------------------------------------------------
    face_weights = _compute_face_weights_mp(batch, mesh, num_workers=num_workers)

    # ------------------------------------------------------------------
    # Part 2: component_points via GPU-batched kernels
    # ------------------------------------------------------------------
    point_offsets, point_values = _compute_component_points_gpu(batch, mesh)

    return _replace_fields(
        batch,
        face_weights=face_weights,
        point_offsets=point_offsets,
        point_values=point_values,
    )


# ======================================================================
# Part 1: Face weights (U-Turn detection) — CPU + multiprocessing
# ======================================================================

# Fork-inherited shared data for face_weight workers.
# Set in parent process BEFORE creating the temporary Pool so that
# forked workers inherit them via copy-on-write (zero pickle overhead).
_FW_MESH_TRIS = None   # (F, 3, 3) float64
_FW_CUBE_IDX = None     # (N, 3) int32
_FW_TRI_OFF = None      # (N+1,) int64
_FW_TRI_VAL = None      # (T,) int32
_FW_STEP = None         # float


def _compute_face_weights_mp(batch: CubeBatch, mesh: MeshTensors,
                             num_workers: int | None = None) -> torch.Tensor:
    """Compute face_weights (N, 12) via per-cube CPU computation.

    Uses a temporary multiprocessing.Pool created AFTER setting module-level
    shared data, so forked workers inherit the data via OS copy-on-write
    with zero pickle overhead.
    """
    global _FW_MESH_TRIS, _FW_CUBE_IDX, _FW_TRI_OFF, _FW_TRI_VAL, _FW_STEP

    import os as _os
    N = batch.num_cubes
    device = batch.device
    R = batch.resolution
    _FW_STEP = 1.0 / R

    _FW_CUBE_IDX = batch.cube_indices.cpu().numpy()
    _FW_TRI_OFF = batch.tri_offsets.cpu().numpy()
    _FW_TRI_VAL = batch.tri_values.cpu().numpy()
    mesh_verts_np = mesh.vertices.cpu().numpy().astype(np.float64)
    mesh_faces_np = mesh.faces.cpu().numpy()
    _FW_MESH_TRIS = mesh_verts_np[mesh_faces_np]  # (F, 3, 3)

    # Default: use most cores (leave 4 for main process + GPU transfers)
    if num_workers is None:
        num_workers = max(1, (_os.cpu_count() or 4) - 4)

    if num_workers > 1 and N > 500:
        from multiprocessing import Pool as _Pool
        chunksize = max(1, N // (num_workers * 4))
        with _Pool(num_workers) as p:
            results = p.map(_fw_worker_indexed, range(N), chunksize=chunksize)
    else:
        results = [_fw_worker_indexed(ci) for ci in range(N)]

    fw_all = np.stack(results, axis=0)  # (N, 12)
    return torch.from_numpy(fw_all).to(device=device, dtype=torch.int32)


def _fw_worker_indexed(ci: int) -> np.ndarray:
    """Compute face weights for cube ci using fork-inherited shared data.

    Each worker only receives a single int (cube index) — no numpy pickle.
    Shared arrays (_FW_MESH_TRIS etc.) are inherited via fork COW.
    """
    fw = np.zeros(12, dtype=np.int32)
    lo = int(_FW_TRI_OFF[ci])
    hi = int(_FW_TRI_OFF[ci + 1])
    if lo >= hi:
        return fw

    ix, iy, iz = _FW_CUBE_IDX[ci]
    base = np.array([ix, iy, iz], dtype=np.float64) * _FW_STEP
    cube_verts = base + _V_OFFSETS * _FW_STEP  # (8, 3)
    cube_tris = _FW_MESH_TRIS[_FW_TRI_VAL[lo:hi]]  # (K, 3, 3) float64

    for t_idx, (vert_ids, edge_ids) in enumerate(zip(FACET_VERTS, FACET_EDGES)):
        v0i, v1i, v2i = vert_ids
        V0 = cube_verts[v0i]
        V1 = cube_verts[v1i]
        V2 = cube_verts[v2i]

        segments = _intersect_facet_with_mesh(V0, V1, V2, cube_tris)
        if not segments:
            continue

        fw[t_idx] = _count_uturns(
            segments, V0, V1, V2, cube_verts,
            vert_ids, edge_ids,
        )

    return fw


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
# Part 2: Component points — GPU-batched SH clip + fan centroid + snap
# ======================================================================

def _compute_component_points_gpu(
    batch: CubeBatch,
    mesh: MeshTensors,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute area-weighted component centroids packed in CSR, using GPU kernels.

    For each cube, for each connected component of registered faces:
    1. Clip each mesh triangle to the cube AABB using sh_clip_aabb()
    2. Compute area-weighted centroid using fan_area_centroid()
    3. Aggregate centroids across triangles within each component (weighted by area)
    4. Snap all centroids to mesh surface using closest_point_on_mesh()

    Returns:
        point_offsets (N+1,) int64 — CSR offsets
        point_values  (P, 3) float32 — one centroid per component
    """
    N = batch.num_cubes
    device = batch.device
    R = batch.resolution
    step = 1.0 / R

    num_components_np = batch.num_components.cpu().numpy()
    comp_face_off_np = batch.comp_face_off.cpu().numpy()
    comp_face_val = batch.comp_face_val  # keep on device

    # Build point_offsets from num_components (CSR: num_components per cube)
    nc_tensor = batch.num_components.to(torch.int64)
    point_offsets = torch.zeros(N + 1, dtype=torch.int64, device=device)
    point_offsets[1:] = torch.cumsum(nc_tensor, dim=0)
    total_points = int(point_offsets[-1].item())

    if total_points == 0:
        return point_offsets, torch.zeros((0, 3), dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Step 1: Enumerate all (cube, component) pairs and their face sets
    # ------------------------------------------------------------------
    # comp_face_off is a CSR over cubes, where each cube's segment contains
    # face ids grouped by component. We need to split each cube's segment
    # into num_components[ci] sub-groups.
    #
    # The comp_face_val layout for cube ci with nc components is:
    #   comp_face_val[comp_face_off[ci]:comp_face_off[ci+1]]
    #   split into nc contiguous groups (from s2_components union-find ordering).
    #
    # We need to re-discover component boundaries. Since s2 stored them
    # contiguously by component, we can use the face_adj connectivity
    # to identify group boundaries.

    cube_indices = batch.cube_indices  # (N, 3) int32, on device
    mesh_faces = mesh.faces  # (F, 3) int32, on device
    mesh_verts = mesh.vertices  # (V, 3) float32/float64, on device
    face_adj = mesh.face_adj  # (F, 3) int32, on device

    # Pre-compute AABB bounds for all cubes: (N, 3)
    cube_min = cube_indices.float() * step  # (N, 3)
    cube_max = (cube_indices.float() + 1.0) * step  # (N, 3)

    # ------------------------------------------------------------------
    # Step 2: For each (cube, component), collect face ids, clip, centroid
    # ------------------------------------------------------------------
    # We process this by iterating over cubes on CPU to build the
    # (cube_component -> face_ids) mapping, then batch the GPU work.

    # Collect all per-component triangle indices and their cube assignments
    comp_cube_idx = []    # which cube each component belongs to
    comp_face_lists = []  # list of face-id arrays per component

    comp_face_off_cpu = comp_face_off_np
    comp_face_val_cpu = comp_face_val.cpu().numpy()
    mesh_faces_cpu = mesh_faces.cpu().numpy()
    face_adj_cpu = face_adj.cpu().numpy()

    for ci in range(N):
        nc = int(num_components_np[ci])
        if nc == 0:
            continue

        lo = int(comp_face_off_cpu[ci])
        hi = int(comp_face_off_cpu[ci + 1])
        face_ids = comp_face_val_cpu[lo:hi]

        if len(face_ids) == 0:
            # Pad with empty components
            for _ in range(nc):
                comp_cube_idx.append(ci)
                comp_face_lists.append(np.array([], dtype=np.int32))
            continue

        # Split face_ids into connected components using face_adj
        components = _get_local_components_np(face_ids, mesh_faces_cpu, face_adj_cpu)

        # Take at most nc components; pad with empty if fewer found
        for k, comp_faces in enumerate(components[:nc]):
            comp_cube_idx.append(ci)
            comp_face_lists.append(np.array(comp_faces, dtype=np.int32))
        # Pad with empty components if union-find found fewer than expected
        for _ in range(nc - len(components[:nc])):
            comp_cube_idx.append(ci)
            comp_face_lists.append(np.array([], dtype=np.int32))

    assert len(comp_cube_idx) == total_points, \
        f"Component count mismatch: got {len(comp_cube_idx)}, expected {total_points}"

    comp_cube_idx_np = np.array(comp_cube_idx, dtype=np.int64)

    # ------------------------------------------------------------------
    # Step 3: Flatten all (component, triangle) pairs for GPU batch clip
    # ------------------------------------------------------------------
    # For each component, we need to clip its triangles against its cube AABB.
    # Flatten into one big batch for sh_clip_aabb.

    # Build flat arrays: for each (component, face) pair, store the face_id and component_id
    flat_face_ids = []
    flat_comp_ids = []
    comp_tri_offsets = np.zeros(total_points + 1, dtype=np.int64)

    for k in range(total_points):
        faces_k = comp_face_lists[k]
        flat_face_ids.append(faces_k)
        flat_comp_ids.extend([k] * len(faces_k))
        comp_tri_offsets[k + 1] = comp_tri_offsets[k] + len(faces_k)

    total_tris = int(comp_tri_offsets[-1])

    if total_tris == 0:
        # All components have no faces — return cube centers as fallback
        cube_centers = (cube_min + cube_max) / 2.0  # (N, 3)
        all_pts = cube_centers[torch.from_numpy(comp_cube_idx_np).to(device)]  # (P, 3)
        return point_offsets, all_pts.float()

    flat_face_ids_np = np.concatenate(flat_face_ids) if flat_face_ids else np.array([], dtype=np.int32)
    flat_comp_ids_np = np.array(flat_comp_ids, dtype=np.int64)

    # Gather triangles for all (component, face) pairs: (total_tris, 3, 3)
    flat_face_ids_t = torch.from_numpy(flat_face_ids_np.astype(np.int64)).to(device)
    flat_comp_ids_t = torch.from_numpy(flat_comp_ids_np).to(device)
    comp_cube_idx_t = torch.from_numpy(comp_cube_idx_np).to(device)

    # Get the cube index for each flat triangle
    tri_cube_idx = comp_cube_idx_t[flat_comp_ids_t]  # (total_tris,)

    # Gather mesh triangles: vertices[faces[face_id]]
    triangles_gpu = mesh.triangles[flat_face_ids_t.long()]  # (total_tris, 3, 3) float32

    # Gather AABB bounds per triangle from their cube
    aabb_min = cube_min[tri_cube_idx.long()]  # (total_tris, 3)
    aabb_max = cube_max[tri_cube_idx.long()]  # (total_tris, 3)

    # ------------------------------------------------------------------
    # Step 4: GPU-batched SH clip + fan centroid
    # ------------------------------------------------------------------
    # sh_clip_aabb expects float32
    triangles_f32 = triangles_gpu.float()
    aabb_min_f32 = aabb_min.float()
    aabb_max_f32 = aabb_max.float()

    poly, v_len = sh_clip_aabb(triangles_f32, aabb_min_f32, aabb_max_f32)
    # poly: (total_tris, MAX_VERTS, 3), v_len: (total_tris,) int32

    centroid_per_tri, area_per_tri = fan_area_centroid(poly, v_len)
    # centroid_per_tri: (total_tris, 3), area_per_tri: (total_tris,)

    # ------------------------------------------------------------------
    # Step 5: Aggregate centroids by component (area-weighted)
    # ------------------------------------------------------------------
    # For each component k, compute:
    #   centroid_k = sum(area_i * centroid_i) / sum(area_i)
    # where i ranges over triangles in component k.

    # Use scatter to aggregate by component
    weighted_centroids = centroid_per_tri * area_per_tri.unsqueeze(1)  # (total_tris, 3)

    # Scatter-add weighted centroids and areas by component index
    comp_weighted_sum = torch.zeros(total_points, 3, device=device, dtype=torch.float32)
    comp_area_sum = torch.zeros(total_points, device=device, dtype=torch.float32)

    comp_weighted_sum.scatter_add_(0, flat_comp_ids_t.unsqueeze(1).expand(-1, 3), weighted_centroids)
    comp_area_sum.scatter_add_(0, flat_comp_ids_t, area_per_tri)

    # Normalize: centroid = weighted_sum / total_area
    degenerate = comp_area_sum < 1e-12
    safe_area = comp_area_sum.clone()
    safe_area[degenerate] = 1.0

    comp_centroids = comp_weighted_sum / safe_area.unsqueeze(1)  # (P, 3)

    # For degenerate components (zero clipped area), use cube center as fallback
    cube_centers = (cube_min + cube_max) / 2.0  # (N, 3)
    fallback_centers = cube_centers[comp_cube_idx_t.long()]  # (P, 3)
    comp_centroids[degenerate] = fallback_centers[degenerate].float()

    # ------------------------------------------------------------------
    # Step 6: Snap centroids to mesh surface using closest_point_on_mesh
    # ------------------------------------------------------------------
    # For each component, snap centroid to the clipped surface.
    # However, building per-component clipped meshes for trimesh is expensive.
    # Instead, we snap to the *original* mesh triangles registered to each component,
    # which is much more GPU-friendly.
    #
    # For non-degenerate components: snap centroid to the component's own triangles.
    # For degenerate components: snap cube center to the component's own triangles.

    # We process snapping per component: for each component k, find closest point
    # on its triangles. To batch this efficiently, we group components by similar
    # triangle count and process in chunks.

    point_values = _snap_centroids_to_components(
        comp_centroids, comp_face_lists, mesh, device,
    )

    return point_offsets, point_values


def _snap_centroids_to_components(
    centroids: torch.Tensor,     # (P, 3) float32
    comp_face_lists: list,        # list of np arrays of face ids per component
    mesh: MeshTensors,
    device: torch.device,
) -> torch.Tensor:
    """Snap each component centroid to its component's mesh surface.

    BATCHED GPU implementation: pad components to max_k triangles, apply
    Ericson §5.1.5 region decomposition across all (centroid, triangle) pairs
    in a single kernel call. Replaces a per-centroid Python loop that caused
    ~275K GPU kernel launches at res=256.
    """
    P = centroids.shape[0]
    if P == 0:
        return centroids

    # Find max triangle count across components
    max_k = 0
    for f in comp_face_lists:
        if len(f) > max_k:
            max_k = len(f)
    if max_k == 0:
        # All components empty — keep centroids as-is (cube centers)
        return centroids.clone()

    # Build padded face ids (P, max_k) with -1 for padding
    face_ids_padded = np.full((P, max_k), -1, dtype=np.int64)
    for k, face_ids in enumerate(comp_face_lists):
        n = len(face_ids)
        if n > 0:
            face_ids_padded[k, :n] = face_ids

    face_ids_t = torch.from_numpy(face_ids_padded).to(device)  # (P, max_k)
    mask = face_ids_t >= 0  # (P, max_k) bool
    safe_ids = face_ids_t.clamp(min=0)  # replace -1 with 0 for safe indexing

    # Gather triangles: (P, max_k, 3, 3)
    all_triangles = mesh.triangles.float()  # (F, 3, 3)
    comp_tris = all_triangles[safe_ids]  # (P, max_k, 3, 3)

    # Batched closest-point-on-triangle (Ericson §5.1.5) for (P, max_k) pairs.
    # Shape trick: each query is paired with its own row of max_k triangles.
    q = centroids.unsqueeze(1)  # (P, 1, 3)
    a = comp_tris[:, :, 0, :]   # (P, max_k, 3)
    b = comp_tris[:, :, 1, :]
    c = comp_tris[:, :, 2, :]

    ab = b - a                  # (P, max_k, 3)
    ac = c - a
    aq = q - a                  # (P, max_k, 3)
    bq = q - b
    cq = q - c

    d1 = (ab * aq).sum(dim=-1)  # (P, max_k)
    d2 = (ac * aq).sum(dim=-1)
    d3 = (ab * bq).sum(dim=-1)
    d4 = (ac * bq).sum(dim=-1)
    d5 = (ab * cq).sum(dim=-1)
    d6 = (ac * cq).sum(dim=-1)

    region_a = (d1 <= 0) & (d2 <= 0)
    region_b = (d3 >= 0) & (d4 <= d3)
    region_c = (d6 >= 0) & (d5 <= d6)

    vc = d1 * d4 - d3 * d2
    edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v_ab = d1 / (d1 - d3 + 1e-30)

    vb = d5 * d2 - d1 * d6
    edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w_ac = d2 / (d2 - d6 + 1e-30)

    va2 = d3 * d6 - d5 * d4
    edge_bc = (va2 <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-30)

    denom = 1.0 / (va2 + vb + vc + 1e-30)
    v_int = vb * denom
    w_int = vc * denom

    closest = a + v_int.unsqueeze(-1) * ab + w_int.unsqueeze(-1) * ac  # (P, max_k, 3)
    closest = torch.where(edge_bc.unsqueeze(-1), b + w_bc.unsqueeze(-1) * (c - b), closest)
    closest = torch.where(edge_ac.unsqueeze(-1), a + w_ac.unsqueeze(-1) * ac, closest)
    closest = torch.where(edge_ab.unsqueeze(-1), a + v_ab.unsqueeze(-1) * ab, closest)
    closest = torch.where(region_c.unsqueeze(-1), c, closest)
    closest = torch.where(region_b.unsqueeze(-1), b, closest)
    closest = torch.where(region_a.unsqueeze(-1), a, closest)

    diff = q - closest                        # (P, max_k, 3)
    sq_dists = (diff * diff).sum(dim=-1)      # (P, max_k)

    # Mask out padding triangles (set distance to +inf)
    sq_dists = torch.where(mask, sq_dists, torch.full_like(sq_dists, float('inf')))

    # For each centroid, pick the triangle with minimum distance
    min_idx = sq_dists.argmin(dim=1)  # (P,)

    # Gather the best closest point per centroid
    # closest: (P, max_k, 3) → pick closest[k, min_idx[k], :] for each k
    best_idx = min_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 3)  # (P, 1, 3)
    snapped = closest.gather(1, best_idx).squeeze(1)  # (P, 3)

    # Components with no triangles (all masked out) keep original centroid
    any_valid = mask.any(dim=1)  # (P,) bool
    result = torch.where(any_valid.unsqueeze(-1), snapped, centroids)
    return result


def _get_local_components_np(
    face_ids: np.ndarray,
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

    face_set = set(int(f) for f in face_ids)
    face_to_idx = {int(f): i for i, f in enumerate(face_ids)}

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
        fid_int = int(fid)
        for e in range(3):
            nbr = int(face_adj[fid_int, e])
            if nbr >= 0 and nbr in face_set:
                union(idx, face_to_idx[nbr])

    # Group by root
    groups: dict[int, list[int]] = {}
    for idx, fid in enumerate(face_ids):
        root = find(idx)
        groups.setdefault(root, []).append(int(fid))

    return list(groups.values())
