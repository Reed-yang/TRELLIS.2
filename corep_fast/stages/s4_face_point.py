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

# Precomputed (12, 3, 3) per-facet vertex-offset lookup.
# FACET_V_OFFSETS[t, k, :] = _V_OFFSETS[FACET_VERTS[t][k], :].
# Avoids materializing per-cube (G, 8, 3) cube_verts in CSR packing.
FACET_V_OFFSETS = _V_OFFSETS[np.asarray(FACET_VERTS, dtype=np.int64)]  # (12, 3, 3) float64


def s4_face_point(batch: CubeBatch, mesh: MeshTensors, pool=None,
                  num_workers: int | None = None,
                  use_gpu_fw: bool | None = None) -> CubeBatch:
    """Compute face_weights and component_points.

    Part A: face_weights via CPU graph BFS, parallelized with multiprocessing
            (default), or GPU-accelerated batch path when ``use_gpu_fw`` is True.
    Part B: component_points via GPU-batched SH clip + fan centroid + closest-point snap.

    Args:
        batch: CubeBatch after s3 (with edge_weights, num_components, comp_face_off/val).
        mesh: MeshTensors with vertices, faces, triangles, face_adj.
        pool: Legacy PersistentWorkerPool (only used to derive num_workers if passed).
        num_workers: Worker count for internal multiprocessing. If None, uses (cpu_count - 4).
        use_gpu_fw: If True, use GPU-accelerated face_weights path (M2 P2).
                    If None, defaults to corep_fast.config.USE_GPU_FW_S4.
    """
    N = batch.num_cubes
    device = batch.device

    if N == 0:
        return batch

    # Derive worker count from legacy pool arg if num_workers not explicit
    if num_workers is None and pool is not None:
        num_workers = pool._num_workers

    # ------------------------------------------------------------------
    # Part 1: face_weights — GPU batch path (M2 P2) or CPU MP fallback
    # ------------------------------------------------------------------
    if use_gpu_fw is None:
        from corep_fast.config import USE_GPU_FW_S4
        use_gpu_fw = USE_GPU_FW_S4

    if use_gpu_fw:
        face_weights = _compute_face_weights_gpu(batch, mesh)
    else:
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
        from corep_fast.utils.persistent_pool import get_pool
        chunksize = max(1, N // (num_workers * 4))
        p = get_pool(num_workers)
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


def _count_uturns_gpu_batched(groups):
    """Batched GPU equivalent of _count_uturns. Returns (G,) int64 tensor.

    Args:
        groups: iterable of (segments, V0, V1, V2, cube_verts, vert_ids, edge_ids)
            tuples — same signature as _count_uturns per group.

    Returns:
        torch.Tensor shape (G,) int64, U-turn count per group (on CUDA device
        if available, else CPU).

    This entrypoint does the Python-side pack of per-group tuples into padded
    tensors, then dispatches to the shared GPU core `_count_uturns_from_packed`.
    Unit tests hit this path (G up to ~50k). Production (G ~1.88M) should use
    `_count_uturns_gpu_batched_csr` to avoid the per-group Python loop.
    """
    import numpy as np
    import torch as _torch

    groups = list(groups)
    G = len(groups)
    if G == 0:
        return _torch.zeros(0, dtype=_torch.int64)

    max_s = max((len(g[0]) for g in groups), default=0)
    if max_s == 0:
        return _torch.zeros(G, dtype=_torch.int64)
    P_MAX = 2 * max_s

    dev = _torch.device('cuda:0' if _torch.cuda.is_available() else 'cpu')

    # ---- Pack into padded tensors ----
    pts_cpu = np.zeros((G, P_MAX, 3), dtype=np.float64)
    pts_valid_cpu = np.zeros((G, P_MAX), dtype=bool)
    fv_cpu = np.zeros((G, 3, 3), dtype=np.float64)
    ed_cpu = np.full((G, 3), -1, dtype=np.int64)
    for gi, (segs, V0, V1, V2, cube_verts, vert_ids, edge_ids) in enumerate(groups):
        v0i, v1i, v2i = vert_ids
        fv_cpu[gi, 0] = cube_verts[v0i]
        fv_cpu[gi, 1] = cube_verts[v1i]
        fv_cpu[gi, 2] = cube_verts[v2i]
        ed_cpu[gi] = np.asarray(edge_ids, dtype=np.int64)
        for si, (a, b) in enumerate(segs):
            pts_cpu[gi, 2 * si] = a
            pts_cpu[gi, 2 * si + 1] = b
            pts_valid_cpu[gi, 2 * si] = True
            pts_valid_cpu[gi, 2 * si + 1] = True

    pts = _torch.from_numpy(pts_cpu).to(dev)
    pts_valid = _torch.from_numpy(pts_valid_cpu).to(dev)
    facet_verts = _torch.from_numpy(fv_cpu).to(dev)
    edge_ids_t = _torch.from_numpy(ed_cpu).to(dev)

    return _count_uturns_from_packed(pts, pts_valid, facet_verts, edge_ids_t)


def _count_uturns_gpu_batched_csr(
    cf_np,          # (G,) int64  — packed cube_id*12 + facet_id
    group_off_np,   # (G+1,) int64 — CSR offsets into A/B
    A_np,           # (S, 3) float64 — segment start points
    B_np,           # (S, 3) float64 — segment end points
    cube_idx_np,    # (N, 3) int (any int dtype) — cube voxel indices
    step: float,    # 1.0 / resolution
):
    """CSR-array batched GPU U-turn counting (production dispatch).

    Memory-bounded chunking wrapper. The core packs padded ``(G, P, P)``
    tensors whose peak memory scales with ``G * P_MAX^2``. At high resolution
    (e.g. 512), a single shot exceeds any single-GPU's VRAM. This wrapper:

      1. Sorts groups by per-group segment count (descending) so the largest
         ``P_MAX`` lives in its own small chunk;
      2. Walks the sorted groups left-to-right, growing each chunk until its
         peak element count ``G_chunk * (2 * chunk_max_s)^2`` hits the budget
         (env ``COREP_FAST_S4_UTURN_CHUNK_ELEMS``, default ~2.5e8 elements ≈
         2 GB per ``(G, P, P)`` i64/f64 tensor);
      3. Runs each chunk through ``_count_uturns_gpu_batched_csr_chunk``
         (the original packing + core call);
      4. Restores the caller's group order.

    Returns (G,) int64 torch.Tensor of U-turn counts (CPU when the core
    ran on CUDA, matching the single-chunk behaviour).
    """
    import os as _os
    import numpy as np
    import torch as _torch

    G = int(cf_np.shape[0])
    if G == 0:
        return _torch.zeros(0, dtype=_torch.int64)

    segs_per_group_np = (group_off_np[1:] - group_off_np[:-1]).astype(np.int64)  # (G,)
    max_s = int(segs_per_group_np.max()) if G > 0 else 0
    if max_s == 0:
        return _torch.zeros(G, dtype=_torch.int64)

    global_p_max = 2 * max_s
    try:
        budget_elems = int(_os.environ.get(
            'COREP_FAST_S4_UTURN_CHUNK_ELEMS', str(250_000_000)))
    except ValueError:
        budget_elems = 250_000_000

    # Fast path: the whole batch fits the budget, skip sort/permute overhead.
    if G * (global_p_max ** 2) <= budget_elems:
        return _count_uturns_gpu_batched_csr_chunk(
            cf_np, group_off_np, A_np, B_np, cube_idx_np, step,
        )

    # Sort groups by segment count descending. Localising large-P groups keeps
    # later (smaller-P) chunks dense, minimising the total number of chunks.
    order = np.argsort(-segs_per_group_np, kind='stable')
    inverse_order = np.argsort(order, kind='stable')

    sorted_segs = segs_per_group_np[order]
    sorted_cf = cf_np[order]

    # Rebuild the CSR in the sorted order. seg_perm[new_idx] = old_seg_idx.
    total_segs = int(group_off_np[-1])
    new_off = np.concatenate(([0], np.cumsum(sorted_segs))).astype(np.int64)
    src_starts = group_off_np[order].astype(np.int64)

    seg_perm = np.empty(total_segs, dtype=np.int64)
    for i in range(G):
        c = int(sorted_segs[i])
        if c == 0:
            continue
        s = int(src_starts[i])
        d = int(new_off[i])
        seg_perm[d:d + c] = np.arange(s, s + c, dtype=np.int64)

    sorted_A = A_np[seg_perm]
    sorted_B = B_np[seg_perm]

    out_np = np.zeros(G, dtype=np.int64)
    g_start = 0
    while g_start < G:
        chunk_max_s = int(sorted_segs[g_start])
        chunk_p_max_sq = (2 * chunk_max_s) ** 2 if chunk_max_s > 0 else 1
        max_chunk_g = max(1, budget_elems // max(1, chunk_p_max_sq))
        g_end = min(G, g_start + int(max_chunk_g))

        chunk_cf = sorted_cf[g_start:g_end]
        chunk_off = new_off[g_start:g_end + 1] - new_off[g_start]
        chunk_A = sorted_A[new_off[g_start]:new_off[g_end]]
        chunk_B = sorted_B[new_off[g_start]:new_off[g_end]]

        chunk_result = _count_uturns_gpu_batched_csr_chunk(
            chunk_cf, chunk_off, chunk_A, chunk_B, cube_idx_np, step,
        )
        # chunk_result is a (G_chunk,) int64 tensor, on CPU when core ran on
        # CUDA (see `_count_uturns_from_packed`), else on the core's device.
        out_np[g_start:g_end] = chunk_result.detach().cpu().numpy()

        # Release cached GPU memory between chunks to reduce fragmentation.
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

        g_start = g_end

    return _torch.from_numpy(out_np[inverse_order])


def _count_uturns_gpu_batched_csr_chunk(
    cf_np,          # (G,) int64  — packed cube_id*12 + facet_id
    group_off_np,   # (G+1,) int64 — CSR offsets into A/B
    A_np,           # (S, 3) float64 — segment start points
    B_np,           # (S, 3) float64 — segment end points
    cube_idx_np,    # (N, 3) int (any int dtype) — cube voxel indices
    step: float,    # 1.0 / resolution
):
    """Single-chunk CSR pack + dispatch to `_count_uturns_from_packed`.

    Optimised CSR packing (Task 9b): all big intermediates built on GPU,
    no per-cube (G, 8, 3) cube_verts materialisation, dtype f32 throughout.
    Reuses the shared core `_count_uturns_from_packed`.

    Returns (G,) int64 torch.Tensor of U-turn counts (on the same device the
    core runs on — CPU copy happens at caller).
    """
    import numpy as np
    import torch as _torch

    G = int(cf_np.shape[0])
    if G == 0:
        return _torch.zeros(0, dtype=_torch.int64)

    # CSR offsets host-side scalar — reused below for max_s determination.
    segs_per_group_np = (group_off_np[1:] - group_off_np[:-1]).astype(np.int64)  # (G,)
    max_s = int(segs_per_group_np.max()) if G > 0 else 0
    if max_s == 0:
        return _torch.zeros(G, dtype=_torch.int64)
    P_MAX = 2 * max_s

    dev = _torch.device('cuda:0' if _torch.cuda.is_available() else 'cpu')

    # ---- Push raw inputs to GPU once (small + medium tensors) ----
    # Opt C: f32 was tried first — failed F3 golden (V count 405176 vs 405214)
    # due to f32 noise vs 1e-8 Phase-A coalescence threshold. Reverted to f64
    # to keep F1-F3 bit-exact. Opt A + Opt B still give the bulk of the win
    # (skip 361 MB cube_verts_all + skip numpy→GPU fancy-index pts packing).
    A_t = _torch.from_numpy(A_np.astype(np.float64, copy=False)).to(dev, non_blocking=True)   # (S, 3) f64
    B_t = _torch.from_numpy(B_np.astype(np.float64, copy=False)).to(dev, non_blocking=True)
    group_off_t = _torch.from_numpy(group_off_np.astype(np.int64, copy=False)).to(dev, non_blocking=True)  # (G+1,)
    segs_per_group_t = group_off_t[1:] - group_off_t[:-1]                                  # (G,)

    # ---- Opt B: GPU-side pts packing ----
    seg_idx_t = _torch.arange(max_s, device=dev, dtype=_torch.int64)                       # (max_s,)
    flat_si_t = group_off_t[:-1, None] + seg_idx_t[None, :]                                # (G, max_s)
    valid_mask_2d_t = seg_idx_t[None, :] < segs_per_group_t[:, None]                       # (G, max_s) bool
    flat_si_clamped_t = _torch.where(valid_mask_2d_t, flat_si_t, _torch.zeros_like(flat_si_t))

    A_gathered_t = A_t[flat_si_clamped_t]                                                  # (G, max_s, 3) f64
    B_gathered_t = B_t[flat_si_clamped_t]                                                  # (G, max_s, 3) f64

    # Build padded (G, P_MAX, 3) directly on device.
    pts = _torch.zeros((G, P_MAX, 3), dtype=_torch.float64, device=dev)
    valid_mask_2d_exp = valid_mask_2d_t.unsqueeze(-1)
    pts[:, 0::2, :] = _torch.where(valid_mask_2d_exp, A_gathered_t, _torch.zeros_like(A_gathered_t))
    pts[:, 1::2, :] = _torch.where(valid_mask_2d_exp, B_gathered_t, _torch.zeros_like(B_gathered_t))

    pts_valid = _torch.zeros((G, P_MAX), dtype=_torch.bool, device=dev)
    pts_valid[:, 0::2] = valid_mask_2d_t
    pts_valid[:, 1::2] = valid_mask_2d_t

    # ---- Opt A: facet_verts via precomputed FACET_V_OFFSETS lookup (no cube_verts_all) ----
    cf_t = _torch.from_numpy(cf_np.astype(np.int64, copy=False)).to(dev, non_blocking=True)
    cube_id_t = cf_t // 12
    facet_id_t = cf_t % 12

    cube_idx_t = _torch.from_numpy(cube_idx_np.astype(np.int64, copy=False)).to(dev, non_blocking=True)  # (N, 3)
    base_t = cube_idx_t[cube_id_t].to(_torch.float64) * step                              # (G, 3) f64

    fvo_t = _torch.from_numpy(FACET_V_OFFSETS.astype(np.float64, copy=False)).to(dev, non_blocking=True)  # (12, 3, 3)
    offsets_per_group_t = fvo_t[facet_id_t]                                               # (G, 3, 3) f64
    facet_verts = base_t[:, None, :] + offsets_per_group_t * step                         # (G, 3, 3) f64

    fe_lut_t = _torch.from_numpy(np.asarray(FACET_EDGES, dtype=np.int64)).to(dev, non_blocking=True)  # (12, 3)
    edge_ids_t = fe_lut_t[facet_id_t]                                                     # (G, 3) int64

    return _count_uturns_from_packed(pts, pts_valid, facet_verts, edge_ids_t)


def _count_uturns_from_packed(pts, pts_valid, facet_verts, edge_ids_t):
    """Core GPU batched U-turn algorithm — phases A/B/C/D.

    Shared by `_count_uturns_gpu_batched` (Python list-of-tuples entry) and
    `_count_uturns_gpu_batched_csr` (production CSR-array entry). Packing is
    done by the caller; this function only runs the GPU algorithm.

    Args:
        pts         (G, P, 3) float64 — padded per-group segment endpoints
        pts_valid   (G, P)    bool    — valid-slot mask
        facet_verts (G, 3, 3) float64 — V0/V1/V2 per group
        edge_ids_t  (G, 3)    int64   — 3 triangle-edge ids per group

    Returns:
        torch.Tensor shape (G,) int64 on CPU (device → CPU copy at end).

    Pipeline (mirrors legacy _count_uturns):
      Phase A — node coalescence via cdist (tol=1e-8, first-occurrence wins)
      Phase B — scatter to build edge_mask (G, P, P) between canonical nodes
      Phase C — label-propagation connected components (diameter-bounded iter)
      Phase C.5 — degree + endpoint identification (degree==1 on self-canonical)
      Phase D — endpoint -> 3 triangle-edge projection, bucket by (comp, edge_id),
                bincount + //2 => U-turn contribution per group
    """
    import torch as _torch

    G, P, _ = pts.shape
    P_MAX = P
    max_s = P // 2
    dev = pts.device

    # ---- Phase A: node coalescence (1e-8 tolerance, first-occurrence wins) ----
    # cdist for valid pts only; padded zeros would spuriously match each other.
    # Mask padded rows/cols by setting their pairwise distance to a huge value.
    d = _torch.cdist(pts, pts)                                # (G, P, P) f64
    valid_pair = pts_valid.unsqueeze(2) & pts_valid.unsqueeze(1)
    # padded-pair distance -> large so match=False
    d = _torch.where(valid_pair, d, _torch.full_like(d, 1.0))
    match = (d < 1e-8)                                         # (G, P, P) bool

    tri_lower = _torch.tril(_torch.ones((P, P), dtype=_torch.bool, device=dev))
    match_lower = match & tri_lower.unsqueeze(0)
    # For each i, smallest j<=i with match -> canonical representative slot.
    j_ar = _torch.arange(P, device=dev, dtype=_torch.int64).view(1, 1, P).expand(G, P, P)
    big_P = _torch.full_like(j_ar, P)
    node_raw = _torch.where(match_lower, j_ar, big_P)
    canonical_idx = node_raw.min(dim=-1).values                # (G, P) in [0..P]
    # Invalid slots -> P sentinel
    canonical_idx = _torch.where(pts_valid, canonical_idx,
                                 _torch.full_like(canonical_idx, P))
    # QW1 (2026-04-21): free Phase-A intermediates before Phase B allocates
    # its own (G,P,P) bool edge_mask. Gated on VRAM_RESCUE so A/B measurable.
    from corep_fast.config import VRAM_RESCUE as _VRAM_RESCUE
    if _VRAM_RESCUE:
        del d, valid_pair, match, match_lower, node_raw, big_P, j_ar, tri_lower
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

    # A slot is a "representative" iff its canonical equals its own index.
    idx_row = _torch.arange(P, device=dev, dtype=_torch.int64).unsqueeze(0)
    self_canonical = (canonical_idx == idx_row) & pts_valid    # (G, P)

    # ---- Phase B: edge_mask between canonical-representative slots ----
    seg_slot_a = _torch.arange(0, P, 2, device=dev, dtype=_torch.int64)   # (max_s,)
    seg_slot_b = _torch.arange(1, P, 2, device=dev, dtype=_torch.int64)
    c_a = canonical_idx.index_select(1, seg_slot_a)                       # (G, max_s)
    c_b = canonical_idx.index_select(1, seg_slot_b)
    seg_valid = pts_valid.index_select(1, seg_slot_a) & pts_valid.index_select(1, seg_slot_b)
    seg_nontrivial = seg_valid & (c_a != c_b)                             # (G, max_s)

    edge_mask = _torch.zeros((G, P, P), dtype=_torch.bool, device=dev)
    g_idx_exp = _torch.arange(G, device=dev, dtype=_torch.int64).unsqueeze(1).expand(G, max_s)
    # Guard canonical indices against the sentinel P (only happens on invalid segs; filtered by mask_flat)
    c_a_safe = c_a.clamp(max=P - 1)
    c_b_safe = c_b.clamp(max=P - 1)
    flat = (g_idx_exp * (P * P) + c_a_safe * P + c_b_safe).reshape(-1)
    flat_sym = (g_idx_exp * (P * P) + c_b_safe * P + c_a_safe).reshape(-1)
    mask_flat = seg_nontrivial.reshape(-1)
    edge_mask_flat = edge_mask.view(-1)
    if mask_flat.any():
        edge_mask_flat[flat[mask_flat]] = True
        edge_mask_flat[flat_sym[mask_flat]] = True
    edge_mask = edge_mask_flat.view(G, P, P)

    # ---- Phase C: label-propagation connected components ----
    labels = _torch.arange(P, device=dev, dtype=_torch.int64).view(1, P).expand(G, P).clone()
    # Non-representative slots -> sentinel P so they never appear as neighbor minima.
    labels = _torch.where(self_canonical, labels, _torch.full_like(labels, P))

    # Diameter of a connected graph on P nodes is at most P-1, so P iterations
    # suffice to propagate the minimum label to every connected component.
    max_iters = P
    for _it in range(max_iters):
        # labels of j broadcast to (G, i, j): for each i, examine neighbors' labels.
        lbl_broadcast = labels.unsqueeze(1).expand(G, P, P)
        big_lbl = _torch.full_like(lbl_broadcast, P)
        nbr_labels = _torch.where(edge_mask, lbl_broadcast, big_lbl)
        min_nbr = nbr_labels.min(dim=-1).values
        new_labels = _torch.minimum(labels, min_nbr)
        new_labels = _torch.where(self_canonical, new_labels,
                                  _torch.full_like(new_labels, P))
        if _torch.equal(new_labels, labels):
            break
        labels = new_labels
    else:
        # Loop completed without hitting the convergence break — labels are
        # possibly unconverged, which would produce wrong U-turn counts.
        raise RuntimeError(
            f"W_SD label-propagation did not converge in {max_iters} iterations "
            f"for P={P}. Increase bound or investigate graph structure."
        )

    # ---- Phase C.5: degree + endpoint identification ----
    degree = edge_mask.sum(dim=-1)                                # (G, P) int
    is_endpoint = (degree == 1) & self_canonical

    # ---- Phase D: project endpoints to 3 facet edges + bucket per (g, comp, edge_id) ----
    A = facet_verts[:, [0, 1, 2], :]                              # (G, 3, 3)
    B = facet_verts[:, [1, 2, 0], :]                              # (G, 3, 3)
    edge_vec = B - A                                              # (G, 3, 3)
    length_sq = (edge_vec * edge_vec).sum(dim=-1)                 # (G, 3)
    length_sq_safe = length_sq.clamp(min=1e-30)

    P_minus_A = pts.unsqueeze(2) - A.unsqueeze(1)                 # (G, P, 3, 3)
    dot_ = (P_minus_A * edge_vec.unsqueeze(1)).sum(dim=-1)        # (G, P, 3)
    t = dot_ / length_sq_safe.unsqueeze(1)                        # (G, P, 3)
    in_range = (t >= -1e-8) & (t <= 1.0 + 1e-8)
    proj = A.unsqueeze(1) + t.unsqueeze(-1) * edge_vec.unsqueeze(1)   # (G, P, 3, 3)
    dist = ((pts.unsqueeze(2) - proj) ** 2).sum(dim=-1).sqrt()    # (G, P, 3)
    on_edge = in_range & (dist < 1e-8)
    # Legacy skips edges with length < 1e-12 (length_sq < 1e-24).
    degenerate_edge = (length_sq < 1e-24).unsqueeze(1).expand(G, P, 3)
    on_edge = on_edge & ~degenerate_edge

    ep_mask = is_endpoint.unsqueeze(-1).expand(G, P, 3)
    hit_mask = ep_mask & on_edge                                  # (G, P, 3)

    # If nothing hit any edge, result is all zeros.
    if not hit_mask.any():
        out = _torch.zeros(G, dtype=_torch.int64, device=dev)
        return out.cpu() if dev.type == 'cuda' else out

    # Flat bucket key = g * (P * max_eid) + label * max_eid + eid.
    # edge_ids may contain any non-negative int; take max over hits only to be safe.
    eid_b = edge_ids_t.view(G, 1, 3).expand(G, P, 3)
    # Only real hits matter for range — mask others to 0 to compute max_eid safely.
    max_eid = int(eid_b[hit_mask].max().item()) + 1
    max_eid = max(max_eid, 1)

    g_idx_b = _torch.arange(G, device=dev, dtype=_torch.int64).view(G, 1, 1).expand(G, P, 3)
    lbl_b = labels.unsqueeze(-1).expand(G, P, 3)

    K_per_g = P * max_eid
    flat_key = g_idx_b * K_per_g + lbl_b * max_eid + eid_b         # (G, P, 3)
    hit_keys = flat_key[hit_mask]

    max_key = int(hit_keys.max().item()) + 1
    counts = _torch.bincount(hit_keys, minlength=max_key)
    uturn_contribs = counts // 2                                   # int64 floor div

    bucket_arange = _torch.arange(max_key, device=dev, dtype=_torch.int64)
    group_of_bucket = bucket_arange // K_per_g
    per_group_uturn = _torch.zeros(G, dtype=_torch.int64, device=dev)
    per_group_uturn.scatter_add_(0, group_of_bucket, uturn_contribs)

    return per_group_uturn.cpu() if dev.type == 'cuda' else per_group_uturn


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

    # Collect all per-component triangle indices and their cube assignments.
    # T6d (W5) path: batched GPU label-propagation UF across all cubes in
    # parallel, replacing the former per-cube Python loop that called
    # _get_local_components_np() 275k times (T0 #2 CPU hotspot: 1030 ms
    # self-time at res=256).

    comp_face_off_cpu = comp_face_off_np

    # Build a padded (N, max_f) view of comp_face_val directly on device.
    face_counts = (batch.comp_face_off[1:] - batch.comp_face_off[:-1]).to(torch.int64)  # (N,)
    max_f = int(face_counts.max().item()) if N > 0 else 0

    if max_f == 0:
        # All cubes empty (degenerate pipeline state).
        comp_cube_idx_np = np.array([], dtype=np.int64)
        comp_face_lists: list = []
    else:
        col_idx = torch.arange(max_f, device=device, dtype=torch.int64).unsqueeze(0)   # (1, max_f)
        counts_exp = face_counts.unsqueeze(1)                                          # (N, 1)
        valid = col_idx < counts_exp                                                   # (N, max_f)
        flat_idx = batch.comp_face_off[:-1].to(torch.int64).unsqueeze(1) + col_idx     # (N, max_f)
        safe_flat = flat_idx.clamp(max=comp_face_val.numel() - 1 if comp_face_val.numel() > 0 else 0)
        padded_fids = torch.where(
            valid,
            comp_face_val.to(torch.int64)[safe_flat],
            torch.full_like(safe_flat, -1),
        )                                                                               # (N, max_f) int64

        # Run batched GPU label propagation.
        batched_labels = _get_local_components_gpu_batched(
            padded_fids, face_adj, face_counts
        )                                                                               # (N, max_f) int64

        # Adapter: convert to list[list[list[int]]] per cube.
        per_cube_components = _labels_to_list_of_lists(
            batched_labels, padded_fids, face_counts
        )

        # Expand per-cube components into flat (comp_cube_idx, comp_face_lists)
        # arrays, honoring num_components[ci]: take at most nc, pad empty if
        # fewer found. Mirrors the former numpy loop semantics exactly.
        comp_cube_idx: list[int] = []
        comp_face_lists = []
        for ci in range(N):
            nc = int(num_components_np[ci])
            if nc == 0:
                continue
            components = per_cube_components[ci]
            if not components:
                for _ in range(nc):
                    comp_cube_idx.append(ci)
                    comp_face_lists.append(np.array([], dtype=np.int32))
                continue
            for comp_faces in components[:nc]:
                comp_cube_idx.append(ci)
                comp_face_lists.append(np.asarray(comp_faces, dtype=np.int32))
            for _ in range(nc - len(components[:nc])):
                comp_cube_idx.append(ci)
                comp_face_lists.append(np.array([], dtype=np.int32))

        comp_cube_idx_np = np.asarray(comp_cube_idx, dtype=np.int64)

    assert len(comp_face_lists) == total_points, \
        f"Component count mismatch: got {len(comp_face_lists)}, expected {total_points}"

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


def _get_local_components_gpu(
    face_ids: "torch.Tensor",  # (n,) int32 — global face ids
    face_adj: "torch.Tensor",  # (F, 3) int32 — GLOBAL face->3-neighbor table, -1 pad
) -> list[list[int]]:
    """Per-cube local connected components via GPU label propagation.

    Bit-identical to `_get_local_components_np` in externally-visible
    output (per T6a audit):
      - Component ordering = first-occurrence of root by slot index
      - Face ordering within component = slot-ascending (input order)

    Algorithm:
      1. Seed labels[j] = j (local slot idx).
      2. Iterate min-propagation over the 3 GLOBAL neighbors in face_adj:
         for each slot j, lookup its 3 neighbors; if a neighbor's global
         face id also appears in face_ids, adopt min(labels[j], labels[k])
         where k is the slot of that neighbor. Converges in <= n iters.
      3. Stable-sort (labels, slot) to emit components in the canonical
         order (ascending root = first-occurrence, ascending slot within).

    All ops deterministic CUDA (sort, min, elementwise, gather). No atomics.
    """
    n = int(face_ids.numel())
    if n == 0:
        return []
    device = face_ids.device

    fids_i64 = face_ids.to(torch.int64)
    # Look up 3 global neighbors per slot: (n, 3)
    neighbors = face_adj[fids_i64].to(torch.int64)  # (n, 3), -1 pad

    # Membership test: for each (j, e), find the slot k in face_ids where
    # face_ids[k] == neighbors[j, e], else -1. Done via broadcast equality.
    # Shape: (n, 3, n) — for small n (typical 2-20) this is cheap.
    fids_bcast = fids_i64.view(1, 1, n)           # (1, 1, n)
    nbr_bcast = neighbors.unsqueeze(-1)            # (n, 3, 1)
    # Also mask out padding (nbr == -1) and self-matches won't arise because
    # face_adj never references a face to itself.
    match = (nbr_bcast == fids_bcast) & (nbr_bcast >= 0)  # (n, 3, n) bool
    # For each (j, e), the first (and only) k where match is True. If no
    # match, nbr_local = n (sentinel). Take argmax over last dim; when all
    # False, argmax returns 0, so guard with any() mask.
    any_match = match.any(dim=-1)                  # (n, 3)
    # Convert bool to int8 for argmax (argmax on bool OK but explicit is fine)
    nbr_local = match.to(torch.int8).argmax(dim=-1)  # (n, 3) int64->int
    nbr_local = torch.where(any_match, nbr_local.to(torch.int64),
                            torch.full_like(nbr_local, n, dtype=torch.int64))  # (n, 3)

    # Label propagation: labels[j] = min slot id reachable from j.
    labels = torch.arange(n, device=device, dtype=torch.int64)  # (n,)
    # sentinel label for "no neighbor"
    SENTINEL = n

    # Gather neighbor labels with sentinel for invalid slots.
    # Pad labels with one extra entry at index n = SENTINEL (== n).
    padded_labels = torch.cat([labels, torch.tensor([SENTINEL], device=device, dtype=torch.int64)])
    for _ in range(n + 1):
        # nbr_labels[j, e] = padded_labels[nbr_local[j, e]]
        nbr_labels = padded_labels[nbr_local]  # (n, 3)
        min_nbr = nbr_labels.min(dim=-1).values  # (n,)
        new_labels = torch.minimum(labels, min_nbr)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        padded_labels = torch.cat([labels, torch.tensor([SENTINEL], device=device, dtype=torch.int64)])

    # Stable-sort by (label, slot). Since we seeded labels=slot and min-prop
    # keeps labels <= slot, labels already encode first-occurrence roots.
    # A stable sort by label preserves slot order within each component,
    # matching numpy's enumerate-order tiebreaker (T6a audit row 3).
    order = torch.argsort(labels, stable=True)  # (n,)
    sorted_labels = labels[order]
    sorted_fids = fids_i64[order]

    # Group by run of equal labels. Compute boundaries via diff.
    # boundaries[i] = True when sorted_labels[i] != sorted_labels[i-1].
    if n == 1:
        return [[int(sorted_fids[0].item())]]
    diff = sorted_labels[1:] != sorted_labels[:-1]
    # Group ids: cumsum of diff.
    first_bnd = torch.zeros(1, dtype=torch.int64, device=device)
    rest_bnd = diff.to(torch.int64).cumsum(0)
    group_ids = torch.cat([first_bnd, rest_bnd])  # (n,), 0..(ncomp-1)

    # Move to CPU once and bucket into list[list[int]].
    sorted_fids_cpu = sorted_fids.cpu().numpy()
    group_ids_cpu = group_ids.cpu().numpy()
    ncomp = int(group_ids_cpu[-1]) + 1
    components: list[list[int]] = [[] for _ in range(ncomp)]
    for i in range(n):
        components[int(group_ids_cpu[i])].append(int(sorted_fids_cpu[i]))
    return components


def _get_local_components_gpu_batched(
    batched_face_ids: "torch.Tensor",    # (N, max_f) int64, -1 pad
    face_adj: "torch.Tensor",            # (F, 3) int32/int64 — GLOBAL adj
    face_counts: "torch.Tensor",         # (N,) int64 — valid count per cube
) -> "torch.Tensor":
    """Batched GPU label propagation over N cubes, each with up to max_f faces.

    Returns:
        labels: (N, max_f) int64 — canonicalized first-occurrence label per
                slot. Padding entries get label = SENTINEL = max_f.

    Matches numpy reference's externally-visible output when combined with a
    stable_sort-by-label downstream step — see T6a audit.

    Memory: builds (N, max_f, max_f) broadcast equality for membership. At
    max_f = 64, N = 275k -> ~1.1 GB int8 (as bool stored byte-wise). We cap
    max_f for safety; outliers can fall back to per-cube if needed.
    """
    device = batched_face_ids.device
    N, M = batched_face_ids.shape
    if N == 0 or M == 0:
        return torch.full(batched_face_ids.shape, 0, device=device, dtype=torch.int64)

    fids = batched_face_ids.to(torch.int64)                    # (N, M)
    mask = fids >= 0                                           # (N, M) bool

    # Guard -1 lookups for face_adj (invalid slots still need a valid index).
    safe_fids = fids.clamp(min=0)
    neighbors = face_adj.to(torch.int64)[safe_fids]            # (N, M, 3)

    # Membership: for each (i, j, e), find slot k in cube i where
    # fids[i, k] == neighbors[i, j, e].
    # Shape (N, M, 3, M) via broadcast eq.
    # Memory: N * M^2 * 3 bool. At M=32, N=275k -> 844 MB — acceptable.
    #          at M=64, N=275k -> 3.4 GB — tight; we expect M typically << 32.
    nbr_bcast = neighbors.unsqueeze(-1)                        # (N, M, 3, 1)
    fids_bcast = fids.unsqueeze(1).unsqueeze(2)                # (N, 1, 1, M)
    match = (nbr_bcast == fids_bcast) & (nbr_bcast >= 0) & mask.unsqueeze(1).unsqueeze(2)
    # (N, M, 3, M) bool
    any_match = match.any(dim=-1)                              # (N, M, 3)
    nbr_local = match.to(torch.int8).argmax(dim=-1).to(torch.int64)  # (N, M, 3)
    # SENTINEL = M (index into padded label vector)
    nbr_local = torch.where(any_match, nbr_local,
                            torch.full_like(nbr_local, M))     # (N, M, 3)
    # Release the big (N, M, 3, M) intermediate.
    del match, nbr_bcast, fids_bcast

    # Label propagation. labels[i, j] = j for valid, M (sentinel) for pad.
    col_idx = torch.arange(M, device=device, dtype=torch.int64).unsqueeze(0).expand(N, M)
    labels = torch.where(mask, col_idx, torch.full_like(col_idx, M))  # (N, M)
    SENTINEL = M

    # Pad labels with 1 extra column (SENTINEL) for neighbor gather.
    pad_col = torch.full((N, 1), SENTINEL, device=device, dtype=torch.int64)
    max_iters = min(M + 1, 64)  # diameter of intra-cube graph, capped
    for _ in range(max_iters):
        padded_labels = torch.cat([labels, pad_col], dim=1)      # (N, M+1)
        # Gather neighbor labels: padded_labels[i, nbr_local[i, j, e]]
        nbr_labels = padded_labels.gather(1, nbr_local.reshape(N, M * 3)).reshape(N, M, 3)
        min_nbr = nbr_labels.min(dim=-1).values                  # (N, M)
        new_labels = torch.minimum(labels, min_nbr)
        # Keep sentinel for pad slots.
        new_labels = torch.where(mask, new_labels, torch.full_like(new_labels, SENTINEL))
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
    return labels


def _labels_to_list_of_lists(
    batched_labels: "torch.Tensor",     # (N, M) int64, SENTINEL=M for pad
    batched_face_ids: "torch.Tensor",   # (N, M) int64, -1 pad
    face_counts: "torch.Tensor",        # (N,) int64
) -> list[list[list[int]]]:
    """Adapter: convert batched label tensor into List[List[List[int]]] with
    per-cube canonical ordering. Shape: outer list is per-cube components,
    each component is a list of face ids in slot-ascending order.
    """
    from corep_fast import config as _cfg  # lazy to avoid circular import

    N, M = batched_labels.shape
    if N == 0:
        return []

    # Stable-sort by labels per row. Padded entries (label=M=SENTINEL) sort last.
    order = torch.argsort(batched_labels, dim=1, stable=True)    # (N, M)
    sorted_labels = batched_labels.gather(1, order)              # (N, M)
    sorted_fids = batched_face_ids.gather(1, order)              # (N, M)

    sorted_labels_cpu = sorted_labels.cpu().numpy()
    sorted_fids_cpu = sorted_fids.cpu().numpy()
    counts_cpu = face_counts.cpu().numpy()

    if not _cfg.LABELS_TO_LIST_VECTORIZED:
        # Legacy path (retained for rollback)
        result: list[list[list[int]]] = []
        for i in range(N):
            n_i = int(counts_cpu[i])
            if n_i == 0:
                result.append([])
                continue
            row_labels = sorted_labels_cpu[i, :n_i]
            row_fids = sorted_fids_cpu[i, :n_i]
            components: list[list[int]] = []
            cur_label = int(row_labels[0])
            cur_comp: list[int] = [int(row_fids[0])]
            for k in range(1, n_i):
                lbl = int(row_labels[k])
                if lbl != cur_label:
                    components.append(cur_comp)
                    cur_comp = []
                    cur_label = lbl
                cur_comp.append(int(row_fids[k]))
            components.append(cur_comp)
            result.append(components)
        return result

    # ---- Vectorized path ----
    # valid_mask[i, k] = k < counts_cpu[i]
    k_idx = np.arange(M, dtype=np.int64)
    valid_mask = k_idx[None, :] < counts_cpu[:, None]  # (N, M) bool

    # Component boundary: slot k starts a new component iff
    # (k == 0 OR sorted_labels[i, k] != sorted_labels[i, k-1]) AND valid_mask[i, k]
    prev_labels = np.concatenate(
        [np.full((N, 1), -1, dtype=np.int64), sorted_labels_cpu[:, :-1]],
        axis=1,
    )  # (N, M)
    is_new_component = (sorted_labels_cpu != prev_labels) & valid_mask  # (N, M)

    comps_per_cube = is_new_component.sum(axis=1).astype(np.int64)  # (N,)

    # Per-slot local component index within cube (only meaningful at valid slots)
    comp_idx_flat = (is_new_component.cumsum(axis=1) - 1).reshape(-1)  # (N*M,)
    fids_flat = sorted_fids_cpu.reshape(-1)
    valid_flat = valid_mask.reshape(-1)

    # Per-cube base offset into global component array
    cumsum_comps = comps_per_cube.cumsum()
    comp_off_per_cube = np.concatenate(
        [np.array([0], dtype=np.int64), cumsum_comps[:-1]]
    )  # (N,)
    cube_idx_flat = np.repeat(np.arange(N, dtype=np.int64), M)
    global_comp_idx = comp_off_per_cube[cube_idx_flat] + comp_idx_flat  # (N*M,)

    # Restrict to valid slots then group by global_comp_idx via split.
    valid_gci = global_comp_idx[valid_flat]
    valid_fids = fids_flat[valid_flat]

    if valid_gci.size == 0:
        return [[] for _ in range(N)]
    split_at = np.flatnonzero(np.diff(valid_gci) > 0) + 1
    fids_per_comp = np.split(valid_fids, split_at)  # list of C numpy arrays

    # Rebuild nested list shape (one tolist() per component, not per fid).
    result_out: list[list[list[int]]] = [None] * N  # type: ignore
    cursor = 0
    for i in range(N):
        c_i = int(comps_per_cube[i])
        if c_i == 0:
            result_out[i] = []
        else:
            result_out[i] = [fids_per_comp[cursor + j].tolist() for j in range(c_i)]
            cursor += c_i
    return result_out


# ======================================================================
# P2: GPU face_weights helpers
# ======================================================================

# Lazily-cached torch versions of FACET_VERTS / _V_OFFSETS, keyed by device.
_FACET_VERTS_T: torch.Tensor | None = None
_V_OFFSETS_T: torch.Tensor | None = None


def _get_facet_constants(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (facet_verts_t, v_offsets_t) tensors on `device`, cached."""
    global _FACET_VERTS_T, _V_OFFSETS_T
    if (
        _FACET_VERTS_T is None
        or _V_OFFSETS_T is None
        or _FACET_VERTS_T.device != device
    ):
        _FACET_VERTS_T = torch.tensor(FACET_VERTS, dtype=torch.int64, device=device)
        _V_OFFSETS_T = torch.tensor(_V_OFFSETS, dtype=torch.float32, device=device)
    return _FACET_VERTS_T, _V_OFFSETS_T


def _csr_expand_to_items(offsets: torch.Tensor, total: int) -> torch.Tensor:
    """CSR offsets (N+1,) → (total,) item-to-group mapping via bucketize.

    For item k (0..total-1), returns the smallest cube j such that
    ``offsets[j] <= k < offsets[j+1]``. We compare against the right
    boundaries (``offsets[1:]``) and need ``right=True`` so that an item
    sitting exactly on a boundary is attributed to the cube whose segment
    starts there (k=offsets[i] -> cube i, not cube i-1).
    """
    if total <= 0:
        return torch.zeros(0, dtype=torch.int64, device=offsets.device)
    arange = torch.arange(total, dtype=offsets.dtype, device=offsets.device)
    return torch.bucketize(arange, offsets[1:], right=True).to(torch.int64)


def _expand_pairs_gpu(batch: CubeBatch, mesh: MeshTensors) -> dict:
    """Expand (cube, facet, mesh_tri) into flat pair tensors on GPU.

    For each registered (cube, mesh_tri) pair (from batch.tri_*), produce
    12 entries — one per facet of the cube. Returns dict with:
        cube_id        : (P,)       int64 — cube index per pair
        facet_id       : (P,)       int64 — facet 0..11
        mesh_id        : (P,)       int64 — mesh triangle index
        facet_vertices : (P, 3, 3)  float32 — V0, V1, V2 in world coords
        mesh_triangles : (P, 3, 3)  float32 — M0, M1, M2
    """
    device = batch.device
    T = batch.tri_values.shape[0]
    if T == 0:
        return dict(
            cube_id=torch.zeros(0, dtype=torch.int64, device=device),
            facet_id=torch.zeros(0, dtype=torch.int64, device=device),
            mesh_id=torch.zeros(0, dtype=torch.int64, device=device),
            facet_vertices=torch.zeros((0, 3, 3), dtype=torch.float32, device=device),
            mesh_triangles=torch.zeros((0, 3, 3), dtype=torch.float32, device=device),
        )

    facet_verts_t, v_offsets_t = _get_facet_constants(device)

    # Cube per tri via CSR
    cube_per_tri = _csr_expand_to_items(batch.tri_offsets, T)  # (T,) int64

    # Cross product with 12 facets
    cube_per_pair = cube_per_tri.repeat_interleave(12)                              # (T*12,)
    facet_per_pair = torch.arange(12, dtype=torch.int64, device=device).repeat(T)   # (T*12,)
    mesh_per_pair = batch.tri_values.to(torch.int64).repeat_interleave(12)          # (T*12,)

    # Facet vertices: cube_base + V_OFFSETS[facet_verts[facet_id]] * step
    step = 1.0 / batch.resolution
    cube_base = batch.cube_indices[cube_per_pair].to(torch.float32) * step          # (P, 3)
    facet_v_idx = facet_verts_t[facet_per_pair]                                     # (P, 3)
    facet_vertices = cube_base.unsqueeze(1) + v_offsets_t[facet_v_idx] * step       # (P, 3, 3)

    # Mesh triangles
    mesh_triangles = mesh.triangles[mesh_per_pair]  # (P, 3, 3)

    return dict(
        cube_id=cube_per_pair,
        facet_id=facet_per_pair,
        mesh_id=mesh_per_pair,
        facet_vertices=facet_vertices,
        mesh_triangles=mesh_triangles,
    )


def _clip_segment_to_triangle_vectorized(
    P1: torch.Tensor, P2: torch.Tensor,
    V0: torch.Tensor, V1: torch.Tensor, V2: torch.Tensor,
    Nc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each pair, clip segment P1->P2 to interior of triangle (V0, V1, V2).

    Parametric Sutherland-Hodgman clip: t in [0, 1] represents P1 + t*(P2-P1).
    For each of the 3 triangle edges, compute the half-plane crossing param
    and narrow [t_min, t_max].

    Mirrors the per-pair `_intersect_facet_with_mesh` clip block (lines
    264-289) but vectorized over P pairs on GPU.

    Args:
        P1, P2:        (P, 3) segment endpoints.
        V0, V1, V2:    (P, 3) facet triangle vertices.
        Nc:            (P, 3) unit facet normal.

    Returns:
        A_out: (P, 3) clipped start endpoint.
        B_out: (P, 3) clipped end endpoint.
        valid: (P,)   bool — True if segment has nonzero length after clip.
    """
    P = P1.shape[0]
    device = P1.device
    dtype = P1.dtype
    dP = P2 - P1

    t_min = torch.zeros(P, device=device, dtype=dtype)
    t_max = torch.ones(P, device=device, dtype=dtype)
    valid = torch.ones(P, dtype=torch.bool, device=device)

    facet_verts = torch.stack([V0, V1, V2], dim=1)  # (P, 3, 3)
    for j in range(3):
        A_j = facet_verts[:, j, :]
        B_j = facet_verts[:, (j + 1) % 3, :]
        edge_vec = B_j - A_j
        # nk: half-plane normal pointing INSIDE the facet (Nc x edge)
        nk = torch.cross(Nc, edge_vec, dim=1)

        Ck_P1 = ((P1 - A_j) * nk).sum(dim=1)
        dot = (dP * nk).sum(dim=1)

        pos_dot = dot > 1e-8
        neg_dot = dot < -1e-8
        zero_dot = ~(pos_dot | neg_dot)

        # Avoid division by zero; we mask the result via pos_dot/neg_dot.
        safe_dot = torch.where(dot.abs() > 1e-12, dot, torch.ones_like(dot))
        t_candidate = -Ck_P1 / safe_dot
        t_min = torch.where(pos_dot, torch.maximum(t_min, t_candidate), t_min)
        t_max = torch.where(neg_dot, torch.minimum(t_max, t_candidate), t_max)
        # If parallel and the segment is on the outside half-space, drop it.
        valid = valid & ~(zero_dot & (Ck_P1 < -1e-8))

    valid = valid & (t_min <= t_max + 1e-8) & ((t_max - t_min) > 1e-8)
    A_out = P1 + t_min.unsqueeze(1) * dP
    B_out = P1 + t_max.unsqueeze(1) * dP
    return A_out, B_out, valid


def _batch_plane_tri_with_clip(
    facet_vertices: torch.Tensor,  # (P, 3, 3) float32
    mesh_triangles: torch.Tensor,  # (P, 3, 3) float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute plane-triangle intersection + clip to facet for each pair.

    Mirrors `_intersect_facet_with_mesh` (CPU numpy version) edge-case-by-edge-case.
    Per pair we:
      1. Build the facet plane (V0, unit normal Nc).
      2. Compute signed distances d0/d1/d2 of mesh vertices to the plane.
      3. Decide which mesh tris cross the plane:
            (has_pos & has_neg) | coplanar_edge
         where eps = 1e-8.
      4. For coplanar-edge tris, return the on-plane edge as the segment.
         For other crossings, walk the 3 edges and emit per-edge candidate
         points: lerp when d_a*d_b<-1e-14, vertex `a` when |d_a|<=1e-8.
         Take the first two distinct points.
      5. Clip the segment to facet interior via vectorized SH.
    """
    P = facet_vertices.shape[0]
    device = facet_vertices.device

    # Promote to float64 so segment endpoints match the CPU reference path
    # exactly (CPU uses float64 throughout). Sub-1e-8 float32 errors here
    # otherwise cause downstream `_count_uturns` graph-node merging to differ.
    fv64 = facet_vertices.to(torch.float64)
    mt64 = mesh_triangles.to(torch.float64)

    V0 = fv64[:, 0, :]
    V1 = fv64[:, 1, :]
    V2 = fv64[:, 2, :]

    E1 = V1 - V0
    E2 = V2 - V0
    Nc_raw = torch.cross(E1, E2, dim=1)
    len_Nc = torch.linalg.norm(Nc_raw, dim=1, keepdim=True).clamp_min(1e-12)
    Nc = Nc_raw / len_Nc

    M0 = mt64[:, 0, :]
    M1 = mt64[:, 1, :]
    M2 = mt64[:, 2, :]

    d0 = ((M0 - V0) * Nc).sum(dim=1)
    d1 = ((M1 - V0) * Nc).sum(dim=1)
    d2 = ((M2 - V0) * Nc).sum(dim=1)

    eps_plane = 1e-8
    eps_cross = 1e-14

    has_pos = (d0 > eps_plane) | (d1 > eps_plane) | (d2 > eps_plane)
    has_neg = (d0 < -eps_plane) | (d1 < -eps_plane) | (d2 < -eps_plane)

    d0_zero = d0.abs() <= eps_plane
    d1_zero = d1.abs() <= eps_plane
    d2_zero = d2.abs() <= eps_plane
    coplanar01 = d0_zero & d1_zero
    coplanar12 = d1_zero & d2_zero
    coplanar20 = d2_zero & d0_zero
    coplanar_edge = coplanar01 | coplanar12 | coplanar20

    intersects_plane = (has_pos & has_neg) | coplanar_edge

    # ---- Per-edge candidate points (3 edges per pair) ----
    # Edge (m_a, m_b, d_a, d_b) produces:
    #   - lerp point if d_a*d_b < -1e-14
    #   - vertex m_a if |d_a| <= 1e-8
    #   - else: no point
    # Order: (M0,M1,d0,d1), (M1,M2,d1,d2), (M2,M0,d2,d0).
    def _edge_cand(m_a, m_b, d_a, d_b):
        cross = (d_a * d_b) < -eps_cross
        on_plane = d_a.abs() <= eps_plane
        # Lerp t = d_a / (d_a - d_b), but only meaningful when cross.
        t = d_a / (d_a - d_b + 1e-30)
        pt_lerp = m_a + t.unsqueeze(1) * (m_b - m_a)
        # When on_plane (and not crossing), use m_a.
        pt = torch.where(cross.unsqueeze(1), pt_lerp, m_a)
        valid_e = cross | on_plane
        return pt, valid_e

    cand0_pt, cand0_v = _edge_cand(M0, M1, d0, d1)
    cand1_pt, cand1_v = _edge_cand(M1, M2, d1, d2)
    cand2_pt, cand2_v = _edge_cand(M2, M0, d2, d0)

    # ---- Coplanar-edge handling (overrides per-edge candidates) ----
    # When coplanar01: segment = (M0, M1); coplanar12: (M1, M2); coplanar20: (M2, M0).
    # CPU spec: extends list with both endpoints in order, then dedup; the
    # first 2 unique points are taken. So pick the first matching edge.
    cop_P1 = torch.where(
        coplanar01.unsqueeze(1), M0,
        torch.where(coplanar12.unsqueeze(1), M1, M2),
    )
    cop_P2 = torch.where(
        coplanar01.unsqueeze(1), M1,
        torch.where(coplanar12.unsqueeze(1), M2, M0),
    )

    # ---- Compose P1, P2 from per-edge candidates with dedup ----
    # Preference order matches CPU (m0/m1 edge first, then m1/m2, then m2/m0).
    # P1 = first valid candidate.
    P1 = torch.where(
        cand0_v.unsqueeze(1), cand0_pt,
        torch.where(cand1_v.unsqueeze(1), cand1_pt, cand2_pt),
    )

    # P2 = second valid distinct candidate.
    # Helper to test if cand_X is "next available distinct from P1".
    def _is_distinct(pt, p1, valid):
        return valid & ((pt - p1).norm(dim=1) >= eps_plane)

    use1 = _is_distinct(cand1_pt, P1, cand1_v)
    use2 = _is_distinct(cand2_pt, P1, cand2_v)
    # Was cand0 used as P1? Only if it was valid. Otherwise P1 = cand1 or cand2.
    p1_is_cand0 = cand0_v
    p1_is_cand1 = (~cand0_v) & cand1_v
    # If P1 came from cand0, P2 is first distinct of (cand1, cand2).
    # If P1 came from cand1, P2 must come from cand2.
    # If P1 came from cand2, no P2 available.
    P2_from_cand0_path = torch.where(use1.unsqueeze(1), cand1_pt, cand2_pt)
    P2_valid_from_cand0_path = use1 | use2
    P2_from_cand1_path = cand2_pt
    P2_valid_from_cand1_path = use2

    P2 = torch.where(
        p1_is_cand0.unsqueeze(1), P2_from_cand0_path,
        torch.where(
            p1_is_cand1.unsqueeze(1), P2_from_cand1_path,
            torch.zeros_like(P1),
        ),
    )
    P2_valid = torch.where(
        p1_is_cand0, P2_valid_from_cand0_path,
        torch.where(p1_is_cand1, P2_valid_from_cand1_path, torch.zeros_like(p1_is_cand0)),
    )
    have_two_pts = (cand0_v.int() + cand1_v.int() + cand2_v.int()) >= 1
    have_two_pts = cand0_v | cand1_v | cand2_v
    have_two_pts = have_two_pts & P2_valid

    # ---- Override with coplanar-edge segment when applicable ----
    P1_final = torch.where(coplanar_edge.unsqueeze(1), cop_P1, P1)
    P2_final = torch.where(coplanar_edge.unsqueeze(1), cop_P2, P2)
    seg_present = (
        intersects_plane & (coplanar_edge | have_two_pts)
    )

    # ---- Clip segment to facet triangle interior ----
    A_clip, B_clip, valid_clip = _clip_segment_to_triangle_vectorized(
        P1_final, P2_final, V0, V1, V2, Nc,
    )

    return A_clip, B_clip, seg_present & valid_clip


# ----------------------------------------------------------------------
# Module globals for fork-COW shared arrays in P2 BFS MP phase.
# Set in the parent process BEFORE spawning the temporary Pool so workers
# inherit them via copy-on-write (no pickle).
# ----------------------------------------------------------------------
_P2_CF = None         # (G,)        int64   — packed cube_id*12 + facet_id per group
_P2_GROUP_OFF = None  # (G+1,)      int64   — CSR offsets into segment arrays
_P2_SEGS_A = None     # (S, 3)      float64 — clipped segment start points
_P2_SEGS_B = None     # (S, 3)      float64 — clipped segment end points
_P2_CUBE_IDX = None   # (N, 3)      int32   — cube voxel indices
_P2_STEP = None       # float                — 1.0 / resolution


def _p2_uturn_worker(gi: int) -> tuple[int, int, int]:
    """Worker: reconstruct facet from packed (cube_id, facet_id), call _count_uturns.

    Reads from fork-inherited globals (no per-task pickle).
    """
    cf = int(_P2_CF[gi])
    cube_id = cf // 12
    facet_id = cf % 12
    lo = int(_P2_GROUP_OFF[gi])
    hi = int(_P2_GROUP_OFF[gi + 1])
    segs = [(_P2_SEGS_A[i], _P2_SEGS_B[i]) for i in range(lo, hi)]

    ix, iy, iz = _P2_CUBE_IDX[cube_id]
    base = np.array([ix, iy, iz], dtype=np.float64) * _P2_STEP
    cube_verts = base + _V_OFFSETS * _P2_STEP
    v_ids = FACET_VERTS[facet_id]
    e_ids = FACET_EDGES[facet_id]
    V0 = cube_verts[v_ids[0]]
    V1 = cube_verts[v_ids[1]]
    V2 = cube_verts[v_ids[2]]

    u = _count_uturns(segs, V0, V1, V2, cube_verts, v_ids, e_ids)
    return cube_id, facet_id, u


def _compute_face_weights_gpu(batch: CubeBatch, mesh: MeshTensors) -> torch.Tensor:
    """GPU-accelerated face_weights computation.

    Pipeline:
      Stage A (GPU): pair expansion + plane-tri + SH clip → (P,) valid segments.
      Stage B (GPU): compact + sort by (cube, facet) → groups.
      Stage C (GPU→CPU): bulk transfer of segments + group offsets.
      Stage D (CPU MP): per-group BFS + U-Turn count (fork-COW shared arrays).
      Stage E (CPU): scatter results into (N, 12) fw array.

    Returns:
        face_weights: (N, 12) int32 on `batch.device` — same semantics as
        `_compute_face_weights_mp`.
    """
    import os as _os

    device = batch.device
    N = batch.num_cubes
    fw = np.zeros((N, 12), dtype=np.int32)

    # Stage A: pair expansion + plane-tri + clip
    pairs = _expand_pairs_gpu(batch, mesh)
    if pairs['cube_id'].numel() == 0:
        return torch.from_numpy(fw).to(device)

    A_seg, B_seg, valid = _batch_plane_tri_with_clip(
        pairs['facet_vertices'], pairs['mesh_triangles'],
    )

    # Stage B: compact + sort + group
    if not bool(valid.any().item()):
        return torch.from_numpy(fw).to(device)

    cf_key = pairs['cube_id'][valid] * 12 + pairs['facet_id'][valid]
    A_kept = A_seg[valid]
    B_kept = B_seg[valid]

    sorted_idx = torch.argsort(cf_key, stable=True)
    sorted_cf = cf_key[sorted_idx]
    sorted_A = A_kept[sorted_idx]
    sorted_B = B_kept[sorted_idx]

    unique_cf, counts = torch.unique_consecutive(sorted_cf, return_counts=True)
    group_offsets = torch.cat([
        torch.zeros(1, dtype=counts.dtype, device=device),
        torch.cumsum(counts, dim=0),
    ])
    G = unique_cf.shape[0]

    # Stage C: Bulk GPU→CPU transfer (one .cpu() per array)
    cf_np = unique_cf.cpu().numpy()
    off_np = group_offsets.cpu().numpy()
    A_np = sorted_A.cpu().numpy().astype(np.float64)
    B_np = sorted_B.cpu().numpy().astype(np.float64)
    cube_idx_np = batch.cube_indices.cpu().numpy()

    # Stage D: Set shared globals, fork MP (legacy path) or batched GPU (W_SD)
    global _P2_CF, _P2_GROUP_OFF, _P2_SEGS_A, _P2_SEGS_B, _P2_CUBE_IDX, _P2_STEP
    _P2_CF = cf_np
    _P2_GROUP_OFF = off_np
    _P2_SEGS_A = A_np
    _P2_SEGS_B = B_np
    _P2_CUBE_IDX = cube_idx_np
    _P2_STEP = 1.0 / batch.resolution

    from corep_fast import config as _cfg
    if _cfg.STAGE_D_GPU and G > 0:
        # W_SD batched GPU path — single CSR call, no MP pool.
        uturn_tensor = _count_uturns_gpu_batched_csr(
            cf_np, off_np, A_np, B_np, cube_idx_np, _P2_STEP,
        )
        uturns_cpu = uturn_tensor.cpu().numpy()
        # cf_np[gi] packs (cube_id * 12 + facet_id); vectorize the scatter.
        cube_ids = (cf_np // 12).astype(np.int64)
        facet_ids = (cf_np % 12).astype(np.int64)
        fw[cube_ids, facet_ids] = uturns_cpu.astype(fw.dtype)
    else:
        # Legacy MP pool path.
        num_workers = max(1, (_os.cpu_count() or 4) - 4)
        if num_workers > 1 and G >= 5000:
            from corep_fast.utils.persistent_pool import get_pool
            chunksize = max(1, G // (num_workers * 4))
            p = get_pool(num_workers)
            results = p.map(_p2_uturn_worker, range(G), chunksize=chunksize)
        else:
            results = [_p2_uturn_worker(gi) for gi in range(G)]
        # Stage E: Scatter into fw
        for ci, fi, u in results:
            fw[ci, fi] = u

    return torch.from_numpy(fw).to(device)
