"""
Stage 7: Rank assignment + Hungarian matching (loop centroid <-> component point).

Re-traces the arc graph from s6 but with rank tracking at each edge crossing,
then matches loop centroids to component points via the Hungarian algorithm.

Output:
  - loop_edge_rank  (E,)  int32  -- rank per edge crossing
  - loop_point_match (L,) int32  -- which component point each loop is assigned to

Public API:
    s7_rank_assign(batch, pool=None) -> CubeBatch
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from corep_fast.constants import (
    CUBE_EDGES, CUBE_EDGE_STARTS, CUBE_EDGE_ENDS, CUBE_FACETS, CUBE_VERTICES,
)
from corep_fast.containers import CubeBatch, CubeStatus, _replace_fields

# ---------------------------------------------------------------------------
# Local topology tables (Python lists, for per-cube CPU work)
# ---------------------------------------------------------------------------

_EDGE_VERTS: List[Tuple[int, int]] = [tuple(e) for e in CUBE_EDGES.tolist()]
_TRIANGLES: List[Tuple[int, int, int]] = [tuple(f) for f in CUBE_FACETS.tolist()]
_CUBE_VERTS_NP: np.ndarray = CUBE_VERTICES.numpy()  # (8, 3) float32


# ---------------------------------------------------------------------------
# Topology helpers (shared with s6)
# ---------------------------------------------------------------------------

def _get_common_vertex(e1: int, e2: int) -> int:
    """Find the shared vertex between two edges."""
    v1, v2 = _EDGE_VERTS[e1]
    u1, u2 = _EDGE_VERTS[e2]
    if v1 in (u1, u2):
        return v1
    if v2 in (u1, u2):
        return v2
    raise ValueError(f"Edges {e1} and {e2} do not share a vertex.")


def _get_ordered_points(e_idx: int, corner_v: int, k: int, weight: int) -> List[int]:
    """Return the k rank indices closest to corner_v on edge e_idx.

    Rank 0 is closest to the first vertex, rank (weight-1) to the second.
    """
    u, v = _EDGE_VERTS[e_idx]
    if corner_v == u:
        return list(range(k))
    elif corner_v == v:
        return [weight - 1 - i for i in range(k)]
    else:
        raise ValueError(f"Vertex {corner_v} is not an endpoint of edge {e_idx}")


# ---------------------------------------------------------------------------
# Fast path: trace arc graph with (edge, rank) nodes — no U-Turns
# ---------------------------------------------------------------------------

def _trace_with_ranks_fast(
    ew: List[int],
) -> List[List[Tuple[int, int]]]:
    """Build the arc adjacency graph with (edge_idx, rank_idx) nodes and trace loops.

    Returns a list of loops, each loop being a list of (edge_idx, rank_idx) tuples.
    Mirrors _collapse_fast from s6 but retains rank information.
    Only valid for cubes with all face_weights == 0.
    """
    adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for e in range(18):
        for p in range(ew[e]):
            adj[(e, p)] = []

    for _t_idx, (e1, e2, e3) in enumerate(_TRIANGLES):
        w1, w2, w3 = ew[e1], ew[e2], ew[e3]

        k12 = (w1 + w2 - w3) // 2
        k23 = (w2 + w3 - w1) // 2
        k31 = (w3 + w1 - w2) // 2

        edge_pairs = [(e1, e2, k12), (e2, e3, k23), (e3, e1, k31)]
        for edge_A, edge_B, arcs_count in edge_pairs:
            if arcs_count == 0:
                continue
            common_v = _get_common_vertex(edge_A, edge_B)
            pts_A = _get_ordered_points(edge_A, common_v, arcs_count, ew[edge_A])
            pts_B = _get_ordered_points(edge_B, common_v, arcs_count, ew[edge_B])
            for i in range(arcs_count):
                adj[(edge_A, pts_A[i])].append((edge_B, pts_B[i]))
                adj[(edge_B, pts_B[i])].append((edge_A, pts_A[i]))

    return _trace_loops_from_adj(adj)


def _trace_loops_from_adj(
    adj: Dict[Tuple[int, int], List[Tuple[int, int]]],
) -> List[List[Tuple[int, int]]]:
    """Trace loops through a 2-regular adjacency graph of (edge, rank) nodes."""
    loops: List[List[Tuple[int, int]]] = []
    visited: Set[Tuple[int, int]] = set()

    for start_node in adj:
        if start_node in visited:
            continue
        if len(adj[start_node]) != 2:
            continue

        current_loop: List[Tuple[int, int]] = []
        curr_node = start_node
        prev_node = None

        while True:
            visited.add(curr_node)
            neighbors = adj[curr_node]
            if len(neighbors) != 2:
                break

            node1, node2 = neighbors[0], neighbors[1]

            if prev_node is None:
                next_node = node1
            else:
                next_node = node2 if node1 == prev_node else node1

            current_loop.append(curr_node)

            prev_node = curr_node
            curr_node = next_node

            if curr_node == start_node:
                break

        if len(current_loop) >= 3:
            loops.append(current_loop)

    return loops


# ---------------------------------------------------------------------------
# Slow path: U-Turn with specific assignment (rank tracking)
# ---------------------------------------------------------------------------

def _trace_with_ranks_uturn_assignment(
    ew: List[int],
    assignment: Tuple[Tuple[int, int, int], ...],
) -> List[List[Tuple[int, int]]]:
    """Build adjacency graph for a specific U-Turn assignment and trace loops with ranks.

    Same as s6's _trace_loops_for_uturn_assignment but returns (edge, rank) tuples
    instead of interleaved [edge, face, edge, face, ...].

    Raises ValueError if the graph is not degree-2 everywhere.
    """
    # adj: (edge, rank) -> list of (facet, edge, rank) — 3-tuple neighbors
    adj_raw: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {
        (e, p): [] for e in range(18) for p in range(ew[e])
    }

    for t_idx, (e1, e2, e3) in enumerate(_TRIANGLES):
        u1, u2, u3 = assignment[t_idx]

        w1_prime = ew[e1] - 2 * u1
        w2_prime = ew[e2] - 2 * u2
        w3_prime = ew[e3] - 2 * u3

        k12 = (w1_prime + w2_prime - w3_prime) // 2
        k23 = (w2_prime + w3_prime - w1_prime) // 2
        k31 = (w3_prime + w1_prime - w2_prime) // 2

        def _get_k_for_vertex(eA, target_v):
            if target_v == _get_common_vertex(e1, e2) and eA in (e1, e2):
                return k12
            if target_v == _get_common_vertex(e2, e3) and eA in (e2, e3):
                return k23
            if target_v == _get_common_vertex(e3, e1) and eA in (e3, e1):
                return k31
            return 0

        def _assign_face_connections(eA, eB, k, v_common):
            if k == 0:
                return
            pts_A = _get_ordered_points(eA, v_common, k, ew[eA])
            pts_B = _get_ordered_points(eB, v_common, k, ew[eB])
            for ii in range(k):
                adj_raw[(eA, pts_A[ii])].append((t_idx, eB, pts_B[ii]))
                adj_raw[(eB, pts_B[ii])].append((t_idx, eA, pts_A[ii]))

        # Standard corner-crossing arcs
        _assign_face_connections(e1, e2, k12, _get_common_vertex(e1, e2))
        _assign_face_connections(e2, e3, k23, _get_common_vertex(e2, e3))
        _assign_face_connections(e3, e1, k31, _get_common_vertex(e3, e1))

        def _assign_uturns(eA, u):
            if u == 0:
                return
            v0, v1 = _EDGE_VERTS[eA]
            k_v0 = _get_k_for_vertex(eA, v0)
            k_v1 = _get_k_for_vertex(eA, v1)
            start_idx = k_v0
            end_idx = ew[eA] - k_v1
            if end_idx - start_idx != 2 * u:
                raise ValueError(f"U-turn bounds mismatch on edge {eA}")
            for ii in range(u):
                p1 = start_idx + 2 * ii
                p2 = start_idx + 2 * ii + 1
                adj_raw[(eA, p1)].append((t_idx, eA, p2))
                adj_raw[(eA, p2)].append((t_idx, eA, p1))

        # U-Turn arcs
        _assign_uturns(e1, u1)
        _assign_uturns(e2, u2)
        _assign_uturns(e3, u3)

    # Convert 3-tuple adj to 2-tuple adj for loop tracing
    adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for node, nbrs in adj_raw.items():
        if len(nbrs) != 2:
            raise ValueError(f"Degree != 2 at node {node}")
        adj[node] = [(nb[1], nb[2]) for nb in nbrs]

    return _trace_loops_from_adj(adj)


def _get_canonical_loop(loop: List[int]) -> Tuple[int, ...]:
    """Canonicalize a cyclic loop for deduplication."""
    n = len(loop)
    best = tuple(loop)
    for i in range(n):
        shifted = tuple(loop[i:] + loop[:i])
        if shifted < best:
            best = shifted
    rev = loop[::-1]
    for i in range(n):
        shifted = tuple(rev[i:] + rev[:i])
        if shifted < best:
            best = shifted
    return best


def _get_canonical_solution(loops: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    """Sort all loops in a solution to form a unique key."""
    return tuple(sorted([_get_canonical_loop(l) for l in loops]))


# ---------------------------------------------------------------------------
# Match traced loops (with ranks) to s6 edge-only loops
# ---------------------------------------------------------------------------

def _match_loops_to_ranks(
    s6_loops: List[List[int]],
    traced_loops: List[List[Tuple[int, int]]],
) -> List[List[int]]:
    """Match each s6 loop (edge-only) to a traced loop (edge+rank) and extract ranks.

    Returns a list of rank lists, one per s6 loop, in the same order.
    Falls back to all-zeros if no match is found (should not happen for valid cubes).
    """
    used_traced = [False] * len(traced_loops)
    results: List[List[int]] = []

    for in_loop in s6_loops:
        k = len(in_loop)
        matched = False

        for i, t_loop in enumerate(traced_loops):
            if used_traced[i] or len(t_loop) != k:
                continue

            t_edges = [p[0] for p in t_loop]

            # Check forward sequence matching (all cyclic shifts)
            for shift in range(k):
                if all(t_edges[(shift + j) % k] == in_loop[j] for j in range(k)):
                    ranks = [t_loop[(shift + j) % k][1] for j in range(k)]
                    results.append(ranks)
                    used_traced[i] = True
                    matched = True
                    break
            if matched:
                break

            # Check backward sequence matching (reversed, all cyclic shifts)
            for shift in range(k):
                if all(t_edges[(shift - j) % k] == in_loop[j] for j in range(k)):
                    ranks = [t_loop[(shift - j) % k][1] for j in range(k)]
                    results.append(ranks)
                    used_traced[i] = True
                    matched = True
                    break
            if matched:
                break

        if not matched:
            # Fallback: assign rank 0 everywhere
            results.append([0] * k)

    return results


# ---------------------------------------------------------------------------
# Centroid computation (CPU fallback per cube)
# ---------------------------------------------------------------------------

def _compute_centroids(
    s6_loops: List[List[int]],
    rank_lists: List[List[int]],
    ew: List[int],
    cube_origin: np.ndarray,
    d: float,
) -> np.ndarray:
    """Compute the 3D centroid of each loop from interpolated edge crossing positions.

    Args:
        s6_loops: list of edge-index lists per loop.
        rank_lists: list of rank lists per loop (parallel to s6_loops).
        ew: edge weights (18,).
        cube_origin: (3,) float -- lower corner of cube in world coords.
        d: 1.0 / resolution.

    Returns:
        (n_loops, 3) float64 array of centroids.
    """
    n_loops = len(s6_loops)
    centroids = np.zeros((n_loops, 3), dtype=np.float64)

    for li in range(n_loops):
        edges = s6_loops[li]
        ranks = rank_lists[li]
        k = len(edges)
        if k == 0:
            continue

        cx, cy, cz = 0.0, 0.0, 0.0
        for e, r in zip(edges, ranks):
            u, v = _EDGE_VERTS[e]
            v1 = cube_origin + _CUBE_VERTS_NP[u] * d
            v2 = cube_origin + _CUBE_VERTS_NP[v] * d
            W = ew[e]
            t = (r + 1) / (W + 1) if W > 0 else 0.5
            pt = v1 + t * (v2 - v1)
            cx += pt[0]
            cy += pt[1]
            cz += pt[2]

        centroids[li] = [cx / k, cy / k, cz / k]

    return centroids


# ---------------------------------------------------------------------------
# Hungarian matching (CPU per cube)
# ---------------------------------------------------------------------------

def _hungarian_match(
    centroids: np.ndarray,
    component_pts: np.ndarray,
) -> List[int]:
    """Optimal assignment of loops to component points via Hungarian algorithm.

    Args:
        centroids: (n_loops, 3) float64.
        component_pts: (n_points, 3) float64.

    Returns:
        List of length n_loops, where result[i] is the local index of the
        matched component point for loop i.
    """
    n_loops = centroids.shape[0]
    n_points = component_pts.shape[0]

    if n_loops == 0:
        return []

    # Build cost matrix: squared Euclidean distance
    diff = centroids[:, None, :] - component_pts[None, :, :]
    cost = (diff ** 2).sum(axis=-1)  # (n_loops, n_points)

    row_ind, col_ind = linear_sum_assignment(cost)

    # Build result: for unmatched loops (if n_loops > n_points), assign 0
    result = [0] * n_loops
    for r, c in zip(row_ind, col_ind):
        result[r] = int(c)

    return result


# ---------------------------------------------------------------------------
# Extract helpers for CSR data
# ---------------------------------------------------------------------------

def _extract_s6_loops(batch: CubeBatch, cube_idx: int) -> List[List[int]]:
    """Extract the list of edge-index loops for a given cube from CSR storage."""
    l_lo = int(batch.loop_cube_off[cube_idx].item())
    l_hi = int(batch.loop_cube_off[cube_idx + 1].item())
    loops = []
    for li in range(l_lo, l_hi):
        e_lo = int(batch.loop_edge_off[li].item())
        e_hi = int(batch.loop_edge_off[li + 1].item())
        loops.append(batch.loop_edge_val[e_lo:e_hi].tolist())
    return loops


def _extract_component_points(batch: CubeBatch, cube_idx: int) -> np.ndarray:
    """Extract (n_points, 3) float array of component points for a given cube."""
    p_lo = int(batch.point_offsets[cube_idx].item())
    p_hi = int(batch.point_offsets[cube_idx + 1].item())
    if p_hi <= p_lo:
        return np.zeros((0, 3), dtype=np.float64)
    return batch.point_values[p_lo:p_hi].cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Phase 1 worker: rank re-tracing (module-level for pickling)
# ---------------------------------------------------------------------------

def _s7_rank_worker(work_item):
    """Process one cube for rank assignment.

    Args:
        work_item: (cube_idx, ew_list, s6_loops, uturn_assign_or_none)
            - cube_idx: int
            - ew_list: list[int] of length 18
            - s6_loops: list[list[int]] — edge-only loops from s6
            - uturn_assign_or_none: tuple[tuple[int,int,int], ...] or None
              None means fast-path (no U-turns); otherwise the (12, 3) assignment.

    Returns:
        (cube_idx, rank_lists)  where rank_lists is list[list[int]]
    """
    cube_idx, ew_i, s6_loops, uturn_assign = work_item

    if not s6_loops:
        return (cube_idx, [])

    if uturn_assign is not None:
        try:
            traced_loops = _trace_with_ranks_uturn_assignment(ew_i, uturn_assign)
        except Exception:
            traced_loops = []
    else:
        traced_loops = _trace_with_ranks_fast(ew_i)

    rank_lists = _match_loops_to_ranks(s6_loops, traced_loops)
    return (cube_idx, rank_lists)


# ---------------------------------------------------------------------------
# Phase 3 worker: Hungarian matching (module-level for pickling)
# ---------------------------------------------------------------------------

def _hungarian_worker(work_item):
    """Process one cube for Hungarian matching.

    Args:
        work_item: (cube_idx, cost_matrix_np)
            - cube_idx: int
            - cost_matrix_np: (n_loops, n_points) float64 cost matrix

    Returns:
        (cube_idx, match_indices)
    """
    cube_idx, cost_matrix = work_item

    if cost_matrix.size == 0:
        return (cube_idx, [])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    n_loops = cost_matrix.shape[0]
    result = [0] * n_loops
    for r, c in zip(row_ind, col_ind):
        result[r] = int(c)
    return (cube_idx, result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def s7_rank_assign(batch: CubeBatch, pool=None) -> CubeBatch:
    """Assign ranks to loop edge crossings and match loops to component points.

    Three-phase pipeline:
      Phase 1: MP rank re-tracing (CPU graph traversal)
      Phase 2: GPU batched centroid interpolation (scatter-mean)
      Phase 3: MP Hungarian matching (scipy per cube)

    Consumes batch.uturn_assignment directly for slow-path cubes (no re-enumeration).

    Args:
        batch: CubeBatch after s6_collapse (must have uturn_assignment populated).
        pool: Optional PersistentWorkerPool for multiprocessing. If None, runs serially.

    Returns:
        Updated CubeBatch with loop_edge_rank and loop_point_match.
    """
    N = batch.num_cubes
    device = batch.device

    if N == 0:
        return batch

    ew_np = batch.edge_weights.cpu().numpy()   # (N, 18) int32
    ci_np = batch.cube_indices.cpu().numpy()    # (N, 3)  int32
    resolution = batch.resolution
    d = 1.0 / resolution

    total_edges = int(batch.loop_edge_val.shape[0])
    total_loops = int(batch.loop_cube_off[-1].item())

    # Precompute uturn_assignment on CPU
    uturn_np = batch.uturn_assignment.cpu().numpy()  # (N, 12, 3) int32

    # Precompute CSR offsets on CPU
    loop_cube_off_cpu = batch.loop_cube_off.cpu()
    loop_edge_off_cpu = batch.loop_edge_off.cpu()
    loop_edge_val_cpu = batch.loop_edge_val.cpu()
    status_cpu = batch.status.cpu()

    # ===================================================================
    # Phase 1: Rank re-tracing (CPU, optionally multiprocessing)
    # ===================================================================

    # Build work items for OK cubes
    rank_work_items = []
    ok_cube_indices = []

    for i in range(N):
        status_i = int(status_cpu[i].item())
        l_lo = int(loop_cube_off_cpu[i].item())
        l_hi = int(loop_cube_off_cpu[i + 1].item())
        n_loops = l_hi - l_lo

        if n_loops == 0 or status_i != CubeStatus.OK:
            continue

        ew_i = ew_np[i].tolist()

        # Extract s6 loops (edge-only)
        s6_loops = []
        for li in range(l_lo, l_hi):
            e_lo = int(loop_edge_off_cpu[li].item())
            e_hi = int(loop_edge_off_cpu[li + 1].item())
            s6_loops.append(loop_edge_val_cpu[e_lo:e_hi].tolist())

        # Determine fast-path vs slow-path from uturn_assignment
        uturn_row = uturn_np[i]  # (12, 3) int32
        if uturn_row[0, 0] == -1:
            # Fast-path: all -1 means no U-turns
            uturn_assign = None
        else:
            # Slow-path: use the stored assignment directly
            uturn_assign = tuple(
                tuple(int(x) for x in uturn_row[t])
                for t in range(12)
            )

        rank_work_items.append((i, ew_i, s6_loops, uturn_assign))
        ok_cube_indices.append(i)

    # Dispatch rank work
    if pool is not None and rank_work_items:
        rank_results = pool.map_chunked(_s7_rank_worker, rank_work_items, chunk_size=500)
    else:
        rank_results = [_s7_rank_worker(item) for item in rank_work_items]

    # Collect rank results into flat arrays
    all_ranks = [0] * total_edges
    all_matches = [0] * total_loops

    # Fill non-OK cubes with defaults
    for i in range(N):
        status_i = int(status_cpu[i].item())
        l_lo = int(loop_cube_off_cpu[i].item())
        l_hi = int(loop_cube_off_cpu[i + 1].item())
        n_loops = l_hi - l_lo

        if n_loops == 0:
            continue

        if status_i != CubeStatus.OK:
            for li in range(l_lo, l_hi):
                e_lo = int(loop_edge_off_cpu[li].item())
                e_hi = int(loop_edge_off_cpu[li + 1].item())
                for k in range(e_lo, e_hi):
                    all_ranks[k] = 0
            for li_off in range(n_loops):
                all_matches[l_lo + li_off] = li_off

    # Write OK-cube rank results into flat array
    for cube_idx, rank_lists in rank_results:
        l_lo = int(loop_cube_off_cpu[cube_idx].item())
        for li_off, ranks in enumerate(rank_lists):
            li = l_lo + li_off
            e_lo = int(loop_edge_off_cpu[li].item())
            for k, r in enumerate(ranks):
                all_ranks[e_lo + k] = r

    # ===================================================================
    # Phase 2: GPU batched centroid interpolation (scatter-mean)
    # ===================================================================

    if total_edges > 0 and ok_cube_indices:
        # Build flat arrays for all edge crossings of OK cubes:
        #   For each crossing at (cube_i, loop_j, position_k):
        #     edge_idx, rank, edge_weight, cube_origin
        # Then compute positions on GPU.

        # Collect per-crossing data
        crossing_loop_ids = []    # which loop each crossing belongs to
        crossing_edge_ids = []    # edge index (0..17)
        crossing_ranks = []       # rank index
        crossing_weights = []     # edge weight
        crossing_cube_ids = []    # which cube (for cube_origin lookup)

        for cube_idx, rank_lists in rank_results:
            l_lo = int(loop_cube_off_cpu[cube_idx].item())
            l_hi = int(loop_cube_off_cpu[cube_idx + 1].item())
            for li_off in range(l_hi - l_lo):
                li = l_lo + li_off
                e_lo = int(loop_edge_off_cpu[li].item())
                e_hi = int(loop_edge_off_cpu[li + 1].item())
                edges = loop_edge_val_cpu[e_lo:e_hi].tolist()
                ranks = rank_lists[li_off] if li_off < len(rank_lists) else [0] * (e_hi - e_lo)
                for pos_k in range(len(edges)):
                    crossing_loop_ids.append(li)
                    crossing_edge_ids.append(edges[pos_k])
                    crossing_ranks.append(ranks[pos_k] if pos_k < len(ranks) else 0)
                    crossing_weights.append(int(ew_np[cube_idx, edges[pos_k]]))
                    crossing_cube_ids.append(cube_idx)

        n_crossings = len(crossing_loop_ids)

        if n_crossings > 0:
            # Move to GPU tensors
            t_loop_ids = torch.tensor(crossing_loop_ids, dtype=torch.int64, device=device)
            t_edge_ids = torch.tensor(crossing_edge_ids, dtype=torch.int64, device=device)
            t_ranks = torch.tensor(crossing_ranks, dtype=torch.float32, device=device)
            t_weights = torch.tensor(crossing_weights, dtype=torch.float32, device=device)
            t_cube_ids = torch.tensor(crossing_cube_ids, dtype=torch.int64, device=device)

            # Edge endpoint coords (18, 3) on GPU
            edge_starts = CUBE_EDGE_STARTS.to(device=device, dtype=torch.float32)  # (18, 3)
            edge_ends = CUBE_EDGE_ENDS.to(device=device, dtype=torch.float32)      # (18, 3)

            # Cube origins: cube_indices / resolution
            cube_indices_gpu = batch.cube_indices.to(dtype=torch.float32)  # (N, 3) on device
            cube_origins = cube_indices_gpu / resolution   # (N, 3)
            step = 1.0 / resolution

            # Gather per-crossing values
            start_pts = edge_starts[t_edge_ids]    # (n_crossings, 3)
            end_pts = edge_ends[t_edge_ids]        # (n_crossings, 3)
            origins = cube_origins[t_cube_ids]     # (n_crossings, 3)

            # Interpolation parameter: t = (rank + 1) / (weight + 1)
            t_param = (t_ranks + 1.0) / (t_weights + 1.0)  # (n_crossings,)
            t_param = t_param.unsqueeze(1)  # (n_crossings, 1)

            # Position = cube_origin + (start + t * (end - start)) * step
            local_pos = start_pts + t_param * (end_pts - start_pts)  # (n_crossings, 3)
            world_pos = origins + local_pos * step                    # (n_crossings, 3)

            # Scatter-mean by loop_id to get (total_loops, 3) centroids
            loop_centroids = torch.zeros(total_loops, 3, dtype=torch.float32, device=device)
            loop_counts = torch.zeros(total_loops, dtype=torch.float32, device=device)

            loop_centroids.scatter_add_(0, t_loop_ids.unsqueeze(1).expand(-1, 3), world_pos)
            loop_counts.scatter_add_(0, t_loop_ids, torch.ones(n_crossings, dtype=torch.float32, device=device))

            # Avoid division by zero
            safe_counts = loop_counts.clamp(min=1.0).unsqueeze(1)
            loop_centroids = loop_centroids / safe_counts  # (total_loops, 3)
        else:
            loop_centroids = torch.zeros(total_loops, 3, dtype=torch.float32, device=device)
    else:
        loop_centroids = torch.zeros(total_loops, 3, dtype=torch.float32, device=device)

    # ===================================================================
    # Phase 3: Hungarian matching (CPU, optionally multiprocessing)
    # ===================================================================

    # Build cost matrices on GPU, then dispatch scipy per cube
    point_offsets_cpu = batch.point_offsets.cpu()
    point_values_gpu = batch.point_values  # (P, 3) on device

    hungarian_work_items = []

    for cube_idx in ok_cube_indices:
        l_lo = int(loop_cube_off_cpu[cube_idx].item())
        l_hi = int(loop_cube_off_cpu[cube_idx + 1].item())
        n_loops = l_hi - l_lo
        if n_loops == 0:
            continue

        p_lo = int(point_offsets_cpu[cube_idx].item())
        p_hi = int(point_offsets_cpu[cube_idx + 1].item())
        n_points = p_hi - p_lo

        if n_points == 0:
            # No component points: identity match
            for li_off in range(n_loops):
                all_matches[l_lo + li_off] = li_off
            continue

        # Get centroids for this cube's loops from GPU tensor
        centroids_i = loop_centroids[l_lo:l_hi]          # (n_loops, 3) on device
        comp_pts_i = point_values_gpu[p_lo:p_hi]          # (n_points, 3) on device

        # Compute cost matrix on GPU: squared Euclidean distance
        diff = centroids_i.unsqueeze(1) - comp_pts_i.unsqueeze(0)  # (n_loops, n_points, 3)
        cost_gpu = (diff ** 2).sum(dim=-1)                          # (n_loops, n_points)
        cost_np = cost_gpu.cpu().numpy().astype(np.float64)

        hungarian_work_items.append((cube_idx, cost_np))

    # Dispatch Hungarian work
    if pool is not None and hungarian_work_items:
        hungarian_results = pool.map_chunked(_hungarian_worker, hungarian_work_items, chunk_size=500)
    else:
        hungarian_results = [_hungarian_worker(item) for item in hungarian_work_items]

    # Write Hungarian results
    for cube_idx, match_indices in hungarian_results:
        l_lo = int(loop_cube_off_cpu[cube_idx].item())
        l_hi = int(loop_cube_off_cpu[cube_idx + 1].item())
        n_loops = l_hi - l_lo
        for li_off in range(min(len(match_indices), n_loops)):
            all_matches[l_lo + li_off] = match_indices[li_off]

    # ===================================================================
    # Pack results
    # ===================================================================

    loop_edge_rank = torch.tensor(all_ranks, dtype=torch.int32, device=device) \
        if total_edges > 0 else torch.zeros(0, dtype=torch.int32, device=device)
    loop_point_match = torch.tensor(all_matches, dtype=torch.int32, device=device) \
        if total_loops > 0 else torch.zeros(0, dtype=torch.int32, device=device)

    return _replace_fields(
        batch,
        loop_edge_rank=loop_edge_rank,
        loop_point_match=loop_point_match,
    )
