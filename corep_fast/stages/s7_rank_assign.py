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
from corep_fast import config as _cfg

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
# GPU batched parallel BFS (W2b)
# ---------------------------------------------------------------------------
#
# Algorithm (per analysis E):
#
#   1. Build a per-cube (edge,rank) adjacency table on GPU.
#      Each node is encoded as `node_id = edge_idx * W_MAX + rank`.
#      Per-cube node space size = 18 * W_MAX = 288. Each node has at most 2
#      neighbors (graph is 2-regular Eulerian).
#
#   2. For each s6 loop, walk along the adjacency starting from each candidate
#      (rank0, neighbor_choice) on the loop's first edge, and verify that the
#      walked edge sequence equals the s6 edge sequence. Up to 2 * ew[e0] <= 24
#      candidates per loop, fully parallelizable.
#
#   3. Output rank per crossing into loop_edge_rank.
#
# This eliminates 275K Python-level graph builds + DFS traversals.

# Maximum per-cube edge weight (12 is the theoretical maximum per the
# triangle-inequality bound; 16 leaves room and aligns to a power of 2.)
_W_MAX: int = 16
_NODES_PER_CUBE: int = 18 * _W_MAX  # 288


def _build_facet_pair_table() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute per-facet, per-edge-pair lookup constants.

    For each of 12 facets and each of 3 edge pairs (e1,e2), (e2,e3), (e3,e1),
    return:
      - pair_eA, pair_eB:  (12, 3) int64  — edge indices A and B
      - pair_a_at_v0, pair_b_at_v0: (12, 3) bool — whether the common vertex
        is the "v0" endpoint of edge A / B respectively.

    These let us, for any facet/pair/cube, derive the rank-ordering of arc
    endpoints from the per-edge weights.
    """
    edge_verts = CUBE_EDGES.tolist()
    facets = CUBE_FACETS.tolist()

    eA = torch.zeros(12, 3, dtype=torch.int64)
    eB = torch.zeros(12, 3, dtype=torch.int64)
    a_at_v0 = torch.zeros(12, 3, dtype=torch.bool)
    b_at_v0 = torch.zeros(12, 3, dtype=torch.bool)

    pair_seq = [(0, 1), (1, 2), (2, 0)]  # k12, k23, k31

    for t_idx, (e1, e2, e3) in enumerate(facets):
        edges = [e1, e2, e3]
        for pi, (a, b) in enumerate(pair_seq):
            ea, eb = edges[a], edges[b]
            va0, va1 = edge_verts[ea]
            vb0, vb1 = edge_verts[eb]
            # Find common vertex
            if va0 in (vb0, vb1):
                cv = va0
            elif va1 in (vb0, vb1):
                cv = va1
            else:
                raise RuntimeError(f"Edges {ea} and {eb} of facet {t_idx} do not share a vertex")
            eA[t_idx, pi] = ea
            eB[t_idx, pi] = eb
            a_at_v0[t_idx, pi] = (cv == va0)
            b_at_v0[t_idx, pi] = (cv == vb0)

    return eA, eB, a_at_v0, b_at_v0


# Cache the table; materialized to GPU in s7_rank_assign_gpu.
_FACET_PAIR_TABLE_CPU: tuple[torch.Tensor, ...] | None = None


def _facet_pair_table(device: torch.device):
    """Lazily build + cache the facet pair lookup table on a given device."""
    global _FACET_PAIR_TABLE_CPU
    if _FACET_PAIR_TABLE_CPU is None:
        _FACET_PAIR_TABLE_CPU = _build_facet_pair_table()
    eA, eB, a_at_v0, b_at_v0 = _FACET_PAIR_TABLE_CPU
    return (
        eA.to(device, non_blocking=True),
        eB.to(device, non_blocking=True),
        a_at_v0.to(device, non_blocking=True),
        b_at_v0.to(device, non_blocking=True),
    )


def _build_uturn_pair_table() -> tuple[torch.Tensor, ...]:
    """For each of 12 facets, return per-edge: (edge_idx, pair_idx_at_v0, pair_idx_at_v1).

    For U-turn ordering on edge eA in facet T, we need to know how many "regular
    arcs" emanate from each of eA's two endpoints — equivalently, which of the
    three triangle pairs (k12, k23, k31) corresponds to each endpoint.

    Returns:
      uturn_edge:        (12, 3) int64 — edge index for each of the 3 edge slots
      uturn_pair_at_v0:  (12, 3) int64 — pair_idx (0/1/2) whose common vertex == eA's v0; -1 if neither
      uturn_pair_at_v1:  (12, 3) int64 — same for v1
    """
    edge_verts = CUBE_EDGES.tolist()
    facets = CUBE_FACETS.tolist()
    pair_seq = [(0, 1), (1, 2), (2, 0)]

    uturn_edge = torch.zeros(12, 3, dtype=torch.int64)
    pair_at_v0 = torch.full((12, 3), -1, dtype=torch.int64)
    pair_at_v1 = torch.full((12, 3), -1, dtype=torch.int64)

    for t_idx, (e1, e2, e3) in enumerate(facets):
        edges = [e1, e2, e3]
        for s_idx, eA in enumerate(edges):
            uturn_edge[t_idx, s_idx] = eA
            v0_A, v1_A = edge_verts[eA]
            for pi, (a_pos, b_pos) in enumerate(pair_seq):
                ea_pi, eb_pi = edges[a_pos], edges[b_pos]
                va0, va1 = edge_verts[ea_pi]
                vb0, vb1 = edge_verts[eb_pi]
                if va0 in (vb0, vb1):
                    cv = va0
                elif va1 in (vb0, vb1):
                    cv = va1
                else:
                    cv = -1
                # We only count the arc count if eA is one of (ea_pi, eb_pi)
                if eA == ea_pi or eA == eb_pi:
                    if cv == v0_A:
                        pair_at_v0[t_idx, s_idx] = pi
                    elif cv == v1_A:
                        pair_at_v1[t_idx, s_idx] = pi

    return uturn_edge, pair_at_v0, pair_at_v1


_UTURN_PAIR_TABLE_CPU: tuple[torch.Tensor, ...] | None = None


def _uturn_pair_table(device: torch.device):
    global _UTURN_PAIR_TABLE_CPU
    if _UTURN_PAIR_TABLE_CPU is None:
        _UTURN_PAIR_TABLE_CPU = _build_uturn_pair_table()
    a, b, c = _UTURN_PAIR_TABLE_CPU
    return (
        a.to(device, non_blocking=True),
        b.to(device, non_blocking=True),
        c.to(device, non_blocking=True),
    )


def _build_adjacency_gpu(
    edge_weights: torch.Tensor,        # (N, 18) int64
    uturn_assignment: torch.Tensor,    # (N, 12, 3) int64; -1 sentinel = fast path
) -> torch.Tensor:
    """Build per-cube (edge,rank) adjacency on GPU.

    Returns adj of shape (N, NODES_PER_CUBE, 2) int32 with -1 for empty slots.
    Each row holds at most 2 neighbors (the graph is 2-regular).

    Encodes node = edge_idx * W_MAX + rank.
    """
    device = edge_weights.device
    N = edge_weights.shape[0]
    NODES = _NODES_PER_CUBE
    W = _W_MAX

    # Node encoding is `edge_idx * W + rank` with rank in [0, edge_weight).
    # If any edge_weight exceeds W, rank can reach >= W and the encoded node
    # overflows into the next edge's slot space, causing OOB indexing into
    # fill_count / adj (which have second-dim size NODES_PER_CUBE = 18 * W).
    # Fail loud and clear instead of emitting an async CUDA device-side assert.
    if N > 0:
        max_w = int(edge_weights.max().item())
        assert max_w <= W, (
            f"edge_weights.max()={max_w} exceeds _W_MAX={W}; "
            f"increase _W_MAX in corep_fast/stages/s7_rank_assign.py "
            f"or reduce resolution / mesh complexity."
        )

    # Output: pre-fill with -1
    adj = torch.full((N, NODES, 2), -1, dtype=torch.int32, device=device)
    # Per-node fill counter (for atomic 2-slot assignment)
    fill_count = torch.zeros((N, NODES), dtype=torch.int32, device=device)

    # ---- Effective edge weights for U-turn cubes ----
    # For fast-path cubes (uturn[i,0,0] == -1), w' = w
    # For slow-path cubes, w' = w - 2 * sum_t(u_for_edge_in_t)
    # uturn_assignment has shape (N,12,3); each row [u1,u2,u3] for triangle t.
    # u for edge in (e1,e2,e3) of triangle t needs scatter to per-edge sum.
    # CUBE_FACETS gives us the (12,3) edge indices.
    facets = CUBE_FACETS.to(device=device, dtype=torch.int64)  # (12, 3) edge idx

    # Detect fast-path cubes: uturn_assignment[i, 0, 0] == -1
    is_fast = (uturn_assignment[:, 0, 0] == -1)  # (N,) bool
    # For fast-path rows, treat uturns as 0
    uturn_clean = torch.where(
        is_fast.view(N, 1, 1).expand(N, 12, 3),
        torch.zeros_like(uturn_assignment),
        uturn_assignment,
    )

    # Scatter sum of u into per-edge accumulator: u_per_edge (N, 18) int64
    u_per_edge = torch.zeros((N, 18), dtype=torch.int64, device=device)
    # Flatten facets edges to (36,) and uturn_clean to (N, 36)
    facets_flat = facets.reshape(36)                   # (36,) edge idx
    u_flat = uturn_clean.reshape(N, 36)                # (N, 36) u-counts
    u_per_edge.scatter_add_(1, facets_flat.unsqueeze(0).expand(N, 36), u_flat)

    # Effective weights
    ew_eff = edge_weights - 2 * u_per_edge   # (N, 18) int64

    # ---- Per-pair k values ----
    eA_tab, eB_tab, a_at_v0_tab, b_at_v0_tab = _facet_pair_table(device)  # (12,3) each

    # Gather per-cube w_a, w_b, w_c for each (T, pair):
    # For pair pi=0 in T: A=e1, B=e2, C=e3 -> k12 = (w1+w2-w3)/2
    # For pair pi=1: A=e2, B=e3, C=e1     -> k23 = (w2+w3-w1)/2
    # For pair pi=2: A=e3, B=e1, C=e2     -> k31 = (w3+w1-w2)/2
    # Need eC for each pair — derive from facets:
    # In facet T with edges (e1,e2,e3):  pair_seq = [(0,1),(1,2),(2,0)]
    # so C = the remaining slot. Slots: pi=0->2, pi=1->0, pi=2->1.
    # We have eA (12,3) = edges[a_pos], eB = edges[b_pos]. For eC:
    facet_edges = facets  # (12, 3) edge idx
    # eC by pi: pi=0 -> facet_edges[:,2], pi=1 -> facet_edges[:,0], pi=2 -> facet_edges[:,1]
    eC_tab = torch.stack(
        [facet_edges[:, 2], facet_edges[:, 0], facet_edges[:, 1]], dim=1
    )  # (12, 3) edge idx for the C edge of each pair

    # Gather weights per (cube, T, pair): shape (N, 12, 3)
    w_a = ew_eff.gather(1, eA_tab.reshape(36).unsqueeze(0).expand(N, 36)).reshape(N, 12, 3)
    w_b = ew_eff.gather(1, eB_tab.reshape(36).unsqueeze(0).expand(N, 36)).reshape(N, 12, 3)
    w_c = ew_eff.gather(1, eC_tab.reshape(36).unsqueeze(0).expand(N, 36)).reshape(N, 12, 3)

    k_pair = (w_a + w_b - w_c).div(2, rounding_mode='floor')  # (N, 12, 3) int64
    # Clamp to valid range [0, W_MAX]
    k_pair = k_pair.clamp(min=0, max=W)

    # Original w_a/w_b for endpoint flipping (we need w_a in original ew, but
    # ranks live in 0..ew[e]-1 (the original weight, not the effective one!))
    # CRITICAL: ranks address into the original (e, rank) node space using
    # original edge_weights, not the effective weight. The slow-path U-turn
    # logic in the CPU code uses `ew[eA]` as the bound for `_get_ordered_points`,
    # i.e. the original full weight.
    # So pts_A endpoints use full edge_weights, not ew_eff.
    w_a_full = edge_weights.gather(1, eA_tab.reshape(36).unsqueeze(0).expand(N, 36)).reshape(N, 12, 3)
    w_b_full = edge_weights.gather(1, eB_tab.reshape(36).unsqueeze(0).expand(N, 36)).reshape(N, 12, 3)

    # ---- Generate arc connections ----
    # For each (cube, T, pair, j), if j < k_pair[cube,T,pair]:
    #   pts_A_j = j if A_at_v0 else (w_a_full - 1 - j)
    #   pts_B_j = j if B_at_v0 else (w_b_full - 1 - j)
    #   node_A = eA * W + pts_A_j
    #   node_B = eB * W + pts_B_j
    #   write adj[node_A] += node_B, adj[node_B] += node_A
    j_idx = torch.arange(W, device=device, dtype=torch.int64)  # (W,)

    # Broadcast: (N, 12, 3, W)
    j_b = j_idx.view(1, 1, 1, W).expand(N, 12, 3, W)
    k_b = k_pair.unsqueeze(-1)             # (N, 12, 3, 1)
    valid_arc = (j_b < k_b)                # (N, 12, 3, W) bool

    # eA / eB / orientation broadcast
    eA_b = eA_tab.view(1, 12, 3, 1).expand(N, 12, 3, W)
    eB_b = eB_tab.view(1, 12, 3, 1).expand(N, 12, 3, W)
    a_at_v0_b = a_at_v0_tab.view(1, 12, 3, 1).expand(N, 12, 3, W)
    b_at_v0_b = b_at_v0_tab.view(1, 12, 3, 1).expand(N, 12, 3, W)
    w_a_full_b = w_a_full.unsqueeze(-1)    # (N, 12, 3, 1)
    w_b_full_b = w_b_full.unsqueeze(-1)

    pts_A = torch.where(a_at_v0_b, j_b, w_a_full_b - 1 - j_b)   # (N,12,3,W)
    pts_B = torch.where(b_at_v0_b, j_b, w_b_full_b - 1 - j_b)

    node_A = eA_b * W + pts_A   # (N, 12, 3, W) int64
    node_B = eB_b * W + pts_B

    # Now scatter arc connections (both directions) into adj.
    # Each arc must claim slot 0 or 1 in adj[cube, src_node, :].
    # Strategy: collect all (cube, src_node, dst_node) for VALID arcs (both
    # directions), then assign slots via grouping. A cleaner serialized approach
    # is to sort by (cube, src_node) and pair up; but simpler is to handle the 6
    # potential incoming edges per node deterministically using fill_count.
    #
    # We process arcs in a fixed order with for-loops over (T, pair, j) to
    # avoid race conditions. There are 12*3*W = 576 iterations -- modest, and
    # each iteration is one batched scatter across N cubes.
    #
    # Performance-wise, 576 small kernel launches is acceptable (sub-millisecond
    # each on modern GPUs). The masked scatter only writes into rows where
    # valid_arc is True.

    cube_arange = torch.arange(N, device=device, dtype=torch.int64)

    for t_idx in range(12):
        for pi in range(3):
            for jj in range(W):
                mask = valid_arc[:, t_idx, pi, jj]   # (N,) bool
                if not mask.any():
                    continue
                idx = cube_arange[mask]              # (M,) cube indices
                nA = node_A[mask, t_idx, pi, jj]     # (M,) int64 src node
                nB = node_B[mask, t_idx, pi, jj]     # (M,) int64 dst node

                # Write A -> B
                slotA = fill_count[idx, nA]          # (M,) int32
                # Use index_put_ with a mask to avoid OOB writes when slotA >= 2
                ok_A = (slotA < 2)
                if ok_A.all():
                    adj[idx, nA, slotA.long()] = nB.to(torch.int32)
                    fill_count[idx, nA] = slotA + 1
                else:
                    idx_ok = idx[ok_A]
                    nA_ok = nA[ok_A]
                    nB_ok = nB[ok_A]
                    slotA_ok = slotA[ok_A]
                    adj[idx_ok, nA_ok, slotA_ok.long()] = nB_ok.to(torch.int32)
                    fill_count[idx_ok, nA_ok] = slotA_ok + 1

                # Write B -> A
                slotB = fill_count[idx, nB]
                ok_B = (slotB < 2)
                if ok_B.all():
                    adj[idx, nB, slotB.long()] = nA.to(torch.int32)
                    fill_count[idx, nB] = slotB + 1
                else:
                    idx_ok = idx[ok_B]
                    nA_ok = nA[ok_B]
                    nB_ok = nB[ok_B]
                    slotB_ok = slotB[ok_B]
                    adj[idx_ok, nB_ok, slotB_ok.long()] = nA_ok.to(torch.int32)
                    fill_count[idx_ok, nB_ok] = slotB_ok + 1

    # ---- U-turn arcs (slow-path cubes only) ----
    # For each (cube i, facet T, edge slot s in [0,1,2]):
    #   u = uturn_assignment[i, T, s]
    #   if u > 0:
    #     eA = facets[T, s]
    #     k_v0 = k for pair adjacent to eA at v0 (or 0 if none)
    #     k_v1 = k for pair adjacent to eA at v1 (or 0 if none)
    #     start = k_v0; end = ew_full[eA] - k_v1
    #     for ii in 0..u-1:
    #       p1 = start + 2*ii; p2 = start + 2*ii + 1
    #       node_p1 = eA*W + p1, node_p2 = eA*W + p2
    #       Add edges (node_p1<->node_p2)
    #
    # Note: U-turn arcs use the ORIGINAL edge_weights (full w), not ew_eff.
    if (~is_fast).any():
        ut_edge_tab, ut_pair_v0_tab, ut_pair_v1_tab = _uturn_pair_table(device)
        # uturn_assignment is (N, 12, 3) — already the u counts per (T, s).
        # We need k_pair gathered at the right pair_idx.

        # k_v0 per (cube, T, s): k_pair[cube, T, ut_pair_v0_tab[T, s]] if pair >= 0 else 0
        # k_v1 same for v1
        # Build via gather
        # ut_pair_v0_tab: (12, 3) int64; replace -1 with 0 then mask out
        pair_v0_safe = ut_pair_v0_tab.clamp(min=0)  # (12, 3)
        pair_v1_safe = ut_pair_v1_tab.clamp(min=0)
        # Gather k for each (T, s): we need k_pair[cube, T, pair_idx]
        # Reshape: k_pair (N, 12, 3) — gather along dim=2
        # idx (1, 12, 3) -> (N, 12, 3)
        idx_v0 = pair_v0_safe.unsqueeze(0).expand(N, 12, 3)
        idx_v1 = pair_v1_safe.unsqueeze(0).expand(N, 12, 3)
        k_v0 = k_pair.gather(2, idx_v0)
        k_v1 = k_pair.gather(2, idx_v1)
        # Mask out invalid pair slots
        valid_v0 = (ut_pair_v0_tab >= 0).unsqueeze(0).expand(N, 12, 3)
        valid_v1 = (ut_pair_v1_tab >= 0).unsqueeze(0).expand(N, 12, 3)
        k_v0 = torch.where(valid_v0, k_v0, torch.zeros_like(k_v0))
        k_v1 = torch.where(valid_v1, k_v1, torch.zeros_like(k_v1))

        # Get ew_full per (cube, T, s) using ut_edge_tab
        ew_at_eA = edge_weights.gather(
            1, ut_edge_tab.reshape(36).unsqueeze(0).expand(N, 36)
        ).reshape(N, 12, 3)

        # Per (cube, T, s): u count
        u_count = uturn_clean  # (N, 12, 3) int64
        start_idx = k_v0       # (N, 12, 3)

        # For each ii in 0..(W//2 - 1), generate up to one U-turn pair per (cube,T,s)
        # ii < u_count
        max_u = W // 2  # safe bound
        for ii in range(max_u):
            mask_u = (u_count > ii) & (~is_fast.view(N, 1, 1).expand(N, 12, 3))
            if not mask_u.any():
                continue
            # Indices of (cube, T, s) where mask_u is True
            sel = mask_u.nonzero(as_tuple=False)  # (M, 3): [cube, T, s]
            if sel.shape[0] == 0:
                continue
            ci_u = sel[:, 0]
            t_u = sel[:, 1]
            s_u = sel[:, 2]

            eA_u = ut_edge_tab[t_u, s_u]                  # (M,) edge idx
            ew_u = ew_at_eA[ci_u, t_u, s_u]               # (M,) full weight
            start_u = start_idx[ci_u, t_u, s_u]           # (M,)
            p1 = start_u + 2 * ii                         # (M,)
            p2 = start_u + 2 * ii + 1
            # Skip if p2 out of range
            in_range = (p2 < ew_u) & (p1 >= 0)
            if not in_range.any():
                continue
            ci_v = ci_u[in_range]
            eA_v = eA_u[in_range]
            p1_v = p1[in_range]
            p2_v = p2[in_range]
            n_p1 = eA_v * W + p1_v
            n_p2 = eA_v * W + p2_v

            # Write n_p1 -> n_p2
            slot = fill_count[ci_v, n_p1]
            ok = (slot < 2)
            ci_ok = ci_v[ok]
            n_p1_ok = n_p1[ok]
            n_p2_ok = n_p2[ok]
            slot_ok = slot[ok]
            adj[ci_ok, n_p1_ok, slot_ok.long()] = n_p2_ok.to(torch.int32)
            fill_count[ci_ok, n_p1_ok] = slot_ok + 1

            # Write n_p2 -> n_p1
            slot = fill_count[ci_v, n_p2]
            ok = (slot < 2)
            ci_ok = ci_v[ok]
            n_p1_ok = n_p1[ok]
            n_p2_ok = n_p2[ok]
            slot_ok = slot[ok]
            adj[ci_ok, n_p2_ok, slot_ok.long()] = n_p1_ok.to(torch.int32)
            fill_count[ci_ok, n_p2_ok] = slot_ok + 1

    return adj  # (N, NODES_PER_CUBE, 2) int32


def _phase1_gpu_rank_assign(
    batch: CubeBatch,
    ok_loop_mask: torch.Tensor,            # (L,) bool — which loops to process
    loop_to_cube: torch.Tensor,            # (L,) int64 — cube index for each loop
) -> torch.Tensor:
    """Compute rank for every crossing in OK cubes via batched walk on per-cube adj.

    Returns:
        loop_edge_rank tensor of shape (E,) int32, where E = batch.loop_edge_val.shape[0].
        Non-OK cube crossings are set to 0.
    """
    device = batch.device
    N = batch.num_cubes
    E = int(batch.loop_edge_val.shape[0])
    L = int(batch.loop_edge_off.shape[0]) - 1
    W = _W_MAX

    # ---- Build adjacency on GPU ----
    edge_weights64 = batch.edge_weights.to(torch.int64)
    uturn64 = batch.uturn_assignment.to(torch.int64)
    adj = _build_adjacency_gpu(edge_weights64, uturn64)  # (N, NODES_PER_CUBE, 2) int32

    # ---- Build per-loop max edge length K and per-loop padded edges ----
    loop_off = batch.loop_edge_off.to(torch.int64)  # (L+1,)
    loop_lengths = (loop_off[1:] - loop_off[:-1])   # (L,)
    K_max = int(loop_lengths.max().item()) if L > 0 else 0
    if K_max == 0 or L == 0 or E == 0:
        return torch.zeros(E, dtype=torch.int32, device=device)

    # Pad each loop's edge sequence to (L, K_max)
    loop_edges_pad = torch.full((L, K_max), -1, dtype=torch.int64, device=device)
    pos_in_loop = torch.arange(K_max, device=device, dtype=torch.int64)  # (K_max,)
    pos_b = pos_in_loop.unsqueeze(0)                                      # (1, K_max)
    valid_pos = pos_b < loop_lengths.unsqueeze(1)                         # (L, K_max)
    flat_idx = loop_off[:-1].unsqueeze(1) + pos_b                         # (L, K_max)
    flat_idx_clamped = flat_idx.clamp(max=E - 1)
    loop_edges_pad = torch.where(
        valid_pos,
        batch.loop_edge_val.to(torch.int64)[flat_idx_clamped],
        torch.full_like(flat_idx, -1),
    )  # (L, K_max) int64

    # Per-loop cube index already provided
    cube_per_loop = loop_to_cube  # (L,) int64

    # ---- Enumerate candidates: for each loop, up to 2 * W = 32 candidates ----
    # candidate = (rank0, neighbor_choice) for the first edge
    # We generate (L, 2*W, K_max) walk traces, then mask invalid and pick first valid.

    # First edge per loop
    e0 = loop_edges_pad[:, 0]   # (L,) int64; -1 if loop empty (filtered by ok_loop_mask)

    # Number of candidates per loop = ew[cube, e0] * 2 (capped at 2*W)
    # But we'll generate all 2*W and mask invalid.

    # For loops with empty ok_loop_mask (non-OK cubes), we skip walk entirely.
    # The output for those crossings will stay 0.

    # rank0 candidates
    r0_idx = torch.arange(W, device=device, dtype=torch.int64)            # (W,)
    nbr_idx = torch.arange(2, device=device, dtype=torch.int64)           # (2,)
    cand_r0 = r0_idx.view(1, W, 1).expand(L, W, 2).reshape(L, W * 2)      # (L, 2W)
    cand_nbr = nbr_idx.view(1, 1, 2).expand(L, W, 2).reshape(L, W * 2)    # (L, 2W)

    # Check: r0 < ew[cube, e0]
    ew_e0 = edge_weights64[cube_per_loop, e0.clamp(min=0)]  # (L,) int64
    cand_r0_valid = cand_r0 < ew_e0.unsqueeze(1)            # (L, 2W) bool

    # Each loop must also be in ok_loop_mask
    cand_r0_valid = cand_r0_valid & ok_loop_mask.unsqueeze(1)

    # ---- Walk along the graph ----
    # Initialize:
    #   cur_node[L, 2W] = e0 * W + cand_r0
    #   prev_node[L, 2W] = -1
    #   walk_rank[L, 2W, K_max] = -1
    #   alive[L, 2W] = cand_r0_valid
    cur_node = e0.unsqueeze(1) * W + cand_r0  # (L, 2W) int64
    cur_node = cur_node.clamp(min=0, max=_NODES_PER_CUBE - 1)  # safety
    prev_node = torch.full_like(cur_node, -1)
    alive = cand_r0_valid.clone()
    walk_rank = torch.zeros((L, W * 2, K_max), dtype=torch.int32, device=device)
    walk_rank[:, :, 0] = cand_r0.to(torch.int32)

    cube_b = cube_per_loop.view(L, 1).expand(L, W * 2)  # (L, 2W) int64

    for step in range(1, K_max):
        # Within bounds
        within = (pos_in_loop[step] < loop_lengths)  # (L,) bool
        within_b = within.unsqueeze(1).expand(L, W * 2)
        # Get the two neighbors of cur_node in each cube
        nbr0 = adj[cube_b, cur_node, 0]  # (L, 2W) int32
        nbr1 = adj[cube_b, cur_node, 1]

        # Step selection:
        #   step 1 (first walk): use cand_nbr (0 or 1)
        #   step 2+: use the neighbor != prev_node
        if step == 1:
            chosen = torch.where(cand_nbr == 0, nbr0, nbr1)  # (L, 2W) int32
        else:
            # Choose neighbor that is NOT prev_node
            chosen = torch.where(
                nbr0.to(torch.int64) == prev_node, nbr1, nbr0
            )

        # Decode chosen node -> (edge, rank)
        chosen_i64 = chosen.to(torch.int64)
        # Mark dead if chosen == -1 (no neighbor)
        no_nbr = (chosen == -1)
        chosen_clamped = chosen_i64.clamp(min=0, max=_NODES_PER_CUBE - 1)
        chosen_edge = chosen_clamped // W   # (L, 2W) int64
        chosen_rank = chosen_clamped % W

        # Compare against expected next edge in loop
        expected_edge = loop_edges_pad[:, step].unsqueeze(1).expand(L, W * 2)  # (L, 2W) int64
        edge_match = (chosen_edge == expected_edge)

        # Step is valid if alive AND within AND not no_nbr AND edge_match
        step_valid = alive & within_b & (~no_nbr) & edge_match
        # Alive after step: alive AND (step_valid OR not within)
        alive = alive & (step_valid | ~within_b)

        # Write rank for valid steps; for invalid OR out-of-bounds keep 0
        # Use scatter-style write through where()
        prev_walk_rank_step = walk_rank[:, :, step]
        new_step_rank = torch.where(step_valid, chosen_rank.to(torch.int32), prev_walk_rank_step)
        walk_rank[:, :, step] = new_step_rank

        # Update prev/cur (only for alive candidates that took a real step)
        prev_node = torch.where(step_valid, cur_node, prev_node)
        cur_node = torch.where(step_valid, chosen_i64, cur_node)

    # ---- Verify cycle closure: for K-length loops, the (K-1)-th edge's
    #      neighbor (other than prev_node) must equal the starting node
    # This is implied by alive[..., K-1] && edge_match for the closing edge,
    # but the loop already implicitly visited the cycle. We accept any walk
    # that completes within bounds.

    # ---- Pick valid candidate per loop with per-cube bijective consumption ----
    # Loops in the same cube with identical edge sequences must claim
    # DISTINCT candidates, mirroring CPU's `used_traced[i] = True` in
    # `_match_loops_to_ranks`. Without this, two s6 loops sharing an edge
    # sequence in one cube would both pick the smallest-rank candidate and
    # produce duplicate ranks, collapsing downstream centroids.
    #
    # Two loops in DIFFERENT cubes — or same cube but different edge
    # sequences — may independently claim the lowest valid candidate (they
    # traverse different cycles in their respective adjacency graphs).

    # Group key: (cube_id, full edge sequence with padding). torch.unique
    # maps each distinct (cube, edges) pair to an int64 group id.
    row_key = torch.cat([
        cube_per_loop.to(torch.int32).unsqueeze(1),  # (L, 1)
        loop_edges_pad.to(torch.int32),              # (L, K_max) — -1 for padding
    ], dim=1)  # (L, K_max + 1) int32
    _, group_ids = torch.unique(row_key, return_inverse=True, dim=0)  # (L,) int64

    # Within-group index: stable-sort by group_ids, then run-length via
    # segment reset. Each group's loops get consecutive indices 0, 1, 2, ...
    sort_idx = torch.argsort(group_ids, stable=True)
    sorted_keys = group_ids[sort_idx]
    same_as_prev = torch.cat([
        torch.zeros(1, dtype=torch.bool, device=device),
        sorted_keys[1:] == sorted_keys[:-1],
    ])
    arange_L_t = torch.arange(L, device=device, dtype=torch.int64)
    reset_pos_sorted = arange_L_t * (~same_as_prev).to(torch.int64)
    last_reset_sorted = torch.cummax(reset_pos_sorted, dim=0).values
    within_group_idx_sorted = arange_L_t - last_reset_sorted
    within_group_idx = torch.empty(L, dtype=torch.int64, device=device)
    within_group_idx.scatter_(0, sort_idx, within_group_idx_sorted)

    # Pick the (within_group_idx[l])-th True in alive[l, :].
    # cumsum over alive gives 1-based rank of each True; the k-th True
    # (0-indexed) is the slot where cumsum equals k+1 AND that slot is alive.
    cumsum_alive = torch.cumsum(alive.to(torch.int32), dim=1)  # (L, 2W) int32
    target = (within_group_idx + 1).to(torch.int32).unsqueeze(1)  # (L, 1)
    matches = (cumsum_alive == target) & alive  # (L, 2W) bool

    # If the group size exceeds the alive-candidate count (e.g. CPU would
    # have exhausted `used_traced`), there is no k-th True — fall back to
    # all-zero ranks, matching CPU's fallback in `_match_loops_to_ranks`.
    any_match = matches.any(dim=1)  # (L,) bool
    first_idx = matches.to(torch.int32).argmax(dim=1)  # (L,) int64; 0 if no match

    # Gather the chosen walk ranks
    chosen_ranks = walk_rank[
        torch.arange(L, device=device), first_idx
    ]  # (L, K_max) int32

    # Zero-out for loops where no candidate was consumed
    chosen_ranks = torch.where(
        any_match.unsqueeze(1),
        chosen_ranks,
        torch.zeros_like(chosen_ranks),
    )

    # ---- CPU fallback for loops the GPU walker could not trace ----
    # The vectorized walker currently struggles with consecutive intra-loop
    # duplicate edges (U-turn patterns: e0 → e0 in the s6 loop sequence). For
    # those rare cubes, delegate to the per-cube CPU worker so ranks match
    # the reference `_match_loops_to_ranks` output. Triggered only when a
    # loop has no alive candidate, so fast-path correctness is untouched.
    # Restrict fallback to OK-status cubes; loops in non-OK cubes never had
    # alive candidates (masked by `ok_loop_mask`), so their `any_match=False`
    # is expected and must be left at zero, matching the identity default
    # path in `s7_rank_assign` at the top of this function.
    from corep_fast.containers import CubeStatus as _CubeStatus
    status_cpu_np = batch.status.cpu().numpy()
    any_match_cpu = any_match.cpu().numpy()
    if not any_match_cpu.all():
        import numpy as _np
        loop_cube_off_np = batch.loop_cube_off.cpu().numpy()
        loop_off_np = batch.loop_edge_off.cpu().numpy()
        loop_edge_val_np = batch.loop_edge_val.cpu().numpy()
        ew_np_all = batch.edge_weights.cpu().numpy()
        ut_np_all = batch.uturn_assignment.cpu().numpy()

        failed_loop_ids = _np.where(~any_match_cpu)[0]
        failed_cubes = set()
        for lid in failed_loop_ids:
            ci = int(_np.searchsorted(loop_cube_off_np[1:], lid, side='right'))
            if int(status_cpu_np[ci]) == _CubeStatus.OK:
                failed_cubes.add(ci)

        chosen_ranks_cpu = chosen_ranks.cpu().numpy()  # (L, K_max) int32
        for cube_i in failed_cubes:
            l_lo = int(loop_cube_off_np[cube_i])
            l_hi = int(loop_cube_off_np[cube_i + 1])
            ew_i = ew_np_all[cube_i].tolist()
            s6_loops = []
            for li in range(l_lo, l_hi):
                e_lo_l = int(loop_off_np[li])
                e_hi_l = int(loop_off_np[li + 1])
                s6_loops.append(loop_edge_val_np[e_lo_l:e_hi_l].tolist())
            uturn_row = ut_np_all[cube_i]
            if uturn_row[0, 0] == -1:
                uturn_assign = None
            else:
                uturn_assign = tuple(
                    tuple(int(x) for x in uturn_row[t]) for t in range(12)
                )
            _, rank_lists = _s7_rank_worker(
                (cube_i, ew_i, s6_loops, uturn_assign)
            )
            for li_off, ranks in enumerate(rank_lists):
                li = l_lo + li_off
                for step_i, r in enumerate(ranks):
                    chosen_ranks_cpu[li, step_i] = r

        chosen_ranks = torch.from_numpy(chosen_ranks_cpu).to(device)

    # ---- Scatter chosen_ranks into flat loop_edge_rank tensor ----
    loop_edge_rank = torch.zeros(E, dtype=torch.int32, device=device)
    # For each loop l, copy chosen_ranks[l, 0:loop_lengths[l]] into
    # loop_edge_rank[loop_off[l] : loop_off[l+1]]
    # Use mask-based scatter:
    flat_pos = loop_off[:-1].unsqueeze(1) + pos_b  # (L, K_max)
    flat_pos_clamped = flat_pos.clamp(max=E - 1)
    flat_pos_flat = flat_pos_clamped.reshape(-1)   # (L*K_max,)
    rank_flat = chosen_ranks.reshape(-1)
    valid_flat = valid_pos.reshape(-1)
    # scatter only where valid_pos True. Use index_put_ with mask.
    loop_edge_rank.scatter_(0, flat_pos_flat[valid_flat], rank_flat[valid_flat])

    return loop_edge_rank


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def s7_rank_assign(batch: CubeBatch, pool=None, num_workers: int | None = None) -> CubeBatch:
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
    import os as _os
    N = batch.num_cubes
    device = batch.device

    if N == 0:
        return batch

    # Derive worker count: explicit > legacy pool > cpu_count - 4
    if num_workers is None:
        if pool is not None:
            num_workers = pool._num_workers
        else:
            num_workers = max(1, (_os.cpu_count() or 4) - 4)

    ew_np = batch.edge_weights.cpu().numpy()   # (N, 18) int32
    ci_np = batch.cube_indices.cpu().numpy()    # (N, 3)  int32
    resolution = batch.resolution
    d = 1.0 / resolution

    total_edges = int(batch.loop_edge_val.shape[0])
    total_loops = int(batch.loop_cube_off[-1].item())

    # Precompute uturn_assignment on CPU
    uturn_np = batch.uturn_assignment.cpu().numpy()  # (N, 12, 3) int32

    # Precompute face_weights on CPU for work item building
    fw_np = batch.face_weights.cpu().numpy()  # (N, 12) int32

    # ===================================================================
    # Phase 1: Rank re-tracing
    #   - W2b GPU path: batched parallel walk on per-cube adjacency tensors
    #   - Legacy CPU MP path: per-cube serial graph traversal
    # ===================================================================

    # Convert CSR to numpy once (avoid per-cube .item() GPU syncs)
    loop_cube_off_np = batch.loop_cube_off.cpu().numpy()
    loop_edge_off_np = batch.loop_edge_off.cpu().numpy()
    loop_edge_val_np = batch.loop_edge_val.cpu().numpy()
    status_np = batch.status.cpu().numpy()

    # Identify OK cubes upfront (used by both Phase 2 and Phase 3)
    ok_cube_indices = []
    for i in range(N):
        status_i = int(status_np[i])
        l_lo = int(loop_cube_off_np[i])
        l_hi = int(loop_cube_off_np[i + 1])
        n_loops = l_hi - l_lo
        if n_loops == 0 or status_i != CubeStatus.OK:
            continue
        ok_cube_indices.append(i)

    all_ranks = [0] * total_edges
    all_matches = [0] * total_loops

    # Fill non-OK cubes' loop_point_match with identity defaults
    for i in range(N):
        status_i = int(status_np[i])
        l_lo = int(loop_cube_off_np[i])
        l_hi = int(loop_cube_off_np[i + 1])
        n_loops = l_hi - l_lo
        if n_loops == 0:
            continue
        if status_i != CubeStatus.OK:
            for li_off in range(n_loops):
                all_matches[l_lo + li_off] = li_off

    # rank_results: per-OK-cube list of rank lists (parallel to ok_cube_indices).
    # Both paths produce this for downstream Phase 2 to consume.
    rank_results: list[tuple[int, list[list[int]]]]

    if _cfg.S7_PHASE1_GPU and total_loops > 0:
        # ---- W2b GPU path ----
        # Build per-loop ok mask (only loops in OK cubes contribute)
        L = total_loops
        loop_to_cube = torch.zeros(L, dtype=torch.int64, device=device)
        ok_loop_mask = torch.zeros(L, dtype=torch.bool, device=device)
        # Compute loop_to_cube via repeat_interleave on loop_cube_off
        loop_cube_off_t = batch.loop_cube_off.to(torch.int64)
        loops_per_cube = (loop_cube_off_t[1:] - loop_cube_off_t[:-1])  # (N,)
        cube_arange = torch.arange(N, device=device, dtype=torch.int64)
        loop_to_cube = torch.repeat_interleave(cube_arange, loops_per_cube)
        # ok mask: cube status == OK
        status_t = batch.status.to(torch.int64)
        cube_ok = (status_t == CubeStatus.OK)  # (N,)
        ok_loop_mask = cube_ok[loop_to_cube]   # (L,) bool

        loop_edge_rank_gpu = _phase1_gpu_rank_assign(
            batch=batch,
            ok_loop_mask=ok_loop_mask,
            loop_to_cube=loop_to_cube,
        )

        # GPU path skips both `all_ranks` Python list and the per-cube
        # `rank_results` construction. The Phase 2 fast path reads
        # loop_edge_rank_gpu directly; the final tensor pack also reuses it.
        rank_results = []  # unused on this path; kept as empty for symmetry
        all_ranks = None   # signal to final pack to use loop_edge_rank_gpu

    else:
        # ---- Legacy CPU MP path ----
        rank_work_items = []
        for i in ok_cube_indices:
            l_lo = int(loop_cube_off_np[i])
            l_hi = int(loop_cube_off_np[i + 1])
            ew_i = ew_np[i].tolist()
            s6_loops = []
            for li in range(l_lo, l_hi):
                e_lo = int(loop_edge_off_np[li])
                e_hi = int(loop_edge_off_np[li + 1])
                s6_loops.append(loop_edge_val_np[e_lo:e_hi].tolist())
            uturn_row = uturn_np[i]
            if uturn_row[0, 0] == -1:
                uturn_assign = None
            else:
                uturn_assign = tuple(
                    tuple(int(x) for x in uturn_row[t])
                    for t in range(12)
                )
            rank_work_items.append((i, ew_i, s6_loops, uturn_assign))

        # Dispatch rank work — use temporary Pool with fork-inherited data
        if num_workers > 1 and len(rank_work_items) > 500:
            from corep_fast.utils.persistent_pool import get_pool
            p = get_pool(num_workers)
            rank_results = p.map(_s7_rank_worker, rank_work_items,
                                 chunksize=max(1, len(rank_work_items) // (num_workers * 4)))
        else:
            rank_results = [_s7_rank_worker(item) for item in rank_work_items]

        # Write OK-cube rank results into flat array
        for cube_idx, rank_lists in rank_results:
            l_lo = int(loop_cube_off_np[cube_idx])
            for li_off, ranks in enumerate(rank_lists):
                li = l_lo + li_off
                e_lo = int(loop_edge_off_np[li])
                for k, r in enumerate(ranks):
                    all_ranks[e_lo + k] = r

    # ===================================================================
    # Phase 2: GPU batched centroid interpolation (scatter-mean)
    # ===================================================================

    # Fast path: GPU Phase 1 already produced a flat (E,) loop_edge_rank tensor,
    # which is exactly the data the downstream code reconstructs piecewise. Use
    # it directly to skip the per-cube/per-loop/per-crossing Python loop.
    if _cfg.S7_PHASE1_GPU and total_edges > 0:
        loop_edge_rank_for_phase2 = loop_edge_rank_gpu  # already on device
        # Build per-crossing (loop_id, edge_id, rank, weight, cube_id) tensors
        # in one shot from CSR.
        loop_off_t = batch.loop_edge_off.to(torch.int64)
        loop_cube_off_t = batch.loop_cube_off.to(torch.int64)
        loops_per_cube = loop_cube_off_t[1:] - loop_cube_off_t[:-1]
        cube_arange = torch.arange(N, device=device, dtype=torch.int64)
        loop_to_cube_t = torch.repeat_interleave(cube_arange, loops_per_cube)  # (L,)

        # Per crossing: cube id and loop id
        crossings_per_loop = loop_off_t[1:] - loop_off_t[:-1]   # (L,)
        loop_arange = torch.arange(total_loops, device=device, dtype=torch.int64)
        t_loop_ids = torch.repeat_interleave(loop_arange, crossings_per_loop)  # (E,)
        t_cube_ids = torch.repeat_interleave(loop_to_cube_t, crossings_per_loop)  # (E,)
        t_edge_ids = batch.loop_edge_val.to(torch.int64)        # (E,)
        t_ranks = loop_edge_rank_for_phase2.to(torch.float32)   # (E,)
        # Weight per crossing = edge_weights[cube, edge]
        t_weights = batch.edge_weights.to(torch.float32)[t_cube_ids, t_edge_ids]

        # Apply ok mask: rank/weight zero for non-OK cubes (no centroid contribution)
        status_t = batch.status.to(torch.int64)
        cube_ok_t = (status_t == CubeStatus.OK)
        cross_ok = cube_ok_t[t_cube_ids]   # (E,) bool
        # For non-OK crossings, set rank=0 and weight=1 to avoid NaN; but we'll
        # still scatter into the loop centroid. To match CPU behavior (which
        # only contributes OK cubes to centroids), we can mask these out from
        # the scatter. Since the CPU path only iterates ok_cube_indices, we
        # filter here too.
        sel = cross_ok
        t_loop_ids = t_loop_ids[sel]
        t_edge_ids = t_edge_ids[sel]
        t_ranks = t_ranks[sel]
        t_weights = t_weights[sel]
        t_cube_ids = t_cube_ids[sel]
        n_crossings = int(t_loop_ids.shape[0])

        if n_crossings > 0:
            edge_starts = CUBE_EDGE_STARTS.to(device=device, dtype=torch.float32)
            edge_ends = CUBE_EDGE_ENDS.to(device=device, dtype=torch.float32)
            cube_indices_gpu = batch.cube_indices.to(dtype=torch.float32)
            cube_origins = cube_indices_gpu / resolution
            step = 1.0 / resolution

            start_pts = edge_starts[t_edge_ids]
            end_pts = edge_ends[t_edge_ids]
            origins = cube_origins[t_cube_ids]
            t_param = ((t_ranks + 1.0) / (t_weights + 1.0)).unsqueeze(1)
            local_pos = start_pts + t_param * (end_pts - start_pts)
            world_pos = origins + local_pos * step

            loop_centroids = torch.zeros(total_loops, 3, dtype=torch.float32, device=device)
            loop_counts = torch.zeros(total_loops, dtype=torch.float32, device=device)
            loop_centroids.scatter_add_(0, t_loop_ids.unsqueeze(1).expand(-1, 3), world_pos)
            loop_counts.scatter_add_(0, t_loop_ids, torch.ones(n_crossings, dtype=torch.float32, device=device))
            loop_centroids = loop_centroids / loop_counts.clamp(min=1.0).unsqueeze(1)
        else:
            loop_centroids = torch.zeros(total_loops, 3, dtype=torch.float32, device=device)
    elif total_edges > 0 and ok_cube_indices:
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
            l_lo = int(loop_cube_off_np[cube_idx])
            l_hi = int(loop_cube_off_np[cube_idx + 1])
            for li_off in range(l_hi - l_lo):
                li = l_lo + li_off
                e_lo = int(loop_edge_off_np[li])
                e_hi = int(loop_edge_off_np[li + 1])
                edges = loop_edge_val_np[e_lo:e_hi].tolist()
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
    # Phase 3: Hungarian matching (CPU, serial scipy — no MP overhead)
    # ===================================================================
    # Key optimization: bulk-transfer loop_centroids and point_values to CPU
    # ONCE, then do cost matrix construction + scipy.linear_sum_assignment
    # serially per cube in numpy. Avoids 275K GPU→CPU syncs and MP pickle overhead.
    # scipy's linear_sum_assignment is C-optimized; for small matrices (~3×5)
    # running serially is faster than MP dispatch overhead.

    point_offsets_np = batch.point_offsets.cpu().numpy()
    loop_centroids_np = loop_centroids.cpu().numpy() if total_loops > 0 \
        else np.zeros((0, 3), dtype=np.float32)
    point_values_np = batch.point_values.cpu().numpy()

    if _cfg.HUNGARIAN_GPU and len(ok_cube_indices) > 0:
        # ---- W_HG batched path: hot 1x1 + 1xK + brute-force buckets on GPU
        # ---- with scipy fallback for the rare hard cases.
        from corep_fast.stages.s7_triton import hungarian_batched

        max_nl, max_np = 5, 8
        B_total = len(ok_cube_indices)

        # Vectorize per-cube nl/npts derivation.
        ok_idx_np = np.asarray(ok_cube_indices, dtype=np.int64)
        l_lo_arr = loop_cube_off_np[ok_idx_np].astype(np.int64)
        l_hi_arr = loop_cube_off_np[ok_idx_np + 1].astype(np.int64)
        p_lo_arr = point_offsets_np[ok_idx_np].astype(np.int64)
        p_hi_arr = point_offsets_np[ok_idx_np + 1].astype(np.int64)
        nl_arr = (l_hi_arr - l_lo_arr).astype(np.int64)
        np_arr = (p_hi_arr - p_lo_arr).astype(np.int64)

        # Cost padded matrix; +inf for padded slots.
        cost_padded = np.full((B_total, max_nl, max_np), np.inf, dtype=np.float32)

        # Hot path vectorized: cubes with nl==1 AND npts==1.
        hot_mask = (nl_arr == 1) & (np_arr == 1)
        if hot_mask.any():
            hi = np.nonzero(hot_mask)[0]
            c_starts = l_lo_arr[hi]
            p_starts = p_lo_arr[hi]
            c_pts = loop_centroids_np[c_starts]   # (H, 3)
            p_pts = point_values_np[p_starts]     # (H, 3)
            d = c_pts - p_pts
            cost_padded[hi, 0, 0] = (d * d).sum(axis=-1).astype(np.float32)

        # Slow path: in-batch eligible (small) + the empty/fallback cases.
        # Cubes excluded from the batched kernel: oversized / rect-reverse shapes
        # plus the nl=0 and npts=0 degenerate cases. The later classification
        # block (Case A / B / C below) further splits these into:
        #   - shape_invalid_m  -> scipy fallback
        #   - np_zero_m        -> identity assignment (Case A, no-op on kernel)
        #   - nl_zero_m        -> nothing to write
        # Kept named broadly so the mask's role (exclusion from the batched
        # kernel, not just "fallbacks") is clear.
        excluded_from_batch_mask = (nl_arr > max_nl) | (np_arr > max_np) | (np_arr < nl_arr) | (nl_arr == 0) | (np_arr == 0)
        # Eligible-but-not-hot mask (still goes through batched kernel):
        eligible_other = ~excluded_from_batch_mask & ~hot_mask
        if eligible_other.any():
            for bi in np.nonzero(eligible_other)[0]:
                nl = int(nl_arr[bi]); npts = int(np_arr[bi])
                l_lo = int(l_lo_arr[bi]); p_lo = int(p_lo_arr[bi])
                c_i = loop_centroids_np[l_lo:l_lo + nl]
                p_i = point_values_np[p_lo:p_lo + npts]
                diff = c_i[:, None, :] - p_i[None, :, :]
                cost_padded[bi, :nl, :npts] = (diff * diff).sum(axis=-1).astype(np.float32)

        # Send to GPU and run.
        cost_padded_t = torch.from_numpy(cost_padded).to(device)
        nl_t = torch.from_numpy(nl_arr).to(device)
        np_t = torch.from_numpy(np_arr).to(device)
        matches = hungarian_batched(cost_padded_t, nl_t, np_t, max_nl, max_np).cpu().numpy()

        # ---- Vectorized apply phase (Task 18b) ----
        # Classify cubes into hot / valid / invalid shape / npts-zero / nl-zero
        # using boolean masks of length B_total, then do bulk fancy assignment.
        #
        # Convert all_matches to a numpy array once for vectorized writes;
        # convert back to list at the end so the tail packing logic is
        # unchanged.
        all_matches_np = np.asarray(all_matches, dtype=np.int64)

        # Shape classification masks (all length B_total).
        nl_zero_m = (nl_arr == 0)
        np_zero_m = (np_arr == 0) & ~nl_zero_m
        shape_invalid_m = (
            (nl_arr > max_nl) | (np_arr > max_np) | (np_arr < nl_arr)
        ) & ~nl_zero_m & ~np_zero_m
        shape_valid_m = ~nl_zero_m & ~np_zero_m & ~shape_invalid_m

        # For shape_valid rows, detect "any -1" in the first nl slots (tie).
        # Vectorize via arange mask: position < nl_arr[bi].
        if shape_valid_m.any():
            valid_idx = np.nonzero(shape_valid_m)[0]
            pos = np.arange(max_nl, dtype=np.int64)
            in_range = pos[None, :] < nl_arr[valid_idx, None]    # (M, max_nl)
            has_sentinel = ((matches[valid_idx] == -1) & in_range).any(axis=1)
            tie_rows = valid_idx[has_sentinel]
            apply_rows = valid_idx[~has_sentinel]
        else:
            tie_rows = np.empty(0, dtype=np.int64)
            apply_rows = np.empty(0, dtype=np.int64)

        # Case A: npts == 0 (and nl > 0) → identity match.
        # For each such row, write all_matches[l_lo + li_off] = li_off for li_off in range(nl).
        if np_zero_m.any():
            rows = np.nonzero(np_zero_m)[0]
            # repeat l_lo by nl; add cumulative offsets 0..nl-1 within each group.
            nl_here = nl_arr[rows]
            l_lo_here = l_lo_arr[rows]
            # Build the full destination index array.
            total = int(nl_here.sum())
            if total > 0:
                # Flat offsets via cumsum: for each row, [0, 1, ..., nl-1].
                ends = np.cumsum(nl_here)
                starts = ends - nl_here
                # li_off = arange(total) - starts[row_of_flat] — compute via repeat.
                row_of_flat = np.repeat(np.arange(len(rows), dtype=np.int64), nl_here)
                li_off = np.arange(total, dtype=np.int64) - starts[row_of_flat]
                dst = l_lo_here[row_of_flat] + li_off
                all_matches_np[dst] = li_off

        # Case B: shape_valid & no tie → apply matches[bi, :nl].
        # 5-iter outer loop over li_off (max_nl = 5), bulk vector write per slot.
        if apply_rows.size > 0:
            nl_ap = nl_arr[apply_rows]
            l_lo_ap = l_lo_arr[apply_rows]
            for li_off in range(max_nl):
                active = (li_off < nl_ap)
                if not active.any():
                    # All remaining li_off slots also empty (mask is monotone).
                    break
                rows_i = apply_rows[active]
                dst = l_lo_ap[active] + li_off
                all_matches_np[dst] = matches[rows_i, li_off]

        # Case C: shape_invalid OR tie → scipy fallback.
        # Build fallback list via cube indices.
        fallback_cubes: List[int] = []
        if shape_invalid_m.any():
            fallback_cubes.extend(ok_idx_np[shape_invalid_m].tolist())
        if tie_rows.size > 0:
            fallback_cubes.extend(ok_idx_np[tie_rows].tolist())

        # Convert back to list for downstream torch.tensor(...) in pack step.
        all_matches = all_matches_np.tolist()

        # Scipy fallback for the rare hard cases.
        for cube_idx in fallback_cubes:
            l_lo = int(loop_cube_off_np[cube_idx])
            l_hi = int(loop_cube_off_np[cube_idx + 1])
            n_loops = l_hi - l_lo
            if n_loops == 0:
                continue
            p_lo = int(point_offsets_np[cube_idx])
            p_hi = int(point_offsets_np[cube_idx + 1])
            n_points = p_hi - p_lo
            if n_points == 0:
                for li_off in range(n_loops):
                    all_matches[l_lo + li_off] = li_off
                continue
            centroids_i = loop_centroids_np[l_lo:l_hi]
            comp_pts_i = point_values_np[p_lo:p_hi]
            diff = centroids_i[:, None, :] - comp_pts_i[None, :, :]
            cost = (diff * diff).sum(axis=-1).astype(np.float64)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if r < n_loops:
                    all_matches[l_lo + int(r)] = int(c)
    else:
        # ---- Legacy scipy path (unchanged) ----
        for cube_idx in ok_cube_indices:
            l_lo = int(loop_cube_off_np[cube_idx])
            l_hi = int(loop_cube_off_np[cube_idx + 1])
            n_loops = l_hi - l_lo
            if n_loops == 0:
                continue

            p_lo = int(point_offsets_np[cube_idx])
            p_hi = int(point_offsets_np[cube_idx + 1])
            n_points = p_hi - p_lo

            if n_points == 0:
                # No component points: identity match
                for li_off in range(n_loops):
                    all_matches[l_lo + li_off] = li_off
                continue

            # Compute cost matrix in numpy (no GPU, no pickle)
            centroids_i = loop_centroids_np[l_lo:l_hi]          # (n_loops, 3)
            comp_pts_i = point_values_np[p_lo:p_hi]              # (n_points, 3)
            diff = centroids_i[:, None, :] - comp_pts_i[None, :, :]
            cost = (diff * diff).sum(axis=-1).astype(np.float64)  # (n_loops, n_points)

            # scipy Hungarian (C-optimized)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if r < n_loops:
                    all_matches[l_lo + int(r)] = int(c)

    # ===================================================================
    # Pack results
    # ===================================================================

    if all_ranks is None:
        # GPU Phase 1 path: reuse the device tensor directly
        loop_edge_rank = loop_edge_rank_gpu if total_edges > 0 \
            else torch.zeros(0, dtype=torch.int32, device=device)
    else:
        loop_edge_rank = torch.tensor(all_ranks, dtype=torch.int32, device=device) \
            if total_edges > 0 else torch.zeros(0, dtype=torch.int32, device=device)
    loop_point_match = torch.tensor(all_matches, dtype=torch.int32, device=device) \
        if total_loops > 0 else torch.zeros(0, dtype=torch.int32, device=device)

    return _replace_fields(
        batch,
        loop_edge_rank=loop_edge_rank,
        loop_point_match=loop_point_match,
    )
