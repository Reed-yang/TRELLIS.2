"""
Stage 6: Normal Curve loop extraction via arc assignment + U-Turn enumeration.

Merges custom/collapse_edge.py (s5) + custom/collapse_face.py (s6):
  - Fast path (face_weights == 0): direct arc assignment + loop trace
  - Slow path (face_weights > 0): U-Turn enumeration + validation

Output:
  - loop_cube_off (N+1,) int64  — per-cube offsets into loop list
  - loop_edge_off (L+1,) int64  — per-loop offsets into edge list
  - loop_edge_val (E,)   int32  — flat array of edge indices per loop
  - status        (N,)   int32  — CubeStatus per cube
  - uturn_assignment (N, 12, 3) int32 — per-facet (u1,u2,u3); -1 for fast-path

Public API:
    s6_collapse(batch, pool=None) -> CubeBatch
"""
from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch

from corep_fast import config as _cfg
from corep_fast.constants import CUBE_EDGES, CUBE_FACETS
from corep_fast.containers import CubeBatch, CubeStatus, _replace_fields

# ---------------------------------------------------------------------------
# Local topology tables (numpy, for per-cube CPU work)
# ---------------------------------------------------------------------------

_EDGE_VERTS: List[Tuple[int, int]] = [tuple(e) for e in CUBE_EDGES.tolist()]
_TRIANGLES: List[Tuple[int, int, int]] = [tuple(f) for f in CUBE_FACETS.tolist()]


# ---------------------------------------------------------------------------
# Topology helpers (mirrors custom/collapse_edge.py)
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
    """Return the k point indices closest to corner_v on edge e_idx.

    Point 0 is closest to the first vertex, point (weight-1) to the second.
    """
    u, v = _EDGE_VERTS[e_idx]
    if corner_v == u:
        return list(range(k))
    elif corner_v == v:
        return [weight - 1 - i for i in range(k)]
    else:
        raise ValueError(f"Vertex {corner_v} is not an endpoint of edge {e_idx}")


# ---------------------------------------------------------------------------
# Static topology tables for the GPU fast-path (W3 / O5)
# ---------------------------------------------------------------------------
# For each facet (e1, e2, e3) and each of the 3 edge-pair orientations
# (e1,e2), (e2,e3), (e3,e1) we precompute:
#   _FACET_PAIRS_NP[f, p, 0] = edge_A, _FACET_PAIRS_NP[f, p, 1] = edge_B
#   _FACET_K_INDEX_NP[f, p]  = which of (k12, k23, k31) governs (0/1/2)
#   _FACET_PA_FLIP_NP[f, p]  = 0 if pts_A[i] = i, 1 if pts_A[i] = w_A - 1 - i
#   _FACET_PB_FLIP_NP[f, p]  = same convention for edge_B
# Shapes: (12, 3, 2) for pairs, (12, 3) for the others.

_FACET_PAIRS_NP = np.zeros((12, 3, 2), dtype=np.int64)
_FACET_K_INDEX_NP = np.zeros((12, 3), dtype=np.int64)
_FACET_PA_FLIP_NP = np.zeros((12, 3), dtype=np.int64)
_FACET_PB_FLIP_NP = np.zeros((12, 3), dtype=np.int64)
for _f_idx, (_e1, _e2, _e3) in enumerate(_TRIANGLES):
    # Pair order matches _collapse_fast: (e1,e2,k12), (e2,e3,k23), (e3,e1,k31)
    for _p_idx, (_eA, _eB) in enumerate([(_e1, _e2), (_e2, _e3), (_e3, _e1)]):
        _cv = _get_common_vertex(_eA, _eB)
        _FACET_PAIRS_NP[_f_idx, _p_idx, 0] = _eA
        _FACET_PAIRS_NP[_f_idx, _p_idx, 1] = _eB
        _FACET_K_INDEX_NP[_f_idx, _p_idx] = _p_idx  # k12=0, k23=1, k31=2
        _FACET_PA_FLIP_NP[_f_idx, _p_idx] = 0 if _cv == _EDGE_VERTS[_eA][0] else 1
        _FACET_PB_FLIP_NP[_f_idx, _p_idx] = 0 if _cv == _EDGE_VERTS[_eB][0] else 1
del _f_idx, _e1, _e2, _e3, _p_idx, _eA, _eB, _cv


# ---------------------------------------------------------------------------
# Fast path: direct arc assignment + loop tracing (no U-Turns)
# ---------------------------------------------------------------------------

def _collapse_fast(ew: List[int]) -> Tuple[List[List[int]], int]:
    """Fast path: all face_weights == 0 → direct arc assignment + loop trace.

    Returns:
        (loops, status) where loops is a list of edge-index lists,
        and status is CubeStatus.OK or CubeStatus.UNSOLVABLE.
    """
    # 1. Validate triangle inequality + parity, build adjacency
    adj: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    for e in range(18):
        for p in range(ew[e]):
            adj[(e, p)] = []

    for t_idx, (e1, e2, e3) in enumerate(_TRIANGLES):
        w1, w2, w3 = ew[e1], ew[e2], ew[e3]

        # Triangle inequality
        if w1 + w2 < w3 or w2 + w3 < w1 or w3 + w1 < w2:
            return [], CubeStatus.UNSOLVABLE
        # Parity
        if (w1 + w2 + w3) % 2 != 0:
            return [], CubeStatus.UNSOLVABLE

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
                adj[(edge_A, pts_A[i])].append((t_idx, edge_B, pts_B[i]))
                adj[(edge_B, pts_B[i])].append((t_idx, edge_A, pts_A[i]))

    # 2. Verify degree-2 invariant
    for node, nbrs in adj.items():
        if len(nbrs) != 2:
            return [], CubeStatus.UNSOLVABLE

    # 3. Trace loops
    loops: List[List[int]] = []
    visited: Set[Tuple[int, int]] = set()

    for start_node in adj:
        if start_node in visited:
            continue

        current_loop: List[int] = []
        curr_node = start_node
        prev_node = None

        while True:
            visited.add(curr_node)
            neighbors = adj[curr_node]

            n1_edge, n1_p = neighbors[0][1], neighbors[0][2]
            n2_edge, n2_p = neighbors[1][1], neighbors[1][2]
            node1 = (n1_edge, n1_p)
            node2 = (n2_edge, n2_p)

            if prev_node is None:
                next_node = node1
            else:
                if node1 == prev_node:
                    next_node = node2
                else:
                    next_node = node1

            # Record the edge index of the current crossing point
            current_loop.append(curr_node[0])

            prev_node = curr_node
            curr_node = next_node

            if curr_node == start_node:
                break

        loops.append(current_loop)

    return loops, CubeStatus.OK


# ---------------------------------------------------------------------------
# Fast path GPU batched (W3 / O5)
# ---------------------------------------------------------------------------

def _fastpath_gpu_build_adjacency(
    edge_weights_fast: torch.Tensor,
    k12_fast: torch.Tensor,
    k23_fast: torch.Tensor,
    k31_fast: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build per-cube adjacency for all fast-path cubes on GPU.

    Inputs (all device tensors, dtype int64 expected for safe arithmetic):
        edge_weights_fast: (M, 18) edge weights (M = N_fast cubes)
        k12_fast/k23_fast/k31_fast: (M, 12) per-facet arc counts

    Returns (numpy host arrays):
        ew_np_fast: (M, 18) int64 — copied edge weights
        point_offset_np: (M, 19) int64 — prefix-sum offsets (point_offset[c, e]
            is the global point id of (e, 0)); point_offset[c, 18] = total
            points for cube c
        adj_np: (M, max_points, 2) int32 — neighbor global point id for each
            point's two slots (-1 if unused / unsolvable)
        degree_ok_np: (M,) bool — True iff every point with id < total_points
            has degree exactly 2

    Algorithm: for each (cube, facet, pair_idx, i), if i < k_pair, emit two
    directed adjacency edges between (cube, global_A) and (cube, global_B).
    Then group by (cube, point) and verify degree == 2.
    """
    device = edge_weights_fast.device
    M = edge_weights_fast.shape[0]
    if M == 0:
        return (
            np.zeros((0, 18), dtype=np.int64),
            np.zeros((0, 19), dtype=np.int64),
            np.zeros((0, 0, 2), dtype=np.int32),
            np.zeros((0,), dtype=bool),
        )

    # Per-cube point offsets: shape (M, 19) where col j = sum(ew[:, :j])
    # point_offset[c, e] is the global point id of (e, p=0) within cube c
    zeros_col = torch.zeros((M, 1), dtype=torch.int64, device=device)
    point_offset = torch.cat(
        [zeros_col, torch.cumsum(edge_weights_fast, dim=1)], dim=1
    )  # (M, 19) int64
    total_points_per_cube = point_offset[:, -1]  # (M,) int64
    max_points = int(total_points_per_cube.max().item()) if M > 0 else 0
    if max_points == 0:
        return (
            edge_weights_fast.cpu().numpy(),
            point_offset.cpu().numpy(),
            np.zeros((M, 0, 2), dtype=np.int32),
            np.ones((M,), dtype=bool),  # vacuously OK
        )

    # max_w bounds the per-pair index loop i ∈ [0, max_w)
    # Each k_pair[c, f] <= min(w_eA, w_eB) but we just bound by global max edge weight
    max_w = int(edge_weights_fast.max().item())
    if max_w == 0:
        # No arcs; degree should be 0 for every point but no points either
        return (
            edge_weights_fast.cpu().numpy(),
            point_offset.cpu().numpy(),
            np.full((M, max_points, 2), -1, dtype=np.int32),
            (total_points_per_cube == 0).cpu().numpy(),
        )

    # Stack k tensors: (M, 12, 3) → k_per_facet_pair[c, f, p]
    k_per_pair = torch.stack([k12_fast, k23_fast, k31_fast], dim=2)  # (M, 12, 3)

    # Static facet pair info on device
    facet_pairs_t = torch.from_numpy(_FACET_PAIRS_NP).to(device)        # (12, 3, 2)
    facet_pa_flip_t = torch.from_numpy(_FACET_PA_FLIP_NP).to(device)    # (12, 3)
    facet_pb_flip_t = torch.from_numpy(_FACET_PB_FLIP_NP).to(device)    # (12, 3)

    eA_idx = facet_pairs_t[..., 0]  # (12, 3)
    eB_idx = facet_pairs_t[..., 1]  # (12, 3)

    # Gather edge weights for A and B sides per facet/pair: (M, 12, 3)
    wA = edge_weights_fast[:, eA_idx]  # (M, 12, 3)
    wB = edge_weights_fast[:, eB_idx]  # (M, 12, 3)

    # Generate i index along last dim → shape (M, 12, 3, max_w)
    i_idx = torch.arange(max_w, device=device, dtype=torch.int64)  # (max_w,)
    i_idx_b = i_idx.view(1, 1, 1, max_w)
    valid_mask = i_idx_b < k_per_pair.unsqueeze(-1)  # (M, 12, 3, max_w)

    # Compute pts_A and pts_B per (M, 12, 3, max_w)
    # if pa_flip == 0: pA = i; else: pA = wA - 1 - i
    pa_flip_b = facet_pa_flip_t.view(1, 12, 3, 1)  # broadcasts to (M, 12, 3, max_w)
    pb_flip_b = facet_pb_flip_t.view(1, 12, 3, 1)
    wA_b = wA.unsqueeze(-1)  # (M, 12, 3, 1)
    wB_b = wB.unsqueeze(-1)
    pA = torch.where(pa_flip_b == 0, i_idx_b.expand_as(valid_mask),
                     wA_b - 1 - i_idx_b)  # (M, 12, 3, max_w)
    pB = torch.where(pb_flip_b == 0, i_idx_b.expand_as(valid_mask),
                     wB_b - 1 - i_idx_b)

    # Per-cube global point id: point_offset[c, eA] + pA, etc.
    # point_offset shape (M, 19); we gather per (cube, facet, pair) edge id
    eA_b = eA_idx.view(1, 12, 3, 1).expand(M, -1, -1, max_w)  # (M, 12, 3, max_w)
    eB_b = eB_idx.view(1, 12, 3, 1).expand(M, -1, -1, max_w)
    # Use gather over edge dim
    # point_offset_a[c, f, p, i] = point_offset[c, eA_b[c, f, p, i]]
    point_offset_a = torch.gather(
        point_offset.unsqueeze(1).unsqueeze(2).expand(-1, 12, 3, -1),
        dim=3,
        index=eA_b,
    )  # (M, 12, 3, max_w)
    point_offset_b = torch.gather(
        point_offset.unsqueeze(1).unsqueeze(2).expand(-1, 12, 3, -1),
        dim=3,
        index=eB_b,
    )  # (M, 12, 3, max_w)
    global_A = point_offset_a + pA  # (M, 12, 3, max_w)
    global_B = point_offset_b + pB

    # We will materialize one entry per directed edge: (cube, global_A, global_B)
    # plus its reverse. Total max entries = M * 12 * 3 * max_w * 2.
    # Filter by valid_mask.
    # To assign neighbor slots, we use sort by (cube, point) and place.
    # Concat both directions:
    cube_idx = torch.arange(M, device=device, dtype=torch.int64).view(M, 1, 1, 1).expand_as(valid_mask)
    src_pts = torch.cat([global_A.flatten(), global_B.flatten()], dim=0)
    dst_pts = torch.cat([global_B.flatten(), global_A.flatten()], dim=0)
    cube_flat = torch.cat([cube_idx.flatten(), cube_idx.flatten()], dim=0)
    valid_flat = torch.cat([valid_mask.flatten(), valid_mask.flatten()], dim=0)

    src_pts = src_pts[valid_flat]
    dst_pts = dst_pts[valid_flat]
    cube_flat = cube_flat[valid_flat]

    # Now for each (cube, src_pt) we need to assign slot 0 or 1 in the adjacency
    # tensor. Sort by (cube, src_pt) so equal keys are adjacent, then count.
    # composite key = cube * (max_points + 1) + src_pt
    composite_key = cube_flat * (max_points + 1) + src_pts
    sorted_key, sort_idx = torch.sort(composite_key, stable=True)
    sorted_dst = dst_pts[sort_idx]
    sorted_cube = cube_flat[sort_idx]
    sorted_src = src_pts[sort_idx]

    # Slot index: within each run of equal keys, position 0..n-1
    # Compute via diff: slot resets when key changes
    if sorted_key.numel() == 0:
        slot = torch.zeros(0, dtype=torch.int64, device=device)
    else:
        same_as_prev = torch.zeros_like(sorted_key, dtype=torch.bool)
        same_as_prev[1:] = sorted_key[1:] == sorted_key[:-1]
        # cumulative count of "same" since last reset
        # We want slot 0,1,2,3 per group
        # A simple approach: for each i, slot[i] = i - last_reset_pos[i]
        not_same = ~same_as_prev
        # cumulative position of last group start
        idx_arange = torch.arange(sorted_key.numel(), device=device, dtype=torch.int64)
        # group_start[i] = max position j <= i where not_same[j] is True
        # Use cummax on idx where not_same else 0 (but we need running max of idx_arange masked by not_same)
        masked_pos = torch.where(not_same, idx_arange, torch.full_like(idx_arange, -1))
        group_start, _ = torch.cummax(masked_pos, dim=0)
        slot = idx_arange - group_start

    # Determine final degree per (cube, src_pt) = max slot + 1 within its group
    # Compute degree by reverse pass via reverse cummax of (slot+1) per group, but
    # easier: degree[c, src_pt] = number of entries with same key
    # Use scatter_add to count
    degree_flat = torch.zeros(M * (max_points + 1), dtype=torch.int64, device=device)
    degree_flat.scatter_add_(0, composite_key,
                             torch.ones_like(composite_key, dtype=torch.int64))
    degree = degree_flat.view(M, max_points + 1)[:, :max_points]  # (M, max_points)

    # Build adjacency tensor (M, max_points, 2) int32, default -1
    adj = torch.full((M, max_points, 2), -1, dtype=torch.int32, device=device)
    # Only fill slot < 2 (degrees > 2 will be detected via degree check later)
    valid_slot = slot < 2
    adj[sorted_cube[valid_slot], sorted_src[valid_slot], slot[valid_slot]] = \
        sorted_dst[valid_slot].to(torch.int32)

    # degree_ok[c] = True iff every point with id < total_points_per_cube[c]
    # has degree exactly 2. Empty points (id >= total_points) are ignored.
    point_id_arange = torch.arange(max_points, device=device, dtype=torch.int64).view(1, max_points)
    point_active = point_id_arange < total_points_per_cube.view(M, 1)  # (M, max_points)
    degree_correct = (degree == 2) | (~point_active)  # only check active points
    degree_ok = degree_correct.all(dim=1)  # (M,)

    # Copy back to host
    return (
        edge_weights_fast.cpu().numpy(),
        point_offset.cpu().numpy(),
        adj.cpu().numpy(),
        degree_ok.cpu().numpy(),
    )


def _fastpath_trace_loops_numpy(
    point_offset_row: np.ndarray,  # (19,) int64 for one cube
    adj_row: np.ndarray,           # (max_points, 2) int32 for one cube
    total_points: int,
) -> List[List[int]]:
    """Trace loops for one fast-path cube using numpy adjacency.

    The two slots in adj_row[p] are p's two neighbors (already verified
    degree==2 by the caller). We then walk loops, recording the edge id
    of each visited point.
    """
    if total_points == 0:
        return []

    # Recover edge id for each global point id via point_offset
    # edge_of_point[p] = largest e s.t. point_offset[e] <= p
    # We just compute it once via searchsorted on point_offset[:18]
    # point_offset_row has shape (19,), monotone non-decreasing
    # Build lookup: for each p in [0, total_points), edge id is np.searchsorted
    # with side='right' minus 1.
    edge_of_point = np.searchsorted(point_offset_row, np.arange(total_points),
                                    side='right') - 1
    edge_of_point = edge_of_point.astype(np.int32)

    visited = np.zeros(total_points, dtype=bool)
    loops: List[List[int]] = []

    for start in range(total_points):
        if visited[start]:
            continue
        loop: List[int] = []
        curr = start
        prev = -1
        while True:
            visited[curr] = True
            n0 = int(adj_row[curr, 0])
            n1 = int(adj_row[curr, 1])
            # Pick the neighbor that is not prev (matches Python reference logic)
            if prev == -1:
                nxt = n0
            else:
                nxt = n1 if n0 == prev else n0
            loop.append(int(edge_of_point[curr]))
            prev = curr
            curr = nxt
            if curr == start:
                break
        loops.append(loop)

    return loops


# ---------------------------------------------------------------------------
# Slow path: U-Turn enumeration (face_weights > 0)
# ---------------------------------------------------------------------------

def _get_canonical_loop(loop: List[int]) -> Tuple[int, ...]:
    """Canonicalize a cyclic loop for deduplication.

    Considers all rotations and the reverse to find the lexicographic minimum.
    """
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


def _trace_loops_for_uturn_assignment(
    ew: List[int],
    assignment: Tuple[Tuple[int, int, int], ...],
) -> List[List[int]]:
    """Build adjacency graph for a specific U-Turn assignment and trace loops.

    Returns list of loops, each loop being a list of edge indices.
    Raises ValueError if the graph is not degree-2 everywhere.
    """
    adj: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {
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
            for i in range(k):
                adj[(eA, pts_A[i])].append((t_idx, eB, pts_B[i]))
                adj[(eB, pts_B[i])].append((t_idx, eA, pts_A[i]))

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
            for i in range(u):
                p1 = start_idx + 2 * i
                p2 = start_idx + 2 * i + 1
                adj[(eA, p1)].append((t_idx, eA, p2))
                adj[(eA, p2)].append((t_idx, eA, p1))

        # U-Turn arcs
        _assign_uturns(e1, u1)
        _assign_uturns(e2, u2)
        _assign_uturns(e3, u3)

    # Trace loops
    loops: List[List[int]] = []
    visited: Set[Tuple[int, int]] = set()

    for start_node in adj:
        if start_node in visited:
            continue

        curr_node = start_node
        prev_node = None
        current_loop: List[int] = []

        while True:
            visited.add(curr_node)
            neighbors = adj[curr_node]
            if len(neighbors) != 2:
                raise ValueError(f"Degree != 2 at node {curr_node}")

            if prev_node is None:
                f_idx, next_e, next_p = neighbors[0]
            else:
                n0_e, n0_p = neighbors[0][1], neighbors[0][2]
                if (n0_e, n0_p) == prev_node:
                    f_idx, next_e, next_p = neighbors[1]
                else:
                    f_idx, next_e, next_p = neighbors[0]

            # Record edge and face
            current_loop.append(curr_node[0])
            current_loop.append(f_idx)

            prev_node = curr_node
            curr_node = (next_e, next_p)

            if curr_node == start_node:
                break

        loops.append(current_loop)

    return loops


def _collapse_with_uturns(ew: List[int], fw: List[int]) -> Tuple[List[List[int]], int]:
    """Slow path: U-Turn enumeration for cubes with face_weights > 0.

    Returns:
        (loops, status) where loops is a list of edge-index lists,
        and status is CubeStatus.OK / AMBIGUOUS / UNSOLVABLE.
    """
    # 1. For each facet, enumerate valid (u1, u2, u3) with u1+u2+u3 = face_weight
    face_valid_assignments: List[List[Tuple[int, int, int]]] = []
    is_possible = True

    for t_idx in range(12):
        valid_for_face: List[Tuple[int, int, int]] = []
        W = fw[t_idx]
        e1, e2, e3 = _TRIANGLES[t_idx]

        for u1 in range(W + 1):
            for u2 in range(W + 1 - u1):
                u3 = W - u1 - u2

                w1 = ew[e1] - 2 * u1
                w2 = ew[e2] - 2 * u2
                w3 = ew[e3] - 2 * u3

                if w1 < 0 or w2 < 0 or w3 < 0:
                    continue
                if w1 + w2 < w3 or w2 + w3 < w1 or w3 + w1 < w2:
                    continue
                if (w1 + w2 + w3) % 2 != 0:
                    continue

                valid_for_face.append((u1, u2, u3))

        if not valid_for_face:
            is_possible = False
            break

        face_valid_assignments.append(valid_for_face)

    if not is_possible:
        return [], CubeStatus.UNSOLVABLE

    # Safeguard against combinatorial explosion
    total_combinations = math.prod(len(v) for v in face_valid_assignments)
    if total_combinations > 100000:
        return [], CubeStatus.BUDGET_EXCEEDED

    # 2. Cartesian product → trace loops → deduplicate
    unique_solutions: Dict[Tuple, List[List[int]]] = {}

    for assignment in itertools.product(*face_valid_assignments):
        try:
            loops_with_faces = _trace_loops_for_uturn_assignment(ew, assignment)
            canonical_sol = _get_canonical_solution(
                [loop[::2] for loop in loops_with_faces]
            )
            if canonical_sol not in unique_solutions:
                unique_solutions[canonical_sol] = [loop[::2] for loop in loops_with_faces]
        except Exception:
            continue

    # 3. Prune backward U-Turns (any edge appearing >= 3 times in a single loop)
    pruned: List[List[List[int]]] = []
    for sol in unique_solutions.values():
        is_valid = True
        for loop in sol:
            if any(count >= 3 for count in Counter(loop).values()):
                is_valid = False
                break
        if is_valid:
            pruned.append(sol)

    # 4. Classify
    num_sols = len(pruned)
    if num_sols == 0:
        return [], CubeStatus.UNSOLVABLE
    elif num_sols == 1:
        return pruned[0], CubeStatus.OK
    else:
        # Ambiguous: return the first solution but mark status
        return pruned[0], CubeStatus.AMBIGUOUS


def _collapse_with_uturns_tracked(
    ew: List[int], fw: List[int],
) -> Tuple[List[List[int]], int, Optional[List[Tuple[int, int, int]]]]:
    """Slow path with assignment tracking for uturn_assignment output.

    Same as _collapse_with_uturns but also returns the winning assignment
    tuple (12 x (u1, u2, u3)) when status is OK.

    Returns:
        (loops, status, assignment_or_none)
        assignment_or_none is a list of 12 (u1,u2,u3) tuples for OK cubes,
        None otherwise.
    """
    # 1. For each facet, enumerate valid (u1, u2, u3) with u1+u2+u3 = face_weight
    face_valid_assignments: List[List[Tuple[int, int, int]]] = []
    is_possible = True

    for t_idx in range(12):
        valid_for_face: List[Tuple[int, int, int]] = []
        W = fw[t_idx]
        e1, e2, e3 = _TRIANGLES[t_idx]

        for u1 in range(W + 1):
            for u2 in range(W + 1 - u1):
                u3 = W - u1 - u2

                w1 = ew[e1] - 2 * u1
                w2 = ew[e2] - 2 * u2
                w3 = ew[e3] - 2 * u3

                if w1 < 0 or w2 < 0 or w3 < 0:
                    continue
                if w1 + w2 < w3 or w2 + w3 < w1 or w3 + w1 < w2:
                    continue
                if (w1 + w2 + w3) % 2 != 0:
                    continue

                valid_for_face.append((u1, u2, u3))

        if not valid_for_face:
            is_possible = False
            break

        face_valid_assignments.append(valid_for_face)

    if not is_possible:
        return [], CubeStatus.UNSOLVABLE, None

    # Safeguard against combinatorial explosion
    total_combinations = math.prod(len(v) for v in face_valid_assignments)
    if total_combinations > 100000:
        return [], CubeStatus.BUDGET_EXCEEDED, None

    # 2. Cartesian product → trace loops → deduplicate
    # Track which assignment produced each canonical solution
    unique_solutions: Dict[Tuple, Tuple[List[List[int]], Tuple]] = {}

    for assignment in itertools.product(*face_valid_assignments):
        try:
            loops_with_faces = _trace_loops_for_uturn_assignment(ew, assignment)
            canonical_sol = _get_canonical_solution(
                [loop[::2] for loop in loops_with_faces]
            )
            if canonical_sol not in unique_solutions:
                unique_solutions[canonical_sol] = (
                    [loop[::2] for loop in loops_with_faces],
                    assignment,
                )
        except Exception:
            continue

    # 3. Prune backward U-Turns (any edge appearing >= 3 times in a single loop)
    pruned: List[Tuple[List[List[int]], Tuple]] = []
    for sol_loops, sol_assignment in unique_solutions.values():
        is_valid = True
        for loop in sol_loops:
            if any(count >= 3 for count in Counter(loop).values()):
                is_valid = False
                break
        if is_valid:
            pruned.append((sol_loops, sol_assignment))

    # 4. Classify
    num_sols = len(pruned)
    if num_sols == 0:
        return [], CubeStatus.UNSOLVABLE, None
    elif num_sols == 1:
        return pruned[0][0], CubeStatus.OK, list(pruned[0][1])
    else:
        # Ambiguous: return the first solution but mark status
        return pruned[0][0], CubeStatus.AMBIGUOUS, None


# ---------------------------------------------------------------------------
# Worker function for multiprocessing (module-level for pickling)
# ---------------------------------------------------------------------------

def _s6_worker(work_item):
    """Process one cube for s6 collapse.

    Args:
        work_item: tuple of (cube_idx, ew_list, fw_list, is_slow_path)

    Returns:
        tuple of (cube_idx, loops, status, assignment_or_none)
        assignment_or_none is a list of 12 (u1,u2,u3) tuples for
        slow-path OK cubes, None otherwise.
    """
    cube_idx, ew_list, fw_list, is_slow_path = work_item

    if is_slow_path:
        loops, status, assignment = _collapse_with_uturns_tracked(ew_list, fw_list)
    else:
        loops, status = _collapse_fast(ew_list)
        assignment = None

    return (cube_idx, loops, status, assignment)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def s6_collapse(batch: CubeBatch, pool=None, num_workers: int | None = None) -> CubeBatch:
    """Extract topological loops via normal curve theory + U-Turn enumeration.

    Merges custom/ collapse_edge.py (s5) + collapse_face.py (s6).

    Phase 1 (GPU): Batched triangle-inequality, parity, arc-count checks
    to partition cubes into fast-path / slow-path / empty.

    Phase 2 (CPU/MP): Graph construction + loop tracing via workers.

    Phase 3: Assembly into CSR arrays + uturn_assignment.

    Args:
        batch: CubeBatch with edge_weights/face_weights populated (after s4).
        pool: Optional PersistentWorkerPool for multiprocessing dispatch.

    Updates loop_cube_off, loop_edge_off, loop_edge_val, status, uturn_assignment.
    """
    import os as _os
    N = batch.num_cubes
    device = batch.device

    if N == 0:
        return batch

    # Derive worker count
    if num_workers is None:
        if pool is not None:
            num_workers = pool._num_workers
        else:
            num_workers = max(1, (_os.cpu_count() or 4) - 4)

    # ==================================================================
    # Phase 1: GPU batched checks — partition cubes
    # ==================================================================
    edge_weights = batch.edge_weights          # (N, 18) int32
    face_weights = batch.face_weights          # (N, 12) int32

    # Partition cubes into fast/slow/empty paths
    has_any_weight = edge_weights.sum(dim=1) > 0              # (N,)
    has_uturns = (face_weights > 0).any(dim=1)                # (N,)
    fast_path_mask = has_any_weight & ~has_uturns             # (N,)
    slow_path_mask = has_any_weight & has_uturns              # (N,)
    # empty_mask = ~has_any_weight  (not needed explicitly)

    # GPU batched triangle-inequality + parity checks on FACETS
    FACETS = CUBE_FACETS.to(device)                           # (12, 3)
    w1 = edge_weights[:, FACETS[:, 0]]                        # (N, 12)
    w2 = edge_weights[:, FACETS[:, 1]]                        # (N, 12)
    w3 = edge_weights[:, FACETS[:, 2]]                        # (N, 12)

    # Triangle inequality
    tri_valid = (w1 + w2 >= w3) & (w2 + w3 >= w1) & (w3 + w1 >= w2)  # (N, 12)

    # Parity check
    parity_ok = ((w1 + w2 + w3) % 2) == 0                    # (N, 12)

    # Arc counts (Normal Curve Theory)
    k12 = (w1 + w2 - w3) // 2                                # (N, 12)
    k23 = (w2 + w3 - w1) // 2
    k31 = (w3 + w1 - w2) // 2

    # Cubes that fail ANY facet check on the fast path are unsolvable
    all_valid = tri_valid.all(dim=1) & parity_ok.all(dim=1)   # (N,)
    # Mark fast-path cubes that fail GPU checks as unsolvable
    # (slow-path cubes may still succeed via U-turn enumeration)
    fast_gpu_fail = fast_path_mask & ~all_valid

    # ==================================================================
    # Phase 2: CPU/MP — build work items and dispatch
    # ==================================================================
    ew_np = edge_weights.cpu().numpy()                        # (N, 18)
    fw_np = face_weights.cpu().numpy()                        # (N, 12)
    fast_mask_np = fast_path_mask.cpu().numpy()
    slow_mask_np = slow_path_mask.cpu().numpy()
    fast_fail_np = fast_gpu_fail.cpu().numpy()

    # ------------------------------------------------------------------
    # W3 / O5: GPU fast-path — process fast-path cubes via batched GPU
    # adjacency build + lightweight CPU loop tracing on numpy arrays.
    # The GPU pre-checks (tri-ineq + parity) already classify cubes as
    # unsolvable via fast_gpu_fail; remaining fast cubes have a valid
    # arc system, so building adjacency and verifying degree==2 suffices.
    # Any cube with degree mismatch is reclassified UNSOLVABLE here.
    # Slow-path cubes still go through the CPU MP worker.
    # ------------------------------------------------------------------
    fastpath_gpu_results: Dict[int, Tuple[List[List[int]], int]] = {}
    if _cfg.S6_FASTPATH_GPU:
        fast_eligible = fast_path_mask & ~fast_gpu_fail   # (N,) bool
        fast_eligible_np = fast_eligible.cpu().numpy()
        fast_idx_t = torch.nonzero(fast_eligible, as_tuple=False).squeeze(1)
        M_fast = int(fast_idx_t.numel())
        if M_fast > 0:
            ew_fast = edge_weights[fast_idx_t].to(torch.int64)        # (M, 18)
            k12_fast = k12[fast_idx_t].to(torch.int64)                # (M, 12)
            k23_fast = k23[fast_idx_t].to(torch.int64)
            k31_fast = k31[fast_idx_t].to(torch.int64)
            (
                _ew_fast_np,
                point_offset_np,
                adj_np,
                degree_ok_np,
            ) = _fastpath_gpu_build_adjacency(
                ew_fast, k12_fast, k23_fast, k31_fast,
            )
            fast_idx_np = fast_idx_t.cpu().numpy()
            total_points_per_cube = point_offset_np[:, -1]
            for local_i in range(M_fast):
                cube_idx = int(fast_idx_np[local_i])
                tp = int(total_points_per_cube[local_i])
                if not bool(degree_ok_np[local_i]):
                    fastpath_gpu_results[cube_idx] = ([], CubeStatus.UNSOLVABLE)
                    continue
                loops = _fastpath_trace_loops_numpy(
                    point_offset_np[local_i], adj_np[local_i], tp,
                )
                fastpath_gpu_results[cube_idx] = (loops, CubeStatus.OK)

    # ------------------------------------------------------------------
    # W3 / O7: tensor-native work-item dispatch.
    # Replace per-cube Python iter with torch.nonzero + bulk numpy slicing.
    # When S6_FASTPATH_GPU is on, fast-path cubes are excluded from the
    # CPU worker queue (handled above); otherwise they go through workers.
    # ------------------------------------------------------------------
    if _cfg.S6_FASTPATH_GPU:
        worker_mask = (fast_path_mask | slow_path_mask) & ~fast_gpu_fail \
            & ~fast_path_mask
        # = slow_path_mask & ~fast_gpu_fail (fast_gpu_fail is a subset of fast_path_mask
        # so the second clause is moot, but kept for explicitness).
    else:
        worker_mask = (fast_path_mask | slow_path_mask) & ~fast_gpu_fail
    worker_idx_t = torch.nonzero(worker_mask, as_tuple=False).squeeze(1)
    worker_idx_np = worker_idx_t.cpu().numpy().astype(np.int64, copy=False)
    n_work = int(worker_idx_np.shape[0])
    if n_work > 0:
        # Bulk gather rows once; .tolist() on the whole row block is much
        # faster than per-cube .tolist() because the inner numpy loop is in C.
        ew_rows = ew_np[worker_idx_np].tolist()       # list[list[int]] length n_work
        fw_rows = fw_np[worker_idx_np].tolist()
        slow_flags = slow_mask_np[worker_idx_np].astype(bool, copy=False).tolist()
        work_items = [
            (int(worker_idx_np[k]), ew_rows[k], fw_rows[k], slow_flags[k])
            for k in range(n_work)
        ]
    else:
        work_items = []

    # Dispatch via temp Pool (fork-inherited) or run serially
    if num_workers > 1 and len(work_items) > 500:
        from multiprocessing import Pool as _Pool
        with _Pool(num_workers) as p:
            results = p.map(_s6_worker, work_items,
                            chunksize=max(1, len(work_items) // (num_workers * 4)))
    else:
        results = [_s6_worker(item) for item in work_items]

    # ==================================================================
    # Phase 3: Assembly — collect results, build CSR + uturn_assignment
    # ==================================================================

    # Initialize per-cube containers
    per_cube_loops: List[List[List[int]]] = [[] for _ in range(N)]
    per_cube_status = np.zeros(N, dtype=np.int32)  # default OK=0
    uturn_np = np.full((N, 12, 3), -1, dtype=np.int32)

    # Mark GPU-failed fast-path cubes (W3 / O7: numpy mask assignment).
    per_cube_status[fast_fail_np] = CubeStatus.UNSOLVABLE

    # Populate fast-path GPU results (W3 / O5)
    for cube_idx, (loops, status) in fastpath_gpu_results.items():
        per_cube_loops[cube_idx] = loops
        per_cube_status[cube_idx] = status

    # Populate from worker results
    for cube_idx, loops, status, assignment in results:
        per_cube_loops[cube_idx] = loops
        per_cube_status[cube_idx] = status
        if assignment is not None and status == CubeStatus.OK:
            for f_idx, (u1, u2, u3) in enumerate(assignment):
                uturn_np[cube_idx, f_idx, 0] = u1
                uturn_np[cube_idx, f_idx, 1] = u2
                uturn_np[cube_idx, f_idx, 2] = u3

    # ------------------------------------------------------------------
    # Pack into two-level CSR (W3 / O7: numpy cumsum-driven, avoids per-loop
    # list.append + extend on hot path).
    #   loop_cube_off[i] .. loop_cube_off[i+1]  → loops for cube i
    #   loop_edge_off[j] .. loop_edge_off[j+1]  → edges for loop j
    #   loop_edge_val[k]                          → edge index
    # ------------------------------------------------------------------
    # Step 1: per-cube loop count (vectorized via numpy)
    n_loops_per_cube = np.fromiter(
        (len(loops) for loops in per_cube_loops),
        dtype=np.int64, count=N,
    )
    cube_offsets_np = np.empty(N + 1, dtype=np.int64)
    cube_offsets_np[0] = 0
    np.cumsum(n_loops_per_cube, out=cube_offsets_np[1:])

    # Step 2: flatten loops into a single Python list-of-lists, then per-loop
    # length and concatenated edge values.
    flat_loops: List[List[int]] = []
    if any(per_cube_loops):
        for loops in per_cube_loops:
            if loops:
                flat_loops.extend(loops)

    n_edges_per_loop = np.fromiter(
        (len(lp) for lp in flat_loops),
        dtype=np.int64, count=len(flat_loops),
    )
    edge_offsets_np = np.empty(len(flat_loops) + 1, dtype=np.int64)
    edge_offsets_np[0] = 0
    np.cumsum(n_edges_per_loop, out=edge_offsets_np[1:])

    total_edges = int(edge_offsets_np[-1]) if edge_offsets_np.size > 0 else 0
    if total_edges > 0:
        # Concatenate all loop edge sequences in one numpy call.
        edge_vals_np = np.concatenate(
            [np.asarray(lp, dtype=np.int32) for lp in flat_loops]
        )
    else:
        edge_vals_np = np.zeros(0, dtype=np.int32)

    loop_cube_off = torch.from_numpy(cube_offsets_np).to(device)
    loop_edge_off = torch.from_numpy(edge_offsets_np).to(device)
    loop_edge_val = torch.from_numpy(edge_vals_np).to(device)
    status_tensor = torch.from_numpy(per_cube_status).to(device)
    uturn_tensor = torch.from_numpy(uturn_np).to(device)

    return _replace_fields(
        batch,
        loop_cube_off=loop_cube_off,
        loop_edge_off=loop_edge_off,
        loop_edge_val=loop_edge_val,
        status=status_tensor,
        uturn_assignment=uturn_tensor,
    )
