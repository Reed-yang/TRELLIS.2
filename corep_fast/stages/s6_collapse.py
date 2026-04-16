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

def s6_collapse(batch: CubeBatch, pool=None) -> CubeBatch:
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
    N = batch.num_cubes
    device = batch.device

    if N == 0:
        return batch

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

    work_items = []
    for i in range(N):
        if fast_fail_np[i]:
            # GPU already determined this cube is unsolvable — skip worker
            continue
        if fast_mask_np[i] or slow_mask_np[i]:
            work_items.append((
                i,
                ew_np[i].tolist(),
                fw_np[i].tolist(),
                bool(slow_mask_np[i]),
            ))

    # Dispatch to pool or run serially
    if pool is not None and work_items:
        results = pool.map_chunked(_s6_worker, work_items, chunk_size=500)
    else:
        results = [_s6_worker(item) for item in work_items]

    # ==================================================================
    # Phase 3: Assembly — collect results, build CSR + uturn_assignment
    # ==================================================================

    # Initialize per-cube containers
    per_cube_loops: List[List[List[int]]] = [[] for _ in range(N)]
    per_cube_status = np.zeros(N, dtype=np.int32)  # default OK=0
    uturn_np = np.full((N, 12, 3), -1, dtype=np.int32)

    # Mark GPU-failed fast-path cubes
    for i in range(N):
        if fast_fail_np[i]:
            per_cube_status[i] = CubeStatus.UNSOLVABLE

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
    # Pack into two-level CSR:
    #   loop_cube_off[i] .. loop_cube_off[i+1]  → loops for cube i
    #   loop_edge_off[j] .. loop_edge_off[j+1]  → edges for loop j
    #   loop_edge_val[k]                          → edge index
    # ------------------------------------------------------------------
    cube_offsets = [0]
    edge_offsets = [0]
    edge_vals: List[int] = []

    for loops in per_cube_loops:
        cube_offsets.append(cube_offsets[-1] + len(loops))
        for loop in loops:
            edge_offsets.append(edge_offsets[-1] + len(loop))
            edge_vals.extend(loop)

    loop_cube_off = torch.tensor(cube_offsets, dtype=torch.int64, device=device)
    loop_edge_off = torch.tensor(edge_offsets, dtype=torch.int64, device=device)
    loop_edge_val = torch.tensor(edge_vals, dtype=torch.int32, device=device) if edge_vals else \
        torch.zeros(0, dtype=torch.int32, device=device)
    status_tensor = torch.tensor(per_cube_status, dtype=torch.int32, device=device)
    uturn_tensor = torch.tensor(uturn_np, dtype=torch.int32, device=device)

    return _replace_fields(
        batch,
        loop_cube_off=loop_cube_off,
        loop_edge_off=loop_edge_off,
        loop_edge_val=loop_edge_val,
        status=status_tensor,
        uturn_assignment=uturn_tensor,
    )
