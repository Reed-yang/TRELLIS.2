"""
Stage 8: Collapse — Torch-accelerated mesh reconstruction from CoReP cube data.

Replaces custom/collapse.py with vectorized Torch operations. The critical
optimization is replacing Python dict/set vertex dedup with torch.unique,
and replacing per-edge Python iteration with batched sort + unique_consecutive.

Reference: spec section 6.8 (s8_collapse Torch strategy).

Public API:
    s8_collapse_to_ply(resolution, cube_data_list, output_filepath, merge_decimals)


===============================================================================
LOCAL-EDGE ENCODING CONVENTIONS — CRITICAL READ BEFORE TOUCHING s8 CODE
===============================================================================

There are TWO independent local-edge encoding conventions in this stage. Mixing
them is the root cause of the 4-cube vectorized-path divergence documented in
my-docs/20260415-corep-fast-stage2-analysis.md §4.1. Every gather that crosses
between the two MUST go through the conversion table at the bottom of this
docstring.

(A) `_EDGE_OFFSET_TABLE` convention — geometry-side / "owner-derivation"
    Used by:
      - `_EDGE_OFFSET_TABLE` (this file, defined below)
      - `compute_global_edge_keys` (computes the global edge ID per cube×edge)
      - `EdgeNeighborTable.neighbor_local_edges` (one slot per neighbor cube,
        the "local edge index" reported when that cube was discovered to
        contribute to this global edge through `_EDGE_OFFSET_TABLE`)
    Definition: index 0..11 over per-cube edges, listed in the order
                  X-bottom-front (0), Z-back-right (1),  X-bottom-back (2),
                  Z-back-left  (3), X-top-front  (4), Z-front-right (5),
                  X-top-back   (6), Z-front-left (7), Y-bottom-left (8),
                  Y-bottom-right (9), Y-top-right (10), Y-top-left (11).
    Each row of `_EDGE_OFFSET_TABLE` stores `(axis, dx, dy, dz)` describing
    *where* this local edge sits within the cube — e.g. axis=2 means a
    Z-aligned edge, (dx,dy,dz) is the corner offset (each in {0,1}).

(B) `custom/collapse.py::get_local_edge(dx, dy, dz)` convention — loop-side
    Used by:
      - `sorted_loops['loop']` entries inside `cube_data` (every value there
        is a "local edge index" produced by get_local_edge)
      - `_process_shared_edge_geometry` (the Python reference implementation)
      - `tensors.loop_edges_flat` (CSR-flat copy of sorted_loops['loop'])
    Definition: given an `edge_axis` (the axis of the SHARED global edge being
    processed) and `(dx, dy, dz)` = `cube_xyz - edge_min_xyz`, returns the
    local edge index 0..11 for THAT specific cube, but indexed by which
    *quadrant* (around the shared edge) the cube occupies.

Why they differ:
    Convention (A) is a pure per-cube enumeration. Convention (B) only emits
    edges that are *geometrically aligned* with `edge_axis`, so for a Y-axis
    shared edge it returns values in {1, 3, 5, 7} (the four Y-aligned
    Top/Bottom-Left/Right edges) — NEVER the {8,9,10,11} that convention (A)
    uses for the same Y-axis edges. The two encodings agree on edge axis but
    use different IDs for Y-axis edges.

Conversion table — given `edge_axis` (of the shared global edge) and the
`(dx, dy, dz)` of one neighbor cube relative to the edge's min corner, the
"loop convention" local edge id (for use against sorted_loops['loop']) is:

    edge_axis = 0 (X):  (dy, dz) → local_edge
        (0, 0) → 6  ('Top-Back')
        (1, 0) → 4  ('Top-Front')
        (0, 1) → 2  ('Bottom-Back')
        (1, 1) → 0  ('Bottom-Front')
    edge_axis = 1 (Y):  (dx, dz) → local_edge
        (0, 0) → 5  ('Top-Right')
        (1, 0) → 7  ('Top-Left')
        (0, 1) → 1  ('Bottom-Right')
        (1, 1) → 3  ('Bottom-Left')
    edge_axis = 2 (Z):  (dx, dy) → local_edge
        (0, 0) → 10 ('Back-Right')
        (1, 0) → 11 ('Back-Left')
        (0, 1) → 9  ('Front-Right')
        (1, 1) → 8  ('Front-Left')

Negative-direction edges (need rank flip):  {2, 6, 3, 7} per custom/collapse.py
"For X-axis edges 2,6 (-X direction) and Y-axis edges 3,7 (-Y direction),
 normalized_rank = (W - 1) - rank".

Where the conversion is currently applied:
    `process_geometry_vectorized` (this file, around line 891) re-derives the
    loop-convention local edge per (edge, neighbor-slot) using the table above
    BEFORE matching against `gathered_edges` from `tensors.loop_edges_flat`.
    Do NOT use `table.neighbor_local_edges` directly to index into loop edges.

UV-slot vs table-slot:
    `EdgeNeighborTable.neighbor_cube_ids` columns 0..3 follow the cyclic
    `_NEIGHBOR_OFFSETS` order. The Python reference `_process_shared_edge_geometry`
    indexes "uv slots" by `(dx,dy)/(dy,dz)/(dx,dz)` directly, with the mapping
    (0,0)→0, (1,0)→1, (1,1)→2, (0,1)→3. The vectorized path therefore also
    permutes table-slot → uv-slot before scatter. See `slot_id` derivation in
    `process_geometry_vectorized`.

If you add a new vectorized consumer of `tensors.loop_edges_flat` or
`tensors.loop_ranks_flat`, you MUST insert the same conversion. The 9
divergent edges observed historically (out of 16,980 4-cube edges on
icosphere@res=64) all came from forgetting this exact step.
===============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np


# ---------------------------------------------------------------------------
# CSR helpers for direct-tensor s8 path (M2 P1)
# ---------------------------------------------------------------------------

def _csr_expand_to_items(offsets: torch.Tensor, total: int) -> torch.Tensor:
    """Given CSR offsets (N+1,), return a (total,) tensor mapping each
    item to its group index.

    Equivalent to np.repeat(np.arange(N), np.diff(offsets)), but on GPU.

    Args:
        offsets: (N+1,) int64/int32 monotonic offsets, starting from 0.
        total: int, the length of the output (should equal offsets[-1]).

    Returns:
        (total,) int64 group index per item.
    """
    device = offsets.device
    if total <= 0:
        return torch.zeros(0, dtype=torch.int64, device=device)
    arange = torch.arange(total, dtype=offsets.dtype, device=device)
    # bucketize with right=True: arange[i] lies in [offsets[b], offsets[b+1])
    # offsets[1:] excludes the leading 0; bucketize returns b.
    # With right=True, a value equal to offsets[b+1] advances to group b+1,
    # matching CSR semantics where offsets[b] is inclusive, offsets[b+1] exclusive.
    return torch.bucketize(arange, offsets[1:], right=True).to(torch.int64)


def _derive_loop_component_point_gpu(batch) -> torch.Tensor:
    """Derive (L, 3) float32 loop_component_point from CubeBatch point data.

    Dict path does: for each loop in cube ci, take point_values[point_offsets[ci] + match_idx]
    if match_idx is valid, else fallback to point_values[point_offsets[ci]] (first point),
    else [0,0,0] if cube has no points.

    Direct path: fully GPU vectorized.
    """
    device = batch.device
    L = int(batch.loop_cube_off[-1].item())
    if L == 0:
        return torch.zeros((0, 3), dtype=torch.float32, device=device)

    # Map each loop to its owning cube (CSR inverse)
    loop_cube = _csr_expand_to_items(batch.loop_cube_off, L)  # (L,) int64

    match = batch.loop_point_match.to(torch.int64)     # (L,)
    cube_po_lo = batch.point_offsets[loop_cube]         # (L,)
    cube_po_hi = batch.point_offsets[loop_cube + 1]     # (L,)
    num_pts = cube_po_hi - cube_po_lo

    valid = (match >= 0) & (match < num_pts) & (num_pts > 0)
    global_idx_matched = cube_po_lo + match
    global_idx_fallback = cube_po_lo  # first point of cube
    empty = num_pts == 0
    global_idx = torch.where(
        valid, global_idx_matched,
        torch.where(empty, torch.zeros_like(cube_po_lo), global_idx_fallback),
    )

    P = batch.point_values.shape[0]
    if P == 0:
        return torch.zeros((L, 3), dtype=torch.float32, device=device)

    gathered = batch.point_values[global_idx.clamp(max=P - 1)]  # (L, 3)
    # Zero out empty-cube loops (match dict path semantics)
    return torch.where(
        empty.unsqueeze(1), torch.zeros_like(gathered), gathered,
    )


def _pad_ragged_loops_gpu(
    loop_edge_off: torch.Tensor,
    loop_edge_val: torch.Tensor,
    loop_edge_rank: torch.Tensor,
    max_loop_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad ragged CSR loop arrays to (L, max_loop_len) with -1, then flatten.

    Args:
        loop_edge_off: (L+1,) CSR offsets.
        loop_edge_val: (E,) int32 edge ids.
        loop_edge_rank: (E,) int32 ranks.
        max_loop_len: int, target padded width.

    Returns:
        edges_flat: (L * max_loop_len,) int32, -1 padded.
        ranks_flat: (L * max_loop_len,) int32, -1 padded.
    """
    device = loop_edge_val.device
    L = loop_edge_off.numel() - 1
    if L <= 0 or max_loop_len <= 0:
        return (
            torch.zeros((0,), dtype=torch.int32, device=device),
            torch.zeros((0,), dtype=torch.int32, device=device),
        )

    E = loop_edge_val.numel()
    if E == 0:
        edges_padded = torch.full((L, max_loop_len), -1, dtype=torch.int32, device=device)
        ranks_padded = edges_padded.clone()
        return edges_padded.reshape(-1), ranks_padded.reshape(-1)

    loop_id = _csr_expand_to_items(loop_edge_off, E)  # (E,) int64
    base = loop_edge_off[loop_id]
    pos = torch.arange(E, dtype=torch.int64, device=device) - base

    edges_padded = torch.full((L, max_loop_len), -1, dtype=torch.int32, device=device)
    ranks_padded = torch.full((L, max_loop_len), -1, dtype=torch.int32, device=device)
    keep = pos < max_loop_len
    edges_padded[loop_id[keep], pos[keep]] = loop_edge_val[keep]
    ranks_padded[loop_id[keep], pos[keep]] = loop_edge_rank[keep]

    return edges_padded.reshape(-1), ranks_padded.reshape(-1)


# ---------------------------------------------------------------------------
# CubeBatch → (vertices, faces) decoder entry point
# ---------------------------------------------------------------------------

def decode_from_cubebatch(
    batch: 'CubeBatch',
    merge_decimals: int = 5,
    use_direct_tensor: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode CubeBatch -> (vertices, faces).

    Args:
        batch: CubeBatch with all stages (s1-s7) populated.
        merge_decimals: Vertex welding precision.
        use_direct_tensor: If True, skip dict intermediate (M2 P1 optimization).
                           If None, use COREP_FAST_S8_DIRECT_TENSOR env var.

    Returns:
        vertices: (V, 3) float32
        faces: (F, 3) int32
    """
    if use_direct_tensor is None:
        from corep_fast.config import USE_DIRECT_TENSOR_S8
        use_direct_tensor = USE_DIRECT_TENSOR_S8

    if use_direct_tensor:
        tensors = _cubebatch_to_tensors_direct(batch)
        return _process_shared_edges_from_tensors(
            resolution=batch.resolution,
            tensors=tensors,
            batch=batch,
            merge_decimals=merge_decimals,
        )

    # Legacy dict path
    cube_data_list = _cubebatch_to_dicts(batch)
    return process_shared_edges_batch(
        resolution=batch.resolution,
        cube_data_list=cube_data_list,
        merge_decimals=merge_decimals,
        use_torch_path=True,
    )


def _cubebatch_to_dicts(batch: 'CubeBatch') -> list[dict]:
    """Convert CubeBatch to list[dict] format for s8 processing.

    All tensors are converted to numpy once upfront to avoid per-element
    .item() GPU sync overhead (275K syncs → 0 syncs).
    """
    from corep_fast.containers import CubeStatus

    N = batch.num_cubes
    result = []

    # Convert all tensors to numpy ONCE (single GPU→CPU transfer per tensor)
    ci_np = batch.cube_indices.cpu().numpy()
    ew_np = batch.edge_weights.cpu().numpy()
    status_np = batch.status.cpu().numpy()
    lco_np = batch.loop_cube_off.cpu().numpy()
    leo_np = batch.loop_edge_off.cpu().numpy()
    lev_np = batch.loop_edge_val.cpu().numpy()
    ler_np = batch.loop_edge_rank.cpu().numpy()
    lpm_np = batch.loop_point_match.cpu().numpy()
    po_np = batch.point_offsets.cpu().numpy()
    pv_np = batch.point_values.cpu().numpy()

    # Pre-convert whole arrays to Python types ONCE with bulk tolist().
    # np.tolist() is a C-optimized bulk conversion that produces native Python
    # ints/floats, ~10x faster than per-element int(x) list comprehension.
    ci_py = ci_np.tolist()           # list[list[int]]
    ew_py = ew_np.tolist()           # list[list[int]]
    status_py = status_np.tolist()   # list[int]
    lco_py = lco_np.tolist()         # list[int]
    leo_py = leo_np.tolist()         # list[int]
    lev_py = lev_np.tolist()         # list[int]
    ler_py = ler_np.tolist()         # list[int]
    lpm_py = lpm_np.tolist()         # list[int]
    po_py = po_np.tolist()           # list[int]
    pv_py = pv_np.tolist()           # list[list[float]]

    OK = int(CubeStatus.OK)

    for i in range(N):
        ci = tuple(ci_py[i])
        ew = ew_py[i]
        is_exception = status_py[i] != OK

        p_lo = po_py[i]
        p_hi = po_py[i + 1]
        comp_pts = pv_py[p_lo:p_hi]

        if is_exception:
            d = {
                'exception': True,
                'cube_indices': ci,
                'sorted_loops': [{'component_point': comp_pts[0] if comp_pts else [0, 0, 0]}],
            }
        else:
            l_lo = lco_py[i]
            l_hi = lco_py[i + 1]

            sorted_loops = []
            for li in range(l_lo, l_hi):
                e_lo = leo_py[li]
                e_hi = leo_py[li + 1]
                edges = lev_py[e_lo:e_hi]
                ranks = ler_py[e_lo:e_hi]

                match_idx = lpm_py[li]
                if match_idx >= 0 and (p_lo + match_idx) < p_hi:
                    cp = pv_py[p_lo + match_idx]
                else:
                    cp = comp_pts[0] if comp_pts else [0, 0, 0]

                sorted_loops.append({
                    'loop': edges,
                    'rank': ranks,
                    'component_point': cp,
                })

            d = {
                'cube_indices': ci,
                'edge_weights': ew,
                'sorted_loops': sorted_loops,
                'exception': False,
            }

        result.append(d)

    return result


# ---------------------------------------------------------------------------
# Edge offset table: for each local edge 0..11, the (axis, dx, dy, dz) offset
# applied to (ix, iy, iz) to produce the global edge coordinate.
#
# axis encoding: 0 = X, 1 = Y, 2 = Z
# Derived from custom/collapse.py::get_global_edge().
# ---------------------------------------------------------------------------

# (axis, dx, dy, dz) for local edges 0..11
_EDGE_OFFSET_TABLE = torch.tensor([
    # local 0: ('x', ix, iy, iz)       → axis=0, dx=0, dy=0, dz=0
    [0, 0, 0, 0],
    # local 1: ('z', ix+1, iy, iz)     → axis=2, dx=1, dy=0, dz=0
    [2, 1, 0, 0],
    # local 2: ('x', ix, iy, iz+1)     → axis=0, dx=0, dy=0, dz=1
    [0, 0, 0, 1],
    # local 3: ('z', ix, iy, iz)       → axis=2, dx=0, dy=0, dz=0
    [2, 0, 0, 0],
    # local 4: ('x', ix, iy+1, iz)     → axis=0, dx=0, dy=1, dz=0
    [0, 0, 1, 0],
    # local 5: ('z', ix+1, iy+1, iz)   → axis=2, dx=1, dy=1, dz=0
    [2, 1, 1, 0],
    # local 6: ('x', ix, iy+1, iz+1)   → axis=0, dx=0, dy=1, dz=1
    [0, 0, 1, 1],
    # local 7: ('z', ix, iy+1, iz)     → axis=2, dx=0, dy=1, dz=0
    [2, 0, 1, 0],
    # local 8: ('y', ix, iy, iz)       → axis=1, dx=0, dy=0, dz=0
    [1, 0, 0, 0],
    # local 9: ('y', ix+1, iy, iz)     → axis=1, dx=1, dy=0, dz=0
    [1, 1, 0, 0],
    # local 10: ('y', ix+1, iy, iz+1)  → axis=1, dx=1, dy=0, dz=1
    [1, 1, 0, 1],
    # local 11: ('y', ix, iy, iz+1)    → axis=1, dx=0, dy=0, dz=1
    [1, 0, 0, 1],
], dtype=torch.int32)  # (12, 4)


# ---------------------------------------------------------------------------
# Neighbor offset table: for each axis, the 4 neighbor cube offsets.
# From custom/collapse.py::get_surrounding_cubes().
#
# axis X: (nx, ny, nz), (nx, ny-1, nz), (nx, ny-1, nz-1), (nx, ny, nz-1)
#   → offsets from (nx, ny, nz): (0,0,0), (0,-1,0), (0,-1,-1), (0,0,-1)
# axis Y: (nx, ny, nz), (nx-1, ny, nz), (nx-1, ny, nz-1), (nx, ny, nz-1)
#   → offsets: (0,0,0), (-1,0,0), (-1,0,-1), (0,0,-1)
# axis Z: (nx, ny, nz), (nx-1, ny, nz), (nx-1, ny-1, nz), (nx, ny-1, nz)
#   → offsets: (0,0,0), (-1,0,0), (-1,-1,0), (0,-1,0)
# ---------------------------------------------------------------------------

_NEIGHBOR_OFFSETS = torch.tensor([
    # axis 0 (X)
    [[0, 0, 0], [0, -1, 0], [0, -1, -1], [0, 0, -1]],
    # axis 1 (Y)
    [[0, 0, 0], [-1, 0, 0], [-1, 0, -1], [0, 0, -1]],
    # axis 2 (Z)
    [[0, 0, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0]],
], dtype=torch.int32)  # (3, 4, 3)


# ---------------------------------------------------------------------------
# Reverse local edge table: for each (axis, neighbor_position 0..3), which
# local edge of that neighbor maps to this shared global edge.
# Derived from custom/collapse.py::get_local_edge() inverted.
#
# For axis X (edge_axis=0):
#   neighbor (dy=0, dz=0) → local_edge 6 (Top-Back)      — position 0 in _NEIGHBOR_OFFSETS
#                            Wait, the offsets for axis X are (0,0,0), (0,-1,0), (0,-1,-1), (0,0,-1)
#   Position 0: offset (0,0,0) = (nx, ny, nz). In get_local_edge: dy=ny-min_y=1, dz=nz-min_z=1
#     → axis X, dy=1, dz=1 → local_edge 0 (Bottom-Front)
#   Position 1: offset (0,-1,0) = (nx, ny-1, nz). dy=ny-1-min_y=0, dz=nz-min_z=1
#     → axis X, dy=0, dz=1 → local_edge 2 (Bottom-Back)
#   Position 2: offset (0,-1,-1) = (nx, ny-1, nz-1). dy=0, dz=0
#     → axis X, dy=0, dz=0 → local_edge 6 (Top-Back)
#   Position 3: offset (0,0,-1) = (nx, ny, nz-1). dy=1, dz=0
#     → axis X, dy=1, dz=0 → local_edge 4 (Top-Front)
#
# For axis Y (edge_axis=1):
#   Position 0: (0,0,0) → dx=1, dz=1 → local_edge 3 (Bottom-Left)
#   Position 1: (-1,0,0) → dx=0, dz=1 → local_edge 1 (Bottom-Right)
#   Position 2: (-1,0,-1) → dx=0, dz=0 → local_edge 5 (Top-Right)
#   Position 3: (0,0,-1) → dx=1, dz=0 → local_edge 7 (Top-Left)
#
# For axis Z (edge_axis=2):
#   Position 0: (0,0,0) → dx=1, dy=1 → local_edge 8 (Front-Left)
#   Position 1: (-1,0,0) → dx=0, dy=1 → local_edge 9 (Front-Right)
#   Position 2: (-1,-1,0) → dx=0, dy=0 → local_edge 10 (Back-Right)
#   Position 3: (0,-1,0) → dx=1, dy=0 → local_edge 11 (Back-Left)
# ---------------------------------------------------------------------------

_REVERSE_LOCAL_EDGE = torch.tensor([
    # axis 0 (X): positions 0,1,2,3
    [0, 2, 6, 4],
    # axis 1 (Y): positions 0,1,2,3
    [3, 1, 5, 7],
    # axis 2 (Z): positions 0,1,2,3
    [8, 9, 10, 11],
], dtype=torch.int32)  # (3, 4)


# ---------------------------------------------------------------------------
# Negative-direction local edges (ranks need flipping).
# From custom/collapse.py: "Edges 2, 6 (-X) and 3, 7 (-Y) go in the
# negative direction."
# ---------------------------------------------------------------------------

_NEGATIVE_DIRECTION_EDGES = frozenset({2, 6, 3, 7})


# ---------------------------------------------------------------------------
# EdgeNeighborTable: CSR-like table mapping unique edges → their neighbor cubes
# ---------------------------------------------------------------------------

@dataclass
class EdgeNeighborTable:
    """
    Mapping from unique global edges to their neighboring occupied cubes.

    Fields:
        neighbor_counts: (E,) int32 — number of occupied neighbors per unique edge
        neighbor_cube_ids: (E, 4) int32 — cube indices (into cube_indices array),
            padded with -1 for absent neighbors
        neighbor_positions: (E, 4) int32 — position 0..3 in the cyclic neighbor order,
            padded with -1
        neighbor_local_edges: (E, 4) int32 — local edge index (0..11) for each neighbor,
            padded with -1
        edge_axes: (E,) int32 — axis (0=X, 1=Y, 2=Z) of each unique edge
    """
    neighbor_counts: torch.Tensor
    neighbor_cube_ids: torch.Tensor
    neighbor_positions: torch.Tensor
    neighbor_local_edges: torch.Tensor
    edge_axes: torch.Tensor


# ---------------------------------------------------------------------------
# CubeDataTensors: dense tensor representation of cube_data_list
# ---------------------------------------------------------------------------

@dataclass
class CubeDataTensors:
    """Dense tensor representation of cube_data_list for vectorized processing.

    All per-cube and per-loop data is flattened into tensors with CSR offsets
    for ragged loop arrays.
    """
    cube_indices: torch.Tensor         # (N, 3) int32
    cube_edge_weights: torch.Tensor    # (N, 18) int32
    cube_exception: torch.Tensor       # (N,) bool
    cube_num_components: torch.Tensor  # (N,) int32
    loop_cube_offsets: torch.Tensor    # (N+1,) int64, cumulative loop count per cube
    loop_component_point: torch.Tensor # (L, 3) float32
    max_loop_len: int                  # max length of any loop (edge count)
    loop_edges_flat: torch.Tensor      # (L * max_loop_len,) int32, -1 padding
    loop_ranks_flat: torch.Tensor      # (L * max_loop_len,) int32, -1 padding


def _cube_data_to_tensors(
    cube_data_list: list[dict],
    device: torch.device,
) -> CubeDataTensors:
    """Convert list-of-dict cube data to dense GPU tensors.

    One-time conversion at the start of s8. After this, all processing is
    vectorized tensor operations.

    Args:
        cube_data_list: list of cube dicts with cube_indices, sorted_loops,
                        edge_weights, exception, num_components fields.
        device: torch device for output tensors.

    Returns:
        CubeDataTensors with all fields populated.
    """
    N = len(cube_data_list)
    if N == 0:
        return CubeDataTensors(
            cube_indices=torch.zeros((0, 3), dtype=torch.int32, device=device),
            cube_edge_weights=torch.zeros((0, 18), dtype=torch.int32, device=device),
            cube_exception=torch.zeros((0,), dtype=torch.bool, device=device),
            cube_num_components=torch.zeros((0,), dtype=torch.int32, device=device),
            loop_cube_offsets=torch.zeros((1,), dtype=torch.int64, device=device),
            loop_component_point=torch.zeros((0, 3), dtype=torch.float32, device=device),
            max_loop_len=1,
            loop_edges_flat=torch.zeros((0,), dtype=torch.int32, device=device),
            loop_ranks_flat=torch.zeros((0,), dtype=torch.int32, device=device),
        )

    cube_indices_list: list[list[int]] = []
    cube_edge_weights_list: list[list[int]] = []
    cube_exception_list: list[bool] = []
    cube_num_components_list: list[int] = []
    loop_points_list: list[list[float]] = []
    loop_edges_nested: list[list[int]] = []
    loop_ranks_nested: list[list[int]] = []
    loops_per_cube: list[int] = []
    max_loop_len = 1

    for data in cube_data_list:
        idx = data.get('cube_indices', (0, 0, 0))
        if isinstance(idx, (list, tuple)) and len(idx) >= 3:
            cube_indices_list.append([int(idx[0]), int(idx[1]), int(idx[2])])
        else:
            cube_indices_list.append([0, 0, 0])

        ew = list(data.get('edge_weights', []) or [])
        if len(ew) < 18:
            ew = ew + [0] * (18 - len(ew))
        else:
            ew = ew[:18]
        cube_edge_weights_list.append([int(x) for x in ew])

        cube_exception_list.append(bool(data.get('exception', False)))
        cube_num_components_list.append(int(data.get('num_components', 0)))

        sorted_loops = data.get('sorted_loops', []) or []
        loops_per_cube.append(len(sorted_loops))
        for loop in sorted_loops:
            pt = loop.get('component_point', [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
            loop_points_list.append([float(pt[0]), float(pt[1]), float(pt[2])])
            edges = list(loop.get('loop', []) or [])
            ranks = list(loop.get('rank', []) or [])
            # Align ranks to edges length
            if len(ranks) < len(edges):
                ranks = ranks + [-1] * (len(edges) - len(ranks))
            elif len(ranks) > len(edges):
                ranks = ranks[:len(edges)]
            loop_edges_nested.append([int(x) for x in edges])
            loop_ranks_nested.append([int(x) for x in ranks])
            if len(edges) > max_loop_len:
                max_loop_len = len(edges)

    L = len(loop_points_list)
    loop_edges_padded = np.full((L, max_loop_len), -1, dtype=np.int32)
    loop_ranks_padded = np.full((L, max_loop_len), -1, dtype=np.int32)
    for i, (edges, ranks) in enumerate(zip(loop_edges_nested, loop_ranks_nested)):
        k = min(len(edges), max_loop_len)
        loop_edges_padded[i, :k] = edges[:k]
        loop_ranks_padded[i, :k] = ranks[:k]

    offsets = np.zeros(N + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(loops_per_cube)

    return CubeDataTensors(
        cube_indices=torch.tensor(cube_indices_list, dtype=torch.int32, device=device),
        cube_edge_weights=torch.tensor(cube_edge_weights_list, dtype=torch.int32, device=device),
        cube_exception=torch.tensor(cube_exception_list, dtype=torch.bool, device=device),
        cube_num_components=torch.tensor(cube_num_components_list, dtype=torch.int32, device=device),
        loop_cube_offsets=torch.from_numpy(offsets).to(device),
        loop_component_point=(
            torch.tensor(loop_points_list, dtype=torch.float32, device=device)
            if L > 0 else torch.zeros((0, 3), dtype=torch.float32, device=device)
        ),
        max_loop_len=max_loop_len,
        loop_edges_flat=(
            torch.from_numpy(loop_edges_padded.reshape(-1)).to(device)
            if L > 0 else torch.zeros((0,), dtype=torch.int32, device=device)
        ),
        loop_ranks_flat=(
            torch.from_numpy(loop_ranks_padded.reshape(-1)).to(device)
            if L > 0 else torch.zeros((0,), dtype=torch.int32, device=device)
        ),
    )


def _cubebatch_to_tensors_direct(batch) -> 'CubeDataTensors':
    """Directly convert CubeBatch CSR tensors to CubeDataTensors on GPU.

    Bypasses the list[dict] intermediate that requires 275K Python iters.
    All operations are vectorized; only 1 scalar sync (for max_loop_len).

    Expected runtime: <50ms at res=256 (vs ~3.6s via dict path).

    Args:
        batch: CubeBatch with all stages (s1-s7) populated.

    Returns:
        CubeDataTensors equivalent to _cube_data_to_tensors(_cubebatch_to_dicts(batch)).
    """
    device = batch.device
    N = batch.num_cubes

    if N == 0:
        return CubeDataTensors(
            cube_indices=torch.zeros((0, 3), dtype=torch.int32, device=device),
            cube_edge_weights=torch.zeros((0, 18), dtype=torch.int32, device=device),
            cube_exception=torch.zeros((0,), dtype=torch.bool, device=device),
            cube_num_components=torch.zeros((0,), dtype=torch.int32, device=device),
            loop_cube_offsets=torch.zeros((1,), dtype=torch.int64, device=device),
            loop_component_point=torch.zeros((0, 3), dtype=torch.float32, device=device),
            max_loop_len=1,
            loop_edges_flat=torch.zeros((0,), dtype=torch.int32, device=device),
            loop_ranks_flat=torch.zeros((0,), dtype=torch.int32, device=device),
        )

    # max_loop_len (single scalar GPU->CPU sync)
    if batch.loop_edge_off.numel() <= 1:
        max_loop_len = 1
    else:
        lengths = batch.loop_edge_off[1:] - batch.loop_edge_off[:-1]
        if lengths.numel() == 0:
            max_loop_len = 1
        else:
            max_loop_len = max(int(lengths.max().item()), 1)

    edges_flat, ranks_flat = _pad_ragged_loops_gpu(
        batch.loop_edge_off, batch.loop_edge_val, batch.loop_edge_rank,
        max_loop_len,
    )
    loop_component_point = _derive_loop_component_point_gpu(batch)

    cube_exception = batch.status.ne(0)
    # Match dict path semantics: _cubebatch_to_dicts does not emit edge_weights
    # for exception cubes, so _cube_data_to_tensors zero-fills them. Preserve
    # that behavior here for parity.
    cube_edge_weights = torch.where(
        cube_exception.unsqueeze(1),
        torch.zeros_like(batch.edge_weights),
        batch.edge_weights,
    )
    # Same for num_components: dict path never emits num_components, so the
    # tensor path reads default 0. We match that for byte-level parity.
    cube_num_components = torch.zeros_like(batch.num_components)

    return CubeDataTensors(
        cube_indices=batch.cube_indices,
        cube_edge_weights=cube_edge_weights,
        cube_exception=cube_exception,
        cube_num_components=cube_num_components,
        loop_cube_offsets=batch.loop_cube_off,
        loop_component_point=loop_component_point,
        max_loop_len=max_loop_len,
        loop_edges_flat=edges_flat,
        loop_ranks_flat=ranks_flat,
    )


def compute_global_edge_keys(
    cube_indices: torch.Tensor,
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For every occupied cube, compute the 12 global edge keys.

    Each global edge is encoded as a single int64:
        key = axis * R^3 + nx * R^2 + ny * R + nz
    where (axis, nx, ny, nz) is the global edge identifier and R = resolution + 2
    (to ensure non-negative coordinates after offsets).

    Args:
        cube_indices: (N, 3) int32 — (ix, iy, iz) per cube.
        resolution: int — grid resolution.

    Returns:
        keys: (N*12,) int64 — global edge key per (cube, local_edge) pair.
        cube_ids: (N*12,) int32 — which cube each entry belongs to.
        local_ids: (N*12,) int32 — which local edge (0..11).
    """
    device = cube_indices.device
    N = cube_indices.shape[0]
    table = _EDGE_OFFSET_TABLE.to(device)  # (12, 4)

    # Expand cube_indices: (N, 1, 3) + offsets (1, 12, 3) → (N, 12, 3)
    ix = cube_indices[:, 0:1].to(torch.int64)  # (N, 1)
    iy = cube_indices[:, 1:2].to(torch.int64)
    iz = cube_indices[:, 2:3].to(torch.int64)

    # offsets for each local edge
    axes = table[:, 0].to(torch.int64)    # (12,)
    dx = table[:, 1].to(torch.int64)      # (12,)
    dy = table[:, 2].to(torch.int64)
    dz = table[:, 3].to(torch.int64)

    # Global edge coordinates: (N, 12)
    nx = ix + dx.unsqueeze(0)   # (N, 12)
    ny = iy + dy.unsqueeze(0)
    nz = iz + dz.unsqueeze(0)
    ax = axes.unsqueeze(0).expand(N, 12)  # (N, 12)

    # Encode as int64 key. Use R = resolution + 2 for headroom.
    R = int(resolution) + 2
    keys = ax * (R * R * R) + nx * (R * R) + ny * R + nz  # (N, 12)

    # Flatten
    keys = keys.reshape(-1)                    # (N*12,)

    # cube_ids: which cube each entry belongs to
    cube_ids = torch.arange(N, dtype=torch.int32, device=device).unsqueeze(1).expand(N, 12).reshape(-1)

    # local_ids: which local edge
    local_ids = torch.arange(12, dtype=torch.int32, device=device).unsqueeze(0).expand(N, 12).reshape(-1)

    return keys, cube_ids, local_ids


def enumerate_unique_edges(
    keys: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Deduplicate global edge keys.

    Args:
        keys: (M,) int64 — all global edge keys from compute_global_edge_keys.

    Returns:
        unique_keys: (E,) int64 — sorted unique edge keys.
        edge_id_per_entry: (M,) int64 — maps each input entry to its unique edge index.
    """
    sorted_keys, sort_idx = torch.sort(keys)
    unique_keys, inverse = torch.unique_consecutive(sorted_keys, return_inverse=True)

    # Map back to original order
    edge_id_per_entry = torch.empty_like(keys, dtype=torch.int64)
    edge_id_per_entry[sort_idx] = inverse.to(torch.int64)

    return unique_keys, edge_id_per_entry


def build_edge_neighbor_table(
    edge_id_per_entry: torch.Tensor,
    cube_ids: torch.Tensor,
    local_ids: torch.Tensor,
    num_unique_edges: int,
) -> EdgeNeighborTable:
    """
    Build a table mapping each unique edge to its occupied neighbor cubes.

    For each unique edge, up to 4 cubes may share it. We record which cubes,
    what their local edge index is, and their position (0..3) in the cyclic
    neighbor ordering.

    Args:
        edge_id_per_entry: (M,) int64 — unique edge index per entry.
        cube_ids: (M,) int32 — cube index per entry.
        local_ids: (M,) int32 — local edge (0..11) per entry.
        num_unique_edges: int — total number of unique edges.

    Returns:
        EdgeNeighborTable with fields filled.
    """
    device = edge_id_per_entry.device
    E = num_unique_edges
    M = edge_id_per_entry.shape[0]

    if M == 0 or E == 0:
        return EdgeNeighborTable(
            neighbor_counts=torch.zeros(E, dtype=torch.int32, device=device),
            neighbor_cube_ids=torch.full((E, 4), -1, dtype=torch.int32, device=device),
            neighbor_positions=torch.full((E, 4), -1, dtype=torch.int32, device=device),
            neighbor_local_edges=torch.full((E, 4), -1, dtype=torch.int32, device=device),
            edge_axes=torch.zeros(E, dtype=torch.int32, device=device),
        )

    # Sort entries by edge_id (stable sort preserves insertion order for tie-breaking)
    sort_idx = torch.argsort(edge_id_per_entry, stable=True)
    sorted_edge_ids = edge_id_per_entry[sort_idx]
    sorted_cube_ids = cube_ids[sort_idx]
    sorted_local_ids = local_ids[sort_idx]

    # Count entries per edge
    neighbor_counts = torch.zeros(E, dtype=torch.int32, device=device)
    ones = torch.ones(M, dtype=torch.int32, device=device)
    neighbor_counts.scatter_add_(0, sorted_edge_ids.to(torch.int64), ones)

    # CSR offsets
    offsets = torch.zeros(E + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(neighbor_counts.to(torch.int64), dim=0)

    # For each entry in sorted order, slot within its group
    entry_pos = torch.arange(M, dtype=torch.int64, device=device)
    slot_per_entry = entry_pos - offsets[sorted_edge_ids.to(torch.int64)]
    valid = slot_per_entry < 4

    # Scatter into (E, 4) tables
    neighbor_cube_ids = torch.full((E, 4), -1, dtype=torch.int32, device=device)
    neighbor_local_edges = torch.full((E, 4), -1, dtype=torch.int32, device=device)

    v_edge = sorted_edge_ids[valid].to(torch.int64)
    v_slot = slot_per_entry[valid]
    flat_idx = v_edge * 4 + v_slot
    neighbor_cube_ids.view(-1).scatter_(0, flat_idx, sorted_cube_ids[valid])
    neighbor_local_edges.view(-1).scatter_(0, flat_idx, sorted_local_ids[valid])

    # Derive edge axes from first entry of each group
    first_entry_pos = offsets[:-1].clamp(max=M - 1)
    first_local_edges = sorted_local_ids[first_entry_pos]
    edge_offset = _EDGE_OFFSET_TABLE.to(device)
    edge_axes = edge_offset[first_local_edges.to(torch.int64), 0].to(torch.int32)

    # Compute neighbor_positions via _REVERSE_LOCAL_EDGE lookup
    rev = _REVERSE_LOCAL_EDGE.to(device)  # (3, 4)
    axis_per_slot = edge_axes.unsqueeze(1).expand(E, 4).to(torch.int64)
    local_per_slot = neighbor_local_edges.to(torch.int64)
    candidates = rev[axis_per_slot]  # (E, 4, 4)
    matches = candidates == local_per_slot.unsqueeze(2)
    positions = matches.to(torch.int32).argmax(dim=2)
    positions = torch.where(
        neighbor_local_edges >= 0,
        positions,
        torch.full_like(positions, -1),
    )

    return EdgeNeighborTable(
        neighbor_counts=neighbor_counts,
        neighbor_cube_ids=neighbor_cube_ids,
        neighbor_positions=positions,
        neighbor_local_edges=neighbor_local_edges,
        edge_axes=edge_axes,
    )


def compute_edge_ownership(
    table: EdgeNeighborTable,
    cube_indices: torch.Tensor,
) -> torch.Tensor:
    """
    For each unique edge, determine which cube owns it (lexicographic minimum).

    Custom/ uses: min(active_neighbors) == idx, comparing tuples lexicographically.
    We replicate this by computing cube_hash = ix * R^2 + iy * R + iz for each
    neighbor and taking argmin.

    Args:
        table: EdgeNeighborTable from build_edge_neighbor_table.
        cube_indices: (N, 3) int32.

    Returns:
        owner_cube_id: (E,) int32 — cube index of the owner for each unique edge.
            -1 if edge has no neighbors (shouldn't happen for occupied cubes).
    """
    device = cube_indices.device
    E = table.neighbor_counts.shape[0]
    R = int(cube_indices.max().item()) + 2

    # Compute hash for all cubes
    cube_hash = (
        cube_indices[:, 0].to(torch.int64) * R * R
        + cube_indices[:, 1].to(torch.int64) * R
        + cube_indices[:, 2].to(torch.int64)
    )  # (N,)

    # For each edge's neighbor slots, look up hash. Pad with MAX for absent slots.
    MAX_HASH = torch.iinfo(torch.int64).max
    neighbor_hashes = torch.full((E, 4), MAX_HASH, dtype=torch.int64, device=device)

    # Fill in valid slots
    valid = table.neighbor_cube_ids >= 0  # (E, 4)
    valid_cids = table.neighbor_cube_ids.clamp(min=0).to(torch.int64)  # (E, 4)
    neighbor_hashes[valid] = cube_hash[valid_cids[valid]]

    # Argmin per edge
    min_slot = neighbor_hashes.argmin(dim=1)  # (E,)
    owner_cube_id = torch.gather(table.neighbor_cube_ids, 1, min_slot.unsqueeze(1)).squeeze(1)

    return owner_cube_id


# ---------------------------------------------------------------------------
# Vectorized geometry processing (Stage 2 Task 3)
# ---------------------------------------------------------------------------

def process_geometry_vectorized(
    table: EdgeNeighborTable,
    tensors: CubeDataTensors,
    device: torch.device,
    max_rank: int = 16,
    exc_pt_per_cube_override: Optional[torch.Tensor] = None,
    exc_present_per_cube_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Vectorized geometry processing: emit fan triangles for all owned edges.

    Handles exception cube wildcards. Does NOT handle conditional promotion
    (complex edges are handled by the Python fallback path in later tasks).

    Args:
        table: EdgeNeighborTable with edges filtered to neighbor_counts >= 2
        tensors: CubeDataTensors with cube and loop data
        device: torch device
        max_rank: max normalized rank to consider
        exc_pt_per_cube_override: Optional (N, 3) float32 tensor giving the
            per-cube first point used for exception-wildcard fallback. When
            provided (direct-tensor path), Step 9 gathers the wildcard point
            from this per-cube tensor instead of `tensors.loop_component_point`
            (which in the dict path holds a virtual first-loop's point; in the
            direct path there is no such virtual loop).
        exc_present_per_cube_override: Optional (N,) bool tensor marking which
            cubes have a valid exception wildcard point. In the dict path the
            virtual first loop guarantees `num_loops >= 1` for any exception
            cube, but in the direct-tensor path exception cubes have
            `num_loops == 0` even though their wildcard point is available
            via `exc_pt_per_cube_override`. Pass this mask to relax the
            `num_loops > 0` gating in Step 9 / rank-0 synthesis. When None,
            the legacy `num_loops > 0` gate applies.

    Returns:
        (T*3, 3) float64 tensor of triangle vertex coordinates.
        Every 3 consecutive rows form one triangle.
    """
    E = table.neighbor_counts.shape[0]
    if E == 0 or tensors.cube_indices.shape[0] == 0:
        return torch.zeros((0, 3), dtype=torch.float64, device=device)

    # --- Step 1: Gather neighbor info per edge ---
    n_cube = table.neighbor_cube_ids           # (E, 4) int32, -1 for missing
    valid_slot = n_cube >= 0                   # (E, 4) bool
    n_cube_safe = n_cube.clamp(min=0).to(torch.int64)

    # --- Step 1b: Re-derive per-slot local edge and uv slot using collapse.py convention ---
    # table.neighbor_local_edges stores _EDGE_OFFSET_TABLE convention (geometry),
    # but sorted_loops['loop'] entries use collapse.py::process_shared_edge_geometry
    # convention, where for Y-axis edges local_edge ∈ {5, 7, 1, 3} (not 8-11).
    # We recompute per-slot local_edge from (dx, dy, dz) = cube_xyz - edge_min_xyz.
    cube_xyz = tensors.cube_indices.to(torch.int64)           # (N, 3)
    # neighbor cube positions per slot, with -1 slots masked to a large sentinel
    slot_xyz = cube_xyz[n_cube_safe]                          # (E, 4, 3) int64
    SENTINEL_MAX = torch.iinfo(torch.int64).max
    masked_xyz = torch.where(
        valid_slot.unsqueeze(2),
        slot_xyz,
        torch.full_like(slot_xyz, SENTINEL_MAX),
    )
    edge_min = masked_xyz.min(dim=1).values                   # (E, 3) int64
    dxyz = torch.where(
        valid_slot.unsqueeze(2),
        slot_xyz - edge_min.unsqueeze(1),
        torch.zeros_like(slot_xyz),
    )                                                         # (E, 4, 3) int64, values in {0, 1}
    dx = dxyz[..., 0]
    dy = dxyz[..., 1]
    dz = dxyz[..., 2]

    # collapse.py::get_local_edge(dx, dy, dz) table:
    #   X axis (edge_axis=0): (dy, dz) → 6, 4, 2, 0 for (0,0),(1,0),(0,1),(1,1)
    #   Y axis (edge_axis=1): (dx, dz) → 5, 7, 1, 3 for (0,0),(1,0),(0,1),(1,1)
    #   Z axis (edge_axis=2): (dx, dy) → 10, 11, 9, 8 for (0,0),(1,0),(0,1),(1,1)
    edge_axes = table.edge_axes.to(torch.int64)               # (E,)
    ax = edge_axes.unsqueeze(1).expand(E, 4)                  # (E, 4)
    # Per-axis mapping. Compute all three then select by axis.
    # X axis: index by (dy, dz)
    x_map = torch.tensor([[6, 2], [4, 0]], dtype=torch.int64, device=device)  # [dy][dz]
    # Y axis: index by (dx, dz)
    y_map = torch.tensor([[5, 1], [7, 3]], dtype=torch.int64, device=device)  # [dx][dz]
    # Z axis: index by (dx, dy)
    z_map = torch.tensor([[10, 9], [11, 8]], dtype=torch.int64, device=device)  # [dx][dy]
    local_x = x_map[dy, dz]
    local_y = y_map[dx, dz]
    local_z = z_map[dx, dy]
    n_loc = torch.where(
        ax == 0, local_x,
        torch.where(ax == 1, local_y, local_z),
    ).to(torch.int32)                                          # (E, 4)
    # Mask invalid slots with -1
    n_loc = torch.where(valid_slot, n_loc, torch.full_like(n_loc, -1))

    # uv slot_id: matches Python's neighbors = [(0,0), (1,0), (1,1), (0,1)]
    #   X axis: uv = (dy, dz)
    #   Y axis: uv = (dx, dz)
    #   Z axis: uv = (dx, dy)
    # uv → slot_id lookup: (0,0)->0, (1,0)->1, (1,1)->2, (0,1)->3
    uv_slot_map = torch.tensor([[0, 3], [1, 2]], dtype=torch.int64, device=device)  # [u][v]
    u_x, v_x = dy, dz
    u_y, v_y = dx, dz
    u_z, v_z = dx, dy
    slot_x = uv_slot_map[u_x, v_x]
    slot_y = uv_slot_map[u_y, v_y]
    slot_z = uv_slot_map[u_z, v_z]
    slot_id = torch.where(
        ax == 0, slot_x,
        torch.where(ax == 1, slot_y, slot_z),
    ).to(torch.int64)                                          # (E, 4)

    # --- Step 2: Per-neighbor exception status + edge weight ---
    cube_exc_flat = tensors.cube_exception     # (N,)
    cube_ew_flat = tensors.cube_edge_weights   # (N, 18)
    cube_exc = cube_exc_flat[n_cube_safe] & valid_slot   # (E, 4)

    n_loc_safe = n_loc.clamp(min=0).to(torch.int64)      # (E, 4)
    cube_ew = cube_ew_flat[n_cube_safe]                  # (E, 4, 18)
    W_at_edge = torch.gather(cube_ew, 2, n_loc_safe.unsqueeze(2)).squeeze(2)  # (E, 4)

    # --- Step 3: Gather loop ranges per neighbor cube ---
    loop_start = tensors.loop_cube_offsets[n_cube_safe]       # (E, 4) int64
    loop_end = tensors.loop_cube_offsets[n_cube_safe + 1]     # (E, 4) int64
    num_loops = (loop_end - loop_start).clamp(min=0)          # (E, 4)
    # Zero out missing slots
    num_loops = torch.where(valid_slot, num_loops, torch.zeros_like(num_loops))
    max_loops = int(num_loops.max().item()) if E > 0 else 0

    if max_loops == 0:
        # No loops in any cube — no crossings possible. Still need to handle exceptions
        # but without any rank presence, exceptions alone don't form full groups of 4.
        return torch.zeros((0, 3), dtype=torch.float64, device=device)

    # Flat loop indices: (E, 4, max_loops)
    loop_range = torch.arange(max_loops, device=device).view(1, 1, max_loops)
    flat_loop_idx = loop_start.unsqueeze(2) + loop_range      # (E, 4, max_loops)
    in_range = loop_range < num_loops.unsqueeze(2)            # (E, 4, max_loops)
    L_total = tensors.loop_component_point.shape[0]
    flat_loop_idx_safe = flat_loop_idx.clamp(min=0, max=max(L_total - 1, 0))

    # --- Step 4: Gather loop edges and ranks ---
    K = tensors.max_loop_len
    if L_total > 0:
        loop_edges_2d = tensors.loop_edges_flat.view(-1, K)   # (L, K)
        loop_ranks_2d = tensors.loop_ranks_flat.view(-1, K)   # (L, K)
    else:
        loop_edges_2d = torch.zeros((1, K), dtype=torch.int32, device=device)
        loop_ranks_2d = torch.zeros((1, K), dtype=torch.int32, device=device)

    # gathered_edges / gathered_ranks: (E, 4, max_loops, K)
    gathered_edges = loop_edges_2d[flat_loop_idx_safe]
    gathered_ranks = loop_ranks_2d[flat_loop_idx_safe]

    # Mask invalid loops
    gathered_edges = torch.where(
        in_range.unsqueeze(3), gathered_edges, torch.full_like(gathered_edges, -1)
    )
    gathered_ranks = torch.where(
        in_range.unsqueeze(3), gathered_ranks, torch.full_like(gathered_ranks, -1)
    )

    # --- Step 5: Find crossings at local_edge ---
    target_local = n_loc.unsqueeze(2).unsqueeze(3)            # (E, 4, 1, 1)
    match = (
        (gathered_edges == target_local) &
        in_range.unsqueeze(3) &
        valid_slot.unsqueeze(2).unsqueeze(3)
    )
    # match: (E, 4, max_loops, K)

    # --- Step 6: Rank normalization ---
    is_neg = (n_loc == 2) | (n_loc == 6) | (n_loc == 3) | (n_loc == 7)  # (E, 4)
    W_broadcast = W_at_edge.unsqueeze(2).unsqueeze(3)                    # (E, 4, 1, 1)
    norm_ranks = torch.where(
        is_neg.unsqueeze(2).unsqueeze(3),
        W_broadcast - 1 - gathered_ranks,
        gathered_ranks,
    )
    rank_valid = match & (norm_ranks >= 0) & (norm_ranks < max_rank)

    # --- Step 7: Gather component points (per loop) ---
    if L_total > 0:
        loop_points_gathered = tensors.loop_component_point[flat_loop_idx_safe]
    else:
        loop_points_gathered = torch.zeros(
            (E, 4, max_loops, 3), dtype=torch.float32, device=device
        )
    # loop_points_gathered: (E, 4, max_loops, 3)

    # --- Step 8: Scatter points into (E, max_rank, 4, 3) ---
    points_by_rank = torch.zeros((E, max_rank, 4, 3), dtype=torch.float32, device=device)
    has_point = torch.zeros((E, max_rank, 4), dtype=torch.bool, device=device)

    if rank_valid.any():
        valid_flat = rank_valid.view(-1)
        idx_flat = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)  # (M,)

        # Recover (e, slot, loop_idx, k) from flat index
        stride_e = 4 * max_loops * K
        stride_s = max_loops * K
        stride_l = K
        e_idx = (idx_flat // stride_e).to(torch.int64)
        rem = idx_flat % stride_e
        s_idx = (rem // stride_s).to(torch.int64)
        rem = rem % stride_s
        l_idx = (rem // stride_l).to(torch.int64)
        # k_idx not needed — the point per match is loop_points_gathered[e, s, l]

        # Map slot index (table slot 0..3) to uv-based fan slot_id (Python neighbor order)
        uv_slot_idx = slot_id[e_idx, s_idx]              # (M,)

        # Rank per match
        r_idx = norm_ranks.view(-1)[idx_flat].to(torch.int64)

        # Points per match
        pts = loop_points_gathered[e_idx, s_idx, l_idx]  # (M, 3)

        # Scatter (later matches overwrite earlier, which is acceptable)
        points_by_rank[e_idx, r_idx, uv_slot_idx] = pts
        has_point[e_idx, r_idx, uv_slot_idx] = True

    # --- Step 9: Exception cube wildcards ---
    # For each (e, slot) where cube_exc is True: use first loop's point as wildcard
    any_has_point = has_point.any(dim=2)  # (E, max_rank)

    if exc_pt_per_cube_override is not None:
        # Direct-tensor path: gather per-cube first point (N, 3) into (E, 4, 3).
        # n_cube_safe clamps -1 slots to 0; invalid slots won't be read because
        # valid_slot masks them out below.
        exc_pt_per_slot = exc_pt_per_cube_override[n_cube_safe]  # (E, 4, 3)
    elif L_total > 0:
        first_loop_safe = loop_start.clamp(min=0, max=L_total - 1).to(torch.int64)
        exc_pt_per_slot = tensors.loop_component_point[first_loop_safe]  # (E, 4, 3)
    else:
        exc_pt_per_slot = torch.zeros((E, 4, 3), dtype=torch.float32, device=device)

    # Permute exc_pt_per_slot / cube_exc / num_loops from table-slot index (0..3 of
    # neighbor_cube_ids) to uv slot_id (Python neighbor order).
    # scatter to uv_ordered tensors using slot_id as the target index per (e, s).
    # valid slots map to a real uv slot; invalid slots (-1 / garbage) won't be read
    # because cube_exc/num_loops for invalid slots are already 0/False.
    slot_id_safe = slot_id.clamp(min=0)                                    # (E, 4)
    exc_pt = torch.zeros((E, 4, 3), dtype=exc_pt_per_slot.dtype, device=device)
    cube_exc_uv = torch.zeros((E, 4), dtype=torch.bool, device=device)
    num_loops_uv = torch.zeros((E, 4), dtype=num_loops.dtype, device=device)
    # Use scatter on last dim: for each (e, table_slot) write to (e, slot_id[e, table_slot])
    e_arange = torch.arange(E, dtype=torch.int64, device=device).unsqueeze(1).expand(E, 4)
    valid_mask_flat = valid_slot.reshape(-1)
    e_flat = e_arange.reshape(-1)[valid_mask_flat]
    uv_flat = slot_id_safe.reshape(-1)[valid_mask_flat]
    exc_pt[e_flat, uv_flat] = exc_pt_per_slot.reshape(-1, 3)[valid_mask_flat]
    cube_exc_uv[e_flat, uv_flat] = cube_exc.reshape(-1)[valid_mask_flat]
    num_loops_uv[e_flat, uv_flat] = num_loops.reshape(-1)[valid_mask_flat]

    # "Has a valid exception wildcard point" per (edge, uv-slot).
    # In the dict path, every exception cube has at least one virtual loop, so
    # num_loops_uv > 0 is sufficient. In the direct-tensor path, exception
    # cubes have num_loops_uv == 0 but the wildcard point lives in
    # exc_pt_per_cube_override; we permute the per-cube presence flag the
    # same way as exc_pt above.
    if exc_present_per_cube_override is not None:
        exc_present_per_slot = exc_present_per_cube_override[n_cube_safe] & valid_slot  # (E, 4)
        exc_present_uv = torch.zeros((E, 4), dtype=torch.bool, device=device)
        exc_present_uv[e_flat, uv_flat] = exc_present_per_slot.reshape(-1)[valid_mask_flat]
        # Combine: cube is an exception AND a wildcard point is materially available.
        exc_active_uv = cube_exc_uv & (exc_present_uv | (num_loops_uv > 0))
    else:
        # Legacy behaviour: rely on virtual loop produced by dict path.
        exc_active_uv = cube_exc_uv & (num_loops_uv > 0)

    # Fill missing slots where: cube has an active exception wildcard, slot is
    # valid, and the rank already has some presence (we don't invent new ranks
    # from exceptions alone here — that is handled by rank-0 synthesis below).
    fill_mask = (
        (~has_point) &
        exc_active_uv.unsqueeze(1) &
        any_has_point.unsqueeze(2)
    )
    points_by_rank = torch.where(
        fill_mask.unsqueeze(3),
        exc_pt.unsqueeze(1).expand(E, max_rank, 4, 3),
        points_by_rank,
    )
    has_point = has_point | fill_mask

    # Rank-0 synthesis: if no rank has any points but exceptions exist
    no_rank = ~any_has_point.any(dim=1)                     # (E,)
    has_exc = exc_active_uv.any(dim=1)                       # (E,)
    synth = no_rank & has_exc                                # (E,)
    if synth.any():
        idx = torch.nonzero(synth, as_tuple=False).squeeze(1)
        valid_exc = exc_active_uv[idx]                       # (M, 4)
        has_point[idx, 0, :] = valid_exc
        pts_rank0 = torch.where(
            valid_exc.unsqueeze(2),
            exc_pt[idx],
            torch.zeros_like(exc_pt[idx]),
        )
        points_by_rank[idx, 0, :, :] = pts_rank0

    # --- Step 10: Emit fan triangles for full rank groups (4 slots present) ---
    full_mask = has_point.all(dim=2)  # (E, max_rank)
    if not full_mask.any():
        return torch.zeros((0, 3), dtype=torch.float64, device=device)

    full_edges, full_ranks = torch.nonzero(full_mask, as_tuple=True)

    pts_4 = points_by_rank[full_edges, full_ranks]  # (F, 4, 3) float32
    # Promote to float64 BEFORE mean to match the Python reference path,
    # which converts float32 point values to Python floats (float64) via
    # tolist() before averaging. Without this promotion, the float32 mean
    # introduces ULP-level drift that crosses merge_decimals=5 buckets and
    # produces visible vertex-set divergence on 4-cube edges.
    pts_4_64 = pts_4.to(torch.float64)
    proj = pts_4_64.mean(dim=1)                      # (F, 3) float64
    p0 = pts_4_64[:, 0]
    p1 = pts_4_64[:, 1]
    p2 = pts_4_64[:, 2]
    p3 = pts_4_64[:, 3]

    # Fan: (proj,p0,p1), (proj,p1,p2), (proj,p2,p3), (proj,p3,p0)
    tris = torch.stack([
        torch.stack([proj, p0, p1], dim=1),
        torch.stack([proj, p1, p2], dim=1),
        torch.stack([proj, p2, p3], dim=1),
        torch.stack([proj, p3, p0], dim=1),
    ], dim=1)  # (F, 4, 3, 3) float64

    return tris.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Kernel 2: Shared-edge geometry processing
# ---------------------------------------------------------------------------

def _process_geometry_batch(grids: list) -> list[np.ndarray]:
    """Process a batch of edge grids, returning non-None numpy arrays."""
    results = []
    for grid in grids:
        r = _process_shared_edge_geometry(grid)
        if r is not None:
            results.append(r)
    return results


def _process_shared_edges_torch(
    resolution: int,
    cube_data_list: list[dict],
    merge_decimals: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized Torch path with Python fallback for conditional promotion edges.

    Drop-in replacement for the Python edge iteration + multiprocessing geometry
    loop. Keeps same input/output contract as process_shared_edges_batch.

    The vectorized Torch path handles the common case. A small subset of edges
    (all-4-cubes-present with num_components > 0 and some edge_weight > 0) may
    trigger custom/collapse.py's conditional-promotion logic; those candidate
    edges are processed via the Python `_process_shared_edge_geometry` fallback
    to preserve bit-exact parity.
    """
    if not cube_data_list:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step A: Convert to tensors (one-time cost)
    tensors = _cube_data_to_tensors(cube_data_list, device)

    # Step B: Edge enumeration (fully vectorized)
    keys, cube_ids, local_ids = compute_global_edge_keys(tensors.cube_indices, resolution)
    unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
    num_unique = unique_keys.shape[0]
    if num_unique == 0:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )
    table = build_edge_neighbor_table(edge_id_per_entry, cube_ids, local_ids, num_unique)

    # Step C: Filter to edges with >=2 neighbors
    keep = table.neighbor_counts >= 2
    if not keep.any():
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )

    kept_table = EdgeNeighborTable(
        neighbor_counts=table.neighbor_counts[keep],
        neighbor_cube_ids=table.neighbor_cube_ids[keep],
        neighbor_positions=table.neighbor_positions[keep],
        neighbor_local_edges=table.neighbor_local_edges[keep],
        edge_axes=table.edge_axes[keep],
    )

    # Step C.1: Identify candidate edges that may diverge from the Python
    # reference path (tensor ops only). Diagnostics across real meshes show
    # that per-edge divergence (Python vs vectorized Torch) only occurs for
    # edges where ALL 4 neighbor cubes are present. Conditional-promotion
    # (custom/collapse.py) is a strict subset of this; other subtle
    # rank / slot-ordering differences also manifest only on 4-cube edges.
    # Routing every 4-cube edge through the Python fallback produces
    # bit-exact parity while leaving the >=95% of partial-neighbor edges on
    # the fast vectorized path.
    #
    # Phase 2 W1: when COREP_FAST_S8_4CUBE_VECTORIZED=1, route 4-cube edges
    # through the vectorized path (encoding mismatch fixed via the loop-
    # convention conversion table re-derived inside process_geometry_vectorized).
    # Hybrid predicate: 4-cube edges with any neighbor having num_loops >= 2
    # still go through the Python fallback because conditional-promotion on
    # multi-loop neighbor cubes is not handled by the vectorized path.
    import os as _os_flag
    s8_4cube_vec = _os_flag.environ.get('COREP_FAST_S8_4CUBE_VECTORIZED', '0') == '1'

    if s8_4cube_vec:
        # Compute max neighbor num_loops per kept edge.
        # tensors.loop_cube_offsets: (N+1,) CSR offsets into loop arrays.
        n_cube_safe_for_pred = kept_table.neighbor_cube_ids.clamp(min=0).to(torch.int64)
        valid_slot_for_pred = kept_table.neighbor_cube_ids >= 0          # (E, 4)
        loop_start_pred = tensors.loop_cube_offsets[n_cube_safe_for_pred]
        loop_end_pred = tensors.loop_cube_offsets[n_cube_safe_for_pred + 1]
        per_slot_num_loops = (loop_end_pred - loop_start_pred).clamp(min=0)
        per_slot_num_loops = torch.where(
            valid_slot_for_pred,
            per_slot_num_loops,
            torch.zeros_like(per_slot_num_loops),
        )                                                                 # (E, 4)
        max_neighbor_num_loops = per_slot_num_loops.max(dim=1).values     # (E,)
        # Hybrid: only 4-cube edges with multi-loop neighbors stay on fallback.
        candidate_mask = (
            (kept_table.neighbor_counts == 4) & (max_neighbor_num_loops >= 2)
        )
    else:
        candidate_mask = kept_table.neighbor_counts == 4                   # (E,)

    # Step D: Non-candidate edges → vectorized Torch path
    non_candidate_mask = ~candidate_mask
    if non_candidate_mask.any():
        nc_table = EdgeNeighborTable(
            neighbor_counts=kept_table.neighbor_counts[non_candidate_mask],
            neighbor_cube_ids=kept_table.neighbor_cube_ids[non_candidate_mask],
            neighbor_positions=kept_table.neighbor_positions[non_candidate_mask],
            neighbor_local_edges=kept_table.neighbor_local_edges[non_candidate_mask],
            edge_axes=kept_table.edge_axes[non_candidate_mask],
        )
        tri_verts_torch = process_geometry_vectorized(nc_table, tensors, device)
    else:
        tri_verts_torch = torch.zeros((0, 3), dtype=torch.float64, device=device)

    # Step E: Candidate edges → Python fallback (with multiprocessing for large batches)
    tri_verts_python_list: list[np.ndarray] = []
    if candidate_mask.any():
        # Build cube_map for Python path (same structure as process_shared_edges_batch)
        cube_map: dict[tuple, list[dict]] = {}
        _empty_18 = [0] * 18
        for data in cube_data_list:
            idx = data.get('cube_indices')
            if idx is None:
                continue
            idx_t = tuple(idx) if not isinstance(idx, tuple) else idx
            data['cube_indices'] = idx_t
            if 'edge_weights' not in data:
                data['edge_weights'] = _empty_18
            if 'sorted_loops' not in data:
                data['sorted_loops'] = []
            if idx_t not in cube_map:
                cube_map[idx_t] = [data]
            else:
                cube_map[idx_t].append(data)

        cand_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
        cand_n_cube = kept_table.neighbor_cube_ids[cand_idx]           # (M, 4)
        cand_ci = tensors.cube_indices.cpu().numpy()                   # (N, 3)
        cand_n_cube_cpu = cand_n_cube.cpu().numpy()                    # (M, 4)

        # Build all grids upfront
        grids: list[list] = []
        M = cand_n_cube_cpu.shape[0]
        for i in range(M):
            slots = cand_n_cube_cpu[i]
            grid: list[list[dict]] = []
            for s_cube in slots:
                s_cube = int(s_cube)
                if s_cube < 0:
                    grid.append([{'cube_indices': (0, 0, 0), 'sorted_loops': []}])
                else:
                    idx_tup = (
                        int(cand_ci[s_cube, 0]),
                        int(cand_ci[s_cube, 1]),
                        int(cand_ci[s_cube, 2]),
                    )
                    entry = cube_map.get(idx_tup)
                    if entry is None:
                        grid.append([{'cube_indices': idx_tup, 'sorted_loops': []}])
                    else:
                        grid.append(entry)
            grids.append(grid)

        # Dispatch via multiprocessing for large batches only.
        # Below ~100K candidates, the MP fork overhead (~0.8s) outweighs
        # the parallelism gain. Benchmarked on H100 128-core node.
        import os as _os
        num_workers = max(1, (_os.cpu_count() or 4) - 4)
        if num_workers > 1 and M >= 100_000:
            from multiprocessing import Pool as _Pool
            chunk_size = max(M // (num_workers * 4), 1)
            with _Pool(num_workers) as p:
                results = p.map(_process_shared_edge_geometry, grids, chunksize=chunk_size)
            for tri_np in results:
                if tri_np is not None and tri_np.shape[0] > 0:
                    tri_verts_python_list.append(tri_np)
        else:
            for grid in grids:
                tri_np = _process_shared_edge_geometry(grid)
                if tri_np is not None and tri_np.shape[0] > 0:
                    tri_verts_python_list.append(tri_np)

    # Step F: Merge Torch and Python triangles
    if tri_verts_torch.shape[0] == 0 and not tri_verts_python_list:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )

    if tri_verts_torch.shape[0] > 0:
        tri_verts_torch_np = tri_verts_torch.cpu().numpy()
    else:
        tri_verts_torch_np = np.zeros((0, 3), dtype=np.float64)

    if tri_verts_python_list:
        tri_verts_python_np = np.concatenate(tri_verts_python_list, axis=0)
        all_tri_np = np.concatenate([tri_verts_torch_np, tri_verts_python_np], axis=0)
    else:
        all_tri_np = tri_verts_torch_np

    # Step G: Vertex welding (reuse existing _weld_and_dedup)
    return _weld_and_dedup(all_tri_np, merge_decimals, device=device)


# ---------------------------------------------------------------------------
# Direct-tensor s8 path (M2 P1): bypasses _cube_data_to_tensors + dict layer.
# ---------------------------------------------------------------------------

def _build_grids_from_cube_map(candidate_mask, kept_table, cube_data_list, tensors):
    """Legacy: build Python grids from cube_data_list (via cube_map dict).

    Used when USE_DIRECT_GRIDS_S8=0 or as fallback inside the direct-tensor
    path. Mirrors the per-candidate grid construction from the
    _process_shared_edges_torch fallback block.
    """
    cube_map: dict[tuple, list[dict]] = {}
    _empty_18 = [0] * 18
    for data in cube_data_list:
        idx = data.get('cube_indices')
        if idx is None:
            continue
        idx_t = tuple(idx) if not isinstance(idx, tuple) else idx
        data['cube_indices'] = idx_t
        if 'edge_weights' not in data:
            data['edge_weights'] = _empty_18
        if 'sorted_loops' not in data:
            data['sorted_loops'] = []
        if idx_t not in cube_map:
            cube_map[idx_t] = [data]
        else:
            cube_map[idx_t].append(data)

    cand_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
    cand_n_cube = kept_table.neighbor_cube_ids[cand_idx].cpu().numpy()  # (M, 4)
    cand_ci = tensors.cube_indices.cpu().numpy()                          # (N, 3)

    M = cand_n_cube.shape[0]
    grids: list[list] = []
    for i in range(M):
        slots = cand_n_cube[i]
        grid: list[list[dict]] = []
        for s_cube in slots:
            s_cube = int(s_cube)
            if s_cube < 0:
                grid.append([{'cube_indices': (0, 0, 0), 'sorted_loops': []}])
            else:
                idx_tup = (
                    int(cand_ci[s_cube, 0]),
                    int(cand_ci[s_cube, 1]),
                    int(cand_ci[s_cube, 2]),
                )
                entry = cube_map.get(idx_tup)
                if entry is None:
                    grid.append([{'cube_indices': idx_tup, 'sorted_loops': []}])
                else:
                    grid.append(entry)
        grids.append(grid)
    return grids


def _build_single_cube_dict(
    ci, cube_indices_cpu, edge_weights_cpu, exception_cpu,
    lco, leo, lev, ler, po, pv, lpm,
):
    """Build a single-cube dict from CPU numpy arrays.

    Only called for cubes that participate in candidate-edge fallback
    (typically ~100K out of 275K at res=256).
    """
    ci_tup = (int(cube_indices_cpu[ci, 0]), int(cube_indices_cpu[ci, 1]),
              int(cube_indices_cpu[ci, 2]))
    is_exc = bool(exception_cpu[ci])
    p_lo = int(po[ci])
    p_hi = int(po[ci + 1])
    comp_pts = pv[p_lo:p_hi].tolist() if p_hi > p_lo else []

    if is_exc:
        return {
            'exception': True,
            'cube_indices': ci_tup,
            'sorted_loops': [{'component_point': comp_pts[0] if comp_pts else [0, 0, 0]}],
        }

    l_lo = int(lco[ci])
    l_hi = int(lco[ci + 1])
    sorted_loops = []
    for li in range(l_lo, l_hi):
        e_lo = int(leo[li])
        e_hi = int(leo[li + 1])
        edges = lev[e_lo:e_hi].tolist()
        ranks = ler[e_lo:e_hi].tolist()
        match_idx = int(lpm[li])
        if 0 <= match_idx < (p_hi - p_lo):
            cp = pv[p_lo + match_idx].tolist()
        elif comp_pts:
            cp = comp_pts[0]
        else:
            cp = [0.0, 0.0, 0.0]
        sorted_loops.append({
            'loop': edges,
            'rank': ranks,
            'component_point': cp,
        })

    return {
        'cube_indices': ci_tup,
        'edge_weights': edge_weights_cpu[ci].tolist(),
        'sorted_loops': sorted_loops,
        'exception': False,
    }


def _build_grids_from_tensors(candidate_mask, kept_table, tensors, batch):
    """Build Python grids directly from CubeBatch CSR, only for candidate cubes.

    Avoids building a full 275K-entry cube_map dict. Instead:
    1. Collect the set of ~100K neighbor cubes referenced by candidates.
    2. Bulk-transfer needed CSR arrays once.
    3. Build dict only for those cubes.

    Expected saving vs _build_grids_from_cube_map: ~0.6s at res=256.
    """
    cand_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
    M = cand_idx.shape[0]
    if M == 0:
        return []

    # Bulk GPU→CPU transfer (once)
    cand_n_cube = kept_table.neighbor_cube_ids[cand_idx].cpu().numpy()  # (M, 4)
    cube_indices_cpu = tensors.cube_indices.cpu().numpy()
    edge_weights_cpu = tensors.cube_edge_weights.cpu().numpy()
    exception_cpu = tensors.cube_exception.cpu().numpy()
    lco = batch.loop_cube_off.cpu().numpy()
    leo = batch.loop_edge_off.cpu().numpy()
    lev = batch.loop_edge_val.cpu().numpy()
    ler = batch.loop_edge_rank.cpu().numpy()
    po = batch.point_offsets.cpu().numpy()
    pv = batch.point_values.cpu().numpy()
    lpm = batch.loop_point_match.cpu().numpy()

    # Collect only referenced neighbor cubes (~100K, not 275K)
    unique_cubes = set()
    for i in range(M):
        for s in range(4):
            ci = int(cand_n_cube[i, s])
            if ci >= 0:
                unique_cubes.add(ci)

    # Build dict only for those cubes
    cube_dicts = {
        ci: _build_single_cube_dict(
            ci, cube_indices_cpu, edge_weights_cpu, exception_cpu,
            lco, leo, lev, ler, po, pv, lpm,
        )
        for ci in unique_cubes
    }

    # Assemble grids
    cand_ci = cube_indices_cpu
    grids = []
    for i in range(M):
        grid = []
        for s in range(4):
            ci = int(cand_n_cube[i, s])
            if ci < 0:
                grid.append([{'cube_indices': (0, 0, 0), 'sorted_loops': []}])
            else:
                grid.append([cube_dicts[ci]])
        grids.append(grid)
    return grids


def _process_shared_edges_from_tensors(
    resolution: int,
    tensors: 'CubeDataTensors',
    batch,
    merge_decimals: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process shared edges directly from CubeDataTensors + CubeBatch.

    Skips Step A (_cube_data_to_tensors) of _process_shared_edges_torch.
    All other logic (enum, vectorized, fallback, merge, weld) is identical.

    Args:
        resolution: grid resolution.
        tensors: CubeDataTensors (already built via _cubebatch_to_tensors_direct).
        batch: CubeBatch, used for per-cube first-point gather and Python
            fallback cube_data construction.
        merge_decimals: vertex welding precision.

    Returns:
        (vertices, faces) tuple.
    """
    device = tensors.cube_indices.device
    N = tensors.cube_indices.shape[0]

    if N == 0:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )

    # Step B: Edge enumeration (identical to _process_shared_edges_torch)
    keys, cube_ids, local_ids = compute_global_edge_keys(tensors.cube_indices, resolution)
    unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
    num_unique = unique_keys.shape[0]
    if num_unique == 0:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )
    table = build_edge_neighbor_table(edge_id_per_entry, cube_ids, local_ids, num_unique)

    # Step C: Filter keep edges (>= 2 neighbors)
    keep = table.neighbor_counts >= 2
    if not keep.any():
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )

    kept_table = EdgeNeighborTable(
        neighbor_counts=table.neighbor_counts[keep],
        neighbor_cube_ids=table.neighbor_cube_ids[keep],
        neighbor_positions=table.neighbor_positions[keep],
        neighbor_local_edges=table.neighbor_local_edges[keep],
        edge_axes=table.edge_axes[keep],
    )

    # 4-cube candidate edges: routed through Python fallback for bit-parity.
    # Phase 2 W1: when COREP_FAST_S8_4CUBE_VECTORIZED=1, hybrid dispatch:
    #   - 4-cube edges where every neighbor has num_loops < 2 → vectorized
    #   - 4-cube edges where any neighbor has num_loops >= 2 → Python fallback
    # Multi-loop neighbors are the case where conditional-promotion in
    # custom/collapse.py can fire and the vectorized path lacks that logic.
    import os as _os_flag
    s8_4cube_vec = _os_flag.environ.get('COREP_FAST_S8_4CUBE_VECTORIZED', '0') == '1'
    if s8_4cube_vec:
        n_cube_safe_for_pred = kept_table.neighbor_cube_ids.clamp(min=0).to(torch.int64)
        valid_slot_for_pred = kept_table.neighbor_cube_ids >= 0
        loop_start_pred = tensors.loop_cube_offsets[n_cube_safe_for_pred]
        loop_end_pred = tensors.loop_cube_offsets[n_cube_safe_for_pred + 1]
        per_slot_num_loops = (loop_end_pred - loop_start_pred).clamp(min=0)
        per_slot_num_loops = torch.where(
            valid_slot_for_pred,
            per_slot_num_loops,
            torch.zeros_like(per_slot_num_loops),
        )
        max_neighbor_num_loops = per_slot_num_loops.max(dim=1).values
        candidate_mask = (
            (kept_table.neighbor_counts == 4) & (max_neighbor_num_loops >= 2)
        )
    else:
        candidate_mask = kept_table.neighbor_counts == 4

    # Build per-cube first-point tensor for Step 9 exception fallback.
    # In the dict path, the virtual first loop of an exception cube carries
    # the fallback point; in the direct path we build it explicitly from CSR.
    po_lo = batch.point_offsets[:-1]
    po_hi = batch.point_offsets[1:]
    has_pt = po_hi > po_lo
    first_pt_per_cube = torch.zeros((N, 3), dtype=torch.float32, device=device)
    if batch.point_values.shape[0] > 0 and has_pt.any():
        first_pt_per_cube[has_pt] = batch.point_values[po_lo[has_pt]]

    # Per-cube "has a wildcard exception point" mask. In the dict path the
    # virtual first loop already implies this for exception cubes, but the
    # direct-tensor path doesn't add a virtual loop, so we plumb the mask
    # explicitly so Step 9 / rank-0 synthesis still fire correctly.
    exc_present_per_cube = tensors.cube_exception & has_pt

    # Step D: Non-candidate edges -> vectorized Torch path
    non_candidate_mask = ~candidate_mask
    if non_candidate_mask.any():
        nc_table = EdgeNeighborTable(
            neighbor_counts=kept_table.neighbor_counts[non_candidate_mask],
            neighbor_cube_ids=kept_table.neighbor_cube_ids[non_candidate_mask],
            neighbor_positions=kept_table.neighbor_positions[non_candidate_mask],
            neighbor_local_edges=kept_table.neighbor_local_edges[non_candidate_mask],
            edge_axes=kept_table.edge_axes[non_candidate_mask],
        )
        # Only relax the exception gating when the W1 flag is on, so the
        # legacy OFF path's behaviour is preserved bit-for-bit.
        tri_verts_torch = process_geometry_vectorized(
            nc_table, tensors, device,
            exc_pt_per_cube_override=first_pt_per_cube,
            exc_present_per_cube_override=(
                exc_present_per_cube if s8_4cube_vec else None
            ),
        )
    else:
        tri_verts_torch = torch.zeros((0, 3), dtype=torch.float64, device=device)

    # Step E: Python fallback for 4-cube candidate edges
    tri_verts_python_list: list[np.ndarray] = []
    if candidate_mask.any():
        from corep_fast.config import USE_DIRECT_GRIDS_S8
        if USE_DIRECT_GRIDS_S8:
            grids = _build_grids_from_tensors(
                candidate_mask, kept_table, tensors, batch,
            )
        else:
            cube_data_list = _cubebatch_to_dicts(batch)
            grids = _build_grids_from_cube_map(
                candidate_mask, kept_table, cube_data_list, tensors,
            )

        import os as _os
        num_workers = max(1, (_os.cpu_count() or 4) - 4)
        M = len(grids)
        if num_workers > 1 and M >= 100_000:
            from multiprocessing import Pool as _Pool
            chunk_size = max(M // (num_workers * 4), 1)
            with _Pool(num_workers) as p:
                results = p.map(_process_shared_edge_geometry, grids, chunksize=chunk_size)
            for tri_np in results:
                if tri_np is not None and tri_np.shape[0] > 0:
                    tri_verts_python_list.append(tri_np)
        else:
            for grid in grids:
                tri_np = _process_shared_edge_geometry(grid)
                if tri_np is not None and tri_np.shape[0] > 0:
                    tri_verts_python_list.append(tri_np)

    # Step F: Merge Torch and Python triangles
    if tri_verts_torch.shape[0] == 0 and not tri_verts_python_list:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )

    if tri_verts_torch.shape[0] > 0:
        tri_verts_torch_np = tri_verts_torch.cpu().numpy()
    else:
        tri_verts_torch_np = np.zeros((0, 3), dtype=np.float64)

    if tri_verts_python_list:
        tri_verts_python_np = np.concatenate(tri_verts_python_list, axis=0)
        all_tri_np = np.concatenate([tri_verts_torch_np, tri_verts_python_np], axis=0)
    else:
        all_tri_np = tri_verts_torch_np

    # Step G: Vertex welding
    return _weld_and_dedup(all_tri_np, merge_decimals, device=device)


def process_shared_edges_batch(
    resolution: int,
    cube_data_list: list[dict],
    merge_decimals: int = 5,
    num_workers: int = 0,
    use_torch_path: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Process all shared edges across all cubes and return (vertices, faces).

    This replaces the full iteration loop in custom/collapse.py::generate_global_mesh().
    It uses the same algorithm but collects all results into tensors.

    Args:
        resolution: Grid resolution.
        cube_data_list: List of dicts from custom/ s7 output.
        merge_decimals: Vertex welding precision.
        num_workers: Number of parallel workers for geometry processing.
                     0 = auto (use os.cpu_count()), 1 = serial (no multiprocessing).
        use_torch_path: If True, dispatch to the fully vectorized Torch path
                        (_process_shared_edges_torch). Default False preserves
                        existing Python + multiprocessing path.

    Returns:
        vertices: (V, 3) float32 — welded vertex coordinates.
        faces: (F, 3) int32 — triangle faces (vertex indices).
    """
    import os as _os
    if not cube_data_list:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )

    # Dispatch to vectorized Torch path if requested
    if use_torch_path:
        return _process_shared_edges_torch(
            resolution, cube_data_list, merge_decimals,
        )

    # 1. Build cube_map: cube_indices_tuple → list[dict]
    #    Reuse original dicts to avoid deep copy overhead. Only normalize cube_indices.
    cube_map: dict[tuple, list[dict]] = {}
    _empty_18 = [0] * 18
    for data in cube_data_list:
        idx = data.get('cube_indices')
        if idx is None:
            continue
        idx = tuple(idx) if not isinstance(idx, tuple) else idx
        data['cube_indices'] = idx
        # Ensure edge_weights and sorted_loops exist
        if 'edge_weights' not in data:
            data['edge_weights'] = _empty_18
        if 'sorted_loops' not in data:
            data['sorted_loops'] = []
        if idx not in cube_map:
            cube_map[idx] = [data]
        else:
            cube_map[idx].append(data)

    # 2. Collect all edge tasks (grid inputs for geometry processing)
    edge_tasks: list[list[list[dict]]] = []

    _empty_sentinel_cache: dict[tuple, list[dict]] = {}

    def get_grid_input(indices_list):
        grid = []
        valid_count = 0
        for idx in indices_list:
            entry = cube_map.get(idx)
            if entry is not None:
                grid.append(entry)
                valid_count += 1
            else:
                sentinel = _empty_sentinel_cache.get(idx)
                if sentinel is None:
                    sentinel = [{'cube_indices': idx, 'sorted_loops': []}]
                    _empty_sentinel_cache[idx] = sentinel
                grid.append(sentinel)
        return grid, valid_count

    cube_set = frozenset(cube_map.keys())
    R = resolution

    for idx in cube_map:
        x, y, z = idx
        local_edges = (
            ('X', x, y, z), ('X', x, y+1, z), ('X', x, y, z+1), ('X', x, y+1, z+1),
            ('Y', x, y, z), ('Y', x+1, y, z), ('Y', x, y, z+1), ('Y', x+1, y, z+1),
            ('Z', x, y, z), ('Z', x+1, y, z), ('Z', x, y+1, z), ('Z', x+1, y+1, z),
        )

        for axis, a, b, c in local_edges:
            if axis == 'X':
                if not (0 <= a < R and 1 <= b < R and 1 <= c < R):
                    continue
                neighbors = ((a, b-1, c-1), (a, b, c-1), (a, b-1, c), (a, b, c))
            elif axis == 'Y':
                if not (1 <= a < R and 0 <= b < R and 1 <= c < R):
                    continue
                neighbors = ((a-1, b, c-1), (a, b, c-1), (a-1, b, c), (a, b, c))
            else:
                if not (1 <= a < R and 1 <= b < R and 0 <= c < R):
                    continue
                neighbors = ((a-1, b-1, c), (a, b-1, c), (a-1, b, c), (a, b, c))

            active = [n for n in neighbors if n in cube_set]
            if not active or min(active) != idx:
                continue

            grid, count = get_grid_input(neighbors)
            if count >= 2:
                edge_tasks.append(grid)

    if not edge_tasks:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )

    # 3. Process geometry — serial or parallel
    if num_workers == 0:
        num_workers = min(_os.cpu_count() or 1, 32)

    if num_workers <= 1 or len(edge_tasks) < 1000:
        # Serial processing for small inputs or when explicitly requested
        tri_vertex_chunks = []
        for grid in edge_tasks:
            r = _process_shared_edge_geometry(grid)
            if r is not None:
                tri_vertex_chunks.append(r)
    else:
        # Parallel processing using multiprocessing.Pool
        from multiprocessing import Pool as _Pool
        chunk_size = max(len(edge_tasks) // num_workers, 1)
        batches = [edge_tasks[i:i+chunk_size]
                    for i in range(0, len(edge_tasks), chunk_size)]
        with _Pool(num_workers) as pool:
            batch_results = pool.map(_process_geometry_batch, batches)
        tri_vertex_chunks = [r for batch in batch_results for r in batch]

    # 4. Vertex welding + face dedup using Torch
    if not tri_vertex_chunks:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )

    all_tri_verts_np = np.concatenate(tri_vertex_chunks, axis=0)
    return _weld_and_dedup(all_tri_verts_np, merge_decimals)


def _process_shared_edge_geometry(
    grid_2x2_lists: list[list[dict]],
) -> Optional[np.ndarray]:
    """
    Process a 2x2 grid of cubes sharing an edge.

    Replicates custom/collapse.py::process_shared_edge_geometry() exactly,
    but returns triangle vertices as a flat numpy array for efficient welding.

    Args:
        grid_2x2_lists: 4 lists of dicts for the 4 neighbor positions.

    Returns:
        numpy array of shape (T*3, 3) float64 where T is number of triangles,
        or None if no triangles produced. Each consecutive 3 rows form one triangle.
    """
    import collections

    # 1. Deduce edge axis
    valid_indices = []
    for cube_list in grid_2x2_lists:
        if cube_list and 'cube_indices' in cube_list[0]:
            valid_indices.append(cube_list[0]['cube_indices'])

    if not valid_indices:
        return [], []

    min_idx = [min(idx[i] for idx in valid_indices) for i in range(3)]
    max_idx = [max(idx[i] for idx in valid_indices) for i in range(3)]

    edge_axis = -1
    for i in range(3):
        if min_idx[i] == max_idx[i]:
            edge_axis = i
            break

    if edge_axis == -1:
        edge_axis = 2

    def get_local_edge(dx, dy, dz):
        if edge_axis == 2:
            if dx == 0 and dy == 0: return 10
            if dx == 1 and dy == 0: return 11
            if dx == 0 and dy == 1: return 9
            if dx == 1 and dy == 1: return 8
        elif edge_axis == 0:
            if dy == 0 and dz == 0: return 6
            if dy == 1 and dz == 0: return 4
            if dy == 0 and dz == 1: return 2
            if dy == 1 and dz == 1: return 0
        elif edge_axis == 1:
            if dx == 0 and dz == 0: return 5
            if dx == 1 and dz == 0: return 7
            if dx == 0 and dz == 1: return 1
            if dx == 1 and dz == 1: return 3
        return -1

    # 2. Extract points by rank
    points_by_rank = collections.defaultdict(dict)
    exception_points = {}
    cube_by_uv = {}
    shared_edge_weight = 0

    for cube_list in grid_2x2_lists:
        if not cube_list:
            continue
        idx = cube_list[0]['cube_indices']
        dx = idx[0] - min_idx[0]
        dy = idx[1] - min_idx[1]
        dz = idx[2] - min_idx[2]

        local_edge = get_local_edge(dx, dy, dz)
        if local_edge == -1:
            continue

        if edge_axis == 2:   uv = (dx, dy)
        elif edge_axis == 0: uv = (dy, dz)
        elif edge_axis == 1: uv = (dx, dz)

        cube_by_uv[uv] = cube_list

        for d in cube_list:
            w = d.get('edge_weights', [0]*18)[local_edge]
            if w > 0:
                shared_edge_weight = max(shared_edge_weight, w)

            if d.get('exception') is True:
                pt = None
                if d.get('sorted_loops') and len(d['sorted_loops']) > 0:
                    pt = d['sorted_loops'][0].get('component_point')
                elif d.get('component_points') and len(d['component_points']) > 0:
                    pt = d['component_points'][0]
                elif d.get('component_point'):
                    pt = d.get('component_point')
                if pt is not None:
                    exception_points[uv] = pt
                continue

            for loop_data in d.get('sorted_loops', []):
                edges = loop_data.get('loop', [])
                ranks = loop_data.get('rank', [])

                for i, edge in enumerate(edges):
                    if edge == local_edge:
                        rank = ranks[i]

                        if local_edge in (2, 6, 3, 7):
                            W = d.get('edge_weights', [0]*18)[local_edge]
                            normalized_rank = (W - 1) - rank
                        else:
                            normalized_rank = rank

                        pt = loop_data.get('component_point')
                        if pt is not None:
                            points_by_rank[normalized_rank][uv] = pt

    # 3. Conditional promotion of disconnected cubes to exceptions
    all_have_components = len(cube_by_uv) == 4 and all(
        any(d.get('num_components', 0) > 0 for d in cl) for cl in cube_by_uv.values()
    )
    all_incomplete = all(len(pt_map) < 4 for pt_map in points_by_rank.values())

    if all_have_components and shared_edge_weight > 0 and all_incomplete:
        neighbors_pairs = [((0,0),(1,0)), ((1,0),(1,1)), ((1,1),(0,1)), ((0,1),(0,0))]
        new_exception_uvs = set()

        for uv1, uv2 in neighbors_pairs:
            connected = False
            for pt_map in points_by_rank.values():
                if uv1 in pt_map and uv2 in pt_map:
                    connected = True
                    break
            if not connected:
                if uv1 in cube_by_uv: new_exception_uvs.add(uv1)
                if uv2 in cube_by_uv: new_exception_uvs.add(uv2)

        for uv in new_exception_uvs:
            if uv not in exception_points:
                pt = None
                for d in cube_by_uv[uv]:
                    if d.get('sorted_loops') and len(d['sorted_loops']) > 0:
                        pt = d['sorted_loops'][0].get('component_point')
                    elif d.get('component_points') and len(d['component_points']) > 0:
                        pt = d['component_points'][0]
                    elif d.get('component_point'):
                        pt = d.get('component_point')
                    if pt is not None:
                        break
                if pt is not None:
                    exception_points[uv] = pt
                for pt_map in points_by_rank.values():
                    pt_map.pop(uv, None)

        empty_ranks = [r for r, pm in points_by_rank.items() if not pm]
        for r in empty_ranks:
            del points_by_rank[r]

    # 4. Distribute exceptions as wildcards
    for rank, pt_map in points_by_rank.items():
        for uv, exc_pt in exception_points.items():
            if uv not in pt_map:
                pt_map[uv] = exc_pt

    if not points_by_rank and len(exception_points) > 0:
        points_by_rank[0] = exception_points

    # 5. Emit triangles — collect as flat list of 9 floats per triangle
    tri_flat: list[list[float]] = []
    neighbors = [(0,0), (1,0), (1,1), (0,1)]

    for rank, pt_map in sorted(points_by_rank.items()):
        rank_pts = list(pt_map.values())
        if not rank_pts:
            continue

        avg_x = sum(p[0] for p in rank_pts) / len(rank_pts)
        avg_y = sum(p[1] for p in rank_pts) / len(rank_pts)
        avg_z = sum(p[2] for p in rank_pts) / len(rank_pts)
        proj_pt = [avg_x, avg_y, avg_z]

        if len(rank_pts) == 4:
            p0 = pt_map[neighbors[0]]
            p1 = pt_map[neighbors[1]]
            p2 = pt_map[neighbors[2]]
            p3 = pt_map[neighbors[3]]
            # 4 fan triangles: proj→p0→p1, proj→p1→p2, proj→p2→p3, proj→p3→p0
            tri_flat.append(proj_pt)
            tri_flat.append(list(p0))
            tri_flat.append(list(p1))
            tri_flat.append(proj_pt)
            tri_flat.append(list(p1))
            tri_flat.append(list(p2))
            tri_flat.append(proj_pt)
            tri_flat.append(list(p2))
            tri_flat.append(list(p3))
            tri_flat.append(proj_pt)
            tri_flat.append(list(p3))
            tri_flat.append(list(p0))

    if not tri_flat:
        return None
    return np.array(tri_flat, dtype=np.float64)  # (T*3, 3)


# ---------------------------------------------------------------------------
# Kernel 3: Vertex welding + face deduplication
# ---------------------------------------------------------------------------

def _weld_and_dedup(
    flat_tri_verts: np.ndarray,
    merge_decimals: int,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vertex welding + face deduplication using torch.unique.

    This replaces the Python dict vertex_to_index and set seen_faces from
    custom/collapse.py::generate_global_mesh() with O(N log N) sort-based dedup.

    Uses GPU if available for massive speedup on torch.unique (97x at 800K vertices).

    Args:
        flat_tri_verts: (T*3, 3) float64 numpy array — every 3 consecutive rows
                       form one triangle's vertices.
        merge_decimals: Number of decimal places for vertex rounding.
        device: torch device. None = auto (GPU if available, else CPU).

    Returns:
        vertices: (V, 3) float32 — unique vertex coordinates (on CPU).
        faces: (F, 3) int32 — triangle face indices (on CPU).
    """
    if flat_tri_verts.shape[0] == 0:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Convert numpy to torch and move to device
    flat_verts = torch.from_numpy(flat_tri_verts).to(device)  # (T*3, 3) float64
    T = flat_verts.shape[0] // 3

    # 2. Round for welding
    scale = 10.0 ** merge_decimals
    rounded = torch.round(flat_verts * scale)

    # 3. Unique vertices via torch.unique on rounded coordinates
    unique_rounded, inverse_indices = torch.unique(rounded, dim=0, return_inverse=True)

    # 4. Recover actual coordinates: first occurrence per unique vertex (vectorized)
    num_unique = unique_rounded.shape[0]
    idx_arange = torch.arange(flat_verts.shape[0], dtype=torch.int64, device=device)
    first_occur = torch.full((num_unique,), flat_verts.shape[0], dtype=torch.int64,
                             device=device)
    first_occur.scatter_reduce_(0, inverse_indices, idx_arange, reduce='amin',
                                include_self=True)
    unique_verts = flat_verts[first_occur].float()  # (V, 3) float32

    # 5. Build face index array
    face_indices = inverse_indices.reshape(T, 3).to(torch.int32)

    # 6. Remove degenerate faces (where any two vertices are the same)
    v0 = face_indices[:, 0]
    v1 = face_indices[:, 1]
    v2 = face_indices[:, 2]
    non_degenerate = (v0 != v1) & (v1 != v2) & (v2 != v0)
    face_indices = face_indices[non_degenerate]

    if face_indices.shape[0] == 0:
        return unique_verts.cpu(), torch.zeros((0, 3), dtype=torch.int32)

    # 7. Canonical face rotation: rotate so minimum vertex index is first
    face_long = face_indices.to(torch.int64)
    min_val, _ = face_long.min(dim=1, keepdim=True)

    # For each face, find which position has the minimum
    is_min = face_long == min_val  # (F, 3)
    # Take the first True position
    min_pos = is_min.to(torch.int64).argmax(dim=1)  # (F,)

    # Rotate: shift so min_pos becomes position 0
    arange3 = torch.arange(3, device=face_long.device).unsqueeze(0)  # (1, 3)
    shifted = (arange3 + min_pos.unsqueeze(1)) % 3  # (F, 3)
    canonical = torch.gather(face_long, 1, shifted)  # (F, 3)

    # 8. Deduplicate canonical faces
    unique_canonical, unique_idx = torch.unique(canonical, dim=0, return_inverse=True)

    # Keep only the first occurrence of each unique canonical face (vectorized)
    F_unique = unique_canonical.shape[0]
    face_arange = torch.arange(face_indices.shape[0], dtype=torch.int64, device=device)
    first_face_occur = torch.full((F_unique,), face_indices.shape[0], dtype=torch.int64,
                                  device=device)
    first_face_occur.scatter_reduce_(0, unique_idx, face_arange, reduce='amin',
                                     include_self=True)

    deduped_faces = face_indices[first_face_occur].to(torch.int32)

    # Move results back to CPU for PLY writing
    return unique_verts.cpu(), deduped_faces.cpu()


# ---------------------------------------------------------------------------
# Kernel 4: PLY writer + public API
# ---------------------------------------------------------------------------

def _write_ply_ascii(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    output_filepath: str,
) -> None:
    """
    Write vertices and faces to an ASCII PLY file.

    Args:
        vertices: (V, 3) float32.
        faces: (F, 3) int32.
        output_filepath: Destination path.
    """
    V = vertices.shape[0]
    F = faces.shape[0]

    verts_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy() if F > 0 else np.zeros((0, 3), dtype=np.int32)

    with open(output_filepath, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {V}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {F}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Vertices — bulk write via numpy
        if V > 0:
            np.savetxt(f, verts_np, fmt='%g')

        # Faces — prepend "3" column and bulk write
        if F > 0:
            prefix = np.full((F, 1), 3, dtype=np.int32)
            face_block = np.hstack([prefix, faces_np])
            np.savetxt(f, face_block, fmt='%d')


def s8_collapse_to_ply(
    resolution: int,
    cube_data_list: list[dict],
    output_filepath: str = "output_mesh.ply",
    merge_decimals: int = 5,
) -> str:
    """
    Public API: Reconstruct mesh from cube data and write PLY.

    Drop-in replacement for custom/collapse.py::reconstruct_mesh().

    This function performs:
    1. reform_intersection_data (extract loops + component_points)
    2. extract_original_cube_edges (filter to edges 0-11)
    3. generate_global_mesh (edge enumeration, geometry, welding, PLY write)

    But steps 1-2 are unnecessary for the sorted_loops format (which already has
    component_point and loop/rank data), so we skip them when sorted_loops is present.

    Args:
        resolution: Grid resolution.
        cube_data_list: List of cube dicts from s7 output.
        output_filepath: Where to write the PLY.
        merge_decimals: Vertex welding precision.

    Returns:
        The output filepath.
    """
    # If cube_data_list uses the old 'loops' + 'component_points' format (not sorted_loops),
    # run reform + extract first
    needs_reform = False
    for data in cube_data_list:
        if 'sorted_loops' not in data and 'loops' in data:
            needs_reform = True
            break

    if needs_reform:
        cube_data_list = _reform_and_extract(cube_data_list)

    # Run geometry processing
    vertices, faces = process_shared_edges_batch(
        resolution=resolution,
        cube_data_list=cube_data_list,
        merge_decimals=merge_decimals,
    )

    # Write PLY
    _write_ply_ascii(vertices, faces, output_filepath)

    return output_filepath


def _reform_and_extract(cube_data_list: list[dict]) -> list[dict]:
    """
    Convert old-format cube data (with 'loops' + 'component_points') to sorted_loops format.

    Replicates custom/collapse.py::reform_intersection_data() +
    extract_original_cube_edges() but maps to sorted_loops format.
    """
    for data in cube_data_list:
        if 'sorted_loops' in data:
            continue

        loops = data.get('loops', [])
        points = data.get('component_points', [])
        num_loops = data.get('num_loops', 0)

        sorted_loops = []
        if num_loops == len(points):
            for i in range(num_loops):
                sorted_loops.append({
                    'component_point': list(points[i]) if isinstance(points[i], (list, tuple)) else points[i],
                    'loop': loops[i] if i < len(loops) else [],
                    'rank': [-1] * len(loops[i]) if i < len(loops) else [],
                })

        data['sorted_loops'] = sorted_loops

    return cube_data_list
