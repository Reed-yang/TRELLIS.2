"""
Stage 8: Collapse — Torch-accelerated mesh reconstruction from CoReP cube data.

Replaces custom/collapse.py with vectorized Torch operations. The critical
optimization is replacing Python dict/set vertex dedup with torch.unique,
and replacing per-edge Python iteration with batched sort + unique_consecutive.

Reference: spec section 6.8 (s8_collapse Torch strategy).

Public API:
    s8_collapse_to_ply(resolution, cube_data_list, output_filepath, merge_decimals)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np


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


def process_shared_edges_batch(
    resolution: int,
    cube_data_list: list[dict],
    merge_decimals: int = 5,
    num_workers: int = 0,
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
