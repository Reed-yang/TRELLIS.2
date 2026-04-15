# CoReP-Fast Stage 2 v2: Full Torch Vectorization (Edge Enum + Geometry)

> Date: 2026-04-15
> Supersedes: 2026-04-15-corep-fast-stage2-triton-design.md (the Triton-first plan is deferred — pure Torch can already hit the 10x target)
> Prerequisite: Stage 1 complete (s8 = 13.9s at res=256, 5.0x vs custom on H100)
> Goal: Achieve 10-20x total speedup on s8_collapse via full Torch vectorization of edge enumeration and geometry processing

---

## 0. Summary

Stage 1 delivered a 5.0x speedup via vectorized vertex welding + multiprocessing + GPU torch.unique. Deep profiling revealed that **63% of remaining s8 time is now spent in Python edge enumeration** (7.3s at res=256), not in geometry processing or welding. This is a surprise — the previous estimate assumed geometry processing was dominant.

### 0.1 New Bottleneck Analysis (res=256, H100, after Stage 1)

| Component | Time | % of s8 | Type | Action |
|-----------|------|---------|------|--------|
| Build cube_map (Python dict) | 0.23s | 2% | Python | Keep as-is |
| **Edge task collection (Python loop)** | **7.30s** | **63%** | Python | **Vectorize (Phase 2a)** |
| **Geometry processing (multiprocessing)** | **3.85s** | **33%** | Python+MP | **Vectorize (Phase 2b)** |
| np.concatenate | 0.10s | 1% | numpy | Keep |
| GPU welding (torch.unique) | 0.11s | 1% | GPU | Already optimal |

**Total s8 at res=256**: 11.6s (excluding Pool startup overhead)

### 0.2 Why Pure Torch (not Triton)

After analyzing the Triton vs Torch tradeoff:

| Criterion | Torch vectorization | Triton kernel |
|-----------|---------------------|---------------|
| Development time | Days | Weeks |
| Debuggability | Full Python introspection | Limited (GPU async) |
| Correctness risk | Low (tensor ops) | High (manual memory mgmt) |
| Expected speedup | 10-20x | 15-30x |
| Memory overhead | Moderate (materialize intermediates) | Low (register state) |

**Decision**: Pure Torch vectorization first. Defer Triton to Stage 3 if Torch doesn't hit the target.

### 0.3 Performance Targets

| Phase | Component | Current | Target | Method |
|-------|-----------|---------|--------|--------|
| 2a | Edge enumeration | 7.3s | ≤0.5s | Torch tensor ops (reuse `compute_global_edge_keys` etc.) |
| 2b | Geometry processing | 3.85s | ≤1.0s | Batch tensor ops, scatter-based rank grouping |
| - | **Total s8** | **13.9s** | **≤3.0s** | (5x speedup within Stage 2) |
| - | vs Custom | 5.0x | **≥25x** | (combined with Stage 1) |

---

## 1. Phase 2a: Vectorized Edge Enumeration

### 1.1 Current Python Code Path

```python
# Current: nested Python loop, 7.3s at res=256
for idx in cube_map:                      # O(N_cubes)
    for axis, a, b, c in local_edges:     # O(12)
        if axis == 'X': bounds_check; neighbors = ...
        elif axis == 'Y': ...
        else: ...
        active = [n for n in neighbors if n in cube_set]
        if not active or min(active) != idx: continue
        grid, count = get_grid_input(neighbors)
        if count >= 2:
            edge_tasks.append(grid)
```

This runs 68K × 12 = 826K Python iterations, producing 264K owned edges. Each iteration does: bounds check, neighbor enumeration, set lookups, min/sort operation, dict lookups.

### 1.2 Torch Strategy

Phase 0 already implemented the tensor-based edge enumeration functions:
- `compute_global_edge_keys(cube_indices, resolution)` → `(keys, cube_ids, local_ids)`
- `enumerate_unique_edges(keys)` → `(unique_keys, edge_id_per_entry)`
- `build_edge_neighbor_table(...)` → `EdgeNeighborTable` (has a Python loop bug — needs fixing)
- `compute_edge_ownership(table, cube_indices)` → `(E,) owner_cube_id`

**Action**: Fix the `build_edge_neighbor_table` Python loop (line 301-327 in s8_collapse.py), then wire the whole chain into `process_shared_edges_batch`.

### 1.3 Required Fix: build_edge_neighbor_table

Current implementation has a Python for-loop over E unique edges (~264K iterations at res=256). Replace with pure tensor ops:

```python
def build_edge_neighbor_table(edge_id_per_entry, cube_ids, local_ids, num_unique_edges):
    """Fully vectorized — no Python loops."""
    device = edge_id_per_entry.device
    E = num_unique_edges
    M = edge_id_per_entry.shape[0]
    
    # Sort entries by edge_id to group them
    sort_idx = torch.argsort(edge_id_per_entry, stable=True)
    sorted_edge_ids = edge_id_per_entry[sort_idx]
    sorted_cube_ids = cube_ids[sort_idx]
    sorted_local_ids = local_ids[sort_idx]
    
    # Count entries per edge (scatter_add)
    neighbor_counts = torch.zeros(E, dtype=torch.int32, device=device)
    ones = torch.ones(M, dtype=torch.int32, device=device)
    neighbor_counts.scatter_add_(0, sorted_edge_ids.to(torch.int64), ones)
    
    # Compute slot index within each group (cumulative count - prior count)
    # For entries sorted by edge_id, the slot index is the position within the run
    # Use torch.arange and segment info to compute this:
    offsets = torch.zeros(E+1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(neighbor_counts.to(torch.int64), dim=0)
    
    # For each sorted entry, compute its position within its group:
    # group_start = offsets[edge_id], slot = entry_pos_in_sorted - group_start
    entry_pos = torch.arange(M, dtype=torch.int64, device=device)
    slot_per_entry = entry_pos - offsets[sorted_edge_ids.to(torch.int64)]
    # Clamp to [0, 3] (max 4 slots) — entries beyond slot 3 are ignored
    valid = slot_per_entry < 4
    
    # Scatter into (E, 4) tables
    neighbor_cube_ids = torch.full((E, 4), -1, dtype=torch.int32, device=device)
    neighbor_local_edges = torch.full((E, 4), -1, dtype=torch.int32, device=device)
    
    # flat_idx = edge_id * 4 + slot
    v_edge = sorted_edge_ids[valid].to(torch.int64)
    v_slot = slot_per_entry[valid]
    flat_idx = v_edge * 4 + v_slot
    
    neighbor_cube_ids.view(-1).scatter_(0, flat_idx, sorted_cube_ids[valid])
    neighbor_local_edges.view(-1).scatter_(0, flat_idx, sorted_local_ids[valid])
    
    # Edge axes: derive from the first local_edge of each group via edge offset table
    # First entry per edge is at position offsets[e]
    first_entry_pos = offsets[:-1].clamp(max=M-1)
    first_local_edges = sorted_local_ids[first_entry_pos]
    edge_axes = _EDGE_OFFSET_TABLE.to(device)[first_local_edges.to(torch.int64), 0].to(torch.int32)
    
    # Neighbor positions: lookup via _REVERSE_LOCAL_EDGE
    # For each (edge, slot), position = index where _REVERSE_LOCAL_EDGE[axis] == local_edge
    # Vectorized: for each neighbor entry, compute (axis, local_edge) → position
    rev = _REVERSE_LOCAL_EDGE.to(device)  # (3, 4)
    # Expand to (E, 4): axis broadcast across slots
    axis_per_slot = edge_axes.unsqueeze(1).expand(E, 4).to(torch.int64)  # (E, 4)
    local_per_slot = neighbor_local_edges.to(torch.int64)  # (E, 4)
    # For each (e, s), find position p such that rev[axis_per_slot[e,s], p] == local_per_slot[e,s]
    # Use broadcasting: (E, 4, 4) match table
    candidates = rev[axis_per_slot]  # (E, 4, 4)
    matches = (candidates == local_per_slot.unsqueeze(2))  # (E, 4, 4)
    positions = matches.to(torch.int32).argmax(dim=2)  # (E, 4)
    # Mask out invalid slots (where neighbor_local_edges == -1)
    positions = torch.where(neighbor_local_edges >= 0, positions,
                            torch.full_like(positions, -1))
    
    return EdgeNeighborTable(
        neighbor_counts=neighbor_counts,
        neighbor_cube_ids=neighbor_cube_ids,
        neighbor_positions=positions,
        neighbor_local_edges=neighbor_local_edges,
        edge_axes=edge_axes,
    )
```

### 1.4 New `process_shared_edges_batch` Fast Path

```python
def process_shared_edges_batch(resolution, cube_data_list, merge_decimals=5, num_workers=0):
    # Step A: Build tensor representation of cube_data_list
    tensors = _cube_data_to_tensors(cube_data_list)  # New function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cube_indices = tensors.cube_indices.to(device)  # (N, 3)
    
    # Step B: Enumerate owned edges (VECTORIZED)
    keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
    unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
    num_unique = unique_keys.shape[0]
    table = build_edge_neighbor_table(edge_id_per_entry, cube_ids, local_ids, num_unique)
    owner = compute_edge_ownership(table, cube_indices)  # (E,)
    
    # Step C: Filter to edges where this cube owns (no action needed — all edges have an owner)
    # Filter to edges with count >= 2 (at least 2 neighbors required)
    keep = table.neighbor_counts >= 2
    
    # Step D: Geometry processing — Phase 2b
    owned_table = filter_table(table, keep)
    tri_verts = process_geometry_vectorized(owned_table, tensors, device)  # (T*3, 3) float64
    
    # Step E: Welding (unchanged)
    return _weld_and_dedup(tri_verts.cpu().numpy(), merge_decimals, device=device)
```

### 1.5 Expected Speedup

- Edge enumeration: 7.3s → <0.3s (tensor ops on GPU) → **>20x**
- No serialization overhead (all tensors stay on GPU)

---

## 2. Phase 2b: Vectorized Geometry Processing

### 2.1 Current Python Logic (per edge)

For each owned edge with 2-4 neighbors:
1. Deduce edge axis from min/max of cube indices
2. For each neighbor cube: extract local edge, compute UV, extract loops
3. For each loop edge crossing local edge: extract rank, normalize
4. Group points by normalized rank into `points_by_rank[rank][uv] = point`
5. Detect conditional promotion case (all 4 cubes have components but none connect)
6. Distribute exception points as wildcards
7. For each full rank group (4 UVs), emit 4 fan triangles around mean projection

### 2.2 Tensor Representation Needed

```python
@dataclass
class CubeDataTensors:
    # Shape: N = number of cubes
    cube_indices: torch.Tensor       # (N, 3) int32
    cube_edge_weights: torch.Tensor  # (N, 18) int32
    cube_exception: torch.Tensor     # (N,) bool
    cube_num_components: torch.Tensor # (N,) int32
    
    # Per-cube loops (CSR): each cube has L_i loops
    loop_cube_offsets: torch.Tensor  # (N+1,) int32
    
    # Per-loop data (total L loops):
    loop_num_edges: torch.Tensor     # (L,) int32 — length of each loop
    loop_component_point: torch.Tensor  # (L, 3) float32
    
    # Per-loop edges+ranks (padded to max_loop_len): each loop has up to K edges
    # Flat layout: for loop l, edges are loop_edges_flat[l*K:(l+1)*K]
    max_loop_len: int
    loop_edges_flat: torch.Tensor    # (L * K,) int32 — edge index (0..17), -1 padding
    loop_ranks_flat: torch.Tensor    # (L * K,) int32 — rank, -1 padding
```

Conversion function `_cube_data_to_tensors(cube_data_list)`:
- Runs once in Python (~0.3-0.5s amortized)
- Iterates cube_data_list, collects loop data into flat arrays
- Handles padding for variable-length loops

### 2.3 Vectorized Algorithm

Given `E` owned edges and their neighbor tables:

```python
def process_geometry_vectorized(owned_table, tensors, device) -> torch.Tensor:
    """
    Returns (T*3, 3) float64 tensor of triangle vertex coordinates.
    """
    E = owned_table.neighbor_counts.shape[0]
    MAX_RANK = 8  # bounded by max edge weight observed in practice
    
    # --- Step 1: Gather neighbor data per edge ---
    # For each edge, we need to know: cube_idx[4], local_edge[4], uv[4], axis
    n_cube = owned_table.neighbor_cube_ids  # (E, 4) — -1 for absent
    n_loc = owned_table.neighbor_local_edges  # (E, 4)
    n_pos = owned_table.neighbor_positions  # (E, 4) — 0..3 or -1
    edge_axes = owned_table.edge_axes  # (E,)
    valid_slot = n_cube >= 0  # (E, 4)
    
    # UV position from neighbor_position and axis (lookup table)
    # _UV_TABLE[axis, position] -> (u, v) in 0..1
    # Derived from the original get_local_edge mapping
    uv_table = _UV_FROM_AXIS_POS.to(device)  # (3, 4, 2) int32
    # (E, 4, 2) UVs
    uvs = uv_table[edge_axes.unsqueeze(1).expand(E, 4).to(torch.int64), n_pos.clamp(min=0).to(torch.int64)]
    # But we need dx/dy/dz based on neighbor position — actually UV is just (slot → 2D grid coord)
    # Standard mapping: slot 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1) in cyclic order  
    # Simpler: uv index = position (0..3), directly maps to standard 4 corners
    
    # --- Step 2: For each edge, find crossings at its 4 neighbor cubes ---
    # This requires iterating per-cube over loops and checking edge membership.
    # We use gather-based approach:
    # 1. For each (edge, slot) pair, find the cube and its loop range
    # 2. For each loop edge, check if it matches the local_edge
    # 3. Collect (rank, point) for matches
    
    # Gather cube loop ranges
    n_cube_long = n_cube.clamp(min=0).to(torch.int64)  # (E, 4)
    cube_loop_start = tensors.loop_cube_offsets[n_cube_long]  # (E, 4)
    cube_loop_end = tensors.loop_cube_offsets[n_cube_long + 1]  # (E, 4)
    cube_num_loops = cube_loop_end - cube_loop_start  # (E, 4)
    max_loops = int(cube_num_loops.max().item()) if E > 0 else 0
    
    # Flat loop indices: (E, 4, max_loops) — -1 where out of range
    loop_offset = torch.arange(max_loops, device=device).view(1, 1, max_loops)
    flat_loop_idx = cube_loop_start.unsqueeze(2) + loop_offset  # (E, 4, max_loops)
    in_range = loop_offset < cube_num_loops.unsqueeze(2)
    flat_loop_idx = torch.where(in_range, flat_loop_idx, torch.full_like(flat_loop_idx, -1))
    
    # For each of these loops, check if any edge matches n_loc[e, slot]
    # Load loop_edges_flat for each loop: (E, 4, max_loops, K)
    K = tensors.max_loop_len
    loop_edges_gathered = tensors.loop_edges_flat.view(-1, K)[flat_loop_idx.clamp(min=0)]
    # Shape: (E, 4, max_loops, K)
    
    # Check match against local edge
    target_local = n_loc.unsqueeze(2).unsqueeze(3)  # (E, 4, 1, 1)
    match_mask = (loop_edges_gathered == target_local) & in_range.unsqueeze(3)
    # match_mask: (E, 4, max_loops, K) — True where edge matches
    
    # For each (e, slot), find the FIRST matching (loop, edge_idx_in_loop)
    # A loop crosses the edge at most once per rank, so this is well-defined
    # Actually a loop can have multiple crossings (rank 0, 1, 2, ...) - we need ALL
    
    # Compute rank for matches via _loop_ranks_flat gather
    loop_ranks_gathered = tensors.loop_ranks_flat.view(-1, K)[flat_loop_idx.clamp(min=0)]
    # (E, 4, max_loops, K)
    
    # Component points per matched loop
    # Loop's component_point: (E, 4, max_loops, 3)
    loop_points_gathered = tensors.loop_component_point[flat_loop_idx.clamp(min=0)]
    # Zero out invalid positions
    loop_points_gathered = torch.where(
        in_range.unsqueeze(3), loop_points_gathered, torch.zeros_like(loop_points_gathered)
    )
    
    # --- Step 3: Rank normalization ---
    # For edges 2, 6, 3, 7: normalized_rank = W - 1 - rank
    # W = cube_edge_weights[cube, local_edge]
    cube_w = tensors.cube_edge_weights[n_cube_long]  # (E, 4, 18)
    W_at_edge = torch.gather(cube_w, 2, n_loc.clamp(min=0).unsqueeze(2).to(torch.int64)).squeeze(2)
    # (E, 4)
    
    is_negative_edge = (n_loc == 2) | (n_loc == 6) | (n_loc == 3) | (n_loc == 7)  # (E, 4)
    # Broadcast W_at_edge across (max_loops, K)
    W_broadcast = W_at_edge.unsqueeze(2).unsqueeze(3)  # (E, 4, 1, 1)
    raw_ranks = loop_ranks_gathered  # (E, 4, max_loops, K)
    normalized_ranks = torch.where(
        is_negative_edge.unsqueeze(2).unsqueeze(3),
        W_broadcast - 1 - raw_ranks,
        raw_ranks
    )
    normalized_ranks = torch.where(match_mask, normalized_ranks, torch.full_like(normalized_ranks, -1))
    
    # --- Step 4: Group points by (edge, rank, slot) ---
    # Build a dense (E, MAX_RANK, 4, 3) tensor of points + (E, MAX_RANK, 4) presence mask
    # Use scatter_add/scatter for this.
    
    points_by_rank = torch.zeros((E, MAX_RANK, 4, 3), dtype=torch.float32, device=device)
    has_point = torch.zeros((E, MAX_RANK, 4), dtype=torch.bool, device=device)
    
    # For each match (e, slot, loop_idx, edge_in_loop_idx):
    #   rank = normalized_ranks[e, slot, loop_idx, edge_in_loop_idx]
    #   point = loop_points_gathered[e, slot, loop_idx]  (per-loop, not per-edge-in-loop)
    #   points_by_rank[e, rank, slot] = point
    
    # Flatten to (E*4*max_loops*K,) and scatter
    # ... (details in implementation)
    
    # --- Step 5: Handle exceptions ---
    # For each exception cube, use its first loop's component_point as wildcard
    cube_exc = tensors.cube_exception[n_cube_long]  # (E, 4)
    # First loop point per neighbor cube
    exc_point_idx = cube_loop_start  # (E, 4) — index of first loop
    exc_points = tensors.loop_component_point[exc_point_idx.clamp(min=0)]  # (E, 4, 3)
    # Mark exception slots
    exc_valid = cube_exc & valid_slot  # (E, 4)
    
    # For each rank with any presence, fill missing slots with exception point
    any_point_per_rank = has_point.any(dim=2)  # (E, MAX_RANK)
    # For each (e, r, slot) where not has_point[e,r,slot] but exc_valid[e, slot]:
    # points_by_rank[e, r, slot] = exc_points[e, slot]; has_point = True
    fill_mask = (~has_point) & exc_valid.unsqueeze(1) & any_point_per_rank.unsqueeze(2)
    points_by_rank = torch.where(
        fill_mask.unsqueeze(3),
        exc_points.unsqueeze(1).expand(E, MAX_RANK, 4, 3),
        points_by_rank
    )
    has_point = has_point | fill_mask
    
    # If no rank has any points and we have exceptions, create a synthetic rank 0
    no_rank = ~any_point_per_rank.any(dim=1)  # (E,)
    # Details... (handled separately)
    
    # --- Step 6: Emit fan triangles ---
    # Full rank groups (has_point.all(dim=2) == True, i.e., all 4 slots present)
    full_mask = has_point.all(dim=2)  # (E, MAX_RANK)
    
    # Compute projection point = mean of 4 slots per full (edge, rank)
    # For partial ranks (1-3 slots), compute mean of present slots (for emitting but not fans)
    # Actually: only full ranks emit triangles; partial ranks just contribute a projection vertex
    
    proj_pts = points_by_rank.mean(dim=2)  # (E, MAX_RANK, 3)
    # But partial ranks need masked mean. For simplicity, only emit triangles for full ranks.
    
    # Gather full (edge, rank) pairs
    full_edges, full_ranks = torch.nonzero(full_mask, as_tuple=True)  # (F,)
    num_full = full_edges.shape[0]
    
    proj = proj_pts[full_edges, full_ranks]  # (F, 3)
    p0 = points_by_rank[full_edges, full_ranks, 0]  # (F, 3)
    p1 = points_by_rank[full_edges, full_ranks, 1]
    p2 = points_by_rank[full_edges, full_ranks, 2]
    p3 = points_by_rank[full_edges, full_ranks, 3]
    
    # 4 fan triangles per full rank:
    #   (proj, p0, p1), (proj, p1, p2), (proj, p2, p3), (proj, p3, p0)
    tris = torch.stack([
        torch.stack([proj, p0, p1], dim=1),
        torch.stack([proj, p1, p2], dim=1),
        torch.stack([proj, p2, p3], dim=1),
        torch.stack([proj, p3, p0], dim=1),
    ], dim=1)  # (F, 4, 3, 3)
    
    tri_verts_flat = tris.reshape(-1, 3)  # (F*12, 3) — each 3 rows = 1 triangle
    return tri_verts_flat.to(torch.float64)
```

### 2.4 Handling of Edge Cases

**Conditional promotion** (the most complex custom/ logic):
- Detected when: all 4 cubes have components + shared_edge_weight > 0 + all rank groups have <4 points
- Action: promote cubes to exceptions, redistribute their points
- **Frequency**: Very rare in typical meshes (<1% of edges at res=256)
- **Strategy**: Handle conditional promotion in Python on a small residual set. Identify edges that MIGHT need promotion by checking: `full_mask.any(dim=1) == False AND (table.neighbor_counts == 4)`. Run Python fallback on those.

**Rank-0 exception fallback**:
- When no loops cross any cube (no ranks present) but exception points exist
- Action: create synthetic rank 0 with just the exception points
- **Frequency**: More common (~5-10% of edges)
- **Strategy**: Vectorizable — detect with `~any_point_per_rank.any(dim=1) AND exc_valid.any(dim=1)`, then set up rank 0 with exception points as the only source

### 2.5 Expected Speedup

- Pure Torch batch: 3.85s → <0.5s → **~8x**
- Combined with edge enum: 11.15s → <0.8s → **~14x on s8**
- Combined with Stage 1: 5.0x × 14/11.6 ≈ **30x vs custom/** (s8 alone)

---

## 3. Implementation Plan (High-Level)

### Phase 2a-1: Fix `build_edge_neighbor_table` vectorization (SMALL, ~1 day)
- Replace Python loop with scatter-based approach
- Preserve existing unit test API
- Verify via existing tests

### Phase 2a-2: Implement `_cube_data_to_tensors` (MEDIUM, ~1 day)
- New function that converts list-of-dict → CubeDataTensors
- Handles variable-length loops with padding
- Unit tests on synthetic + real data

### Phase 2a-3: Wire Torch edge enumeration into `process_shared_edges_batch` (MEDIUM)
- Add a `use_torch_edges: bool = True` flag to toggle
- Existing Python path kept as fallback + validation
- A/B test: vertex/face counts must match exactly

### Phase 2b-1: Implement `process_geometry_vectorized` basic version (LARGE, ~3 days)
- Handle simple case: normal cubes with regular rank crossings
- Skip conditional promotion (use Python fallback for complex edges)
- Output (T*3, 3) tensor ready for welding

### Phase 2b-2: Handle exceptions + rank-0 fallback (MEDIUM)
- Add vectorized exception cube handling
- Add synthetic rank-0 when no crossings but exceptions exist

### Phase 2b-3: Conditional promotion fallback (SMALL)
- Detect candidate edges needing promotion
- Run Python path on residual set only
- Merge results

### Phase 2c: Integration + Benchmark (SMALL)
- Replace MP path with vectorized path as default
- Benchmark at res=128, 256, 512
- Confirm ≥25x total speedup vs custom/

---

## 4. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Rank normalization bugs | Medium | High | Compare intermediate tensors against Python path |
| Exception handling edge cases missed | Medium | Medium | Use Python fallback for any mismatch |
| Memory explosion for (E, MAX_RANK, 4, 3) | Low | Medium | At res=256 E=264K, MAX_RANK=8 → 100MB — fine |
| Loop padding wasting memory | Low | Low | Loop lengths are bounded (K ≤ 18 in practice) |
| Floating point mismatches from GPU parallel sum | Low | Medium | Match custom/ exactly by using same ordering (stable sort) |

---

## 5. Exit Criteria

All four required:
1. **Correctness**: A/B tests pass (exact vertex/face count match) at res=128, 256
2. **Performance**: Total s8 ≤ 3.0s at res=256 on H100 (vs 13.9s current)
3. **Regression**: All 121 existing tests pass
4. **Clean integration**: New Torch path is default; Python path kept for validation
